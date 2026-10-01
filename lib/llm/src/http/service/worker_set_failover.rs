// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Illustrative, pre-dispatch revision failover for HTTP chat completions. A zero-worker
//! selection failure may restart raw-request preprocessing once in another revision. Streams,
//! transport errors, overload, and explicit worker targets are never replayed here.

use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context, ManyOut};

use crate::discovery::worker_set_selection::WorkerSetAffinity;
use crate::discovery::{ChatEngineSelection, ModelManagerError, worker_set_selection};
use crate::protocols::openai::{
    ParsingOptions,
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
};
use crate::types::Annotated;

type ChatStream = ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>;

#[derive(Debug)]
pub(super) enum DispatchError {
    Selection(ModelManagerError),
    Policy(anyhow::Error),
    Generate(anyhow::Error),
}

fn allows_revision_retry(request: &NvCreateChatCompletionRequest) -> bool {
    request.nvext.as_ref().is_none_or(|ext| {
        ext.backend_instance_id.is_none()
            && ext.prefill_worker_id.is_none()
            && ext.decode_worker_id.is_none()
            && ext.dp_rank.is_none()
            && ext.prefill_dp_rank.is_none()
    })
}

fn no_endpoints(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<KvSchedulerError>(),
            Some(KvSchedulerError::NoEndpoints)
        )
    })
}

/// Keep revision selection outside the per-revision engine's migration operator. Returning the
/// successful revision's parsing options is essential: revisions can advertise different parsers.
pub(super) async fn generate_chat(
    request: Context<NvCreateChatCompletionRequest>,
    affinity: Option<WorkerSetAffinity>,
    mut select: impl FnMut() -> Result<ChatEngineSelection, ModelManagerError>,
) -> Result<(ChatStream, ParsingOptions, bool), DispatchError> {
    let select_revision = |excluded: Option<&str>, select: &mut _| {
        worker_set_selection::with_affinity(affinity.clone(), || {
            worker_set_selection::excluding_namespace(excluded, select)
        })
    };
    let mut selection = select_revision(None, &mut select).map_err(DispatchError::Selection)?;
    let can_retry = affinity.is_some() && allows_revision_retry(&request);
    // Only the opt-in affinity path retains a raw request. Unique context state disables retry
    // rather than silently dropping ownership-bearing values from the replacement attempt.
    let attempt = can_retry
        .then(|| request.fork_shared(request.content().clone()))
        .flatten();
    let (parsing, can_defer) = response_options(selection.parsing, &request)?;
    let Some(attempt) = attempt else {
        return selection
            .engine
            .generate(request)
            .await
            .map(|stream| (stream, parsing, can_defer))
            .map_err(DispatchError::Generate);
    };
    let attempt_context = attempt.context();
    match selection.engine.generate(attempt).await {
        Ok(stream) => Ok((stream, parsing, can_defer)),
        Err(error) => {
            // NoEndpoints is a selection failure, not evidence that a dispatched request failed.
            // Require both signals so a nonempty revision's policy failures cannot escape it.
            if !no_endpoints(&error)
                || selection.worker_set.worker_count() != 0
                || request.context().is_stopped()
                || request.context().is_killed()
            {
                return Err(DispatchError::Generate(error));
            }
            attempt_context.stop();
            let retired = selection.worker_set.namespace().to_owned();
            selection = match select_revision(Some(&retired), &mut select) {
                Ok(selection) => selection,
                // Preserve the original failure when no replacement can serve the request.
                Err(_) => return Err(DispatchError::Generate(error)),
            };
            if request.context().is_stopped() || request.context().is_killed() {
                return Err(DispatchError::Generate(error));
            }
            tracing::debug!(from_namespace = %retired,
                to_namespace = %selection.worker_set.namespace(), "Reselecting retired revision before dispatch");
            let (parsing, can_defer) = response_options(selection.parsing, &request)?;
            // Move the original context into the final attempt, retaining its cancellation,
            // session identity, raw request, lifecycle metadata, and all request constraints.
            selection
                .engine
                .generate(request)
                .await
                .map(|stream| (stream, parsing, can_defer))
                .map_err(DispatchError::Generate)
        }
    }
}

fn response_options(
    parsing: ParsingOptions,
    request: &NvCreateChatCompletionRequest,
) -> Result<(ParsingOptions, bool), DispatchError> {
    use crate::preprocessor::OpenAIPreprocessor;
    let parsing = super::apply_request_tool_call_parsing_options(parsing, request)
        .map_err(|error| DispatchError::Policy(error.into()))?
        .with_parallel_tool_calls(request.inner.parallel_tool_calls)
        .with_move_reasoning_to_content_when_empty(
            OpenAIPreprocessor::wants_reasoning_as_content_when_empty(
                request.chat_template_args.as_ref(),
            ),
        );
    let can_defer = OpenAIPreprocessor::stream_can_defer_all_output(
        parsing.tool_call_parser.as_deref(),
        parsing.reasoning_parser.as_deref(),
        request.chat_template_args.as_ref(),
    );
    Ok((parsing, can_defer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{Model, WorkerSet};
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId};
    use async_trait::async_trait;
    use dynamo_runtime::engine::AsyncEngine;
    use dynamo_runtime::pipeline::{AsyncEngineContext, ResponseStream};
    use futures::{StreamExt, stream};
    use std::sync::{Arc, Mutex};
    use tokio::sync::watch;

    #[derive(Clone, Copy)]
    enum Behavior {
        Success,
        Retire,
        Overload,
        Transport,
        StreamError,
        Cancel,
        LiveNoEndpoints,
    }

    struct Engine {
        name: &'static str,
        behavior: Behavior,
        workers: watch::Sender<Vec<u64>>,
        seen: Arc<Mutex<Vec<String>>>,
        parent: Arc<dyn AsyncEngineContext>,
    }

    #[async_trait]
    impl AsyncEngine<Context<NvCreateChatCompletionRequest>, ChatStream, anyhow::Error> for Engine {
        async fn generate(
            &self,
            request: Context<NvCreateChatCompletionRequest>,
        ) -> anyhow::Result<ChatStream> {
            self.seen.lock().unwrap().push(self.name.into());
            assert_eq!(request.id(), "original-id");
            assert_eq!(
                request.metadata().get("tenant").map(String::as_str),
                Some("tenant-a")
            );
            assert_eq!(
                request
                    .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                    .unwrap()
                    .as_str(),
                "session-a"
            );
            assert_eq!(
                *request.get::<bool>("preserved-future-context-key").unwrap(),
                true
            );
            assert_eq!(request.inner.messages.len(), 1);
            match self.behavior {
                Behavior::Retire | Behavior::Cancel => {
                    self.workers.send_replace(vec![]);
                    if matches!(self.behavior, Behavior::Cancel) {
                        self.parent.stop();
                    }
                    Err(KvSchedulerError::NoEndpoints.into())
                }
                Behavior::Overload => {
                    self.workers.send_replace(vec![]);
                    Err(KvSchedulerError::AllEligibleWorkersOverloaded.into())
                }
                Behavior::Transport => {
                    self.workers.send_replace(vec![]);
                    Err(dynamo_runtime::error::DynamoError::builder()
                        .error_type(dynamo_runtime::error::ErrorType::CannotConnect)
                        .message("test transport error")
                        .build()
                        .into())
                }
                Behavior::LiveNoEndpoints => Err(KvSchedulerError::NoEndpoints.into()),
                Behavior::StreamError => {
                    self.workers.send_replace(vec![]);
                    Ok(ResponseStream::new(
                        Box::pin(stream::iter(vec![Annotated::from_error("late failure")])),
                        request.context(),
                    ))
                }
                Behavior::Success => Ok(ResponseStream::new(
                    Box::pin(stream::empty()),
                    request.context(),
                )),
            }
        }
    }

    fn request() -> Context<NvCreateChatCompletionRequest> {
        let body: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4
        }))
        .unwrap();
        let mut request = Context::with_id_and_metadata(
            body,
            "original-id".into(),
            std::collections::BTreeMap::from([("tenant".into(), "tenant-a".into())]),
        );
        request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("session-a"),
        );
        request.insert("preserved-future-context-key", true);
        request
    }

    fn model(
        request: &Context<NvCreateChatCompletionRequest>,
        first: Behavior,
        second: Behavior,
    ) -> (Model, Arc<Mutex<Vec<String>>>) {
        let model = Model::new("m".into());
        let seen = Arc::new(Mutex::new(Vec::new()));
        for (name, behavior, parser) in [("old", first, "deepseek_r1"), ("new", second, "qwen3")] {
            let mut card = ModelDeploymentCard::default();
            card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
            card.runtime_config.reasoning_parser = Some(parser.into());
            let (workers, rx) = watch::channel(vec![1]);
            let mut ws = WorkerSet::new(name.into(), name.into(), card);
            ws.set_instance_watcher(rx);
            ws.chat_engine = Some(Arc::new(Engine {
                name,
                behavior,
                workers,
                seen: seen.clone(),
                parent: request.context(),
            }));
            model.add_worker_set(name.into(), Arc::new(ws));
        }
        (model, seen)
    }

    fn affinity() -> Option<WorkerSetAffinity> {
        Some(WorkerSetAffinity {
            key: Some("session-a".into()),
            pinned_namespace: Some("old".into()),
        })
    }

    #[tokio::test]
    async fn last_worker_retires_after_selection_and_chat_reselects_with_matching_parser() {
        let request = request();
        let (model, seen) = model(&request, Behavior::Retire, Behavior::Success);
        let (_, parsing, _) =
            generate_chat(request, affinity(), || model.get_chat_engine_selection())
                .await
                .unwrap();
        assert_eq!(*seen.lock().unwrap(), ["old", "new"]);
        assert_eq!(parsing.reasoning_parser.as_deref(), Some("qwen3"));
    }

    #[tokio::test]
    async fn failover_is_bounded_and_never_replays_other_failures() {
        for (first, second, expected) in [
            (Behavior::Retire, Behavior::Retire, vec!["old", "new"]),
            (Behavior::Overload, Behavior::Success, vec!["old"]),
            (Behavior::Transport, Behavior::Success, vec!["old"]),
            (Behavior::LiveNoEndpoints, Behavior::Success, vec!["old"]),
            (Behavior::Cancel, Behavior::Success, vec!["old"]),
        ] {
            let request = request();
            let (model, seen) = model(&request, first, second);
            assert!(
                generate_chat(request, affinity(), || model.get_chat_engine_selection())
                    .await
                    .is_err()
            );
            assert_eq!(*seen.lock().unwrap(), expected);
        }
    }

    #[tokio::test]
    async fn stream_errors_are_returned_without_revision_replay() {
        let request = request();
        let (model, seen) = model(&request, Behavior::StreamError, Behavior::Success);
        let (mut stream, _, _) =
            generate_chat(request, affinity(), || model.get_chat_engine_selection())
                .await
                .unwrap();
        assert!(stream.next().await.unwrap().error.is_some());
        assert_eq!(*seen.lock().unwrap(), ["old"]);
    }

    #[tokio::test]
    async fn explicit_worker_targets_and_unique_state_disable_replay() {
        for field in [
            "backend_instance_id",
            "prefill_worker_id",
            "decode_worker_id",
            "dp_rank",
            "prefill_dp_rank",
            "unique",
        ] {
            let mut request = request();
            if field == "unique" {
                request.insert_unique("owned", 7_u32);
            } else {
                request.nvext =
                    Some(serde_json::from_value(serde_json::json!({field: 0})).unwrap());
            }
            let (model, seen) = model(&request, Behavior::Retire, Behavior::Success);
            assert!(
                generate_chat(request, affinity(), || model.get_chat_engine_selection())
                    .await
                    .is_err()
            );
            assert_eq!(*seen.lock().unwrap(), ["old"], "{field}");
        }
    }

    #[tokio::test]
    async fn no_replacement_preserves_original_error_and_disabled_mode_does_not_retry() {
        for enabled in [true, false] {
            let request = request();
            let (model, seen) = model(&request, Behavior::Retire, Behavior::Success);
            if enabled {
                model.remove_worker_set("new");
            }
            let result = generate_chat(request, enabled.then(|| affinity().unwrap()), || {
                // Keep the initial choice deterministic even when the feature is disabled.
                worker_set_selection::with_affinity(affinity(), || {
                    model.get_chat_engine_selection()
                })
            })
            .await;
            assert!(
                matches!(result, Err(DispatchError::Generate(ref error)) if no_endpoints(error))
            );
            assert_eq!(*seen.lock().unwrap(), ["old"]);
        }
    }
}
