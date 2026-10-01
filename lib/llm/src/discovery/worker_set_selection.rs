// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-set selection across Dynamo namespaces that serve the same model.
//!
//! During a rollout several namespaces (revisions) serve a model side by side. With
//! `DYN_WORKER_SET_SELECTION=affinity_rendezvous`, every frontend replica sends a request carrying an
//! affinity key to the same namespace (weighted rendezvous hashing, so shifting weights only moves the
//! keys that must move), and a request the gateway already pinned to this frontend's revision stays in
//! the frontend's own namespace.

use std::sync::OnceLock;

use axum::http::HeaderMap;
use dynamo_protocols::types::ChatCompletionRequestMessage;
use dynamo_protocols::types::anthropic::{AnthropicMessage, AnthropicRole, SystemContent};
use dynamo_runtime::pipeline::Context;
use serde::Serialize;

use crate::protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId};

pub const WORKER_SET_SELECTION_ENV: &str = "DYN_WORKER_SET_SELECTION";
pub const WORKER_SET_PIN_HEADER_ENV: &str = "DYN_WORKER_SET_PIN_HEADER";
const NAMESPACE_ENV: &str = "DYN_NAMESPACE";

pub const WORKER_SET_PINNED_CONTEXT_KEY: &str = "dynamo.llm.worker_set_pinned";

/// Upper bound on the serialized conversation start hashed into a derived key.
const DERIVED_KEY_MAX_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectionMode {
    #[default]
    WeightedRandom,
    AffinityRendezvous,
}

impl SelectionMode {
    fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some("affinity_rendezvous") => Self::AffinityRendezvous,
            None | Some("") | Some("weighted_random") => Self::WeightedRandom,
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "unknown {WORKER_SET_SELECTION_ENV}, using weighted_random"
                );
                Self::WeightedRandom
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SelectionConfig {
    pub mode: SelectionMode,
    pub own_namespace: Option<String>,
    pub pin_header: Option<String>,
}

impl SelectionConfig {
    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let non_empty = |name: &str| {
            lookup(name)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        Self {
            mode: SelectionMode::parse(non_empty(WORKER_SET_SELECTION_ENV).as_deref()),
            own_namespace: non_empty(NAMESPACE_ENV),
            pin_header: non_empty(WORKER_SET_PIN_HEADER_ENV)
                .map(|header| header.to_ascii_lowercase()),
        }
    }

    pub fn global() -> &'static SelectionConfig {
        static CONFIG: OnceLock<SelectionConfig> = OnceLock::new();
        CONFIG.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }

    /// Whether the gateway pinned this request to the frontend's revision.
    pub fn is_pinned(&self, headers: &HeaderMap) -> bool {
        self.mode == SelectionMode::AffinityRendezvous
            && self
                .pin_header
                .as_deref()
                .is_some_and(|header| headers.contains_key(header))
    }
}

/// Resolved worker-set affinity for one request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerSetAffinity {
    pub key: Option<String>,
    pub pinned_namespace: Option<String>,
}

tokio::task_local! {
    static REQUEST_AFFINITY: WorkerSetAffinity;
    static EXCLUDED_NAMESPACE: String;
}

/// A failed revision must not be selected again while discovery converges. This exclusion is
/// request-local and never withdraws a revision for other requests.
pub(crate) fn excluding_namespace<R>(namespace: Option<&str>, f: impl FnOnce() -> R) -> R {
    match namespace {
        Some(namespace) => EXCLUDED_NAMESPACE.sync_scope(namespace.to_owned(), f),
        None => f(),
    }
}

pub(crate) fn namespace_is_excluded(namespace: &str) -> bool {
    EXCLUDED_NAMESPACE
        .try_with(|excluded| excluded == namespace)
        .unwrap_or(false)
}

/// Affinity for `request` from its gateway pin and session id, falling back to `body_key`. `None` when
/// affinity selection is disabled or the request carries nothing to be sticky on.
pub fn request_affinity<T: Send + Sync + 'static>(
    config: &SelectionConfig,
    request: &mut Context<T>,
    body_key: impl FnOnce(&T) -> Option<String>,
) -> Option<WorkerSetAffinity> {
    if config.mode != SelectionMode::AffinityRendezvous {
        return None;
    }
    let pinned = request
        .get::<bool>(WORKER_SET_PINNED_CONTEXT_KEY)
        .is_ok_and(|pinned| *pinned);
    let pinned_namespace = pinned.then(|| config.own_namespace.clone()).flatten();
    let key = request
        .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .ok()
        .map(|session| session.as_str().to_string())
        .or_else(|| body_key(request));
    // Use the same identity for revision selection and the existing worker-router policy.
    if let Some(key) = key.as_ref() {
        request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new(key.clone()),
        );
    }
    (key.is_some() || pinned_namespace.is_some()).then_some(WorkerSetAffinity {
        key,
        pinned_namespace,
    })
}

/// OpenAI body identity, shared by Chat, Completions and Responses. Empty values fall through.
pub fn openai_identity_key(
    prompt_cache_key: Option<&str>,
    safety_identifier: Option<&str>,
    user: Option<&str>,
) -> Option<String> {
    non_empty(prompt_cache_key)
        .or_else(|| non_empty(safety_identifier))
        .or_else(|| non_empty(user))
}

/// Runs `f` (an engine lookup) with `affinity` visible to worker-set selection.
pub fn with_affinity<R>(affinity: Option<WorkerSetAffinity>, f: impl FnOnce() -> R) -> R {
    match affinity {
        Some(affinity) => REQUEST_AFFINITY.sync_scope(affinity, f),
        None => f(),
    }
}

/// Index into `candidates` (namespace, worker count) for the request in scope, or `None` to fall back to
/// weighted random selection.
pub(crate) fn choose_for_current_request(candidates: &[(&str, usize)]) -> Option<usize> {
    REQUEST_AFFINITY
        .try_with(|affinity| choose(affinity, candidates))
        .ok()
        .flatten()
}

pub fn choose(affinity: &WorkerSetAffinity, candidates: &[(&str, usize)]) -> Option<usize> {
    if let Some(own) = affinity.pinned_namespace.as_deref()
        && let Some(index) = candidates
            .iter()
            .position(|(namespace, weight)| *namespace == own && *weight > 0)
    {
        return Some(index);
    }
    let key = affinity.key.as_deref()?;
    candidates
        .iter()
        .enumerate()
        .filter(|(_, (_, weight))| *weight > 0)
        .map(|(index, (namespace, weight))| (index, rendezvous_score(key, namespace, *weight)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

/// Weighted rendezvous score (larger wins): `-w / ln(u)` with `u` uniform in (0, 1) derived from the key
/// and namespace, so each namespace wins a key with probability proportional to its weight.
fn rendezvous_score(key: &str, namespace: &str, weight: usize) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key.as_bytes());
    hasher.update(&[0]);
    hasher.update(namespace.as_bytes());
    let hash = hasher.finalize();
    let bits = u64::from_le_bytes(
        hash.as_bytes()[..8]
            .try_into()
            .expect("blake3 output is 32 bytes"),
    );
    let unit = ((bits >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    -(weight as f64) / unit.ln()
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value.filter(|value| !value.is_empty()).map(str::to_string)
}

fn derived_key<S: Serialize + ?Sized>(conversation_start: &S) -> Option<String> {
    let bytes = serde_json::to_vec(conversation_start).ok()?;
    let start = &bytes[..bytes.len().min(DERIVED_KEY_MAX_BYTES)];
    Some(blake3::hash(start).to_hex().to_string())
}

/// Messages up to and including the first user message, or all of them when there is none.
fn conversation_start<M>(messages: &[M], is_user: impl Fn(&M) -> bool) -> &[M] {
    let end = messages
        .iter()
        .position(is_user)
        .map_or(messages.len(), |index| index + 1);
    &messages[..end]
}

/// Chat affinity key: `user`, else a hash of the conversation up to and including the first user
/// message.
pub fn chat_affinity_key(
    user: Option<&str>,
    messages: &[ChatCompletionRequestMessage],
) -> Option<String> {
    non_empty(user).or_else(|| {
        if messages.is_empty() {
            return None;
        }
        derived_key(conversation_start(messages, |message| {
            matches!(message, ChatCompletionRequestMessage::User(_))
        }))
    })
}

/// Completions fallback: `user`, else the first 64 KiB of the JSON-serialized prompt.
/// Below this cap, appending to the prompt changes the key; callers need an explicit identity
/// for stable affinity across growing completion prompts.
pub fn completion_affinity_key<P: Serialize>(user: Option<&str>, prompt: &P) -> Option<String> {
    non_empty(user).or_else(|| derived_key(prompt))
}

/// Anthropic affinity key: `metadata.user_id`, else a hash of the system prompt and the conversation up
/// to and including the first user message.
pub fn anthropic_affinity_key(
    metadata: Option<&serde_json::Value>,
    system: Option<&SystemContent>,
    messages: &[AnthropicMessage],
) -> Option<String> {
    let user_id = metadata
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(serde_json::Value::as_str);
    non_empty(user_id).or_else(|| {
        if system.is_none() && messages.is_empty() {
            return None;
        }
        let start = conversation_start(messages, |message| message.role == AnthropicRole::User);
        derived_key(&(system.map(|system| system.text.as_str()), start))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::anthropic::AnthropicMessageContent;
    use dynamo_protocols::types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    };

    fn affinity_config() -> SelectionConfig {
        SelectionConfig {
            mode: SelectionMode::AffinityRendezvous,
            own_namespace: Some("inference-rev-1".to_string()),
            pin_header: Some("x-gateway-session-bucket".to_string()),
        }
    }

    fn keyed(key: &str) -> WorkerSetAffinity {
        WorkerSetAffinity {
            key: Some(key.to_string()),
            pinned_namespace: None,
        }
    }

    fn system(text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: ChatCompletionRequestSystemMessageContent::Text(text.to_string()),
            name: None,
            tools: None,
        })
    }

    fn user(text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(text.to_string()),
            name: None,
        })
    }

    fn assistant(text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                text.to_string(),
            )),
            ..Default::default()
        })
    }

    fn anthropic(role: AnthropicRole, text: &str) -> AnthropicMessage {
        AnthropicMessage {
            role,
            content: AnthropicMessageContent::Text {
                content: text.to_string(),
            },
        }
    }

    #[test]
    fn selection_mode_parse() {
        let cases = [
            (None, SelectionMode::WeightedRandom),
            (Some("weighted_random"), SelectionMode::WeightedRandom),
            (
                Some("affinity_rendezvous"),
                SelectionMode::AffinityRendezvous,
            ),
            (
                Some(" affinity_rendezvous "),
                SelectionMode::AffinityRendezvous,
            ),
            (Some("bogus"), SelectionMode::WeightedRandom),
        ];
        for (value, want) in cases {
            assert_eq!(SelectionMode::parse(value), want, "value {value:?}");
        }
    }

    #[test]
    fn selection_config_from_lookup() {
        let config = SelectionConfig::from_lookup(|name| match name {
            WORKER_SET_SELECTION_ENV => Some("affinity_rendezvous".to_string()),
            NAMESPACE_ENV => Some("inference-rev-1".to_string()),
            WORKER_SET_PIN_HEADER_ENV => Some("X-Gateway-Session-Bucket".to_string()),
            _ => None,
        });
        assert_eq!(config, affinity_config());

        let empty_values = SelectionConfig::from_lookup(|_| Some(String::new()));
        assert_eq!(empty_values, SelectionConfig::default());
    }

    #[test]
    fn is_pinned_requires_affinity_mode_and_header() {
        let mut with_header = HeaderMap::new();
        with_header.insert("x-gateway-session-bucket", "42".parse().unwrap());

        assert!(affinity_config().is_pinned(&with_header));
        assert!(!affinity_config().is_pinned(&HeaderMap::new()));

        let weighted = SelectionConfig {
            mode: SelectionMode::WeightedRandom,
            ..affinity_config()
        };
        assert!(!weighted.is_pinned(&with_header));

        let no_pin_header = SelectionConfig {
            pin_header: None,
            ..affinity_config()
        };
        assert!(!no_pin_header.is_pinned(&with_header));
    }

    #[test]
    fn request_affinity_sources() {
        struct Case {
            name: &'static str,
            config: SelectionConfig,
            session: bool,
            pinned: bool,
            body_key: Option<&'static str>,
            want: Option<WorkerSetAffinity>,
        }
        let cases = [
            Case {
                name: "session id wins over body key",
                config: affinity_config(),
                session: true,
                pinned: false,
                body_key: Some("body"),
                want: Some(keyed("session-1")),
            },
            Case {
                name: "body key without session id",
                config: affinity_config(),
                session: false,
                pinned: false,
                body_key: Some("body"),
                want: Some(keyed("body")),
            },
            Case {
                name: "pinned request keeps its key and pins to own namespace",
                config: affinity_config(),
                session: true,
                pinned: true,
                body_key: None,
                want: Some(WorkerSetAffinity {
                    key: Some("session-1".to_string()),
                    pinned_namespace: Some("inference-rev-1".to_string()),
                }),
            },
            Case {
                name: "nothing to be sticky on",
                config: affinity_config(),
                session: false,
                pinned: false,
                body_key: None,
                want: None,
            },
            Case {
                name: "weighted random ignores everything",
                config: SelectionConfig::default(),
                session: true,
                pinned: true,
                body_key: Some("body"),
                want: None,
            },
        ];

        for case in cases {
            let mut ctx = Context::new(());
            if case.session {
                ctx.insert(
                    SESSION_AFFINITY_CONTEXT_KEY,
                    SessionAffinityId::new("session-1"),
                );
            }
            if case.pinned {
                ctx.insert(WORKER_SET_PINNED_CONTEXT_KEY, true);
            }
            let got = request_affinity(&case.config, &mut ctx, |_| {
                case.body_key.map(str::to_string)
            });
            assert_eq!(got, case.want, "{}", case.name);
            let router_key = ctx
                .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                .ok()
                .map(|key| key.as_str().to_string());
            let expected = if case.session {
                Some("session-1".to_string())
            } else {
                case.want.as_ref().and_then(|affinity| affinity.key.clone())
            };
            assert_eq!(router_key, expected, "{} router context", case.name);
        }
    }

    #[test]
    fn disabled_affinity_does_not_derive_or_promote_body_key() {
        let mut context = Context::new(());
        assert_eq!(
            request_affinity(&SelectionConfig::default(), &mut context, |_| {
                panic!("disabled affinity must not derive a key")
            }),
            None
        );
        assert!(
            context
                .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                .is_err()
        );
    }

    #[test]
    fn header_identity_skips_body_derivation() {
        let mut context = Context::new(());
        context.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("header"),
        );
        assert_eq!(
            request_affinity(&affinity_config(), &mut context, |_| {
                panic!("header affinity must take precedence")
            }),
            Some(keyed("header"))
        );
    }

    #[test]
    fn pinned_body_identity_survives_context_mapping() {
        let mut context = Context::new(());
        context.insert(WORKER_SET_PINNED_CONTEXT_KEY, true);
        let affinity =
            request_affinity(&affinity_config(), &mut context, |_| Some("cache".into())).unwrap();
        assert_eq!(
            affinity.pinned_namespace.as_deref(),
            Some("inference-rev-1")
        );
        assert_eq!(affinity.key.as_deref(), Some("cache"));
        let mapped = context.map(|_| "converted request");
        assert_eq!(
            mapped
                .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                .unwrap()
                .as_str(),
            "cache"
        );
    }

    #[test]
    fn openai_identity_precedence() {
        for (cache, safety, user, expected) in [
            (Some("cache"), Some("safety"), Some("user"), Some("cache")),
            (Some(""), Some("safety"), Some("user"), Some("safety")),
            (None, Some(""), Some("user"), Some("user")),
            (Some(""), None, Some(""), None),
        ] {
            assert_eq!(
                openai_identity_key(cache, safety, user),
                expected.map(str::to_string)
            );
        }
    }

    #[test]
    fn openai_body_identity_is_supported_and_reaches_router() {
        use crate::protocols::openai::{
            chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
        };
        for (cache, safety, user, expected) in [
            (Some("cache"), Some("safety"), Some("user"), "cache"),
            (Some(""), Some("safety"), Some("user"), "safety"),
            (None, Some(""), Some("user"), "user"),
        ] {
            let chat: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "test", "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": cache, "safety_identifier": safety, "user": user,
            }))
            .unwrap();
            assert!(chat.unsupported_fields.is_empty());
            let completion: NvCreateCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "test", "prompt": "hello",
                "prompt_cache_key": cache, "safety_identifier": safety, "user": user,
            }))
            .unwrap();
            assert!(completion.unsupported_fields.is_empty());
            let body_key = openai_identity_key(
                chat.inner.prompt_cache_key.as_deref(),
                chat.common.safety_identifier.as_deref(),
                chat.inner.user.as_deref(),
            );
            assert_eq!(
                body_key,
                openai_identity_key(
                    completion.prompt_cache_key.as_deref(),
                    completion.common.safety_identifier.as_deref(),
                    completion.inner.user.as_deref()
                )
            );
            let mut context = Context::new(completion);
            let affinity = request_affinity(&affinity_config(), &mut context, |_| body_key);
            assert_eq!(affinity.unwrap().key.as_deref(), Some(expected));
            assert_eq!(
                context
                    .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                    .unwrap()
                    .as_str(),
                expected
            );
        }
    }

    #[test]
    fn with_affinity_scopes_choice_to_closure() {
        let candidates = [("inference-rev-0", 1), ("inference-rev-1", 1)];
        let pinned = WorkerSetAffinity {
            key: None,
            pinned_namespace: Some("inference-rev-1".to_string()),
        };

        assert_eq!(
            with_affinity(Some(pinned), || choose_for_current_request(&candidates)),
            Some(1)
        );
        assert_eq!(
            with_affinity(None, || choose_for_current_request(&candidates)),
            None
        );
        assert_eq!(choose_for_current_request(&candidates), None);
    }

    #[test]
    fn choose_pinned_and_unkeyed() {
        let candidates = [("inference-rev-0", 3), ("inference-rev-1", 1)];
        let cases = [
            (
                "pinned to present own namespace",
                WorkerSetAffinity {
                    key: Some("session-1".to_string()),
                    pinned_namespace: Some("inference-rev-1".to_string()),
                },
                Some(1),
            ),
            ("no key and no pin", WorkerSetAffinity::default(), None),
            (
                "pinned to absent namespace without key",
                WorkerSetAffinity {
                    key: None,
                    pinned_namespace: Some("inference-rev-9".to_string()),
                },
                None,
            ),
        ];
        for (name, affinity, want) in cases {
            assert_eq!(choose(&affinity, &candidates), want, "{name}");
        }
    }

    #[test]
    fn choose_pinned_to_absent_namespace_falls_back_to_key() {
        let candidates = [("inference-rev-0", 1), ("inference-rev-1", 1)];
        let affinity = WorkerSetAffinity {
            key: Some("session-1".to_string()),
            pinned_namespace: Some("inference-rev-9".to_string()),
        };
        assert_eq!(
            choose(&affinity, &candidates),
            choose(&keyed("session-1"), &candidates)
        );
    }

    #[test]
    fn choose_is_independent_of_candidate_order() {
        let forward = [
            ("inference-rev-0", 2),
            ("inference-rev-1", 3),
            ("inference-rev-2", 5),
        ];
        let reversed = [
            ("inference-rev-2", 5),
            ("inference-rev-1", 3),
            ("inference-rev-0", 2),
        ];
        for i in 0..500 {
            let affinity = keyed(&format!("session-{i}"));
            let a = forward[choose(&affinity, &forward).unwrap()].0;
            let b = reversed[choose(&affinity, &reversed).unwrap()].0;
            assert_eq!(a, b, "session-{i}");
        }
    }

    #[test]
    fn choose_spreads_keys_by_weight() {
        let candidates = [("inference-rev-0", 3), ("inference-rev-1", 1)];
        let total = 20_000;
        let on_rev0 = (0..total)
            .filter(|i| choose(&keyed(&format!("session-{i}")), &candidates) == Some(0))
            .count();
        let share = on_rev0 as f64 / total as f64;
        assert!((share - 0.75).abs() < 0.02, "rev-0 share {share}");
    }

    #[test]
    fn choose_moves_keys_only_toward_the_grown_namespace() {
        let before = [("inference-rev-0", 4), ("inference-rev-1", 1)];
        let after = [("inference-rev-0", 4), ("inference-rev-1", 3)];
        let mut moved = 0;
        for i in 0..5_000 {
            let affinity = keyed(&format!("session-{i}"));
            let was = before[choose(&affinity, &before).unwrap()].0;
            let now = after[choose(&affinity, &after).unwrap()].0;
            if was != now {
                assert_eq!(
                    (was, now),
                    ("inference-rev-0", "inference-rev-1"),
                    "session-{i}"
                );
                moved += 1;
            }
        }
        assert!(moved > 0);
    }

    #[test]
    fn choose_skips_zero_weight() {
        let candidates = [("inference-rev-0", 0), ("inference-rev-1", 1)];
        for i in 0..200 {
            assert_eq!(
                choose(&keyed(&format!("session-{i}")), &candidates),
                Some(1)
            );
        }
        assert_eq!(choose(&keyed("session-1"), &[("inference-rev-0", 0)]), None);
    }

    #[test]
    fn chat_affinity_key_prefers_user_then_conversation_start() {
        let first_turn = vec![system("be brief"), user("hello")];
        let later_turn = vec![
            system("be brief"),
            user("hello"),
            assistant("hi"),
            user("more"),
        ];
        let other_conversation = vec![system("be brief"), user("goodbye")];

        assert_eq!(
            chat_affinity_key(Some("user-1"), &first_turn),
            Some("user-1".to_string())
        );

        let first = chat_affinity_key(None, &first_turn);
        assert!(first.is_some());
        assert_eq!(first, chat_affinity_key(None, &later_turn));
        assert_ne!(first, chat_affinity_key(None, &other_conversation));
        assert_eq!(chat_affinity_key(Some(""), &first_turn), first);
        assert_eq!(chat_affinity_key(None, &[]), None);
    }

    #[test]
    fn completion_affinity_key_prefers_user_then_prompt_start() {
        // The cap does not make short growing prompts session-stable.
        assert_ne!(
            completion_affinity_key(None, &"first"),
            completion_affinity_key(None, &"first second")
        );
        let long_prefix = "x".repeat(DERIVED_KEY_MAX_BYTES);
        let turn_one = format!("{long_prefix} first");
        let turn_two = format!("{long_prefix} first second");

        assert_eq!(
            completion_affinity_key(Some("user-1"), &"hi"),
            Some("user-1".to_string())
        );
        let one = completion_affinity_key(None, &turn_one);
        assert!(one.is_some());
        assert_eq!(one, completion_affinity_key(None, &turn_two));
        assert_ne!(
            completion_affinity_key(None, &"a"),
            completion_affinity_key(None, &"b")
        );
    }

    #[test]
    fn anthropic_affinity_key_prefers_user_id_then_conversation_start() {
        let metadata = serde_json::json!({"user_id": "user-1"});
        let first_turn = vec![anthropic(AnthropicRole::User, "hello")];
        let later_turn = vec![
            anthropic(AnthropicRole::User, "hello"),
            anthropic(AnthropicRole::Assistant, "hi"),
            anthropic(AnthropicRole::User, "more"),
        ];
        let system_a = SystemContent {
            text: "be brief".to_string(),
            cache_control: None,
        };
        let system_b = SystemContent {
            text: "be verbose".to_string(),
            cache_control: None,
        };

        assert_eq!(
            anthropic_affinity_key(Some(&metadata), None, &first_turn),
            Some("user-1".to_string())
        );
        let first = anthropic_affinity_key(None, Some(&system_a), &first_turn);
        assert!(first.is_some());
        assert_eq!(
            first,
            anthropic_affinity_key(None, Some(&system_a), &later_turn)
        );
        assert_ne!(
            first,
            anthropic_affinity_key(None, Some(&system_b), &first_turn)
        );
        assert_eq!(
            anthropic_affinity_key(
                Some(&serde_json::json!({"other": 1})),
                Some(&system_a),
                &first_turn
            ),
            first
        );
    }
}
