// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod crd;
mod daemon;
mod utils;

pub use crd::{DynamoWorkerMetadata, DynamoWorkerMetadataSpec};
// hash_pod_name/hash_container_name are used by C bindings and the Rust EPP
// for pod- and container-level worker ID mapping.
pub use utils::{hash_container_name, hash_pod_name};

use crd::{apply_cr, build_cr};
use daemon::DiscoveryDaemon;
use utils::PodInfo;

use crate::CancellationToken;
use crate::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryMetadata,
    DiscoveryQuery, DiscoverySpec, DiscoveryStream, MAX_JSON_SAFE_PUBLISHER_ID,
    ModelCardInstanceId, reconcile_discovery_snapshot, resync_discovery_events,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use kube::{
    Api, Client as KubeClient,
    api::{DeleteParams, Preconditions},
};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

/// A container restart preserves its Pod and discovery identity but may change its TCP port.
/// Retire the previous incarnation before starting a reflector that could advertise it again.
async fn clear_stale_metadata(kube_client: KubeClient, pod_info: &PodInfo) -> Result<()> {
    let cr_name = pod_info.target.cr_name();
    let api: Api<DynamoWorkerMetadata> = Api::namespaced(kube_client, &pod_info.pod_namespace);
    let Some(existing) = api.get_opt(&cr_name).await? else {
        return Ok(());
    };
    let owners = existing
        .metadata
        .owner_references
        .as_deref()
        .unwrap_or_default();
    anyhow::ensure!(
        owners.len() == 1
            && owners[0].api_version == "v1"
            && owners[0].kind == "Pod"
            && owners[0].name == pod_info.pod_name
            && owners[0].uid == pod_info.pod_uid,
        "refusing to clear discovery metadata {cr_name}: not exclusively owned by the starting Pod"
    );
    let uid = existing
        .metadata
        .uid
        .context("discovery metadata has no UID")?;
    let resource_version = existing
        .metadata
        .resource_version
        .context("discovery metadata has no resourceVersion")?;
    // Do not delete a replacement or a concurrent publisher's update after the ownership check.
    let params = DeleteParams {
        preconditions: Some(Preconditions {
            uid: Some(uid),
            resource_version: Some(resource_version),
        }),
        ..Default::default()
    };
    match api.delete(&cr_name, &params).await {
        Ok(_) => {}
        Err(kube::Error::Api(error)) if error.code == 404 => {}
        Err(error) => return Err(error).context("failed to clear stale discovery metadata"),
    }
    // A finalizer or concurrent replacement must not leave stale state visible to startup.
    anyhow::ensure!(
        api.get_opt(&cr_name).await?.is_none(),
        "discovery metadata {cr_name} still exists after cleanup; refusing to start with stale state"
    );
    tracing::info!("Deleted stale CR: {cr_name}");
    Ok(())
}

fn validate_kubernetes_publisher_id(publisher_id: u64) -> Result<()> {
    if publisher_id > MAX_JSON_SAFE_PUBLISHER_ID {
        anyhow::bail!(
            "Kubernetes discovery publisher ID {publisher_id} exceeds the JSON-safe maximum \
             {MAX_JSON_SAFE_PUBLISHER_ID}"
        );
    }

    Ok(())
}

async fn update_model_taints_and_persist<F, Fut>(
    metadata: &Arc<RwLock<DiscoveryMetadata>>,
    id: ModelCardInstanceId,
    taints: HashSet<String>,
    persist: F,
) -> Result<bool>
where
    F: FnOnce(DiscoveryMetadata) -> Fut + Send + 'static,
    Fut: Future<Output = Result<DiscoveryMetadata>> + Send + 'static,
{
    let metadata = Arc::clone(metadata);
    // Once started, persistence and the matching local commit must outlive request cancellation.
    // Dropping the JoinHandle detaches this task instead of cancelling the remote-commit/local-
    // state critical section.
    tokio::spawn(async move {
        let mut metadata = metadata.write().await;
        let mut candidate = metadata.clone();
        let changed = candidate.update_model_taints(&id, taints)?;

        // Persist even a local no-op. This repairs an authoritative CR that may differ after an
        // earlier commit/ack ambiguity instead of trusting potentially stale local metadata.
        let persisted = persist(candidate).await?;
        *metadata = persisted;
        Ok(changed)
    })
    .await
    .map_err(|error| anyhow::anyhow!("model taint persistence task failed: {error}"))?
}

/// Kubernetes-based discovery client
#[derive(Clone)]
pub struct KubeDiscoveryClient {
    instance_id: u64,
    metadata: Arc<RwLock<DiscoveryMetadata>>,
    list_state: Arc<RwLock<HashMap<u64, Arc<DiscoveryMetadata>>>>,
    event_tx: broadcast::Sender<DiscoveryEvent>,
    kube_client: KubeClient,
    pod_info: PodInfo,
}

impl KubeDiscoveryClient {
    /// Create a new Kubernetes discovery client
    ///
    /// # Arguments
    /// * `metadata` - Shared metadata store (also used by system server)
    /// * `cancel_token` - Cancellation token for shutdown
    pub async fn new(
        metadata: Arc<RwLock<DiscoveryMetadata>>,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let pod_info = PodInfo::from_env()?;
        let instance_id = pod_info.target.instance_id();
        let cr_name = pod_info.target.cr_name();

        tracing::info!(
            "Initializing KubeDiscoveryClient: mode={:?}, target={:?}, cr_name={}, instance_id={:x}, namespace={}, pod_uid={}",
            pod_info.mode,
            pod_info.target,
            cr_name,
            instance_id,
            pod_info.pod_namespace,
            pod_info.pod_uid
        );

        let kube_client = KubeClient::try_default()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Kubernetes client: {}", e))?;

        // Both discovery modes survive an in-Pod container restart. Keep their readiness
        // sources unchanged, but never let the new runtime inherit its predecessor's record.
        clear_stale_metadata(kube_client.clone(), &pod_info).await?;

        let list_state = Arc::new(RwLock::new(HashMap::new()));
        let (event_tx, _) = broadcast::channel::<DiscoveryEvent>(4096);

        let daemon = DiscoveryDaemon::new(kube_client.clone(), pod_info.clone(), cancel_token)?;
        let daemon_list_state = list_state.clone();
        let daemon_event_tx = event_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = daemon.run(daemon_list_state, daemon_event_tx).await {
                tracing::error!("Discovery daemon failed: {e}");
            }
        });

        tracing::info!("Discovery daemon started");

        Ok(Self {
            instance_id,
            metadata,
            list_state,
            event_tx,
            kube_client,
            pod_info,
        })
    }
}

#[async_trait]
impl Discovery for KubeDiscoveryClient {
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        match &spec {
            DiscoverySpec::EventChannel { publisher_id, .. }
            | DiscoverySpec::EventSource { publisher_id, .. } => {
                validate_kubernetes_publisher_id(*publisher_id)?;
            }
            _ => {}
        }
        let instance = spec.into_instance(self.instance_id());
        let instance_id = instance.instance_id();

        tracing::debug!(
            "Registering discovery instance: {:?}, instance_id={:x}",
            instance,
            instance_id
        );

        // Write to local metadata and persist to CR
        // IMPORTANT: Hold the write lock across the CR write to prevent race conditions
        let mut metadata = self.metadata.write().await;

        // Clone state for rollback in case CR persistence fails
        let original_state = metadata.clone();

        let registered_instance = match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                tracing::info!(
                    "Registering endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    instance_id
                );
                metadata.register_endpoint(instance.clone())?;
                instance.clone()
            }
            DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                ..
            } => {
                tracing::info!(
                    "Registering model card: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    namespace,
                    component,
                    endpoint,
                    instance_id
                );
                metadata.register_model_card(instance.clone())?
            }
            DiscoveryInstance::EventChannel { scope, topic, .. } => {
                tracing::info!(
                    "Registering event channel: scope={:?}, topic={}, instance_id={:x}",
                    scope,
                    topic,
                    instance_id
                );
                metadata.register_event_channel(instance.clone())?;
                instance.clone()
            }
            DiscoveryInstance::EventSource { scope, topic, .. } => {
                tracing::info!(
                    "Registering event source: scope={:?}, topic={}, publisher_id={:x}",
                    scope,
                    topic,
                    instance_id
                );
                metadata.register_event_source(instance.clone())?;
                instance.clone()
            }
        };

        // Build and apply the CR with the updated metadata
        // This persists the metadata to Kubernetes for other pods to discover
        let cr_name = self.pod_info.target.cr_name();
        let cr = build_cr(
            &cr_name,
            &self.pod_info.pod_name,
            &self.pod_info.pod_uid,
            &metadata,
        )?;

        if let Err(e) = apply_cr(&self.kube_client, &self.pod_info.pod_namespace, &cr).await {
            // Rollback local state on CR persistence failure
            tracing::warn!(
                "Failed to persist metadata to CR, rolling back local state: {}",
                e
            );
            *metadata = original_state;
            return Err(e);
        }

        tracing::debug!("Persisted metadata to DynamoWorkerMetadata CR");

        Ok(registered_instance)
    }

    async fn update_model_taints_internal(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        let kube_client = self.kube_client.clone();
        let pod_namespace = self.pod_info.pod_namespace.clone();
        let cr_name = self.pod_info.target.cr_name();
        let pod_name = self.pod_info.pod_name.clone();
        let pod_uid = self.pod_info.pod_uid.clone();
        let changed = update_model_taints_and_persist(
            &self.metadata,
            id,
            taints,
            move |candidate| async move {
                let cr = build_cr(&cr_name, &pod_name, &pod_uid, &candidate)?;
                apply_cr(&kube_client, &pod_namespace, &cr).await?;
                Ok(candidate)
            },
        )
        .await?;
        if !changed {
            return Ok(());
        }

        tracing::debug!("Persisted model taint update to DynamoWorkerMetadata CR");
        Ok(())
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let instance_id = instance.instance_id();

        // Write to local metadata and persist to CR
        // IMPORTANT: Hold the write lock across the CR write to prevent race conditions
        let mut metadata = self.metadata.write().await;

        // Clone state for rollback in case CR persistence fails
        let original_state = metadata.clone();

        match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                tracing::info!(
                    "Unregistering endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    instance_id
                );
                metadata.unregister_endpoint(&instance)?;
            }
            DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                ..
            } => {
                tracing::info!(
                    "Unregistering model card: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    namespace,
                    component,
                    endpoint,
                    instance_id
                );
                metadata.unregister_model_card(&instance)?;
            }
            DiscoveryInstance::EventChannel { scope, topic, .. } => {
                tracing::info!(
                    "Unregistering event channel: scope={:?}, topic={}, instance_id={:x}",
                    scope,
                    topic,
                    instance_id
                );
                metadata.unregister_event_channel(&instance)?;
            }
            DiscoveryInstance::EventSource { scope, topic, .. } => {
                tracing::info!(
                    "Unregistering event source: scope={:?}, topic={}, publisher_id={:x}",
                    scope,
                    topic,
                    instance_id
                );
                metadata.unregister_event_source(&instance)?;
            }
        }

        // Build and apply the CR with the updated metadata
        // This persists the removal to Kubernetes for other pods to see
        let cr_name = self.pod_info.target.cr_name();
        let cr = build_cr(
            &cr_name,
            &self.pod_info.pod_name,
            &self.pod_info.pod_uid,
            &metadata,
        )?;

        if let Err(e) = apply_cr(&self.kube_client, &self.pod_info.pod_namespace, &cr).await {
            // Rollback local state on CR persistence failure
            tracing::warn!(
                "Failed to persist metadata removal to CR, rolling back local state: {}",
                e
            );
            *metadata = original_state;
            return Err(e);
        }

        tracing::debug!("Persisted metadata removal to DynamoWorkerMetadata CR");

        Ok(())
    }

    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        tracing::debug!("KubeDiscoveryClient::list called with query={:?}", query);

        let state = self.list_state.read().await;
        let instances: Vec<DiscoveryInstance> =
            state.values().flat_map(|m| m.filter(&query)).collect();

        tracing::info!(
            "KubeDiscoveryClient::list returning {} instances for query={:?}",
            instances.len(),
            query
        );

        Ok(instances)
    }

    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream> {
        use broadcast::error::RecvError;
        use tokio::sync::mpsc;

        tracing::info!(
            "KubeDiscoveryClient::list_and_watch started for query={:?}",
            query
        );

        let (out_tx, out_rx) = mpsc::unbounded_channel();
        let stream_id = uuid::Uuid::new_v4();

        // Acquire read lock, subscribe to broadcast, then read initial state.
        // The write lock (held by the daemon while updating list_state and sending events)
        // is mutually exclusive with our read lock, so no events can slip between
        // our subscription point and our initial state read.
        // This runs before the return, so a caller that lists afterwards cannot observe an
        // instance that a removal deletes before the subscription.
        let (initial_instances, mut broadcast_rx) = {
            let state = self.list_state.read().await;
            let rx = self.event_tx.subscribe();
            let initial = state
                .values()
                .flat_map(|m| m.filter(&query))
                .collect::<Vec<_>>();
            (initial, rx)
        };
        let list_state = self.list_state.clone();

        tokio::spawn(async move {
            tracing::debug!(
                stream_id = %stream_id,
                initial_count = initial_instances.len(),
                "Watch started for query={:?}",
                query
            );

            let mut known: HashMap<DiscoveryInstanceId, DiscoveryInstance> = initial_instances
                .iter()
                .map(|i| (i.id(), i.clone()))
                .collect();

            for instance in &initial_instances {
                tracing::info!(
                    stream_id = %stream_id,
                    instance_id = format!("{:x}", instance.instance_id()),
                    "Emitting initial Added event"
                );
                if out_tx
                    .send(Ok(DiscoveryEvent::Added(instance.clone())))
                    .is_err()
                {
                    return;
                }
            }

            loop {
                let recv_result = if let Some(ref token) = cancel_token {
                    tokio::select! {
                        result = broadcast_rx.recv() => result,
                        _ = token.cancelled() => {
                            tracing::info!(stream_id = %stream_id, "Watch cancelled via cancel token");
                            break;
                        }
                    }
                } else {
                    broadcast_rx.recv().await
                };

                match recv_result {
                    Ok(event) => {
                        let forwarded = match &event {
                            DiscoveryEvent::Added(instance) => {
                                if instance.matches(&query) {
                                    let id = instance.id();
                                    if known.get(&id) != Some(instance) {
                                        known.insert(id.clone(), instance.clone());
                                        Some(("added", id))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            DiscoveryEvent::Removed(id) => {
                                known.remove(id).is_some().then(|| ("removed", id.clone()))
                            }
                            DiscoveryEvent::ModelTaintsUpdated(update) => {
                                let id = DiscoveryInstanceId::Model(update.id.clone());
                                known
                                    .contains_key(&id)
                                    .then_some(("model_taints_updated", id))
                            }
                            // The daemon publishes incremental events only.
                            DiscoveryEvent::Resync(_) => None,
                        };
                        if let Some((event_kind, instance_id)) = forwarded {
                            tracing::info!(
                                stream_id = %stream_id,
                                event_kind,
                                ?instance_id,
                                "Emitting discovery event"
                            );
                            if out_tx.send(Ok(event)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!(
                            stream_id = %stream_id,
                            dropped = n,
                            "Broadcast receiver lagged, reconciling from list_state"
                        );
                        let state = list_state.read().await;
                        let current: HashMap<DiscoveryInstanceId, DiscoveryInstance> = state
                            .values()
                            .flat_map(|m| m.filter(&query))
                            .map(|i| (i.id(), i))
                            .collect();
                        drop(state);
                        for event in resync_discovery_events(&mut known, current) {
                            if out_tx.send(Ok(event)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(RecvError::Closed) => {
                        tracing::info!(
                            stream_id = %stream_id,
                            "Broadcast channel closed (daemon stopped)"
                        );
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(
            tokio_stream::wrappers::UnboundedReceiverStream::new(out_rx),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::TransportType;
    use crate::discovery::{EventScope, EventTransport, ModelTaintsUpdate};
    use axum::{
        Json, Router,
        body::Body,
        extract::State,
        http::{Method, Request, StatusCode},
    };
    use serde_json::{Value, json};
    use std::{collections::VecDeque, sync::Mutex};
    use utils::{KubeDiscoveryMode, KubeDiscoveryTarget};

    fn restart_pod(container: Option<&str>) -> PodInfo {
        PodInfo {
            pod_name: "pod".into(),
            pod_namespace: "ns".into(),
            pod_uid: "pod-uid".into(),
            system_port: 9090,
            mode: if container.is_some() {
                KubeDiscoveryMode::Container
            } else {
                KubeDiscoveryMode::Pod
            },
            target: match container {
                Some(name) => KubeDiscoveryTarget::Container("pod".into(), name.into()),
                None => KubeDiscoveryTarget::Pod("pod".into()),
            },
        }
    }

    fn stale_record(pod: &PodInfo) -> Value {
        let mut metadata = DiscoveryMetadata::new();
        metadata
            .register_endpoint(endpoint_instance(
                pod.target.instance_id(),
                "127.0.0.1:1234",
            ))
            .unwrap();
        let mut record = build_cr(
            &pod.target.cr_name(),
            &pod.pod_name,
            &pod.pod_uid,
            &metadata,
        )
        .unwrap();
        record.metadata.uid = Some("old-record".into());
        record.metadata.resource_version = Some("42".into());
        serde_json::to_value(record).unwrap()
    }

    fn api_status(code: u16) -> Value {
        let reason = match code {
            404 => "NotFound",
            409 => "Conflict",
            403 => "Forbidden",
            _ => "fixture",
        };
        json!({"apiVersion":"v1", "kind":"Status", "code":code,
            "status":if code < 400 { "Success" } else { "Failure" },
            "reason":reason, "message":"fixture"})
    }

    // Exercise the real kube request serialization and error handling without a live API server.
    async fn exercise_cleanup(
        pod: PodInfo,
        steps: Vec<(Method, u16, Value)>,
    ) -> (Result<()>, Vec<Value>) {
        struct ApiScript {
            responses: VecDeque<(u16, Value)>,
            requests: Vec<(Method, String, Value)>,
        }
        let methods: Vec<_> = steps.iter().map(|step| step.0.clone()).collect();
        let state = Arc::new(Mutex::new(ApiScript {
            responses: steps
                .into_iter()
                .map(|(_, code, body)| (code, body))
                .collect(),
            requests: Vec::new(),
        }));
        let router = Router::new()
            .fallback(
                |State(state): State<Arc<Mutex<ApiScript>>>, request: Request<Body>| async move {
                    let (parts, body) = request.into_parts();
                    let bytes = axum::body::to_bytes(body, 65536).await.unwrap();
                    let body = if bytes.is_empty() {
                        Value::Null
                    } else {
                        serde_json::from_slice(&bytes).unwrap()
                    };
                    let mut script = state.lock().unwrap();
                    script
                        .requests
                        .push((parts.method, parts.uri.path().to_string(), body));
                    let (code, body) = script
                        .responses
                        .pop_front()
                        .unwrap_or((500, api_status(500)));
                    (StatusCode::from_u16(code).unwrap(), Json(body))
                },
            )
            .with_state(state.clone());
        let result = clear_stale_metadata(KubeClient::new(router, "ns"), &pod).await;
        let mut script = state.lock().unwrap();
        assert!(script.responses.is_empty(), "unconsumed API responses");
        assert_eq!(
            script.requests.len(),
            methods.len(),
            "unexpected API requests"
        );
        let path = format!(
            "/apis/nvidia.com/v1alpha1/namespaces/ns/dynamoworkermetadatas/{}",
            pod.target.cr_name()
        );
        for ((method, actual_path, _), expected) in script.requests.iter().zip(methods) {
            assert_eq!(*method, expected);
            assert_eq!(*actual_path, path);
        }
        (
            result,
            std::mem::take(&mut script.requests)
                .into_iter()
                .map(|(_, _, body)| body)
                .collect(),
        )
    }

    #[tokio::test]
    async fn stale_metadata_cleanup_fences_both_discovery_modes() {
        for container in [None, Some("main"), Some("engine-0")] {
            let pod = restart_pod(container);
            let record = stale_record(&pod);
            let (result, requests) = exercise_cleanup(
                pod,
                vec![
                    (Method::GET, 200, record),
                    (Method::DELETE, 200, api_status(200)),
                    (Method::GET, 404, api_status(404)),
                ],
            )
            .await;
            result.unwrap();
            assert_eq!(
                requests[1]["preconditions"],
                json!({"uid":"old-record", "resourceVersion":"42"})
            );
        }
    }

    #[tokio::test]
    async fn stale_metadata_cleanup_allows_fresh_start() {
        let (result, _) =
            exercise_cleanup(restart_pod(None), vec![(Method::GET, 404, api_status(404))]).await;
        result.unwrap();
    }

    #[tokio::test]
    async fn stale_metadata_cleanup_rejects_foreign_or_unfenced_records() {
        for field in [
            "owner",
            "pod_uid",
            "pod_name",
            "co_owner",
            "uid",
            "resourceVersion",
        ] {
            let pod = restart_pod(None);
            let mut record = stale_record(&pod);
            match field {
                "owner" => record["metadata"]["ownerReferences"] = json!([]),
                "pod_uid" => record["metadata"]["ownerReferences"][0]["uid"] = json!("other-pod"),
                "pod_name" => record["metadata"]["ownerReferences"][0]["name"] = json!("other"),
                "co_owner" => {
                    let owner = record["metadata"]["ownerReferences"][0].clone();
                    record["metadata"]["ownerReferences"]
                        .as_array_mut()
                        .unwrap()
                        .push(owner);
                }
                key => {
                    record["metadata"].as_object_mut().unwrap().remove(key);
                }
            }
            let (result, _) = exercise_cleanup(pod, vec![(Method::GET, 200, record)]).await;
            assert!(result.is_err(), "accepted invalid {field}");
        }
    }

    #[tokio::test]
    async fn stale_metadata_cleanup_fails_closed_on_api_errors_or_races() {
        for code in [403, 409, 500] {
            let pod = restart_pod(None);
            let record = stale_record(&pod);
            let (result, _) = exercise_cleanup(
                pod,
                vec![
                    (Method::GET, 200, record),
                    (Method::DELETE, code, api_status(code)),
                ],
            )
            .await;
            assert!(result.is_err(), "ignored delete status {code}");
        }
        let (result, _) =
            exercise_cleanup(restart_pod(None), vec![(Method::GET, 403, api_status(403))]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn stale_metadata_cleanup_accepts_already_deleted_record() {
        let pod = restart_pod(None);
        let record = stale_record(&pod);
        let (result, _) = exercise_cleanup(
            pod,
            vec![
                (Method::GET, 200, record),
                (Method::DELETE, 404, api_status(404)),
                (Method::GET, 404, api_status(404)),
            ],
        )
        .await;
        result.unwrap();
    }

    #[tokio::test]
    async fn stale_metadata_cleanup_rejects_pending_deletion_or_replacement() {
        for uid in ["old-record", "replacement-record"] {
            let pod = restart_pod(None);
            let record = stale_record(&pod);
            let mut remaining = record.clone();
            remaining["metadata"]["uid"] = json!(uid);
            let (result, _) = exercise_cleanup(
                pod,
                vec![
                    (Method::GET, 200, record),
                    (Method::DELETE, 200, api_status(200)),
                    (Method::GET, 200, remaining),
                ],
            )
            .await;
            assert!(result.is_err(), "started while {uid} was still present");
        }
    }

    fn endpoint_instance(instance_id: u64, transport: &str) -> DiscoveryInstance {
        DiscoveryInstance::Endpoint(crate::component::Instance {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
            instance_id,
            transport: TransportType::Tcp(transport.to_string()),
            device_type: None,
            request_plane_codec: None,
        })
    }

    fn model_with_taint(taint: &str) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 7,
            card_json: serde_json::json!({
                "runtime_config": {"taints": [taint]}
            }),
            model_suffix: None,
        }
    }

    #[test]
    fn publisher_ids_must_fit_kubernetes_json_safe_range() {
        assert!(validate_kubernetes_publisher_id(MAX_JSON_SAFE_PUBLISHER_ID).is_ok());
        assert!(validate_kubernetes_publisher_id(MAX_JSON_SAFE_PUBLISHER_ID + 1).is_err());
        assert!(validate_kubernetes_publisher_id(u64::MAX).is_err());
    }

    #[test]
    fn snapshot_diff_emits_updated_instance_when_transport_changes() {
        let original = endpoint_instance(1, "127.0.0.1:8000");
        let updated = endpoint_instance(1, "127.0.0.1:9000");
        let known = HashMap::from([(original.id(), original)]);
        let current = HashMap::from([(updated.id(), updated.clone())]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        assert_eq!(events, vec![DiscoveryEvent::Added(updated.clone())]);
        assert_eq!(reconciled.get(&updated.id()), Some(&updated));
    }

    #[test]
    fn snapshot_diff_ignores_same_id_event_channel_changes() {
        let event_channel = |endpoint: &str| DiscoveryInstance::EventChannel {
            scope: EventScope::Namespace {
                name: "ns".to_string(),
            },
            topic: "topic".to_string(),
            instance_id: 1,
            transport: EventTransport::zmq(endpoint),
        };
        let original = event_channel("tcp://127.0.0.1:8000");
        let updated = event_channel("tcp://127.0.0.1:9000");
        let known = HashMap::from([(original.id(), original.clone())]);
        let current = HashMap::from([(updated.id(), updated)]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        assert!(events.is_empty());
        assert_eq!(reconciled.get(&original.id()), Some(&original));
    }

    #[test]
    fn snapshot_diff_emits_added_and_removed_instances() {
        let removed_instance = endpoint_instance(1, "127.0.0.1:8000");
        let added_instance = endpoint_instance(2, "127.0.0.1:9000");
        let removed_id = removed_instance.id();
        let added_id = added_instance.id();
        let known = HashMap::from([(removed_id.clone(), removed_instance)]);
        let current = HashMap::from([(added_id.clone(), added_instance.clone())]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        assert_eq!(events.len(), 2);
        assert!(events.contains(&DiscoveryEvent::Removed(removed_id.clone())));
        assert!(events.contains(&DiscoveryEvent::Added(added_instance.clone())));
        assert!(!reconciled.contains_key(&removed_id));
        assert_eq!(reconciled.get(&added_id), Some(&added_instance));
    }
    #[test]
    fn changed_model_taints_emit_scoped_event() {
        let old = model_with_taint("old");
        let updated = model_with_taint("updated");
        let known = HashMap::from([(old.id(), old)]);
        let current = HashMap::from([(updated.id(), updated.clone())]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        let DiscoveryInstanceId::Model(id) = updated.id() else {
            unreachable!()
        };
        assert_eq!(
            events,
            vec![DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id,
                taints: vec!["updated".to_string()],
            })]
        );
        assert_eq!(reconciled.get(&updated.id()), Some(&updated));
    }

    #[tokio::test]
    async fn model_taint_persistence_completes_after_caller_cancellation() {
        let model = model_with_taint("old");
        let DiscoveryInstanceId::Model(id) = model.id() else {
            unreachable!()
        };
        let mut initial = DiscoveryMetadata::new();
        initial.register_model_card(model).unwrap();
        let metadata = Arc::new(RwLock::new(initial));
        let task_metadata = metadata.clone();
        let remote = Arc::new(RwLock::new(DiscoveryMetadata::new()));
        let task_remote = remote.clone();
        let (remote_committed_tx, remote_committed_rx) = tokio::sync::oneshot::channel();
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();

        let task = tokio::spawn(async move {
            update_model_taints_and_persist(
                &task_metadata,
                id,
                HashSet::from(["new".to_string()]),
                move |candidate| async move {
                    *task_remote.write().await = candidate.clone();
                    remote_committed_tx.send(()).unwrap();
                    ack_rx.await.unwrap();
                    Ok(candidate)
                },
            )
            .await
        });

        remote_committed_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        ack_tx.send(()).unwrap();

        let stored = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let stored = metadata.read().await.get_all_model_cards().pop().unwrap();
                let DiscoveryInstance::Model { card_json, .. } = &stored else {
                    unreachable!()
                };
                if card_json["runtime_config"]["taints"] == serde_json::json!(["new"]) {
                    break stored;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached persistence did not commit local metadata");
        let DiscoveryInstance::Model { card_json, .. } = stored else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["new"])
        );
        let remote = remote.read().await.get_all_model_cards().pop().unwrap();
        let DiscoveryInstance::Model { card_json, .. } = remote else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["new"])
        );
    }

    #[tokio::test]
    async fn local_noop_reapplies_authoritative_model_taints() {
        let local_model = model_with_taint("old");
        let DiscoveryInstanceId::Model(id) = local_model.id() else {
            unreachable!()
        };
        let mut initial = DiscoveryMetadata::new();
        initial.register_model_card(local_model).unwrap();
        let metadata = Arc::new(RwLock::new(initial));
        let persisted = Arc::new(RwLock::new(None));
        let task_persisted = persisted.clone();

        let changed = update_model_taints_and_persist(
            &metadata,
            id,
            HashSet::from(["old".to_string()]),
            move |candidate| async move {
                *task_persisted.write().await = Some(candidate.clone());
                Ok(candidate)
            },
        )
        .await
        .unwrap();

        assert!(!changed);
        let reapplied = persisted
            .read()
            .await
            .clone()
            .expect("no-op was not persisted");
        let DiscoveryInstance::Model { card_json, .. } =
            reapplied.get_all_model_cards().pop().unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["old"])
        );
    }
}
