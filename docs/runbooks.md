# Runbooks

Each alert in `k8s/monitoring/prometheus-rules.yaml` links to its section here
through `runbook_url`. Sections follow the same shape: what fired, who is
affected, how to find the cause, and how to mitigate.

Commands assume the `vortex-inference` namespace and a shell at the repository
root. Useful entry points:

```bash
make status                      # pods, ScaledObject, HPA, services
make logs                        # logfmt logs from all inference pods
make metrics                     # raw vortex_* metrics from one pod
kubectl describe scaledobject vortex-inference-scaler -n vortex-inference
kubectl get events -n vortex-inference --sort-by=.lastTimestamp
```

Every request carries a request ID (`X-Request-ID` response header, `request_id`
log field, `cmpl-<id>` completion id), so a client report can be traced to its
log lines with `make logs | grep request_id=<id>`.

---

## VortexInferenceHighP95Latency

**Meaning.** P95 time spent in the inference engine has been above 10s for 5
minutes. Completed and timed-out requests count; rejected prompts and
cancellations do not.

**Impact.** Interactive clients see slow responses. Not yet an outage.

**Diagnosis.**
1. Check whether the queue is the cause or the engine itself:
   `histogram_quantile(0.95, sum(rate(vortex_inference_queue_wait_seconds_bucket[5m])) by (le))`.
   High queue wait with normal inference time means capacity, not speed.
2. Check replicas: `make status`. If replicas are below `maxReplicaCount` while
   the queue grows, see [PersistentQueuePressure](#vortexinferencepersistentqueuepressure).
3. Check what clients are asking for: the `tokens_completion` field in
   `Inference complete.` log lines. Large `max_tokens` values dominate latency
   on CPU.
4. Check CPU throttling for the pods in Grafana or with `kubectl top pods`.

**Mitigation.** Scale out if KEDA has not (see below), lower the per-request
`max_tokens` ceiling in `CompletionRequest`, or raise CPU requests together with
`VORTEX_N_THREADS`.

## VortexInferenceCriticalP99Latency

**Meaning.** P99 inference time has been above 30s for 3 minutes. Requests are
approaching the 120s server-side deadline.

**Impact.** A visible share of requests is about to start failing with 504.

**Diagnosis.** As for [HighP95Latency](#vortexinferencehighp95latency), plus:
- Look for `Inference exceeded its deadline.` log lines; their `max_tokens`
  field shows whether a few oversized requests are responsible.
- Check for Pending pods: `kubectl get pods -n vortex-inference`. A Pending
  replica usually means node CPU or memory is exhausted, or the ResourceQuota
  is rejecting it (`kubectl describe resourcequota -n vortex-inference`).

**Mitigation.** Restore scale-out capacity first. If the node is full, the
cluster needs more capacity; reducing `maxReplicaCount` will not help.

## VortexInferenceHighErrorRate

**Meaning.** More than 1% of requests ended in `error` or `timeout` over 5
minutes. Client errors (`rejected`, `cancelled`) are excluded.

**Impact.** Some clients are receiving 500 or 504 responses.

**Diagnosis.**
1. Split by status: `sum by (status) (rate(vortex_inference_requests_total[5m]))`.
2. For `timeout`, follow [CriticalP99Latency](#vortexinferencecriticalp99latency).
3. For `error`, find the traceback: `make logs | grep 'msg="Inference failed."'`.
   The `exc` field holds the full traceback on one line.
4. Correlate with deployments: `kubectl rollout history deployment/vortex-inference -n vortex-inference`.

**Mitigation.** Roll back a recent bad deployment with
`kubectl rollout undo deployment/vortex-inference -n vortex-inference`.
For OOM-related errors see [PodHighMemoryUsage](#vortexinferencepodhighmemoryusage).

## VortexInferenceCriticalErrorRate

**Meaning.** More than 5% of requests failed over 3 minutes.

**Impact.** The service is degraded for a significant fraction of traffic.

**Diagnosis and mitigation.** As for [HighErrorRate](#vortexinferencehigherrorrate),
with rollback as the first action if the onset coincides with a deployment.

## VortexInferenceNoReadyPods

**Meaning.** The Deployment has had zero Ready replicas for 2 minutes.

**Impact.** Full outage: every request fails at the Service.

**Diagnosis.**
1. `kubectl get pods -n vortex-inference` and `kubectl describe pod <pod>`.
2. `CreateContainerConfigError`: the `vortex-inference-api-key` Secret is
   missing. Recreate it with `make api-key`.
3. `CrashLoopBackOff`: read the previous container's logs with
   `kubectl logs <pod> -n vortex-inference --previous`. A
   `Model failed to load.` line with a missing path means the PVC is empty or
   the file name does not match `VORTEX_MODEL_PATH`; an OOM kill
   (`kubectl describe pod`, reason `OOMKilled`) means the model and KV cache
   exceed the 4Gi limit.
4. Pods Running but not Ready: the readiness probe is failing. A worker
   thread stalled past its deadline also fails readiness; see
   [FrequentRestarts](#vortexinferencefrequentrestarts).

**Mitigation.** Fix the specific cause above. For a bad image, roll back.

## VortexInferencePersistentQueuePressure

**Meaning.** More than 5 requests have been active across all pods for 10
minutes. KEDA should have scaled out well before this.

**Impact.** Rising latency and, eventually, timeouts.

**Diagnosis.**
1. `kubectl describe scaledobject vortex-inference-scaler -n vortex-inference`:
   check the `Ready` and `Active` conditions and recent events.
2. `kubectl get hpa -n vortex-inference`: compare current and desired replicas.
   Desired at the maximum means capacity is exhausted, not that KEDA is broken.
3. If desired replicas are higher than current, the new pods are blocked.
   Check for Pending pods and quota rejections:
   `kubectl get events -n vortex-inference | grep -i quota`.
4. If KEDA reports errors querying Prometheus, verify the server address in
   `k8s/keda-autoscaler.yaml` matches the Prometheus service.

**Mitigation.** Restore KEDA's metric source, free node capacity, or raise
`maxReplicaCount` together with the ResourceQuota (see `k8s/namespace.yaml`
for the sizing formula).

## VortexInferencePodHighMemoryUsage

**Meaning.** A pod's working set has been above 90% of its 4Gi limit for 5
minutes.

**Impact.** The next long-context request may push it over the limit and get
the container OOM-killed, dropping its queue.

**Diagnosis.** Compare the pod's memory with its context settings
(`VORTEX_N_CTX` in the ConfigMap). The KV cache is allocated for the full
context window at startup, so memory usage is mostly fixed by `n_ctx` and the
model size rather than by traffic.

**Mitigation.** Lower `VORTEX_N_CTX`, use a smaller quantization, or raise the
memory limit together with the ResourceQuota and LimitRange in
`k8s/namespace.yaml`. Apply with `make deploy`.

## VortexInferenceFrequentRestarts

**Meaning.** A container restarted more than twice in 15 minutes.

**Impact.** Each restart drops the pod's queue and costs 60-90s of model load.

**Diagnosis.**
1. `kubectl describe pod <pod> -n vortex-inference`: the last state's reason
   (`OOMKilled`, `Error`) and the events (liveness probe failures).
2. `kubectl logs <pod> -n vortex-inference --previous` for the crashed
   container.
3. Events such as `Liveness probe failed: HTTP probe failed with statuscode: 503`
   mean `/health/live` reported a stalled worker: a llama.cpp call ignored its
   stop signal for more than 60s past the deadline (the probe body says
   `Inference worker is stalled past its deadline.`). The restart is the intended recovery; recurring stalls point to a
   specific request shape. Find the request ID of the last request on that
   pod before the restart and reproduce it.

**Mitigation.** For OOM, see [PodHighMemoryUsage](#vortexinferencepodhighmemoryusage).
For model load failures, see [NoReadyPods](#vortexinferencenoreadypods).
