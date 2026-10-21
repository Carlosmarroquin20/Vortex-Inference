# Runbooks

Each alert in `k8s/base/monitoring/prometheus-rules.yaml` links to its section here
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

## VortexInferenceSlowTimeToFirstToken

**Meaning.** For 5 minutes, the P95 of streamed requests has waited more than
10s from arrival to their first token. Buffered requests do not contribute.

**Impact.** Streaming clients see a long pause before any output, the part of
latency users notice most. Generation speed after the first token may be fine.

**Diagnosis.**
1. Split the wait into its two parts. Queue wait:
   `histogram_quantile(0.95, sum(rate(vortex_inference_queue_wait_seconds_bucket[5m])) by (le))`.
   If it accounts for most of the time to first token, the pods are saturated:
   check replicas with `make status` and see
   [PersistentQueuePressure](#vortexinferencepersistentqueuepressure).
2. If the queue wait is low, the engine is slow to start. For `llama_cpp` that
   is prompt processing: look at `tokens_prompt` in `Inference complete.` log
   lines with `streamed=True`; long prompts or long conversations dominate on
   CPU. For remote backends it is the provider: compare with its status page,
   and check whether the model thinks before answering
   (`VORTEX_ANTHROPIC_EFFORT`).
3. Check `vortex_inference_time_per_output_token_seconds`. If decode speed has
   degraded at the same time, the cause is shared (CPU throttling, a slower
   node, an overloaded provider) rather than specific to prompt processing.

**Mitigation.** Scale out for queueing; cap prompt length or conversation size
for slow prompt processing; for a remote provider, lower the effort setting or
choose a faster model.

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
   `k8s/base/keda-autoscaler.yaml` matches the Prometheus service.

**Mitigation.** Restore KEDA's metric source, free node capacity, or raise
`maxReplicaCount` together with the ResourceQuota (see `k8s/base/namespace.yaml`
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
`k8s/base/namespace.yaml`. Apply with `make deploy`.

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

---

# Asynchronous jobs

Jobs are queued in RabbitMQ (`vortex-rabbitmq`), tracked in Redis
(`vortex-redis`) and run by `vortex-worker`, which KEDA scales from zero on the
length of the `vortex.jobs` queue. A job's ID is also the `request_id` of every
worker log line about it, so `kubectl logs -l app.kubernetes.io/name=vortex-worker
-n vortex-inference | grep request_id=<job id>` follows one job.

```bash
kubectl describe scaledobject vortex-worker-scaler -n vortex-inference
kubectl get pods -n vortex-inference -l app.kubernetes.io/part-of=vortex-inference
kubectl exec -n vortex-inference vortex-rabbitmq-0 -- \
  rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers
```

## VortexJobsDeadLettered

**Meaning.** `vortex.jobs.dead` holds at least one message. A message lands
there when the worker could not parse it, when processing it raised an
unexpected error (the job is marked failed with a 500), or when it was
delivered more than `VORTEX_JOB_MAX_ATTEMPTS` times without an
acknowledgement, i.e. it kept killing its worker.

**Impact.** Those jobs will not run again on their own. Clients polling them
see `failed`, or `running` for a job whose worker died every time.

**Diagnosis.**
1. Look at the messages without removing them (`ackmode` requeues them):
   ```bash
   kubectl port-forward -n vortex-inference svc/vortex-rabbitmq 15672 &
   # credentials: kubectl get secret vortex-inference-jobs -n vortex-inference \
   #   -o jsonpath='{.data.rabbitmq-password}' | base64 -d
   curl -u vortex:<password> -H 'content-type: application/json' \
     -d '{"count":5,"ackmode":"reject_requeue_true","encoding":"auto"}' \
     http://127.0.0.1:15672/api/queues/%2F/vortex.jobs.dead/get
   ```
   The `message_id` property is the job ID.
2. Search the worker logs for that ID: `Job processing failed unexpectedly`
   with a traceback is a bug; `Unreadable job message` is a message something
   other than the API published; no log line at all, with restarts on the
   worker pods, means the job crashed or OOM-killed its worker (see
   [FrequentRestarts](#vortexinferencefrequentrestarts); the same diagnosis
   applies to `vortex-worker` pods).

**Mitigation.** Fix the cause first. Then either purge the queue
(`rabbitmqctl purge_queue vortex.jobs.dead`) or move the messages back to
`vortex.jobs` with a shovel (enabling `rabbitmq_shovel_management` adds a
"Move messages" form to the queue's page in the management UI). A job that
was marked failed stays failed, since the worker skips finished jobs; clients
resubmit those. Records expire after `VORTEX_JOB_TTL_SECONDS`.

## VortexJobsNotConsumed

**Meaning.** `vortex.jobs` has had messages ready and no consumer for 10
minutes.

**Impact.** No job is making progress. Submissions still succeed, so the
backlog keeps growing.

**Diagnosis.**
1. `kubectl describe scaledobject vortex-worker-scaler -n vortex-inference`:
   `Ready` and `Active` conditions, and events. A scaler error usually means the
   KEDA operator cannot reach the management API (the `vortex-rabbitmq`
   NetworkPolicy admits only the `keda` namespace on port 15672) or the
   `rabbitmq-management-url` in the `vortex-inference-jobs` Secret is wrong.
2. `kubectl get deployment vortex-worker -n vortex-inference`: if replicas are
   wanted but none are ready, check the pods' events for `FailedScheduling`
   (quota or node capacity; workers run at the lowest priority and are the
   first to stay Pending), image errors, or a failing startup probe (model
   missing, RabbitMQ unreachable: the worker retries the connection and logs
   `RabbitMQ unreachable. Retrying.`).

**Mitigation.** Restore whatever step above failed. Messages are durable;
nothing is lost while workers are down.

## VortexJobsSlowToStart

**Meaning.** The P95 time from a job's submission to its first attempt has
been above 10 minutes for 15 minutes, while workers are running.

**Impact.** Jobs finish late. This is a capacity signal, not a failure.

**Diagnosis.** Compare queue depth with the worker count in the dashboard's
Jobs row. Workers at `maxReplicaCount` with a growing queue mean too little
capacity; workers below it mean KEDA cannot add pods (check events for
`FailedScheduling`: on a full node the API's replicas take precedence).

**Mitigation.** Raise `maxReplicaCount` in `k8s/base/worker.yaml` if the node
has room, or `VORTEX_MAX_CONCURRENT_REQUESTS` for remote backends (and the
scaler's `value` with it).

## VortexJobSubmissionsFailing

**Meaning.** `POST /v1/jobs` has been answering 503 for 5 minutes because the
API cannot reach the job store (`component="store"`, Redis) or the queue
(`component="queue"`, RabbitMQ). Logged as `Job store unavailable.` or
`Job queue unavailable.` with the reason.

**Impact.** Clients cannot submit jobs; they are told to retry
(`Retry-After: 5`). Completions and reading existing jobs are unaffected for a
queue outage; a store outage also fails `GET /v1/jobs/{id}`.

**Diagnosis.**
1. `kubectl get pods -n vortex-inference -l app.kubernetes.io/name=vortex-redis`
   (or `vortex-rabbitmq`): restarts, readiness, events.
2. RabbitMQ under a memory or disk alarm blocks publishers, and the API times
   out waiting for the confirm: `rabbitmq-diagnostics alarms` in the pod.
3. Redis refusing writes with `OOM command not allowed` has reached its
   `maxmemory` (set to refuse rather than evict, see `k8s/base/redis.yaml`):
   too many jobs within the TTL.

**Mitigation.** Restart or resize the failing component. For a full Redis,
lower `VORTEX_JOB_TTL_SECONDS` or raise `maxmemory` together with the
container's memory limit.

## Operating RabbitMQ and Redis

**Rotating the RabbitMQ password.** RabbitMQ applies `RABBITMQ_DEFAULT_PASS`
only on its first boot, so change it in the broker first, then in the Secret,
then restart the clients:

```bash
kubectl exec -n vortex-inference vortex-rabbitmq-0 -- rabbitmqctl change_password vortex <new>
kubectl delete secret vortex-inference-jobs -n vortex-inference
# recreate it with the new RabbitMQ password and the current Redis one, then:
kubectl rollout restart deployment/vortex-inference deployment/vortex-worker -n vortex-inference
```

**Changing `VORTEX_JOB_MAX_ATTEMPTS`.** It is the jobs queue's
`x-delivery-limit`, which a quorum queue cannot change in place; the API and
workers then fail to declare the queue (`PRECONDITION_FAILED`). Stop
submissions, let the workers drain the queue, delete it
(`rabbitmqctl delete_queue vortex.jobs`) and deploy the new value; the queue is
recreated on first use.
