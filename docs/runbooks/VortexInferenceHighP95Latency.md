# VortexInferenceHighP95Latency

**Meaning.** P95 time spent in the inference engine has been above 10s for 5
minutes. Completed and timed-out requests count; rejected prompts and
cancellations do not.

**Impact.** Interactive clients see slow responses. Not yet an outage.

**Diagnosis.**
1. Check whether the queue is the cause or the engine itself:
   `histogram_quantile(0.95, sum(rate(vortex_inference_queue_wait_seconds_bucket[5m])) by (le))`.
   High queue wait with normal inference time means capacity, not speed.
2. Check replicas: `make status`. If replicas are below `maxReplicaCount` while
   the queue grows, see [PersistentQueuePressure](VortexInferencePersistentQueuePressure.md).
3. Check what clients are asking for: the `tokens_completion` field in
   `Inference complete.` log lines. Large `max_tokens` values dominate latency
   on CPU.
4. Check CPU throttling for the pods in Grafana or with `kubectl top pods`.

**Mitigation.** Scale out if KEDA has not (see below), lower the per-request
`max_tokens` ceiling in `CompletionRequest`, or raise CPU requests together with
`VORTEX_N_THREADS`.
