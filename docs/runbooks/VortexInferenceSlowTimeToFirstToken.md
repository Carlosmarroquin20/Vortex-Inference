# VortexInferenceSlowTimeToFirstToken

**Meaning.** For 5 minutes, the P95 of streamed requests has waited more than
10s from arrival to their first token. Buffered requests do not contribute.

**Impact.** Streaming clients see a long pause before any output, the part of
latency users notice most. Generation speed after the first token may be fine.

**Diagnosis.**
1. Split the wait into its two parts. Queue wait:
   `histogram_quantile(0.95, sum(rate(vortex_inference_queue_wait_seconds_bucket[5m])) by (le))`.
   If it accounts for most of the time to first token, the pods are saturated:
   check replicas with `make status` and see
   [PersistentQueuePressure](VortexInferencePersistentQueuePressure.md).
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
