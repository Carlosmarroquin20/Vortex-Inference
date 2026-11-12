# VortexInferenceCriticalP99Latency

**Meaning.** P99 inference time has been above 30s for 3 minutes. Requests are
approaching the 120s server-side deadline.

**Impact.** A visible share of requests is about to start failing with 504.

**Diagnosis.** As for [HighP95Latency](VortexInferenceHighP95Latency.md), plus:
- Look for `Inference exceeded its deadline.` log lines; their `max_tokens`
  field shows whether a few oversized requests are responsible.
- Check for Pending pods: `kubectl get pods -n $NS`. A Pending
  replica usually means node CPU or memory is exhausted, or the ResourceQuota
  is rejecting it (`kubectl describe resourcequota -n $NS`).

**Mitigation.** Restore scale-out capacity first. If the node is full, the
cluster needs more capacity; reducing `maxReplicaCount` will not help.
