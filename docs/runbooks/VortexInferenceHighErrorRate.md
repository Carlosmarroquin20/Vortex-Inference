# VortexInferenceHighErrorRate

**Meaning.** More than 1% of requests ended in `error` or `timeout` over 5
minutes. Client errors (`rejected`, `cancelled`) are excluded.

**Impact.** Some clients are receiving 500 or 504 responses.

**Diagnosis.**
1. Split by status: `sum by (status) (rate(vortex_inference_requests_total[5m]))`.
2. For `timeout`, follow [CriticalP99Latency](VortexInferenceCriticalP99Latency.md).
3. For `error`, find the traceback: `make logs | grep 'msg="Inference failed."'`.
   The `exc` field holds the full traceback on one line.
4. Correlate with releases. In staging and prod every release is a commit:
   `git log --oneline -- k8s/overlays/<env>/` lists them, newest first.
   Locally: `kubectl rollout history deployment/vortex-inference -n $NS`.

**Mitigation.** If the onset matches a release, roll it back with
`git revert` of its `deploy(<env>)` commit (see
[Rolling back](README.md#rolling-back)); a canary that is still in progress is
aborted by its analysis without that. Locally:
`kubectl rollout undo deployment/vortex-inference -n $NS`.
For OOM-related errors see [PodHighMemoryUsage](VortexInferencePodHighMemoryUsage.md).
