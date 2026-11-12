# VortexJobsSlowToStart

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
