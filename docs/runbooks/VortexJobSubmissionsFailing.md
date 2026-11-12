# VortexJobSubmissionsFailing

**Meaning.** `POST /v1/jobs` has been answering 503 for 5 minutes because the
API cannot reach the job store (`component="store"`, Redis) or the queue
(`component="queue"`, RabbitMQ). Logged as `Job store unavailable.` or
`Job queue unavailable.` with the reason.

**Impact.** Clients cannot submit jobs; they are told to retry
(`Retry-After: 5`). Completions and reading existing jobs are unaffected for a
queue outage; a store outage also fails `GET /v1/jobs/{id}`.

**Diagnosis.**
1. `kubectl get pods -n $NS -l app.kubernetes.io/name=vortex-redis`
   (or `vortex-rabbitmq`): restarts, readiness, events.
2. RabbitMQ under a memory or disk alarm blocks publishers, and the API times
   out waiting for the confirm: `rabbitmq-diagnostics alarms` in the pod.
3. Redis refusing writes with `OOM command not allowed` has reached its
   `maxmemory` (set to refuse rather than evict, see `k8s/base/redis.yaml`):
   too many jobs within the TTL.

**Mitigation.** Restart or resize the failing component. For a full Redis,
lower `VORTEX_JOB_TTL_SECONDS` or raise `maxmemory` together with the
container's memory limit.
