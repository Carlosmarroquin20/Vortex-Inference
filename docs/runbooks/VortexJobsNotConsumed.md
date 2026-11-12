# VortexJobsNotConsumed

**Meaning.** `vortex.jobs` has had messages ready and no consumer for 10
minutes.

**Impact.** No job is making progress. Submissions still succeed, so the
backlog keeps growing.

**Diagnosis.**
1. `kubectl describe scaledobject vortex-worker-scaler -n $NS`:
   `Ready` and `Active` conditions, and events. A scaler error usually means the
   KEDA operator cannot reach the management API (the `vortex-rabbitmq`
   NetworkPolicy admits only the `keda` namespace on port 15672) or the
   `rabbitmq-management-url` in the `vortex-inference-jobs` Secret is wrong.
2. `kubectl get deployment vortex-worker -n $NS`: if replicas are
   wanted but none are ready, check the pods' events for `FailedScheduling`
   (quota or node capacity; workers run at the lowest priority and are the
   first to stay Pending), image errors, or a failing startup probe (model
   missing, RabbitMQ unreachable: the worker retries the connection and logs
   `RabbitMQ unreachable. Retrying.`).

**Mitigation.** Restore whatever step above failed. Messages are durable;
nothing is lost while workers are down.
