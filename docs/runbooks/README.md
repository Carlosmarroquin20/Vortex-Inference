# Runbooks

One short runbook per alert in `k8s/base/monitoring/prometheus-rules.yaml`,
linked from the alert through `runbook_url`. Each follows the same shape:
what fired, who is affected, how to find the cause, how to mitigate.

| Alert | Severity |
|-------|----------|
| [VortexInferenceHighP95Latency](VortexInferenceHighP95Latency.md) | warning |
| [VortexInferenceCriticalP99Latency](VortexInferenceCriticalP99Latency.md) | critical |
| [VortexInferenceSlowTimeToFirstToken](VortexInferenceSlowTimeToFirstToken.md) | warning |
| [VortexInferenceHighErrorRate](VortexInferenceHighErrorRate.md) | warning |
| [VortexInferenceCriticalErrorRate](VortexInferenceCriticalErrorRate.md) | critical |
| [VortexInferenceNoReadyPods](VortexInferenceNoReadyPods.md) | critical |
| [VortexInferencePersistentQueuePressure](VortexInferencePersistentQueuePressure.md) | warning |
| [VortexInferencePodHighMemoryUsage](VortexInferencePodHighMemoryUsage.md) | warning |
| [VortexInferenceFrequentRestarts](VortexInferenceFrequentRestarts.md) | warning |
| [VortexPrimaryBackendBypassed](VortexPrimaryBackendBypassed.md) | warning |
| [VortexJobsDeadLettered](VortexJobsDeadLettered.md) | warning |
| [VortexJobsNotConsumed](VortexJobsNotConsumed.md) | critical |
| [VortexJobsSlowToStart](VortexJobsSlowToStart.md) | warning |
| [VortexJobSubmissionsFailing](VortexJobSubmissionsFailing.md) | warning |

## Where the alert came from

Commands use `$NS` for the namespace. Under GitOps every alert carries a
`namespace` label (Prometheus scopes each environment's rules to it), so
`vortex-staging` or `vortex-prod` is in the alert itself; locally it is
`vortex-inference`.

```bash
NS=vortex-prod
kubectl get pods,rollout,scaledobject,hpa -n $NS     # what is running and how many
kubectl get events -n $NS --sort-by=.lastTimestamp   # what changed recently
kubectl logs -n $NS -l app.kubernetes.io/name=vortex-inference --prefix --tail=100
git log --oneline -5 -- k8s/overlays/${NS#vortex-}/  # the latest releases (GitOps)
```

Every request carries a request ID (`X-Request-ID` response header,
`request_id` log field, `cmpl-<id>` completion id), so a client report can be
traced to its log lines by grepping for `request_id=<id>`.

## Rolling back

In staging and prod, a release is a commit that changes the image in the
environment's overlay (`deploy(staging): sha-…`, `deploy(prod): sha-…`).
Rolling back is reverting that commit:

```bash
git log --oneline -- k8s/overlays/prod/        # find the release commit
git revert <sha> && git push origin main
```

Argo CD applies the revert within a minute (sooner with a refresh in its UI).
The previous image is already on the nodes and its pod template matches the
Rollout's stable revision, so the rollback skips the canary steps.

`kubectl argo rollouts undo`, `kubectl rollout undo` or editing the Rollout
in place would also restore the old pods, but only until Argo CD's self-heal
puts back what git says, a few seconds later. Git is the desired state; a
rollback that does not change git does not last, and one that does is
reviewed, recorded and repeatable like any release. See
[Rolling back](../../README.md#rolling-back) in the README.

A canary that is still in progress needs none of this: a failed analysis
aborts it and the stable pods keep serving. Revert the commit afterwards so
git stops asking for the release that failed.

Locally (`make deploy`, no Argo CD), `kubectl rollout undo
deployment/vortex-inference -n vortex-inference` is the rollback.

## Operating RabbitMQ and Redis

**Rotating the RabbitMQ password.** RabbitMQ applies `RABBITMQ_DEFAULT_PASS`
only on its first boot, so change it in the broker first, then in the
Secret, then restart the clients:

```bash
kubectl exec -n $NS vortex-rabbitmq-0 -- rabbitmqctl change_password vortex <new>
kubectl delete secret vortex-inference-jobs -n $NS   # locally; recreate it with <new>
kubectl rollout restart deployment/vortex-inference deployment/vortex-worker -n $NS
```

Under GitOps the Secret comes from git: `python scripts/seal_secrets.py
environment <env>` generates and seals new credentials, all of them, and
the commit delivers them. Read the new password from the decrypted Secret
once Argo CD has synced, run `change_password` with it, then restart
Redis, the Rollout and the workers so every component reads the new
values.

**Changing `VORTEX_JOB_MAX_ATTEMPTS`.** It is the jobs queue's
`x-delivery-limit`, which a quorum queue cannot change in place; the API and
workers then fail to declare the queue (`PRECONDITION_FAILED`). Stop
submissions, let the workers drain the queue, delete it
(`rabbitmqctl delete_queue vortex.jobs`) and deploy the new value; the queue
is recreated on first use.
