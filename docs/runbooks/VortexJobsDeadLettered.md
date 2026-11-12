# VortexJobsDeadLettered

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
   kubectl port-forward -n $NS svc/vortex-rabbitmq 15672 &
   # credentials: kubectl get secret vortex-inference-jobs -n $NS \
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
   [FrequentRestarts](VortexInferenceFrequentRestarts.md); the same diagnosis
   applies to `vortex-worker` pods).

**Mitigation.** Fix the cause first. Then either purge the queue
(`rabbitmqctl purge_queue vortex.jobs.dead`) or move the messages back to
`vortex.jobs` with a shovel (enabling `rabbitmq_shovel_management` adds a
"Move messages" form to the queue's page in the management UI). A job that
was marked failed stays failed, since the worker skips finished jobs; clients
resubmit those. Records expire after `VORTEX_JOB_TTL_SECONDS`.
