"""
Prometheus metrics exposed on /metrics.

Defined at import time in the default registry, so every module that
imports them updates the same series. The alerting rules, the KEDA trigger
and the Grafana dashboard depend on these names and labels; renaming one is
a breaking change for all three.
"""

from prometheus_client import Counter, Gauge, Histogram

# Naming convention: <project>_<subsystem>_<unit>_<suffix>
# Namespace "vortex_inference" prevents label collisions in shared Prometheus
# instances where multiple services expose identically-named metrics.

INFERENCE_REQUESTS_TOTAL = Counter(
    "vortex_inference_requests_total",
    "Total inference requests, partitioned by terminal status.",
    ["status"],  # "success" | "error" | "timeout" | "cancelled" | "rejected"
)

INFERENCE_DURATION_SECONDS = Histogram(
    "vortex_inference_duration_seconds",
    "Time spent in the inference engine, from acquiring the inference permit "
    "until the request finished, partitioned by terminal status. Queue wait "
    "is excluded (see vortex_inference_queue_wait_seconds). Timeouts are "
    "recorded too: omitting them would hide the slowest requests from the "
    "latency SLO exactly when it is being breached. Buckets are calibrated "
    "for interactive LLM workloads; the sub-100ms range is intentionally "
    "sparse because CPU inference cannot achieve it.",
    ["status"],  # same values as vortex_inference_requests_total
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, float("inf")],
)

QUEUE_WAIT_SECONDS = Histogram(
    "vortex_inference_queue_wait_seconds",
    "Time a request waited for the inference permit before starting. "
    "Sustained growth here with flat inference duration means the pod is "
    "saturated and KEDA should be adding replicas.",
    buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, float("inf")],
)

TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "vortex_inference_time_to_first_token_seconds",
    "Streamed requests only: time from the request's arrival to its first text "
    "chunk, i.e. the wait a client sees before output starts. It includes the "
    "queue wait for a permit (vortex_inference_queue_wait_seconds), so a "
    "rise with flat queue wait points at the engine: prompt processing for "
    "local models, the network and the provider's own queueing and thinking "
    "for remote backends.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, float("inf")],
)

TIME_PER_OUTPUT_TOKEN_SECONDS = Histogram(
    "vortex_inference_time_per_output_token_seconds",
    "Streamed requests only: average gap between output tokens after the "
    "first, observed once per completed stream with at least two tokens. "
    "This is the decode speed a reader experiences; its inverse is the "
    "per-request tokens per second, without the prompt processing that "
    "dominates time to first token.",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, float("inf")],
)

TOKENS_GENERATED_TOTAL = Counter(
    "vortex_inference_tokens_generated_total",
    "Cumulative token count by type. Use rate() over a time window for TPS "
    "rather than reading this counter directly — it prevents misleading spikes "
    "from appearing as sustained throughput in dashboards.",
    ["type"],  # "prompt" | "completion"
)

INFERENCE_ACTIVE_REQUESTS = Gauge(
    "vortex_inference_active_requests",
    "Requests currently held by the inference engine (including queue wait). "
    "THIS IS THE KEDA SCALING SIGNAL. The ScaledObject queries "
    "sum(vortex_inference_active_requests) and divides by the configured "
    "threshold to determine target replica count. A sustained value above "
    "threshold triggers a scale-out event; dropping to zero starts the "
    "cooldown period before scale-in.",
)

TOKENS_PER_SECOND = Gauge(
    "vortex_inference_tokens_per_second",
    "Throughput of the most recently completed request. This is a trailing "
    "point-in-time indicator. For capacity planning, prefer "
    "rate(vortex_inference_tokens_generated_total[5m]) instead.",
)

MODEL_LOAD_DURATION_SECONDS = Gauge(
    "vortex_model_load_duration_seconds",
    "Wall-clock time to load the model at pod startup. Elevated values indicate "
    "memory contention during init — a risk factor in pod restart storms where "
    "multiple replicas start simultaneously on a memory-constrained node.",
)

MODEL_WARMUP_DURATION_SECONDS = Gauge(
    "vortex_model_warmup_duration_seconds",
    "Time the startup warm-up generation took (llama_cpp only): mostly "
    "reading the memory-mapped weights from disk, which the first request "
    "would otherwise pay for. Close to a normal one-token request when the "
    "file was already in the node's page cache.",
)

BACKEND_FALLBACKS_TOTAL = Counter(
    "vortex_backend_fallbacks_total",
    "Requests served by the fallback backend instead of the primary, by "
    "reason: overloaded, upstream_error, timeout, prompt_too_long, or "
    "circuit_open when the primary was skipped after repeated failures.",
    ["reason"],
)

JOBS_SUBMITTED_TOTAL = Counter(
    "vortex_jobs_submitted_total",
    "Asynchronous jobs accepted and confirmed by RabbitMQ, by endpoint. "
    "Idempotent replays of an earlier submission are not counted.",
    ["url"],
)

JOB_SUBMISSION_FAILURES_TOTAL = Counter(
    "vortex_job_submission_failures_total",
    "Job submissions answered 503 because a dependency failed: the Redis "
    "store or the RabbitMQ queue.",
    ["component"],  # "store" | "queue"
)

# Worker metrics (python -m app.worker). The worker runs the backend too, but
# does not record the vortex_inference_* request series: those describe the
# synchronous API, which the latency and error SLOs are defined on.

WORKER_JOBS_TOTAL = Counter(
    "vortex_worker_jobs_total",
    "Job deliveries handled by the worker, by how each one ended: succeeded "
    "and failed are final; retried means a transient failure with attempts "
    "left; deferred means Redis was unreachable and the job was put back "
    "without using an attempt; skipped means the job had already finished "
    "or expired; rejected means the message was unreadable and was "
    "dead-lettered.",
    ["outcome"],
)

WORKER_JOB_DURATION_SECONDS = Histogram(
    "vortex_worker_job_duration_seconds",
    "Time one attempt spent in the backend, by its result: succeeded, "
    "failed or retried.",
    ["outcome"],
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, float("inf")],
)

WORKER_JOB_WAIT_SECONDS = Histogram(
    "vortex_worker_job_wait_seconds",
    "Time from a job's submission until its first attempt started: the "
    "backlog a client waits behind. Second resolution, as the job records "
    "store their timestamps.",
    buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, float("inf")],
)

WORKER_JOBS_IN_PROGRESS = Gauge(
    "vortex_worker_jobs_in_progress",
    "Jobs the worker has received and not yet acknowledged.",
)
