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
    "Streamed requests only: time from acquiring the inference permit to the "
    "first text chunk. Queue wait is excluded, as in "
    "vortex_inference_duration_seconds; the delay a client sees before the "
    "first token is the sum of both. For local models it is dominated by "
    "prompt processing, for remote backends by the network and the "
    "provider's own queueing and thinking.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, float("inf")],
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
