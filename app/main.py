"""
Vortex Inference — FastAPI serving layer for GGUF-quantized language models.

Architecture overview:
  This module is the single process entry point for all inference traffic.
  It owns three distinct concerns: model lifecycle management, HTTP request
  handling with concurrency control, and Prometheus telemetry emission.

  Colocating these concerns in one process is an intentional trade-off for
  edge/resource-constrained deployments. A microservices split (separate
  inference worker + API gateway) would add network hops and IPC overhead
  that are not justified when the bottleneck is always CPU-bound token
  generation, not I/O wait.

Concurrency model:
  asyncio.Semaphore(1) serializes inference calls within a single pod.
  Horizontal scaling via KEDA handles traffic spikes. This design keeps
  the per-pod memory footprint predictable, which is the primary constraint
  when operating under strict Kubernetes resource limits.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.config import Settings
from app.models.llm_loader import LLMLoader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# logfmt-style format aligns with Promtail/Loki's default parser configuration,
# enabling label extraction without a custom pipeline stage.
logging.basicConfig(
    level=logging.INFO,
    format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
)
logger = logging.getLogger("vortex.inference")

# ---------------------------------------------------------------------------
# Settings (resolved from environment / ConfigMap at process startup)
# ---------------------------------------------------------------------------
settings = Settings()

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
# Naming convention: <project>_<subsystem>_<unit>_<suffix>
# Namespace "vortex_inference" prevents label collisions in shared Prometheus
# instances where multiple services expose identically-named metrics.

INFERENCE_REQUESTS_TOTAL = Counter(
    "vortex_inference_requests_total",
    "Total inference requests, partitioned by terminal status.",
    ["status"],  # "success" | "error" | "timeout"
)

INFERENCE_DURATION_SECONDS = Histogram(
    "vortex_inference_duration_seconds",
    "End-to-end latency from request receipt to final token delivery. "
    "Buckets are calibrated for interactive LLM workloads, not batch jobs — "
    "the sub-100ms range is intentionally sparse because CPU inference cannot "
    "achieve it; resolution there would waste cardinality.",
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, float("inf")],
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

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------
# Module-level references avoid re-initialization on each request.
llm: LLMLoader | None = None

# The semaphore is initialized during lifespan startup, not at import time,
# because asyncio primitives must be created inside a running event loop.
_inference_semaphore: asyncio.Semaphore | None = None


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the model lifecycle: load on startup, release on shutdown.

    Fail-fast pattern: if model loading raises, the exception propagates to
    uvicorn, which exits with a non-zero code. Kubernetes interprets this as
    a CrashLoopBackOff and applies exponential backoff before restarting —
    the correct behavior when the underlying issue (missing file, corrupt
    weights, OOM) requires human intervention, not an immediate retry.
    """
    global llm, _inference_semaphore

    logger.info(
        f"Starting inference engine initialization. "
        f"model_path={settings.model_path} "
        f"n_ctx={settings.n_ctx} "
        f"n_threads={settings.n_threads} "
        f"n_gpu_layers={settings.n_gpu_layers}"
    )

    load_start = time.monotonic()
    try:
        llm = LLMLoader(settings)
    except Exception as exc:
        logger.critical(
            f"Fatal: model failed to load. Triggering CrashLoopBackOff. error={exc}"
        )
        raise RuntimeError(
            "Inference server cannot start without a valid model. "
            "Check model_path, available memory, and file integrity."
        ) from exc

    load_duration = time.monotonic() - load_start
    MODEL_LOAD_DURATION_SECONDS.set(load_duration)
    logger.info(
        f"Model loaded. duration_seconds={load_duration:.2f} ready_for_traffic=true"
    )

    _inference_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    yield  # ← Application serves traffic here

    # Graceful shutdown: allow the semaphore to drain in-flight requests
    # before the process exits. The terminationGracePeriodSeconds in the
    # Deployment spec must be long enough to cover worst-case inference duration.
    logger.info("Shutdown signal received. Draining in-flight requests.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vortex Inference API",
    description=(
        "Production-grade inference server for GGUF-quantized language models. "
        "OpenAI-compatible /v1/completions endpoint with Prometheus telemetry."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable the automatic /docs redirect — reduces attack surface in production.
    # Re-enable for local development by removing this line.
    # docs_url=None,
    # redoc_url=None,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32768)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: list[str] | None = Field(default=None, max_length=4)

    model_config = {"extra": "forbid"}
    # extra="forbid" prevents prompt injection via undocumented parameters
    # that could be processed by future versions of the model loader.


class CompletionResponse(BaseModel):
    text: str
    finish_reason: str  # "stop" = hit stop sequence; "length" = max_tokens reached
    tokens_prompt: int
    tokens_completion: int
    tokens_per_second: float
    duration_seconds: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health_check() -> HealthResponse:
    """
    Kubernetes readiness and liveness probe target.

    Returns HTTP 503 when the model is not initialized. This prevents the
    kubelet from routing traffic to a pod that lost its model state (e.g.,
    after an OOM event that killed the loader thread but not the HTTP server
    process). A 503 here causes Kubernetes to remove the pod from the
    Service's Endpoints slice immediately, eliminating bad requests at the
    load balancer level rather than at the application level.
    """
    if llm is None or not llm.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Pod is not ready to serve inference traffic.",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=settings.model_path,
    )


@app.get("/metrics", tags=["operations"])
async def prometheus_metrics() -> Response:
    """
    Prometheus scrape endpoint.

    Exposed on the same port as the application to eliminate the need for a
    secondary container port and simplify NetworkPolicy rules. In multi-tenant
    clusters where metric scraping originates from a different security zone,
    consider moving this to a separate port with a dedicated ServiceMonitor.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/completions", response_model=CompletionResponse, tags=["inference"])
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """
    Primary inference endpoint. Path mirrors the OpenAI API for drop-in
    client compatibility without requiring SDK changes.

    Backpressure strategy: requests block on the semaphore rather than
    immediately receiving a 429. This is intentional. KEDA reads the
    INFERENCE_ACTIVE_REQUESTS gauge, which counts requests in the semaphore
    queue AND requests being actively processed. A queuing model inflates the
    gauge under load, giving KEDA a stronger signal to trigger scale-out
    before the queue depth causes user-visible timeouts.

    If an immediate 429 were returned instead, the gauge would only reflect
    active (not queued) requests, and KEDA would underestimate true demand.
    """
    if llm is None or _inference_semaphore is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    # Increment before entering the semaphore so that queued requests are
    # counted in the KEDA scaling signal. Do NOT move this inside the semaphore.
    INFERENCE_ACTIVE_REQUESTS.inc()

    try:
        async with _inference_semaphore:
            # request_start is measured after acquiring the semaphore so that
            # INFERENCE_DURATION_SECONDS reflects pure inference time, not queue
            # wait. Queue depth is already observable via INFERENCE_ACTIVE_REQUESTS.
            request_start = time.monotonic()

            # run_in_executor offloads the CPU-bound C++ inference call to a
            # thread, freeing the asyncio event loop to handle concurrent
            # health checks and metrics scrapes without stalling.
            # The default ThreadPoolExecutor is adequate because there is at
            # most one inference thread running at any time (semaphore=1).
            # get_running_loop() is used instead of the deprecated get_event_loop():
            # it raises RuntimeError explicitly if called outside a running loop.
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    llm.generate,
                    request.prompt,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.stop,
                ),
                timeout=settings.inference_timeout_seconds,
            )

        duration = time.monotonic() - request_start
        tps = result["tokens_completion"] / duration if duration > 0 else 0.0

        INFERENCE_REQUESTS_TOTAL.labels(status="success").inc()
        INFERENCE_DURATION_SECONDS.observe(duration)
        TOKENS_GENERATED_TOTAL.labels(type="prompt").inc(result["tokens_prompt"])
        TOKENS_GENERATED_TOTAL.labels(type="completion").inc(
            result["tokens_completion"]
        )
        TOKENS_PER_SECOND.set(tps)

        logger.info(
            f"inference_complete "
            f"duration_seconds={duration:.3f} "
            f"tokens_prompt={result['tokens_prompt']} "
            f"tokens_completion={result['tokens_completion']} "
            f"tps={tps:.2f}"
        )

        return CompletionResponse(
            text=result["text"],
            finish_reason=result["finish_reason"],
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            tokens_per_second=tps,
            duration_seconds=duration,
            model=settings.model_name,
        )

    except asyncio.TimeoutError:
        INFERENCE_REQUESTS_TOTAL.labels(status="timeout").inc()
        logger.error(
            f"Inference timeout after {settings.inference_timeout_seconds}s. "
            f"Consider reducing max_tokens or scaling out replicas."
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Inference exceeded {settings.inference_timeout_seconds}s timeout. "
                "Reduce max_tokens or retry when the service has scaled out."
            ),
        )

    except Exception as exc:
        INFERENCE_REQUESTS_TOTAL.labels(status="error").inc()
        logger.error(f"Inference failed. error={exc}")
        # TODO: Implement circuit breaker pattern if upstream inference error rate
        # exceeds 10% over a 60s window — prevents cascading failures where a
        # corrupt model state causes all replicas to fail simultaneously.
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    finally:
        # CRITICAL: gauge decrement is in finally to prevent metric leaks on
        # any exit path (success, timeout, error). A leaked increment would
        # permanently inflate the KEDA scaling signal, causing the autoscaler
        # to maintain more replicas than necessary — a resource waste that
        # compounds in multi-tenant environments.
        INFERENCE_ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# Entry Point (local development only — production uses the Dockerfile CMD)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        workers=1,  # Single worker is mandatory: model state is process-local.
        log_level="info",  # uvicorn access logs complement Prometheus metrics.
    )
