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
import hmac
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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

# Strong references to in-flight drain tasks. The event loop holds tasks only
# weakly, so an unreferenced create_task() result can be garbage-collected
# before it completes, and with it the only code path that releases the
# semaphore permit. The set is also what shutdown waits on.
_drain_tasks: set[asyncio.Task[None]] = set()


async def _drain_and_release(fut: asyncio.Future, sem: asyncio.Semaphore) -> None:
    """
    Waits for an abandoned inference thread to exit, then releases its permit.

    The permit must outlive the request that abandoned the thread: releasing
    it earlier would let the next queued request enter the non-thread-safe
    Llama object concurrently with the call that is still unwinding.
    """
    try:
        await fut
    except Exception:
        # The client has already been answered; the failure is only worth logging.
        logger.exception("Abandoned inference thread raised while draining.")
    finally:
        sem.release()
        INFERENCE_ACTIVE_REQUESTS.dec()
        logger.info("Abandoned inference thread drained. Semaphore released.")


def _hand_off_permit(fut: asyncio.Future, sem: asyncio.Semaphore) -> None:
    task = asyncio.get_running_loop().create_task(_drain_and_release(fut, sem))
    _drain_tasks.add(task)
    task.add_done_callback(_drain_tasks.discard)


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

    if settings.api_key is None:
        logger.warning(
            "VORTEX_API_KEY is not set. /v1/completions accepts unauthenticated "
            "requests; this is only acceptable for local development."
        )
    else:
        logger.info("API key authentication enabled for /v1/completions.")

    yield  # ← Application serves traffic here

    # Graceful shutdown. uvicorn has stopped accepting connections and waited
    # for open requests; any request it had to cancel handed its worker
    # thread to a drain task. Waiting for those threads matters because the
    # Llama object is freed when the process tears down, and native code still
    # running against it would crash instead of exiting cleanly.
    # terminationGracePeriodSeconds must cover this wait plus the preStop sleep.
    pending = set(_drain_tasks)
    if pending:
        logger.info(
            f"Shutdown signal received. Waiting for {len(pending)} "
            "abandoned inference thread(s) to exit."
        )
        _, still_running = await asyncio.wait(
            pending, timeout=settings.inference_timeout_seconds
        )
        if still_running:
            logger.error(
                f"{len(still_running)} inference thread(s) did not exit before "
                "the shutdown deadline."
            )
    logger.info("Shutdown complete.")


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
    # openapi_url is disabled together with the UIs; otherwise the schema
    # would remain reachable at /openapi.json with the docs pages turned off.
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
# auto_error=False lets the dependency decide: when no key is configured the
# header is irrelevant, and when one is configured a missing header and a
# wrong token both map to the same 401 instead of FastAPI's default 403.
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """
    Enforces the shared bearer token when VORTEX_API_KEY is configured.

    Bearer is used rather than a custom header so that OpenAI SDK clients,
    which always send `Authorization: Bearer <key>`, work unchanged.
    /health and /metrics stay unauthenticated: kubelet probes and the
    Prometheus scraper cannot attach credentials without extra plumbing, and
    neither endpoint triggers inference.
    """
    if settings.api_key is None:
        return

    expected = settings.api_key.get_secret_value().encode()
    provided = credentials.credentials.encode() if credentials else b""

    # Constant-time comparison so response timing does not leak how many
    # leading characters of a guessed key were correct.
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
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
    # Reports the logical model name only. The on-disk path is an internal
    # deployment detail and this endpoint is unauthenticated by design.
    status: str
    model_loaded: bool
    model: str


class LivenessResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# Margin past the inference deadline before a still-running worker thread is
# treated as wedged rather than slow to observe its stop event.
STALL_GRACE_SECONDS = 60.0


def _inference_stalled() -> bool:
    """
    True when a worker thread has overrun the request deadline by a wide margin.

    On timeout the stop event halts llama.cpp at the next token boundary,
    which takes well under a second. A thread still inside the native call
    STALL_GRACE_SECONDS after the deadline is wedged: it holds the only
    inference permit, so every queued request will time out behind it and
    only a process restart recovers the pod.
    """
    if llm is None:
        return False
    return llm.busy_seconds() > settings.inference_timeout_seconds + STALL_GRACE_SECONDS


@app.get("/health/live", response_model=LivenessResponse, tags=["operations"])
async def liveness() -> LivenessResponse:
    """
    Liveness probe target: fails only when a restart would fix the problem.

    A slow request or a long queue is not a liveness failure; KEDA handles
    load by scaling out, and restarting a busy pod would discard its queue
    and pay the model load cost again. The two signals that do warrant a
    restart are a blocked event loop (this handler cannot answer, so the
    probe times out) and a wedged inference thread.
    """
    if _inference_stalled():
        raise HTTPException(
            status_code=503,
            detail="Inference worker is stalled past its deadline.",
        )
    return LivenessResponse(status="alive")


@app.get("/health/ready", response_model=HealthResponse, tags=["operations"])
@app.get("/health", response_model=HealthResponse, include_in_schema=False)
async def readiness() -> HealthResponse:
    """
    Readiness and startup probe target: gates traffic routing.

    Not ready until the model is loaded, and not ready while the inference
    worker is stalled, so the pod leaves the Service endpoints within one
    readiness window, well before liveness restarts it. /health is kept as
    an alias for existing clients and scripts.
    """
    if llm is None or not llm.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Pod is not ready to serve inference traffic.",
        )
    if _inference_stalled():
        raise HTTPException(
            status_code=503,
            detail="Inference worker is stalled past its deadline.",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model=settings.model_name,
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


@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
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

    # Increment before acquiring the semaphore so queued requests contribute to
    # the KEDA scaling signal. Do NOT move this inside the semaphore.
    INFERENCE_ACTIVE_REQUESTS.inc()

    _semaphore_acquired = False
    _drain_pending = False

    try:
        await _inference_semaphore.acquire()
        _semaphore_acquired = True

        # request_start is measured after acquiring the semaphore so that
        # INFERENCE_DURATION_SECONDS reflects pure inference time, not queue
        # wait. Queue depth is already observable via INFERENCE_ACTIVE_REQUESTS.
        request_start = time.monotonic()

        stop_event = threading.Event()
        loop = asyncio.get_running_loop()

        # asyncio.shield prevents wait_for from cancelling the executor future
        # on timeout. The thread must run to completion before the semaphore is
        # released — the Llama object is not thread-safe, and releasing the
        # semaphore while the thread is still inside llm.generate() would allow
        # a second request to enter, corrupting shared C++ model state.
        fut = loop.run_in_executor(
            None,
            llm.generate,
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.stop,
            stop_event,
        )

        try:
            result = await asyncio.wait_for(
                asyncio.shield(fut), timeout=settings.inference_timeout_seconds
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            # Both paths leave the worker thread running inside llama.cpp.
            # Signal it to stop at the next token boundary via the
            # stopping_criteria callback in LLMLoader.generate(), and hand the
            # permit to a drain task that releases it only once the thread
            # has exited. The response (or the cancellation) proceeds
            # immediately instead of blocking on the draining thread.
            #
            # CancelledError must be handled here explicitly: it derives from
            # BaseException, so without this clause it bypassed the old
            # timeout-only branch and the finally block released the permit
            # while the thread was still inside the non-thread-safe model.
            stop_event.set()
            _hand_off_permit(fut, _inference_semaphore)
            _drain_pending = True

            if isinstance(exc, asyncio.CancelledError):
                INFERENCE_REQUESTS_TOTAL.labels(status="cancelled").inc()
                logger.warning(
                    "Inference cancelled before completion. Worker thread "
                    "signalled to stop."
                )
                raise

            INFERENCE_REQUESTS_TOTAL.labels(status="timeout").inc()
            logger.error(
                f"Inference timeout after {settings.inference_timeout_seconds}s. "
                f"Consider reducing max_tokens or scaling out replicas."
            )
            timeout_s = settings.inference_timeout_seconds
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Inference exceeded {timeout_s}s timeout. "
                    "Reduce max_tokens or retry when the service has scaled out."
                ),
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

    except HTTPException:
        raise

    except Exception as exc:
        INFERENCE_REQUESTS_TOTAL.labels(status="error").inc()
        # Full traceback goes to the server log only. Exception messages from
        # llama.cpp and the OS routinely include file paths and allocator
        # state, which should not be returned to API clients.
        logger.exception("Inference failed.")
        raise HTTPException(
            status_code=500, detail="Inference failed due to an internal error."
        ) from exc

    finally:
        # On the timeout and cancellation paths, _drain_and_release() owns the
        # semaphore release and gauge decrement once the thread exits. All
        # other paths (success, error, cancellation while still queued) are
        # handled here.
        if not _drain_pending:
            if _semaphore_acquired:
                _inference_semaphore.release()
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
