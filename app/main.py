"""
Vortex Inference — FastAPI serving layer for language models.

Completions come from a pluggable backend (app/backends): a GGUF model run
in-process by llama.cpp, or an OpenAI-compatible API.

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
  An asyncio.Semaphore sized by max_concurrent_requests bounds inference
  calls within a pod. With llama.cpp it is 1, serializing access to the
  model; horizontal scaling via KEDA handles traffic spikes, and the per-pod
  memory footprint stays predictable under strict Kubernetes limits. Remote
  backends are I/O-bound and can run with a higher limit.
"""

import asyncio
import contextlib
import contextvars
import functools
import hmac
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

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
from pydantic import BaseModel, Field, StringConstraints, model_validator
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from app.backends import (
    BackendError,
    InferenceBackend,
    PromptTooLongError,
    RequestRejectedError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    create_backend,
)
from app.config import Settings
from app.logging_config import (
    RequestContextMiddleware,
    configure_logging,
    request_id_var,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# logfmt output aligns with Promtail/Loki's built-in logfmt parser, enabling
# label extraction without a custom pipeline stage. See app/logging_config.py.
configure_logging()
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

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------
# Module-level references avoid re-initialization on each request.
backend: InferenceBackend | None = None

# The semaphore is initialized during lifespan startup, not at import time,
# because asyncio primitives must be created inside a running event loop.
_inference_semaphore: asyncio.Semaphore | None = None

# Dedicated worker pool, sized to the semaphore. asyncio's default executor
# has min(32, cpu_count + 4) threads, which on a 2-CPU pod would cap a remote
# backend at 6 concurrent calls whatever max_concurrent_requests says. A
# separate pool also keeps inference from competing with any other use of
# the default executor. One thread per permit is enough: a permit is held
# until its thread exits, including after a timeout.
_inference_executor: ThreadPoolExecutor | None = None

# Strong references to in-flight drain tasks. The event loop holds tasks only
# weakly, so an unreferenced create_task() result can be garbage-collected
# before it completes, and with it the only code path that releases the
# semaphore permit. The set is also what shutdown waits on.
_drain_tasks: set[asyncio.Task[None]] = set()


async def _drain_and_release(
    fut: asyncio.Future, sem: asyncio.Semaphore, abandoned: bool
) -> None:
    """
    Waits for an inference thread to exit, then releases its permit.

    The permit must outlive the request that abandoned the thread: releasing
    it earlier would let the next queued request enter the backend (for
    llama.cpp, a non-thread-safe model) concurrently with the call that is
    still unwinding. Streamed requests always release this way, since their
    thread can outlive the response whether or not it was abandoned.
    """
    try:
        await fut
    except Exception:
        # The client has already been answered; the failure is only worth logging.
        logger.exception("Inference thread raised while draining.")
    finally:
        sem.release()
        INFERENCE_ACTIVE_REQUESTS.dec()
        if abandoned:
            logger.info("Abandoned inference thread drained. Semaphore released.")


def _hand_off_permit(
    fut: asyncio.Future, sem: asyncio.Semaphore, abandoned: bool = True
) -> None:
    task = asyncio.get_running_loop().create_task(
        _drain_and_release(fut, sem, abandoned)
    )
    _drain_tasks.add(task)
    task.add_done_callback(_drain_tasks.discard)


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the backend lifecycle: create on startup, drain on shutdown.

    Fail-fast pattern: if backend creation raises, the exception propagates to
    uvicorn, which exits with a non-zero code. Kubernetes interprets this as
    a CrashLoopBackOff and applies exponential backoff before restarting —
    the correct behavior when the underlying issue (missing file, corrupt
    weights, OOM) requires human intervention, not an immediate retry.
    """
    global backend, _inference_semaphore, _inference_executor

    logger.info(
        "Starting inference engine initialization.",
        extra={"backend": settings.backend, "model": settings.model_name},
    )

    load_start = time.monotonic()
    try:
        backend = create_backend(settings)
    except Exception as exc:
        logger.critical(
            "Inference backend failed to start. Exiting so Kubernetes applies "
            "CrashLoopBackOff.",
            exc_info=True,
        )
        raise RuntimeError(
            f"Inference server cannot start without a working {settings.backend} "
            "backend. For llama_cpp, check model_path, available memory, and "
            "file integrity."
        ) from exc

    load_duration = time.monotonic() - load_start
    MODEL_LOAD_DURATION_SECONDS.set(load_duration)
    logger.info(
        "Model loaded. Ready for traffic.",
        extra={"duration_seconds": round(load_duration, 2)},
    )

    _inference_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    _inference_executor = ThreadPoolExecutor(
        max_workers=settings.max_concurrent_requests,
        thread_name_prefix="inference",
    )

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
    # backend is torn down with the process; for llama.cpp, native code still
    # running against a freed model would crash instead of exiting cleanly.
    # terminationGracePeriodSeconds must cover this wait plus the preStop sleep.
    pending = set(_drain_tasks)
    if pending:
        logger.info(
            "Shutdown signal received. Waiting for abandoned inference threads.",
            extra={"threads": len(pending)},
        )
        _, still_running = await asyncio.wait(
            pending, timeout=settings.inference_timeout_seconds
        )
        if still_running:
            logger.error(
                "Inference threads did not exit before the shutdown deadline.",
                extra={"threads": len(still_running)},
            )
    # wait=False: a thread that outlived the deadline above is not waited on
    # again; the process exits regardless.
    _inference_executor.shutdown(wait=False, cancel_futures=True)
    backend.close()
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
app.add_middleware(RequestContextMiddleware)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
# auto_error=False lets the dependency decide: when no key is configured the
# header is irrelevant, and when one is configured a missing header and a
# wrong token both map to the same 401 instead of FastAPI's default 403.
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ],
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
class StreamOptions(BaseModel):
    # Sends one extra chunk before [DONE] with usage and no choices.
    include_usage: bool = False

    model_config = {"extra": "forbid"}


class CompletionRequest(BaseModel):
    # Accepted because the OpenAI SDKs always send it. The server hosts a
    # single model, so the value is not used for routing.
    model: str | None = Field(default=None, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=32768)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    # Each stop sequence is matched against the generated text after every
    # token, so both count and length are bounded. Empty strings are rejected:
    # they match immediately and would end generation before the first token.
    stop: (
        list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] | None
    ) = Field(default=None, max_length=4)
    # stream=true answers with server-sent events in OpenAI's chunk format.
    stream: bool = False
    stream_options: StreamOptions | None = None

    model_config = {"extra": "forbid"}
    # extra="forbid" prevents prompt injection via undocumented parameters
    # that could be processed by future versions of the model loader.

    @model_validator(mode="after")
    def stream_options_require_stream(self) -> "CompletionRequest":
        # Matches OpenAI, which rejects stream_options on a buffered request
        # instead of silently ignoring it.
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options is only allowed when stream is true.")
        return self


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: None = None
    finish_reason: str | None  # "stop" = stop sequence/EOS; "length" = max_tokens


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionPerformance(BaseModel):
    tokens_per_second: float
    duration_seconds: float


class CompletionResponse(BaseModel):
    """
    Mirrors OpenAI's text_completion object so the official SDKs and other
    OpenAI-compatible clients parse responses natively. `performance` is a
    server-specific extension; those clients ignore unknown fields.
    """

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage
    performance: CompletionPerformance


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

# Client-facing text for BackendError, by status. The exception message holds
# the upstream details and goes to the log only.
_BACKEND_ERROR_DETAIL = {
    502: "The inference backend failed to produce a completion.",
    503: "The inference backend is overloaded. Retry later.",
    504: "The inference backend did not respond in time.",
}


def _timeout_detail() -> str:
    return (
        f"Inference exceeded {settings.inference_timeout_seconds}s timeout. "
        "Reduce max_tokens or retry when the service has scaled out."
    )


def _classify_failure(exc: Exception) -> tuple[str, HTTPException]:
    """
    Maps a backend failure to its metric outcome and client response, and
    logs it. Shared by buffered and streamed requests, so a failure is
    reported the same way whichever the client asked for.
    """
    if isinstance(exc, RequestRejectedError):
        # A client error, not a service failure: counted separately so it
        # does not inflate the error-ratio SLO.
        details = (
            {"tokens_prompt": exc.prompt_tokens, "n_ctx": exc.n_ctx}
            if isinstance(exc, PromptTooLongError)
            else {}
        )
        logger.info("Request rejected.", extra={"reason": str(exc), **details})
        return "rejected", HTTPException(status_code=400, detail=str(exc))

    if isinstance(exc, BackendError):
        # Upstream timeouts share the "timeout" status with local deadline
        # overruns so the latency SLO sees both.
        outcome = "timeout" if exc.status_code == 504 else "error"
        logger.error(
            "Inference backend failed.",
            extra={"reason": str(exc), "status_code": exc.status_code},
        )
        headers = (
            {"Retry-After": exc.retry_after}
            if exc.status_code == 503 and exc.retry_after
            else None
        )
        return outcome, HTTPException(
            status_code=exc.status_code,
            detail=_BACKEND_ERROR_DETAIL.get(
                exc.status_code, _BACKEND_ERROR_DETAIL[502]
            ),
            headers=headers,
        )

    # Full traceback goes to the server log only. Exception messages from
    # llama.cpp and the OS routinely include file paths and allocator state,
    # which should not be returned to API clients.
    logger.error("Inference failed.", exc_info=exc)
    return "error", HTTPException(
        status_code=500, detail="Inference failed due to an internal error."
    )


def _record_outcome(outcome: str, request_start: float) -> None:
    INFERENCE_REQUESTS_TOTAL.labels(status=outcome).inc()
    INFERENCE_DURATION_SECONDS.labels(status=outcome).observe(
        time.monotonic() - request_start
    )


def _record_usage(tokens_prompt: int, tokens_completion: int, duration: float) -> float:
    tps = tokens_completion / duration if duration > 0 else 0.0
    TOKENS_GENERATED_TOTAL.labels(type="prompt").inc(tokens_prompt)
    TOKENS_GENERATED_TOTAL.labels(type="completion").inc(tokens_completion)
    TOKENS_PER_SECOND.set(tps)
    return tps


def _inference_stalled() -> bool:
    """
    True when a worker thread has overrun the request deadline by a wide margin.

    On timeout the stop event halts llama.cpp at the next token boundary,
    which takes well under a second. A thread still inside the native call
    STALL_GRACE_SECONDS after the deadline is wedged: it holds the only
    inference permit, so every queued request will time out behind it and
    only a process restart recovers the pod.
    """
    if backend is None:
        return False
    return (
        backend.busy_seconds()
        > settings.inference_timeout_seconds + STALL_GRACE_SECONDS
    )


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
    if backend is None or not backend.is_ready():
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
    responses={200: {"content": {"text/event-stream": {}}}},
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
async def create_completion(
    request: CompletionRequest,
) -> CompletionResponse | StreamingResponse:
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
    if backend is None or _inference_semaphore is None or _inference_executor is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    if request.stream:
        return await _stream_completion(request)

    # Increment before acquiring the semaphore so queued requests contribute to
    # the KEDA scaling signal. Do NOT move this inside the semaphore.
    INFERENCE_ACTIVE_REQUESTS.inc()

    _semaphore_acquired = False
    _drain_pending = False

    # Set once the permit is held and on every terminal path respectively;
    # the finally block records latency from them so each outcome is
    # observed exactly once.
    request_start: float | None = None
    outcome: str | None = None

    try:
        enqueued_at = time.monotonic()
        await _inference_semaphore.acquire()
        _semaphore_acquired = True

        # request_start is measured after acquiring the semaphore so that
        # INFERENCE_DURATION_SECONDS reflects pure inference time; queue wait
        # is recorded separately.
        request_start = time.monotonic()
        QUEUE_WAIT_SECONDS.observe(request_start - enqueued_at)

        stop_event = threading.Event()
        loop = asyncio.get_running_loop()

        # asyncio.shield prevents wait_for from cancelling the executor future
        # on timeout. The thread must run to completion before the semaphore is
        # released: releasing it while the thread is still inside
        # backend.generate() would let another request enter, and the llama.cpp
        # model is not thread-safe, so shared C++ state would be corrupted.
        #
        # run_in_executor does not propagate context variables to the worker
        # thread, so the call is wrapped in a copy of the current context to
        # keep the request ID on log lines emitted from inside generate().
        ctx = contextvars.copy_context()
        fut = loop.run_in_executor(
            _inference_executor,
            functools.partial(ctx.run, backend.generate),
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
        except (TimeoutError, asyncio.CancelledError) as exc:
            # Both paths leave the worker thread running inside the backend.
            # Signal it to stop through stop_event (for llama.cpp, at the next
            # token boundary via its stopping_criteria callback), and hand the
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
                outcome = "cancelled"
                INFERENCE_REQUESTS_TOTAL.labels(status="cancelled").inc()
                logger.warning(
                    "Inference cancelled before completion. Worker thread "
                    "signalled to stop."
                )
                raise

            outcome = "timeout"
            INFERENCE_REQUESTS_TOTAL.labels(status="timeout").inc()
            logger.error(
                "Inference exceeded its deadline. Worker thread signalled to stop.",
                extra={
                    "timeout_seconds": settings.inference_timeout_seconds,
                    "max_tokens": request.max_tokens,
                },
            )
            raise HTTPException(status_code=504, detail=_timeout_detail()) from exc

        duration = time.monotonic() - request_start
        outcome = "success"
        INFERENCE_REQUESTS_TOTAL.labels(status="success").inc()
        tps = _record_usage(
            result["tokens_prompt"], result["tokens_completion"], duration
        )

        logger.info(
            "Inference complete.",
            extra={
                "duration_seconds": round(duration, 3),
                "tokens_prompt": result["tokens_prompt"],
                "tokens_completion": result["tokens_completion"],
                "tokens_per_second": round(tps, 2),
                "finish_reason": result["finish_reason"],
            },
        )

        # Reusing the request ID makes a completion traceable to its log lines.
        completion_id = request_id_var.get() or uuid.uuid4().hex
        return CompletionResponse(
            id=f"cmpl-{completion_id}",
            created=int(time.time()),
            model=settings.model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=result["text"],
                    finish_reason=result["finish_reason"],
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=result["tokens_prompt"],
                completion_tokens=result["tokens_completion"],
                total_tokens=result["tokens_prompt"] + result["tokens_completion"],
            ),
            performance=CompletionPerformance(
                tokens_per_second=round(tps, 3),
                duration_seconds=round(duration, 3),
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        outcome, http_exc = _classify_failure(exc)
        INFERENCE_REQUESTS_TOTAL.labels(status=outcome).inc()
        raise http_exc from exc

    finally:
        if request_start is not None and outcome is not None:
            INFERENCE_DURATION_SECONDS.labels(status=outcome).observe(
                time.monotonic() - request_start
            )

        # On the timeout and cancellation paths, _drain_and_release() owns the
        # semaphore release and gauge decrement once the thread exits. All
        # other paths (success, error, cancellation while still queued) are
        # handled here.
        if not _drain_pending:
            if _semaphore_acquired:
                _inference_semaphore.release()
            INFERENCE_ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
# A streamed request holds its permit for the whole generation, exactly like a
# buffered one: the worker thread drives the backend's generator and passes
# each event to the event loop through a queue, and the permit is released
# only when that thread exits. The response is committed (status 200, SSE
# headers) only after the first event arrives, so failures that happen before
# any output (a prompt that does not fit, an upstream 429) still get their
# proper status code. After that, a failure can only end the stream with an
# error event, which is what OpenAI's clients expect.

_STREAM_HEADERS = {
    # Proxies must pass events through as they are produced, not buffer the
    # response; X-Accel-Buffering is the nginx ingress spelling of that.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    """An exception raised in the worker thread, carried to the event loop."""

    exc: Exception


class _ClosingStreamingResponse(StreamingResponse):
    """
    StreamingResponse that always closes its body iterator.

    Starlette stops iterating when the client disconnects but leaves the
    generator suspended, so its cleanup (which stops generation and records
    the outcome) would run whenever the garbage collector finalizes it.
    Closing it here makes that cleanup immediate and deterministic.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.body_iterator.aclose()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _sse_error(exc: HTTPException) -> str:
    # The shape OpenAI uses for errors inside a stream; its SDKs raise an
    # APIError when they read it.
    error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return _sse(
        {"error": {"message": exc.detail, "type": error_type, "code": exc.status_code}}
    )


async def _next_event(
    events: asyncio.Queue, deadline: float
) -> StreamEvent | _StreamFailure:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return await asyncio.wait_for(events.get(), timeout=remaining)


async def _stream_completion(request: CompletionRequest) -> StreamingResponse:
    """Serves a completion as server-sent events in OpenAI's chunk format."""
    # Counted while queued, as for buffered requests: this is KEDA's signal.
    INFERENCE_ACTIVE_REQUESTS.inc()
    enqueued_at = time.monotonic()
    try:
        await _inference_semaphore.acquire()
    except BaseException:
        INFERENCE_ACTIVE_REQUESTS.dec()
        raise
    request_start = time.monotonic()
    QUEUE_WAIT_SECONDS.observe(request_start - enqueued_at)
    # One deadline for the whole stream, measured like the buffered timeout,
    # so a stream cannot hold the permit longer than a buffered request.
    deadline = request_start + settings.inference_timeout_seconds

    loop = asyncio.get_running_loop()
    events: asyncio.Queue[StreamEvent | _StreamFailure] = asyncio.Queue()
    stop_event = threading.Event()

    def produce() -> None:
        # Runs in the worker thread. The backend's generator is created,
        # consumed and closed here, so all of its code runs in the thread that
        # holds the permit, whether it completes, fails, or the request is
        # abandoned (stop_event).
        try:
            source = backend.stream(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.stop,
                stop_event,
            )
            with contextlib.closing(source):
                for event in source:
                    loop.call_soon_threadsafe(events.put_nowait, event)
                    if stop_event.is_set():
                        return
        except Exception as exc:
            loop.call_soon_threadsafe(events.put_nowait, _StreamFailure(exc))

    # Context is copied so log lines from the worker keep the request ID.
    ctx = contextvars.copy_context()
    fut = loop.run_in_executor(_inference_executor, functools.partial(ctx.run, produce))
    _hand_off_permit(fut, _inference_semaphore, abandoned=False)

    try:
        first = await _next_event(events, deadline)
    except TimeoutError as exc:
        stop_event.set()
        _record_outcome("timeout", request_start)
        logger.error(
            "Inference produced no output before its deadline.",
            extra={"timeout_seconds": settings.inference_timeout_seconds},
        )
        raise HTTPException(status_code=504, detail=_timeout_detail()) from exc
    except asyncio.CancelledError:
        stop_event.set()
        _record_outcome("cancelled", request_start)
        logger.warning("Streamed inference cancelled before its first output.")
        raise

    if isinstance(first, _StreamFailure):
        outcome, http_exc = _classify_failure(first.exc)
        _record_outcome(outcome, request_start)
        raise http_exc from first.exc

    include_usage = bool(
        request.stream_options and request.stream_options.include_usage
    )
    completion_id = f"cmpl-{request_id_var.get() or uuid.uuid4().hex}"
    return _ClosingStreamingResponse(
        _sse_body(
            first,
            events,
            deadline,
            stop_event,
            request_start,
            completion_id,
            include_usage,
        ),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


async def _sse_body(
    first: StreamEvent,
    events: asyncio.Queue,
    deadline: float,
    stop_event: threading.Event,
    request_start: float,
    completion_id: str,
    include_usage: bool,
) -> AsyncIterator[str]:
    created = int(time.time())

    def chunk(text: str, finish_reason: str | None) -> str:
        return _sse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": settings.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    outcome: str | None = None
    first_token_at: float | None = None
    event: StreamEvent | _StreamFailure = first
    try:
        while True:
            if isinstance(event, TextDelta):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                    TIME_TO_FIRST_TOKEN_SECONDS.observe(first_token_at - request_start)
                yield chunk(event.text, None)

            elif isinstance(event, StreamEnd):
                # Generation is complete at this point; a client that leaves
                # while the trailer is sent does not turn it into a cancel.
                outcome = "success"
                duration = time.monotonic() - request_start
                tps = _record_usage(
                    event.tokens_prompt, event.tokens_completion, duration
                )
                yield chunk("", event.finish_reason)
                if include_usage:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": settings.model_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": event.tokens_prompt,
                                "completion_tokens": event.tokens_completion,
                                "total_tokens": event.tokens_prompt
                                + event.tokens_completion,
                            },
                        }
                    )
                yield "data: [DONE]\n\n"

                logger.info(
                    "Inference complete.",
                    extra={
                        "streamed": True,
                        "duration_seconds": round(duration, 3),
                        "time_to_first_token_seconds": (
                            round(first_token_at - request_start, 3)
                            if first_token_at is not None
                            else None
                        ),
                        "tokens_prompt": event.tokens_prompt,
                        "tokens_completion": event.tokens_completion,
                        "tokens_per_second": round(tps, 2),
                        "finish_reason": event.finish_reason,
                    },
                )
                return

            else:
                outcome, http_exc = _classify_failure(event.exc)
                yield _sse_error(http_exc)
                return

            event = await _next_event(events, deadline)

    except TimeoutError:
        outcome = "timeout"
        logger.error(
            "Streamed inference exceeded its deadline. Worker thread signalled "
            "to stop.",
            extra={"timeout_seconds": settings.inference_timeout_seconds},
        )
        yield _sse_error(HTTPException(status_code=504, detail=_timeout_detail()))

    except (asyncio.CancelledError, GeneratorExit):
        # The client went away mid-stream. Nothing more can be sent; stopping
        # generation and recording the outcome happen below.
        if outcome is None:
            outcome = "cancelled"
            logger.warning("Client disconnected mid-stream. Worker thread signalled.")
        raise

    finally:
        # Harmless once generation has ended; otherwise it stops the worker
        # at the next token, and the drain task then releases the permit.
        stop_event.set()
        if outcome is not None:
            _record_outcome(outcome, request_start)


# ---------------------------------------------------------------------------
# Entry Point (local development only — production uses the Dockerfile CMD)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        # Loopback only: this entry point is for local development, where
        # authentication is off unless VORTEX_API_KEY is set. Binding all
        # interfaces would expose an open inference API to the local network.
        # The container image passes --host 0.0.0.0 explicitly in its CMD.
        host="127.0.0.1",
        port=settings.port,
        workers=1,  # Single worker is mandatory: model state is process-local.
        log_level="info",
        # RequestContextMiddleware writes the access log with request IDs;
        # uvicorn's own access log would duplicate every line without them.
        access_log=False,
    )
