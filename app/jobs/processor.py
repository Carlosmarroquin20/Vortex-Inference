"""
Running one job: from a queued message to a stored result.

JobProcessor decides what happens to each delivery and returns that decision
instead of acting on the message itself, so the AMQP consumer (app/worker.py)
stays a thin adapter and every branch here is testable without a broker.

The job record in Redis is the source of truth for progress. A delivery is
acknowledged only after the record reflects its outcome; if Redis cannot be
written, the job goes back to RabbitMQ and runs again later. Delivery is
therefore at least once: a worker that dies after storing a result but
before acknowledging leaves a message that the next worker skips, because
the record already says the job finished.
"""

import asyncio
import contextvars
import functools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import ValidationError

from app import tracing
from app.backends import BackendError, InferenceBackend
from app.completions import (
    Failure,
    Finished,
    Generation,
    chat_completion_response,
    classify_failure,
    completion_response,
    timeout_detail,
)
from app.jobs.models import Job, JobError, JobMessage
from app.jobs.store import JobStore, JobStoreUnavailable
from app.metrics import (
    TOKENS_GENERATED_TOTAL,
    WORKER_JOB_DURATION_SECONDS,
    WORKER_JOB_WAIT_SECONDS,
)
from app.schemas import ChatCompletionRequest, CompletionRequest

logger = logging.getLogger("vortex.worker")

# Backend failures worth another attempt: the provider was overloaded, rate
# limited, unreachable or slow. Rejections (400) and internal errors (500)
# would fail the same way again, and so would a local deadline overrun: the
# job asks for more generation than inference_timeout_seconds allows.
_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class InferenceTimeout(Exception):
    """The generation overran inference_timeout_seconds and was stopped."""


class Runner(Protocol):
    async def run(self, generation: Generation) -> Finished: ...

    def close(self) -> None: ...


class BackendRunner:
    """
    Runs generations on the backend in a dedicated thread pool.

    The pool has one thread per concurrent job, and the worker's prefetch
    matches it, so a job never waits for a thread. A generation that overruns
    its deadline is told to stop and awaited before run() returns: the
    thread is still inside the backend until then, and for llama.cpp the
    next job must not enter the model while it is.
    """

    def __init__(
        self, backend: InferenceBackend, concurrency: int, timeout_seconds: float
    ) -> None:
        self._backend = backend
        self._timeout = timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="inference"
        )

    async def run(self, generation: Generation) -> Finished:
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        # Copied so the job ID on log lines from inside the backend survives
        # the hop to the worker thread.
        ctx = contextvars.copy_context()
        started = time.monotonic()
        fut = loop.run_in_executor(
            self._executor,
            functools.partial(ctx.run, self._backend.generate),
            generation.prompt,
            generation.max_tokens,
            generation.temperature,
            generation.top_p,
            generation.stop,
            stop_event,
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), self._timeout)
        except (TimeoutError, asyncio.CancelledError) as exc:
            stop_event.set()
            try:
                await asyncio.shield(fut)
            except Exception:
                logger.warning(
                    "Stopped generation raised while draining.", exc_info=True
                )
            if isinstance(exc, TimeoutError):
                raise InferenceTimeout from exc
            raise
        duration = time.monotonic() - started
        tokens = result["tokens_completion"]
        return Finished(
            result=result,
            duration=duration,
            tokens_per_second=tokens / duration if duration > 0 else 0.0,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True, slots=True)
class Decision:
    """What the consumer does with a delivery."""

    # ack: done with it. retry: publish it to the retry queue for
    # retry_attempt, then ack. reject: dead-letter it.
    action: Literal["ack", "retry", "reject"]
    # For vortex_worker_jobs_total.
    outcome: Literal[
        "succeeded", "failed", "retried", "deferred", "skipped", "rejected"
    ]
    retry_attempt: int = 1
    # The parsed message, for retries; None when it could not be read.
    message: JobMessage | None = None


def _now() -> int:
    return int(time.time())


class JobProcessor:
    def __init__(
        self,
        store: JobStore,
        runner: Runner,
        model_name: str,
        max_attempts: int,
        timeout_seconds: float,
        provider: str = "unknown",
    ) -> None:
        self._store = store
        self._runner = runner
        self._model = model_name
        # The backend's name, for the generation span (gen_ai.provider.name).
        self._provider = provider
        self._max_attempts = max_attempts
        self._timeout = timeout_seconds

    async def process(self, body: bytes) -> Decision:
        try:
            message = JobMessage.model_validate_json(body)
        except ValidationError as exc:
            # Not something this code ever publishes; retrying cannot help, so
            # it goes to the dead-letter queue for someone to look at.
            logger.error(
                "Unreadable job message. Dead-lettering it.",
                extra={"reason": str(exc)[:300]},
            )
            return Decision("reject", "rejected")

        try:
            return await self._process(message)
        except JobStoreUnavailable as exc:
            # Nothing can be recorded, so nothing is decided: the job returns
            # through the first retry queue without using an attempt, and
            # runs again once Redis is back. The delay keeps a Redis outage
            # from turning into a tight redelivery loop.
            logger.warning(
                "Job store unavailable. Deferring the job.",
                extra={"reason": str(exc)},
            )
            return Decision("retry", "deferred", retry_attempt=1, message=message)
        except Exception:
            # A bug rather than a property of the job's input or the
            # upstream. Requeueing is not an option: RabbitMQ 4 does not count
            # a requeue towards the delivery limit, so the message would be
            # redelivered forever. It is dead-lettered for inspection and the
            # job marked failed, so the client gets an answer.
            logger.exception("Job processing failed unexpectedly. Dead-lettering it.")
            await self._fail_after_error(message.job_id)
            return Decision("reject", "failed")

    async def _fail_after_error(self, job_id: str) -> None:
        try:
            job = await self._store.get(job_id)
            if job is not None and job.status not in ("succeeded", "failed"):
                await self._finish_failed(
                    job,
                    JobError(
                        status_code=500,
                        message="The job failed due to an internal error.",
                    ),
                )
        except Exception:
            # The message is dead-lettered either way; the record then
            # stays as it was until it expires.
            logger.warning("Could not mark the job failed.", exc_info=True)

    async def _process(self, message: JobMessage) -> Decision:
        job = await self._store.get(message.job_id)
        if job is None:
            # Expired while queued, or removed because its submission failed;
            # nobody can read a result for it any more.
            logger.warning("Job record not found. Dropping the message.")
            return Decision("ack", "skipped")
        if job.status in ("succeeded", "failed"):
            # A redelivery of a job that already finished: its worker stored
            # the result but did not get to acknowledge the message.
            logger.info("Job already finished. Skipping.", extra={"status": job.status})
            return Decision("ack", "skipped")
        if job.attempts >= self._max_attempts:
            # Every attempt so far was started and never recorded an outcome:
            # each one's worker was killed or crashed mid-job. RabbitMQ's
            # delivery limit would dead-letter the message shortly; marking
            # the job failed here gives the client an answer instead of a job
            # stuck in "running".
            await self._finish_failed(
                job,
                JobError(
                    status_code=500,
                    message=(
                        f"The job was interrupted on each of its {job.attempts} "
                        "attempts."
                    ),
                ),
            )
            logger.error(
                "Job interrupted on every attempt. Marked failed.",
                extra={"attempts": job.attempts},
            )
            return Decision("ack", "failed")

        try:
            generation = self._generation(message)
        except ValidationError as exc:
            # The API validated this body when it was submitted; a worker
            # running a newer version can still disagree, and it would on
            # every attempt.
            await self._finish_failed(
                job,
                JobError(status_code=400, message=f"Invalid job body: {exc}"[:1000]),
            )
            logger.error("Job body failed validation. Marked failed.")
            return Decision("ack", "failed")

        if job.attempts == 0:
            WORKER_JOB_WAIT_SECONDS.observe(max(0, _now() - job.created_at))
        job = job.model_copy(
            update={
                "status": "running",
                "started_at": _now(),
                "attempts": job.attempts + 1,
                "error": None,
            }
        )
        await self._store.save(job)
        logger.info("Job started.", extra={"attempt": job.attempts, "url": job.url})

        started = time.monotonic()
        span = tracing.start_generation_span(generation, self._model, self._provider)
        try:
            done = await self._runner.run(generation)
        except asyncio.CancelledError:
            tracing.end_generation_span(span, outcome="cancelled")
            raise
        except InferenceTimeout:
            logger.error(
                "Job exceeded the inference deadline.",
                extra={"timeout_seconds": self._timeout},
            )
            failure = Failure("timeout", 504, timeout_detail(self._timeout))
            retryable = False
        except Exception as exc:
            failure = classify_failure(exc)
            retryable = (
                isinstance(exc, BackendError)
                and failure.status_code in _RETRYABLE_STATUSES
            )
        else:
            tracing.end_generation_span(
                span,
                tokens_prompt=done.result["tokens_prompt"],
                tokens_completion=done.result["tokens_completion"],
                finish_reason=done.result["finish_reason"],
                response_model=done.result.get("model"),
            )
            return await self._succeed(job, message, done)

        tracing.end_generation_span(
            span, outcome=failure.outcome, status_code=failure.status_code
        )

        duration = time.monotonic() - started
        error = JobError(status_code=failure.status_code, message=failure.detail)
        if retryable and job.attempts < self._max_attempts:
            # Back to "queued", keeping the error so a client polling the job
            # can see why it is waiting again.
            await self._store.save(
                job.model_copy(update={"status": "queued", "error": error})
            )
            WORKER_JOB_DURATION_SECONDS.labels(outcome="retried").observe(duration)
            logger.warning(
                "Job attempt failed. Retrying.",
                extra={"attempt": job.attempts, "status_code": failure.status_code},
            )
            return Decision(
                "retry", "retried", retry_attempt=job.attempts, message=message
            )

        await self._finish_failed(job, error)
        WORKER_JOB_DURATION_SECONDS.labels(outcome="failed").observe(duration)
        logger.error(
            "Job failed.",
            extra={"attempt": job.attempts, "status_code": failure.status_code},
        )
        return Decision("ack", "failed")

    def _generation(self, message: JobMessage) -> Generation:
        if message.url == "/v1/completions":
            return Generation.for_completion(
                CompletionRequest.model_validate(message.body)
            )
        return Generation.for_chat(ChatCompletionRequest.model_validate(message.body))

    async def _succeed(self, job: Job, message: JobMessage, done: Finished) -> Decision:
        # The job ID's random part doubles as the completion ID, so a result
        # can be traced back to its job.
        completion_id = message.job_id.removeprefix("job_")
        if message.url == "/v1/completions":
            response = completion_response(f"cmpl-{completion_id}", self._model, done)
        else:
            response = chat_completion_response(
                f"chatcmpl-{completion_id}", self._model, done
            )
        await self._store.save(
            job.model_copy(
                update={
                    "status": "succeeded",
                    "finished_at": _now(),
                    "result": response.model_dump(mode="json"),
                }
            )
        )
        TOKENS_GENERATED_TOTAL.labels(type="prompt").inc(done.result["tokens_prompt"])
        TOKENS_GENERATED_TOTAL.labels(type="completion").inc(
            done.result["tokens_completion"]
        )
        WORKER_JOB_DURATION_SECONDS.labels(outcome="succeeded").observe(done.duration)
        logger.info(
            "Job succeeded.",
            extra={
                "attempt": job.attempts,
                "duration_seconds": round(done.duration, 3),
                "tokens_prompt": done.result["tokens_prompt"],
                "tokens_completion": done.result["tokens_completion"],
                "finish_reason": done.result["finish_reason"],
            },
        )
        return Decision("ack", "succeeded")

    async def _finish_failed(self, job: Job, error: JobError) -> None:
        await self._store.save(
            job.model_copy(
                update={"status": "failed", "finished_at": _now(), "error": error}
            )
        )
