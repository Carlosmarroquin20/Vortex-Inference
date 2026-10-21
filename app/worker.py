"""
Job worker: python -m app.worker

Consumes the jobs queue, runs each job on the configured backend and stores
the result in Redis (app/jobs/processor.py decides what happens to each
message; this module applies those decisions to RabbitMQ). It runs from the
same image and settings as the API, so it loads the same backend and model.

Each worker takes at most max_concurrent_requests jobs at a time: that is
its prefetch count, so RabbitMQ hands further messages to other workers
instead of queueing them behind a busy one.

On SIGTERM or SIGINT the worker stops taking new jobs, lets the ones in
progress finish (each is bounded by inference_timeout_seconds) and exits.
Jobs it could not finish in that window are stopped and left unacknowledged,
and RabbitMQ redelivers them to another worker.
"""

import asyncio
import contextlib
import logging
import signal
import time

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractIncomingMessage,
    AbstractRobustConnection,
)
from prometheus_client import start_http_server

from app.backends import create_backend
from app.config import Settings
from app.jobs.processor import BackendRunner, Decision, JobProcessor
from app.jobs.queue import JobTopology, declare_topology, job_message, job_topology
from app.jobs.store import RedisJobStore
from app.logging_config import configure_logging, request_id_var
from app.metrics import WORKER_JOBS_IN_PROGRESS, WORKER_JOBS_TOTAL

logger = logging.getLogger("vortex.worker")

# Margin past the inference deadline for in-progress jobs to store their
# results during shutdown. terminationGracePeriodSeconds must cover the
# deadline plus this margin.
SHUTDOWN_GRACE_SECONDS = 15.0

# Backoff between attempts to reach RabbitMQ at startup. Retrying here
# rather than exiting avoids reloading the model on every attempt.
_CONNECT_RETRY_INITIAL = 1.0
_CONNECT_RETRY_MAX = 30.0


class Worker:
    def __init__(
        self,
        url: str,
        topology: JobTopology,
        processor: JobProcessor,
        concurrency: int,
        drain_timeout_seconds: float,
        publish_timeout_seconds: float,
    ) -> None:
        self._url = url
        self._topology = topology
        self._processor = processor
        self._concurrency = concurrency
        self._drain_timeout = drain_timeout_seconds
        self._publish_timeout = publish_timeout_seconds
        self._channel: AbstractChannel | None = None
        self._in_progress: set[asyncio.Task] = set()

    async def run(self, stop: asyncio.Event) -> None:
        """Consumes until stop is set, then drains and disconnects."""
        connection = await self._connect(stop)
        if connection is None:
            return
        try:
            channel = await connection.channel(
                publisher_confirms=True, on_return_raises=True
            )
            await channel.set_qos(prefetch_count=self._concurrency)
            declared = await declare_topology(channel, self._topology)
            self._channel = channel
            consumer_tag = await declared.queue.consume(self._on_message)
            logger.info(
                "Worker consuming.",
                extra={"queue": self._topology.queue, "prefetch": self._concurrency},
            )

            await stop.wait()

            # No new deliveries from here on. Messages delivered before the
            # cancel took effect are already in progress and drained below.
            await declared.queue.cancel(consumer_tag)
            await self._drain()
        finally:
            await connection.close()

    async def _connect(self, stop: asyncio.Event) -> AbstractRobustConnection | None:
        delay = _CONNECT_RETRY_INITIAL
        while not stop.is_set():
            try:
                # connect_robust reconnects on its own once connected; only
                # the first connection has to be retried by hand.
                return await aio_pika.connect_robust(
                    self._url,
                    timeout=self._publish_timeout,
                    client_properties={"connection_name": "vortex-worker"},
                )
            except Exception as exc:
                logger.warning(
                    "RabbitMQ unreachable. Retrying.",
                    extra={"reason": repr(exc), "retry_in_seconds": delay},
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                delay = min(delay * 2, _CONNECT_RETRY_MAX)
        return None

    async def _drain(self) -> None:
        if not self._in_progress:
            return
        logger.info(
            "Shutdown requested. Waiting for jobs in progress.",
            extra={"jobs": len(self._in_progress)},
        )
        _, pending = await asyncio.wait(
            set(self._in_progress), timeout=self._drain_timeout
        )
        if pending:
            # Cancelling stops each generation at its next token; the
            # messages stay unacknowledged and are redelivered.
            logger.error(
                "Jobs did not finish before the shutdown deadline. Returning "
                "them to the queue.",
                extra={"jobs": len(pending)},
            )
            for task in pending:
                task.cancel()
            await asyncio.wait(pending)

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        # aiormq runs each delivery in its own task, so up to prefetch_count
        # of these run at once.
        task = asyncio.current_task()
        self._in_progress.add(task)
        WORKER_JOBS_IN_PROGRESS.inc()
        # The job ID goes in the request_id field so that a job's log lines,
        # including the backend's, can be followed with one query.
        token = request_id_var.set(message.message_id)
        try:
            decision = await self._processor.process(message.body)
            await self._settle(message, decision)
            WORKER_JOBS_TOTAL.labels(outcome=decision.outcome).inc()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A bug or a broker failure while settling. Requeued, so the
            # delivery limit eventually dead-letters a message that keeps
            # failing this way.
            logger.exception("Job handling failed. Requeueing the message.")
            with contextlib.suppress(Exception):
                await message.nack(requeue=True)
        finally:
            request_id_var.reset(token)
            WORKER_JOBS_IN_PROGRESS.dec()
            self._in_progress.discard(task)

    async def _settle(
        self, message: AbstractIncomingMessage, decision: Decision
    ) -> None:
        if decision.action == "reject":
            await message.reject(requeue=False)
            return
        if decision.action == "retry":
            # Published before the original is acknowledged: a crash between
            # the two leaves a duplicate, which the record's status turns
            # into a skip, rather than a lost job.
            await self._channel.default_exchange.publish(
                job_message(decision.message),
                routing_key=self._topology.retry_queue(decision.retry_attempt),
                mandatory=True,
                timeout=self._publish_timeout,
            )
        await message.ack()


async def run_worker(settings: Settings) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows event loops have no add_signal_handler; local
            # development only, the image runs on Linux.
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

    logger.info(
        "Starting job worker.",
        extra={"backend": settings.backend, "model": settings.model_name},
    )
    load_start = time.monotonic()
    backend = create_backend(settings)
    logger.info(
        "Model loaded.",
        extra={"duration_seconds": round(time.monotonic() - load_start, 2)},
    )

    metrics_server, _ = start_http_server(settings.worker_metrics_port)
    store = RedisJobStore(
        settings.redis_url.get_secret_value(),
        ttl_seconds=settings.job_ttl_seconds,
        timeout_seconds=settings.jobs_service_timeout_seconds,
    )
    runner = BackendRunner(
        backend,
        concurrency=settings.max_concurrent_requests,
        timeout_seconds=settings.inference_timeout_seconds,
    )
    worker = Worker(
        settings.rabbitmq_url.get_secret_value(),
        topology=job_topology(settings),
        processor=JobProcessor(
            store,
            runner,
            model_name=settings.model_name,
            max_attempts=settings.job_max_attempts,
            timeout_seconds=settings.inference_timeout_seconds,
        ),
        concurrency=settings.max_concurrent_requests,
        drain_timeout_seconds=settings.inference_timeout_seconds
        + SHUTDOWN_GRACE_SECONDS,
        publish_timeout_seconds=settings.jobs_service_timeout_seconds,
    )
    try:
        await worker.run(stop)
    finally:
        runner.close()
        backend.close()
        await store.close()
        metrics_server.shutdown()
        logger.info("Worker stopped.")


def main() -> None:
    configure_logging()
    settings = Settings()
    if not settings.jobs_enabled:
        # The URLs are only validated as required when jobs are enabled.
        raise SystemExit(
            "The worker needs VORTEX_JOBS_ENABLED=true, VORTEX_RABBITMQ_URL "
            "and VORTEX_REDIS_URL."
        )
    asyncio.run(run_worker(settings))


if __name__ == "__main__":
    main()
