"""
Publishing jobs to RabbitMQ.

Topology, declared identically by the API and the workers so either can
start first:

    exchange vortex.jobs (direct, durable)
      -> queue vortex.jobs (quorum)          consumed by the workers
           dead-letters to exchange vortex.jobs.dlx
             -> queue vortex.jobs.dead (quorum)   inspected by operators

    queue vortex.jobs.retry.10s, .20s, ... (quorum, one per delay)
      dead-letters to exchange vortex.jobs once its TTL expires

Quorum queues replicate messages across RabbitMQ nodes and count each
message's deliveries: once a job has been delivered job_max_attempts times
without being acknowledged (its worker crashed or was killed every time), it
is dead-lettered instead of being redelivered forever.

A worker retries a transient failure by publishing the job to the retry
queue for that attempt and acknowledging the original. The retry queue has
no consumers; the message waits out the queue's TTL and is dead-lettered
back to the jobs exchange. Each delay gets its own queue because RabbitMQ
expires messages only from the head of a queue: with mixed per-message TTLs,
a long delay at the head would hold back every shorter one behind it.
"""

import asyncio
from contextvars import Context
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractQueue,
    AbstractRobustConnection,
)

from app.config import Settings
from app.jobs.models import JobMessage

# Upper bound on a single retry delay, however many attempts are allowed.
MAX_RETRY_DELAY_SECONDS = 3600


class JobQueueUnavailable(RuntimeError):
    """RabbitMQ could not be reached or did not confirm the message."""


class JobQueue(Protocol):
    async def publish(self, message: JobMessage) -> None:
        """Returns once the broker has confirmed the message is stored."""
        ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class JobTopology:
    """Names and arguments of the jobs' exchanges and queues."""

    queue: str
    max_attempts: int
    retry_delay_seconds: int

    @property
    def dead_letter_exchange(self) -> str:
        return f"{self.queue}.dlx"

    @property
    def dead_letter_queue(self) -> str:
        return f"{self.queue}.dead"

    def retry_delay(self, attempt: int) -> int:
        """Seconds to wait before the attempt after `attempt` (1-based)."""
        return min(
            self.retry_delay_seconds * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS
        )

    def retry_queue(self, attempt: int) -> str:
        # Named after the delay rather than the attempt: changing
        # job_retry_delay_seconds then declares new queues instead of
        # colliding with existing ones whose TTL can no longer change.
        return f"{self.queue}.retry.{self.retry_delay(attempt)}s"

    def retry_queues(self) -> dict[str, int]:
        """Every retry queue, by name, with its delay in seconds."""
        return {
            self.retry_queue(attempt): self.retry_delay(attempt)
            for attempt in range(1, self.max_attempts)
        }


def job_topology(settings: Settings) -> JobTopology:
    return JobTopology(
        queue=settings.jobs_queue,
        max_attempts=settings.job_max_attempts,
        retry_delay_seconds=settings.job_retry_delay_seconds,
    )


class DeclaredTopology(NamedTuple):
    exchange: AbstractExchange
    queue: AbstractQueue


async def declare_topology(
    channel: AbstractChannel, topology: JobTopology
) -> DeclaredTopology:
    """
    Declares the exchanges and queues and returns the jobs exchange and queue.

    Declarations are idempotent while the arguments match. A queue that
    exists with different arguments (for example another delivery limit)
    is refused by the broker with PRECONDITION_FAILED; quorum queue
    arguments cannot change in place, so it has to be deleted and
    recreated deliberately.
    """
    dlx = await channel.declare_exchange(
        topology.dead_letter_exchange, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dead = await channel.declare_queue(
        topology.dead_letter_queue, durable=True, arguments={"x-queue-type": "quorum"}
    )
    await dead.bind(dlx, routing_key=topology.queue)

    exchange = await channel.declare_exchange(
        topology.queue, aio_pika.ExchangeType.DIRECT, durable=True
    )
    jobs = await channel.declare_queue(
        topology.queue,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-delivery-limit": topology.max_attempts,
            "x-dead-letter-exchange": topology.dead_letter_exchange,
            "x-dead-letter-routing-key": topology.queue,
        },
    )
    await jobs.bind(exchange, routing_key=topology.queue)

    for name, delay in topology.retry_queues().items():
        await channel.declare_queue(
            name,
            durable=True,
            arguments={
                "x-queue-type": "quorum",
                "x-message-ttl": delay * 1000,
                "x-dead-letter-exchange": topology.queue,
                "x-dead-letter-routing-key": topology.queue,
                # A quorum queue dead-letters at most once by default, so a
                # message expiring while the target is unavailable would be
                # dropped. At-least-once keeps it until the jobs queue has
                # confirmed it, and RabbitMQ requires reject-publish overflow
                # for that strategy.
                "x-dead-letter-strategy": "at-least-once",
                "x-overflow": "reject-publish",
            },
        )
    return DeclaredTopology(exchange, jobs)


class RabbitMQJobQueue:
    """
    Publishes with publisher confirms over a robust connection.

    The connection is opened on first use rather than at startup, so the
    synchronous endpoints keep serving while RabbitMQ is down; job
    submissions fail with 503 until it is back, and aio-pika reconnects on
    its own afterwards.
    """

    def __init__(self, url: str, topology: JobTopology, timeout_seconds: float) -> None:
        self._url = url
        self._topology = topology
        self._timeout = timeout_seconds
        self._lock = asyncio.Lock()
        self._connection: AbstractRobustConnection | None = None
        self._exchange: AbstractExchange | None = None

    async def publish(self, message: JobMessage) -> None:
        try:
            exchange = await self._ready()
            # mandatory with on_return_raises: a message no queue would take
            # is an error, not a silent drop. Persistent so it survives a
            # broker restart; the confirm means it has been written.
            await exchange.publish(
                job_message(message),
                routing_key=self._topology.queue,
                mandatory=True,
                timeout=self._timeout,
            )
        except JobQueueUnavailable:
            raise
        except Exception as exc:
            # aio-pika surfaces connection, channel and confirm failures as
            # several unrelated exception types; to the API they all mean
            # the job was not stored.
            raise JobQueueUnavailable(f"RabbitMQ publish failed: {exc!r}") from exc

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()

    async def _ready(self) -> AbstractExchange:
        async with self._lock:
            if self._exchange is None:
                # Connected in a task with an empty context: the connection's
                # background tasks (reader, reconnection) copy the context they
                # are created in, and created inside the first request they
                # would tag every later connection log line with that
                # request's ID.
                await asyncio.create_task(self._connect(), context=Context())
            return self._exchange

    async def _connect(self) -> None:
        connection = await aio_pika.connect_robust(
            self._url,
            timeout=self._timeout,
            client_properties={"connection_name": "vortex-api"},
        )
        try:
            channel = await connection.channel(
                publisher_confirms=True, on_return_raises=True
            )
            declared = await declare_topology(channel, self._topology)
        except BaseException:
            await connection.close()
            raise
        self._exchange = declared.exchange
        self._connection = connection


def job_message(message: JobMessage) -> aio_pika.Message:
    """The AMQP message for a job, as published by the API and by retries."""
    return aio_pika.Message(
        body=message.model_dump_json().encode("utf-8"),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        message_id=message.job_id,
    )
