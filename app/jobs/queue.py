"""
Publishing jobs to RabbitMQ.

Topology, declared identically by the API and the workers so either can
start first:

    exchange vortex.jobs (direct, durable)
      -> queue vortex.jobs (quorum)          consumed by the workers
           dead-letters to exchange vortex.jobs.dlx
             -> queue vortex.jobs.dead (quorum)   inspected by operators

Quorum queues replicate messages across RabbitMQ nodes and count each
message's deliveries: once a job has been delivered job_max_attempts times
without being acknowledged (its worker crashed or was killed every time), it
is dead-lettered instead of being redelivered forever.
"""

import asyncio
from contextvars import Context
from typing import Protocol

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection

from app.jobs.models import JobMessage


class JobQueueUnavailable(RuntimeError):
    """RabbitMQ could not be reached or did not confirm the message."""


class JobQueue(Protocol):
    async def publish(self, message: JobMessage) -> None:
        """Returns once the broker has confirmed the message is stored."""
        ...

    async def close(self) -> None: ...


def dead_letter_names(queue: str) -> tuple[str, str]:
    """Exchange and queue that collect jobs which ran out of attempts."""
    return f"{queue}.dlx", f"{queue}.dead"


async def declare_topology(
    channel: AbstractChannel, queue: str, max_attempts: int
) -> AbstractExchange:
    """
    Declares the exchanges and queues and returns the jobs exchange.

    Declarations are idempotent while the arguments match. A queue that
    exists with different arguments (for example another delivery limit)
    is refused by the broker with PRECONDITION_FAILED; quorum queue
    arguments cannot change in place, so it has to be deleted and
    recreated deliberately.
    """
    dlx_name, dead_name = dead_letter_names(queue)
    dlx = await channel.declare_exchange(
        dlx_name, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dead = await channel.declare_queue(
        dead_name, durable=True, arguments={"x-queue-type": "quorum"}
    )
    await dead.bind(dlx, routing_key=queue)

    exchange = await channel.declare_exchange(
        queue, aio_pika.ExchangeType.DIRECT, durable=True
    )
    jobs = await channel.declare_queue(
        queue,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-delivery-limit": max_attempts,
            "x-dead-letter-exchange": dlx_name,
            "x-dead-letter-routing-key": queue,
        },
    )
    await jobs.bind(exchange, routing_key=queue)
    return exchange


class RabbitMQJobQueue:
    """
    Publishes with publisher confirms over a robust connection.

    The connection is opened on first use rather than at startup, so the
    synchronous endpoints keep serving while RabbitMQ is down; job
    submissions fail with 503 until it is back, and aio-pika reconnects on
    its own afterwards.
    """

    def __init__(
        self, url: str, queue: str, max_attempts: int, timeout_seconds: float
    ) -> None:
        self._url = url
        self._queue = queue
        self._max_attempts = max_attempts
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
                aio_pika.Message(
                    body=message.model_dump_json().encode("utf-8"),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    message_id=message.job_id,
                ),
                routing_key=self._queue,
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
            self._exchange = await declare_topology(
                channel, self._queue, self._max_attempts
            )
        except BaseException:
            await connection.close()
            raise
        self._connection = connection
