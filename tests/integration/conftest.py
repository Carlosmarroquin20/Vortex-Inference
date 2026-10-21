"""
Integration tests against a real RabbitMQ and Redis.

They run only when both URLs are set, which `make test-integration` and the
CI integration job do after starting tests/integration/compose.yaml; in a
plain `pytest` run they are reported as skipped. The backend stays a fake:
these tests are about the broker and store semantics the jobs rely on
(confirms, routing, TTL queues, delivery limits, atomic SET NX), which the
in-memory fakes in the unit suite can only assume.

Each test declares its own uniquely named queues and deletes them
afterwards, so tests do not see each other's messages and a failed run
leaves nothing behind that a later one would trip over.
"""

import asyncio
import os
import uuid

import aio_pika
import pytest

from app.jobs.queue import JobTopology

RABBITMQ_URL = os.environ.get("VORTEX_IT_RABBITMQ_URL")
REDIS_URL = os.environ.get("VORTEX_IT_REDIS_URL")


def pytest_collection_modifyitems(config, items):
    here = os.path.dirname(__file__)
    skip = pytest.mark.skip(
        reason="VORTEX_IT_RABBITMQ_URL and VORTEX_IT_REDIS_URL are not set; "
        "run `make test-integration`."
    )
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.integration)
            if not (RABBITMQ_URL and REDIS_URL):
                item.add_marker(skip)


@pytest.fixture
def topology():
    """A fresh set of queues, removed after the test."""
    declared = JobTopology(
        f"it.{uuid.uuid4().hex[:12]}", max_attempts=2, retry_delay_seconds=1
    )
    yield declared
    asyncio.run(_delete(declared))


async def _delete(topology: JobTopology) -> None:
    connection = await aio_pika.connect(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        for queue in (
            topology.queue,
            topology.dead_letter_queue,
            *topology.retry_queues(),
        ):
            await channel.queue_delete(queue)
        for exchange in (topology.queue, topology.dead_letter_exchange):
            await channel.exchange_delete(exchange)
