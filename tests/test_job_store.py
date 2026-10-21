"""RedisJobStore against fakeredis, and the RabbitMQ topology declaration."""

import asyncio

import fakeredis
import pytest

from app.jobs.models import Job
from app.jobs.queue import declare_topology
from app.jobs.store import JobStoreUnavailable, RedisJobStore

TTL = 3600


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def store(server) -> RedisJobStore:
    client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    return RedisJobStore("redis://unused", TTL, 1.0, client=client)


def job(**overrides) -> Job:
    values = {"id": "job_" + "1" * 32, "url": "/v1/completions", "status": "queued"}
    return Job(**{**values, **overrides})


def test_round_trip_with_ttl(store):
    async def scenario():
        await store.create(job())
        stored = await store.get(job().id)
        ttl = await store._redis.ttl(f"vortex:job:{job().id}")
        return stored, ttl

    stored, ttl = run(scenario())

    assert stored == job()
    assert 0 < ttl <= TTL


def test_create_refuses_an_existing_id(store):
    async def scenario():
        await store.create(job())
        await store.create(job(status="running"))

    with pytest.raises(ValueError, match="already exists"):
        run(scenario())


def test_save_replaces_and_refreshes_the_ttl(store):
    async def scenario():
        await store.create(job())
        await store._redis.expire(f"vortex:job:{job().id}", 10)
        await store.save(job(status="succeeded", result={"ok": True}))
        return await store.get(job().id), await store._redis.ttl(
            f"vortex:job:{job().id}"
        )

    stored, ttl = run(scenario())

    assert stored.status == "succeeded"
    assert stored.result == {"ok": True}
    assert ttl > 10


def test_missing_and_deleted_jobs(store):
    async def scenario():
        missing = await store.get("job_" + "9" * 32)
        await store.create(job())
        await store.delete(job().id)
        return missing, await store.get(job().id)

    assert run(scenario()) == (None, None)


def test_idempotency_keys_are_hashed_and_claimed_once(store):
    async def scenario():
        first = await store.claim_idempotency_key("client-key", "job_a")
        second = await store.claim_idempotency_key("client-key", "job_b")
        keys = await store._redis.keys("vortex:idempotency:*")
        await store.release_idempotency_key("client-key")
        third = await store.claim_idempotency_key("client-key", "job_c")
        return first, second, keys, third

    first, second, keys, third = run(scenario())

    assert (first, second, third) == (None, "job_a", None)
    # The client's key never appears in Redis as is.
    assert len(keys) == 1 and "client-key" not in keys[0]


def test_redis_failures_become_store_unavailable(server, store):
    server.connected = False

    with pytest.raises(JobStoreUnavailable):
        run(store.get(job().id))


class RecordingChannel:
    """Captures declarations instead of sending them to a broker."""

    def __init__(self) -> None:
        self.exchanges: dict[str, dict] = {}
        self.queues: dict[str, dict] = {}
        self.bindings: list[tuple[str, str, str]] = []

    async def declare_exchange(self, name, type, **kwargs):
        self.exchanges[name] = {"type": type, **kwargs}
        return name

    async def declare_queue(self, name, **kwargs):
        self.queues[name] = kwargs
        channel = self

        class Queue:
            async def bind(self, exchange, routing_key):
                channel.bindings.append((name, exchange, routing_key))

        return Queue()


def test_topology_uses_quorum_queues_with_a_delivery_limit():
    channel = RecordingChannel()

    exchange = run(declare_topology(channel, "vortex.jobs", max_attempts=3))

    assert exchange == "vortex.jobs"
    assert channel.queues["vortex.jobs"] == {
        "durable": True,
        "arguments": {
            "x-queue-type": "quorum",
            "x-delivery-limit": 3,
            "x-dead-letter-exchange": "vortex.jobs.dlx",
            "x-dead-letter-routing-key": "vortex.jobs",
        },
    }
    assert channel.queues["vortex.jobs.dead"]["arguments"] == {"x-queue-type": "quorum"}
    assert all(e["durable"] for e in channel.exchanges.values())
    assert set(channel.bindings) == {
        ("vortex.jobs", "vortex.jobs", "vortex.jobs"),
        ("vortex.jobs.dead", "vortex.jobs.dlx", "vortex.jobs"),
    }


class FakeExchange:
    def __init__(self) -> None:
        self.published: list = []
        self.error: Exception | None = None

    async def publish(self, message, routing_key, **kwargs):
        if self.error is not None:
            raise self.error
        self.published.append((message, routing_key, kwargs))


class FakeConnection:
    def __init__(self, exchange: FakeExchange, channel_error=None) -> None:
        self.exchange = exchange
        self.channel_error = channel_error
        self.channel_kwargs = None
        self.closed = False

    async def channel(self, **kwargs):
        if self.channel_error is not None:
            raise self.channel_error
        self.channel_kwargs = kwargs
        return RecordingChannel()

    async def close(self):
        self.closed = True


@pytest.fixture
def broker(monkeypatch):
    """Replaces aio_pika.connect_robust; records each connection attempt."""
    import app.jobs.queue as queue_module

    state = {"connects": [], "exchange": FakeExchange(), "fail_connect": None}

    async def fake_connect(url, **kwargs):
        if state["fail_connect"] is not None:
            raise state["fail_connect"]
        connection = FakeConnection(state["exchange"])
        state["connects"].append((url, kwargs, connection))
        return connection

    async def fake_topology(channel, queue, max_attempts):
        return state["exchange"]

    monkeypatch.setattr(queue_module.aio_pika, "connect_robust", fake_connect)
    monkeypatch.setattr(queue_module, "declare_topology", fake_topology)
    return state


def make_queue():
    from app.jobs.queue import RabbitMQJobQueue

    return RabbitMQJobQueue("amqp://u:p@host/", "vortex.jobs", 3, timeout_seconds=2.0)


def message(job_id="job_" + "2" * 32):
    from app.jobs.models import JobMessage

    return JobMessage(job_id=job_id, url="/v1/completions", body={"prompt": "hi"})


def test_publish_is_persistent_mandatory_and_confirmed(broker):
    import aio_pika

    queue = make_queue()
    run(queue.publish(message()))

    ((url, kwargs, connection),) = broker["connects"]
    assert url == "amqp://u:p@host/"
    assert kwargs["timeout"] == 2.0
    assert connection.channel_kwargs == {
        "publisher_confirms": True,
        "on_return_raises": True,
    }
    ((sent, routing_key, options),) = broker["exchange"].published
    assert routing_key == "vortex.jobs"
    assert options == {"mandatory": True, "timeout": 2.0}
    assert sent.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert sent.content_type == "application/json"
    assert sent.message_id == "job_" + "2" * 32


def test_concurrent_first_publishes_open_one_connection(broker):
    queue = make_queue()

    async def scenario():
        await asyncio.gather(*(queue.publish(message()) for _ in range(5)))

    run(scenario())

    assert len(broker["connects"]) == 1
    assert len(broker["exchange"].published) == 5


def test_connection_failures_are_retried_on_the_next_publish(broker):
    from app.jobs.queue import JobQueueUnavailable

    queue = make_queue()
    broker["fail_connect"] = ConnectionError("refused")
    with pytest.raises(JobQueueUnavailable, match="refused"):
        run(queue.publish(message()))

    broker["fail_connect"] = None
    run(queue.publish(message()))
    assert len(broker["exchange"].published) == 1


def test_publish_errors_become_queue_unavailable(broker):
    from app.jobs.queue import JobQueueUnavailable

    queue = make_queue()
    broker["exchange"].error = RuntimeError("NACK from broker")

    with pytest.raises(JobQueueUnavailable, match="NACK"):
        run(queue.publish(message()))


def test_failed_channel_setup_closes_the_connection(broker, monkeypatch):
    import app.jobs.queue as queue_module
    from app.jobs.queue import JobQueueUnavailable

    connections = []

    async def connect_then_fail(url, **kwargs):
        connection = FakeConnection(FakeExchange(), channel_error=OSError("reset"))
        connections.append(connection)
        return connection

    monkeypatch.setattr(queue_module.aio_pika, "connect_robust", connect_then_fail)

    with pytest.raises(JobQueueUnavailable):
        run(make_queue().publish(message()))
    assert connections[0].closed is True
