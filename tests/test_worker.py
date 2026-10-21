"""The job worker: JobProcessor's decisions, BackendRunner, and the AMQP consumer."""

import asyncio
import threading
import time

import pytest

from app.backends import BackendError, RequestRejectedError
from app.completions import Finished
from app.jobs.health import WorkerHealth
from app.jobs.models import Job, JobMessage
from app.jobs.processor import (
    BackendRunner,
    Decision,
    InferenceTimeout,
    JobProcessor,
)
from app.jobs.queue import DeclaredTopology, JobTopology
from tests.fakes import InMemoryJobStore
from tests.helpers import metric

JOB_ID = "job_" + "a" * 32
TOPOLOGY = JobTopology("vortex.jobs", max_attempts=3, retry_delay_seconds=10)
RESULT = {
    "text": " world",
    "finish_reason": "stop",
    "tokens_prompt": 3,
    "tokens_completion": 2,
}


def run(coro):
    return asyncio.run(coro)


class FakeRunner:
    """Runner that returns RESULT or raises the queued errors, in order."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.generations: list = []

    async def run(self, generation):
        self.generations.append(generation)
        if self.errors:
            raise self.errors.pop(0)
        return Finished(result=dict(RESULT), duration=0.5, tokens_per_second=4.0)

    def close(self) -> None:
        pass


def body(url="/v1/completions", job_id=JOB_ID, **fields) -> bytes:
    payload = fields or (
        {"prompt": "hello", "max_tokens": 8}
        if url == "/v1/completions"
        else {"messages": [{"role": "user", "content": "hi"}]}
    )
    return JobMessage(job_id=job_id, url=url, body=payload).model_dump_json().encode()


def processor(store, runner, max_attempts=3) -> JobProcessor:
    return JobProcessor(
        store,
        runner,
        model_name="test-model",
        max_attempts=max_attempts,
        timeout_seconds=5,
    )


def stored(store, url="/v1/completions", **fields) -> InMemoryJobStore:
    store.jobs[JOB_ID] = Job(id=JOB_ID, url=url, status="queued", **fields)
    return store


# ---------------------------------------------------------------------------
# JobProcessor
# ---------------------------------------------------------------------------


def test_successful_completion_job_stores_the_endpoint_response():
    store = stored(InMemoryJobStore(), created_at=int(time.time()) - 30)
    waits = metric("vortex_worker_job_wait_seconds_count")

    decision = run(processor(store, FakeRunner()).process(body()))

    assert decision == Decision("ack", "succeeded")
    job = store.jobs[JOB_ID]
    assert (job.status, job.attempts, job.error) == ("succeeded", 1, None)
    assert job.started_at is not None and job.finished_at is not None
    assert job.result["id"] == "cmpl-" + "a" * 32
    assert job.result["object"] == "text_completion"
    assert job.result["model"] == "test-model"
    assert job.result["choices"][0]["text"] == " world"
    assert job.result["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert metric("vortex_worker_job_wait_seconds_count") == waits + 1


def test_chat_job_runs_the_conversation():
    store = stored(InMemoryJobStore(), url="/v1/chat/completions")
    runner = FakeRunner()

    run(processor(store, runner).process(body("/v1/chat/completions")))

    (generation,) = runner.generations
    assert generation.prompt == [{"role": "user", "content": "hi"}]
    result = store.jobs[JOB_ID].result
    assert result["id"] == "chatcmpl-" + "a" * 32
    assert result["object"] == "chat.completion"
    message = result["choices"][0]["message"]
    assert (message["role"], message["content"]) == ("assistant", " world")


def test_transient_failure_is_retried_with_its_error_visible():
    store = stored(InMemoryJobStore())

    decision = run(
        processor(store, FakeRunner(BackendError("429", status_code=503))).process(
            body()
        )
    )

    assert decision.action == "retry" and decision.outcome == "retried"
    assert decision.retry_attempt == 1
    assert decision.message.job_id == JOB_ID
    job = store.jobs[JOB_ID]
    assert (job.status, job.attempts) == ("queued", 1)
    assert job.error.status_code == 503


def test_retry_succeeds_and_clears_the_error():
    store = stored(InMemoryJobStore())
    runner = FakeRunner(BackendError("timeout", status_code=504))
    jobs = processor(store, runner)

    run(jobs.process(body()))
    decision = run(jobs.process(body()))

    assert decision.outcome == "succeeded"
    job = store.jobs[JOB_ID]
    assert (job.status, job.attempts, job.error) == ("succeeded", 2, None)


def test_transient_failure_on_the_last_attempt_fails_the_job():
    store = stored(InMemoryJobStore(), attempts=2)

    decision = run(processor(store, FakeRunner(BackendError("down"))).process(body()))

    assert decision == Decision("ack", "failed")
    job = store.jobs[JOB_ID]
    assert (job.status, job.attempts) == ("failed", 3)
    assert job.error.status_code == 502


@pytest.mark.parametrize(
    "error, status",
    [
        (RequestRejectedError("prompt too long"), 400),
        (InferenceTimeout(), 504),
        (RuntimeError("segfault-ish /models/path"), 500),
    ],
)
def test_permanent_failures_are_not_retried(error, status):
    store = stored(InMemoryJobStore())

    decision = run(processor(store, FakeRunner(error)).process(body()))

    assert decision == Decision("ack", "failed")
    job = store.jobs[JOB_ID]
    assert (job.status, job.error.status_code) == ("failed", status)
    # Internal details stay in the log, as on the synchronous endpoints.
    assert "/models/path" not in job.error.message


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_redelivery_of_a_finished_job_is_skipped(status):
    store = stored(InMemoryJobStore())
    store.jobs[JOB_ID] = store.jobs[JOB_ID].model_copy(update={"status": status})
    runner = FakeRunner()

    decision = run(processor(store, runner).process(body()))

    assert decision == Decision("ack", "skipped")
    assert runner.generations == []


def test_message_without_a_record_is_dropped():
    runner = FakeRunner()

    decision = run(processor(InMemoryJobStore(), runner).process(body()))

    assert decision == Decision("ack", "skipped")
    assert runner.generations == []


def test_job_interrupted_on_every_attempt_is_failed_without_running():
    store = stored(InMemoryJobStore(), attempts=3)
    store.jobs[JOB_ID] = store.jobs[JOB_ID].model_copy(update={"status": "running"})
    runner = FakeRunner()

    decision = run(processor(store, runner).process(body()))

    assert decision == Decision("ack", "failed")
    assert runner.generations == []
    job = store.jobs[JOB_ID]
    assert job.status == "failed"
    assert job.error.status_code == 500
    assert "interrupted" in job.error.message


def test_body_the_worker_cannot_validate_fails_the_job():
    store = stored(InMemoryJobStore())

    decision = run(
        processor(store, FakeRunner()).process(body(prompt="x", max_tokens=-1))
    )

    assert decision == Decision("ack", "failed")
    assert store.jobs[JOB_ID].error.status_code == 400


def test_unreadable_message_is_dead_lettered():
    decision = run(processor(InMemoryJobStore(), FakeRunner()).process(b"{not json"))

    assert decision == Decision("reject", "rejected")


def test_store_outage_defers_without_using_an_attempt():
    store = stored(InMemoryJobStore())
    store.unavailable = True

    decision = run(processor(store, FakeRunner()).process(body()))

    assert (decision.action, decision.outcome, decision.retry_attempt) == (
        "retry",
        "deferred",
        1,
    )
    assert decision.message.job_id == JOB_ID
    assert store.jobs[JOB_ID].attempts == 0


def test_unexpected_error_fails_the_job_and_dead_letters_it():
    class Malformed(FakeRunner):
        async def run(self, generation):
            # A result missing its fields breaks response building.
            return Finished(result={}, duration=0.1, tokens_per_second=0.0)

    store = stored(InMemoryJobStore())

    decision = run(processor(store, Malformed()).process(body()))

    assert decision == Decision("reject", "failed")
    job = store.jobs[JOB_ID]
    assert (job.status, job.error.status_code) == ("failed", 500)


def test_unexpected_error_is_dead_lettered_even_if_the_record_cannot_be_updated():
    class BrokenAfterFirstSave(InMemoryJobStore):
        saves = 0

        async def save(self, job):
            self.saves += 1
            if self.saves > 1:
                raise RuntimeError("serializer bug")
            await super().save(job)

    store = stored(BrokenAfterFirstSave())

    decision = run(processor(store, FakeRunner()).process(body()))

    assert decision == Decision("reject", "failed")
    assert store.jobs[JOB_ID].status == "running"


def test_store_outage_after_the_run_defers_the_job():
    store = stored(InMemoryJobStore())

    class FailingSaveAfterRun(FakeRunner):
        async def run(self, generation):
            store.unavailable = True
            return await super().run(generation)

    decision = run(processor(store, FailingSaveAfterRun()).process(body()))

    assert decision.outcome == "deferred"


# ---------------------------------------------------------------------------
# BackendRunner
# ---------------------------------------------------------------------------


class SlowBackend:
    """generate() runs until stop_event is set, like llama.cpp's stopping criteria."""

    def __init__(self) -> None:
        self.exited = threading.Event()

    def generate(self, prompt, max_tokens, temperature, top_p, stop, stop_event):
        stop_event.wait(10)
        time.sleep(0.05)
        self.exited.set()
        return dict(RESULT)


def generation():
    from app.completions import Generation

    return Generation(prompt="hi", max_tokens=4, temperature=0.7, top_p=0.9, stop=None)


def test_runner_returns_the_result_and_throughput():
    class Instant:
        def generate(self, *args):
            return dict(RESULT)

    runner = BackendRunner(Instant(), concurrency=1, timeout_seconds=5)
    try:
        done = run(runner.run(generation()))
    finally:
        runner.close()

    assert done.result == RESULT
    assert done.duration >= 0


def test_runner_timeout_stops_the_thread_before_returning():
    backend = SlowBackend()
    runner = BackendRunner(backend, concurrency=1, timeout_seconds=0.1)
    try:
        with pytest.raises(InferenceTimeout):
            run(runner.run(generation()))
        # The thread had left the backend by the time run() raised.
        assert backend.exited.is_set()
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# Worker (AMQP consumer)
# ---------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, data: bytes = b"", message_id: str = JOB_ID) -> None:
        self.body = data
        self.message_id = message_id
        self.headers: dict = {}
        self.settled: list = []

    async def ack(self):
        self.settled.append("ack")

    async def reject(self, requeue):
        self.settled.append(("reject", requeue))

    async def nack(self, requeue):
        self.settled.append(("nack", requeue))


class FakeExchange:
    def __init__(self, events: list) -> None:
        self.events = events

    async def publish(self, message, routing_key, **kwargs):
        self.events.append(("publish", routing_key, message.message_id, kwargs))


class FakeChannel:
    def __init__(self) -> None:
        self.events: list = []
        self.default_exchange = FakeExchange(self.events)
        self.qos = None

    async def set_qos(self, prefetch_count):
        self.qos = prefetch_count


class FakeQueue:
    def __init__(self) -> None:
        self.callback = None
        self.cancelled = False
        self.consuming = asyncio.Event()

    async def consume(self, callback):
        self.callback = callback
        self.consuming.set()
        return "ctag"

    async def cancel(self, tag):
        self.cancelled = True


class StaticProcessor:
    def __init__(self, decision=None, error=None, gate=None) -> None:
        self.decision = decision
        self.error = error
        self.gate = gate

    async def process(self, data):
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.decision


def make_worker(processor, drain_timeout=1.0):
    from app.worker import Worker

    return Worker(
        "amqp://u:p@host/",
        topology=TOPOLOGY,
        processor=processor,
        concurrency=2,
        drain_timeout_seconds=drain_timeout,
        publish_timeout_seconds=1.0,
    )


def message_for_retry() -> JobMessage:
    return JobMessage(job_id=JOB_ID, url="/v1/completions", body={"prompt": "hi"})


@pytest.mark.parametrize(
    "decision, expected",
    [
        (Decision("ack", "succeeded"), ["ack"]),
        (Decision("reject", "rejected"), [("reject", False)]),
    ],
)
def test_settling_ack_and_reject(decision, expected):
    worker = make_worker(StaticProcessor(decision))
    worker._channel = FakeChannel()
    message = FakeMessage()
    before = metric("vortex_worker_jobs_total", outcome=decision.outcome)

    run(worker._on_message(message))

    assert message.settled == expected
    assert metric("vortex_worker_jobs_total", outcome=decision.outcome) == before + 1


def test_retry_is_published_to_its_delay_queue_before_the_ack():
    decision = Decision(
        "retry", "retried", retry_attempt=2, message=message_for_retry()
    )
    worker = make_worker(StaticProcessor(decision))
    channel = FakeChannel()
    worker._channel = channel
    message = FakeMessage()

    async def scenario():
        original_ack = message.ack

        async def ack():
            channel.events.append("ack")
            await original_ack()

        message.ack = ack
        await worker._on_message(message)

    run(scenario())

    (publish, ack) = channel.events
    assert publish[:3] == ("publish", "vortex.jobs.retry.20s", JOB_ID)
    assert publish[3]["mandatory"] is True
    assert ack == "ack"


def test_failed_settling_requeues_the_message():
    decision = Decision(
        "retry", "retried", retry_attempt=1, message=message_for_retry()
    )
    worker = make_worker(StaticProcessor(decision))
    channel = FakeChannel()

    async def broken_publish(*args, **kwargs):
        raise ConnectionError("channel closed")

    channel.default_exchange.publish = broken_publish
    worker._channel = channel
    message = FakeMessage()

    run(worker._on_message(message))

    assert message.settled == [("nack", True)]


def test_job_id_tags_the_log_context_only_while_handling():
    from app.logging_config import request_id_var

    seen = []

    class Recording(StaticProcessor):
        async def process(self, data):
            seen.append(request_id_var.get())
            return Decision("ack", "succeeded")

    worker = make_worker(Recording())
    worker._channel = FakeChannel()

    async def scenario():
        await worker._on_message(FakeMessage())
        return request_id_var.get()

    assert run(scenario()) is None
    assert seen == [JOB_ID]


@pytest.fixture
def broker(monkeypatch):
    """Replaces connect_robust and the topology declaration for Worker.run()."""
    import app.worker as worker_module

    state = {
        "channel": FakeChannel(),
        "queue": FakeQueue(),
        "failures": 0,
        "connects": 0,
        "closed": False,
    }

    class Connection:
        async def channel(self, **kwargs):
            state["channel_kwargs"] = kwargs
            return state["channel"]

        async def close(self):
            state["closed"] = True

    async def connect(url, **kwargs):
        state["connects"] += 1
        if state["failures"]:
            state["failures"] -= 1
            raise ConnectionError("refused")
        state["connect_kwargs"] = kwargs
        return Connection()

    async def topology(channel, topology):
        return DeclaredTopology("exchange", state["queue"])

    monkeypatch.setattr(worker_module.aio_pika, "connect_robust", connect)
    monkeypatch.setattr(worker_module, "declare_topology", topology)
    monkeypatch.setattr(worker_module, "_CONNECT_RETRY_INITIAL", 0.01)
    return state


def test_run_consumes_with_prefetch_and_drains_on_stop(broker):
    gate = asyncio.Event()
    worker = make_worker(StaticProcessor(Decision("ack", "succeeded"), gate=gate))
    health = WorkerHealth(backend=None, timeout_seconds=5)
    worker._health = health
    message = FakeMessage()

    async def scenario():
        stop = asyncio.Event()
        running = asyncio.create_task(worker.run(stop))
        await broker["queue"].consuming.wait()
        assert health.ready()
        handling = asyncio.create_task(broker["queue"].callback(message))
        await asyncio.sleep(0.01)
        stop.set()
        await asyncio.sleep(0.05)
        # Stopped consuming, but still waiting for the job in progress.
        assert broker["queue"].cancelled and not running.done()
        assert not health.ready()
        gate.set()
        await running
        await handling

    run(scenario())

    assert broker["channel"].qos == 2
    assert broker["channel_kwargs"] == {
        "publisher_confirms": True,
        "on_return_raises": True,
    }
    assert message.settled == ["ack"]
    assert broker["closed"] is True


def test_jobs_past_the_drain_deadline_are_cancelled_unacknowledged(broker):
    worker = make_worker(
        StaticProcessor(Decision("ack", "succeeded"), gate=asyncio.Event()),
        drain_timeout=0.05,
    )
    message = FakeMessage()

    async def scenario():
        stop = asyncio.Event()
        running = asyncio.create_task(worker.run(stop))
        await broker["queue"].consuming.wait()
        handling = asyncio.create_task(broker["queue"].callback(message))
        await asyncio.sleep(0.01)
        stop.set()
        await running
        return handling

    handling = run(scenario())

    assert handling.cancelled()
    assert message.settled == []
    assert broker["closed"] is True


def test_startup_retries_until_rabbitmq_answers(broker):
    broker["failures"] = 2
    worker = make_worker(StaticProcessor(Decision("ack", "succeeded")))

    async def scenario():
        stop = asyncio.Event()
        running = asyncio.create_task(worker.run(stop))
        await broker["queue"].consuming.wait()
        stop.set()
        await running

    run(scenario())

    assert broker["connects"] == 3
    assert broker["connect_kwargs"]["client_properties"] == {
        "connection_name": "vortex-worker"
    }


def test_stop_during_startup_retries_exits_without_connecting(broker):
    broker["failures"] = 1000
    worker = make_worker(StaticProcessor(Decision("ack", "succeeded")))

    async def scenario():
        stop = asyncio.Event()
        running = asyncio.create_task(worker.run(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(running, timeout=2)

    run(scenario())

    assert broker["queue"].callback is None


def test_worker_refuses_to_start_with_jobs_disabled():
    from app.worker import main as worker_main

    with pytest.raises(SystemExit, match="VORTEX_JOBS_ENABLED"):
        worker_main()


def test_run_worker_wires_settings_and_cleans_up(monkeypatch):
    import app.worker as worker_module
    from app.config import Settings

    settings = Settings(
        jobs_enabled=True,
        rabbitmq_url="amqp://u:p@rabbit/",
        redis_url="redis://:p@redis/0",
        max_concurrent_requests=3,
        inference_timeout_seconds=30,
    )
    events = []

    class Backend:
        def close(self):
            events.append("backend closed")

    class HealthServer:
        def shutdown(self):
            events.append("health stopped")

    async def fake_run(self, stop):
        events.append(("run", self._url, self._concurrency, self._drain_timeout))
        assert self._health is not None

    monkeypatch.setattr(worker_module, "create_backend", lambda s: Backend())
    monkeypatch.setattr(worker_module, "serve", lambda health, port: HealthServer())
    monkeypatch.setattr(worker_module.Worker, "run", fake_run)

    run(worker_module.run_worker(settings))

    assert events == [
        ("run", "amqp://u:p@rabbit/", 3, 30 + worker_module.SHUTDOWN_GRACE_SECONDS),
        "backend closed",
        "health stopped",
    ]
