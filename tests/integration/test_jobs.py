"""Jobs end to end over a real RabbitMQ and Redis, with a scripted backend."""

import asyncio
import time
from contextlib import asynccontextmanager

import aio_pika
import pytest

from app.backends import BackendError
from app.completions import Finished
from app.jobs.models import Job, JobMessage, new_job_id
from app.jobs.processor import JobProcessor
from app.jobs.queue import JobQueueUnavailable, JobTopology, RabbitMQJobQueue
from app.jobs.store import RedisJobStore
from app.worker import Worker
from tests.integration.conftest import RABBITMQ_URL, REDIS_URL

# Upper bound on any single scenario, so a broken broker interaction fails
# the test instead of hanging the job.
SCENARIO_TIMEOUT = 60

RESULT = {
    "text": " done",
    "finish_reason": "stop",
    "tokens_prompt": 2,
    "tokens_completion": 1,
}


def run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, SCENARIO_TIMEOUT)

    return asyncio.run(bounded())


class ScriptedRunner:
    """Raises the queued errors in order, then succeeds after `delay` seconds."""

    def __init__(self, *errors: Exception, delay: float = 0.0) -> None:
        self.errors = list(errors)
        self.delay = delay
        self.calls = 0

    async def run(self, generation):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        await asyncio.sleep(self.delay)
        return Finished(result=dict(RESULT), duration=self.delay, tokens_per_second=1.0)

    def close(self) -> None:
        pass


def new_store() -> RedisJobStore:
    return RedisJobStore(REDIS_URL, ttl_seconds=300, timeout_seconds=5)


def new_queue(topology: JobTopology) -> RabbitMQJobQueue:
    return RabbitMQJobQueue(RABBITMQ_URL, topology, timeout_seconds=5)


async def submit(store, queue, body=None) -> str:
    """What POST /v1/jobs does: the record first, then the confirmed publish."""
    job = Job(id=new_job_id(), url="/v1/completions", status="queued")
    await store.create(job)
    await queue.publish(
        JobMessage(
            job_id=job.id,
            url="/v1/completions",
            body=body or {"prompt": "hi", "max_tokens": 4},
        )
    )
    return job.id


async def wait_for_job(store, job_id, predicate) -> Job:
    while True:
        job = await store.get(job_id)
        if job is not None and predicate(job):
            return job
        await asyncio.sleep(0.05)


async def wait_for_status(store, job_id, statuses=("succeeded", "failed")) -> Job:
    return await wait_for_job(store, job_id, lambda job: job.status in statuses)


async def queue_state(name: str) -> tuple[int, int]:
    """(ready messages, consumers) as the broker reports them."""
    connection = await aio_pika.connect(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(name, passive=True)
        result = queue.declaration_result
        return result.message_count, result.consumer_count


async def wait_for_depth(name: str, depth: int) -> None:
    # Polled: the broker's queue state has no change notification.
    while True:
        if (await queue_state(name))[0] == depth:
            return
        await asyncio.sleep(0.1)


@asynccontextmanager
async def running_worker(topology, store, processor=None, runner=None, drain=10.0):
    processor = processor or JobProcessor(
        store,
        runner or ScriptedRunner(),
        model_name="it-model",
        max_attempts=topology.max_attempts,
        timeout_seconds=5,
    )
    worker = Worker(
        RABBITMQ_URL,
        topology=topology,
        processor=processor,
        concurrency=2,
        drain_timeout_seconds=drain,
        publish_timeout_seconds=5,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        # Consuming once the broker lists the worker as a consumer; the
        # queue exists by then because the worker declared it.
        while True:
            try:
                if (await queue_state(topology.queue))[1] > 0:
                    break
            except aio_pika.exceptions.ChannelClosed:
                pass
            if task.done():
                task.result()
            await asyncio.sleep(0.05)
        yield stop
    finally:
        stop.set()
        await task


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def test_submitted_job_runs_and_its_result_is_stored(topology):
    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(topology, store):
                job_id = await submit(store, queue)
                job = await wait_for_status(store, job_id)
            depth, _ = await queue_state(topology.queue)
            return job, depth
        finally:
            await queue.close()
            await store.close()

    job, depth = run(scenario())

    assert (job.status, job.attempts) == ("succeeded", 1)
    assert job.result["object"] == "text_completion"
    assert job.result["choices"][0]["text"] == " done"
    assert depth == 0


def test_transient_failure_waits_in_the_retry_queue(topology):
    runner = ScriptedRunner(BackendError("rate limited", status_code=503))

    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(topology, store, runner=runner):
                job_id = await submit(store, queue)
                queued_again = await wait_for_job(
                    store,
                    job_id,
                    lambda job: job.status == "queued" and job.error is not None,
                )
                # Parked in the 1s delay queue rather than redelivered at once.
                await wait_for_depth(topology.retry_queue(1), 1)
                started = time.monotonic()
                job = await wait_for_status(store, job_id)
                return queued_again, time.monotonic() - started, job
        finally:
            await queue.close()
            await store.close()

    queued_again, waited, job = run(scenario())

    assert queued_again.error.status_code == 503
    assert waited >= 0.3
    assert (job.status, job.attempts, job.error) == ("succeeded", 2, None)
    assert runner.calls == 2


def test_attempts_run_out_and_the_job_fails(topology):
    runner = ScriptedRunner(*(BackendError("down") for _ in range(5)))

    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(topology, store, runner=runner):
                job_id = await submit(store, queue)
                return await wait_for_status(store, job_id)
        finally:
            await queue.close()
            await store.close()

    job = run(scenario())

    assert (job.status, job.attempts) == ("failed", topology.max_attempts)
    assert job.error.status_code == 502
    assert runner.calls == topology.max_attempts


def test_unreadable_message_goes_to_the_dead_letter_queue(topology):
    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(topology, store):
                connection = await aio_pika.connect(RABBITMQ_URL)
                async with connection:
                    channel = await connection.channel()
                    exchange = await channel.get_exchange(topology.queue)
                    await exchange.publish(
                        aio_pika.Message(b"{not json"), routing_key=topology.queue
                    )
                await wait_for_depth(topology.dead_letter_queue, 1)
        finally:
            await queue.close()
            await store.close()

    run(scenario())


def test_delivery_limit_dead_letters_a_job_that_keeps_killing_its_worker(topology):
    """
    Each "worker" takes the message and disappears without acknowledging
    it, as one killed mid-job does. The quorum queue counts those failed
    deliveries and dead-letters the message once they exceed the limit.
    """

    async def scenario():
        store, queue = new_store(), new_queue(topology)
        deliveries = 0
        try:
            await submit(store, queue)
            while True:
                if (await queue_state(topology.dead_letter_queue))[0] == 1:
                    return deliveries
                connection = await aio_pika.connect(RABBITMQ_URL)
                async with connection:
                    channel = await connection.channel()
                    jobs = await channel.get_queue(topology.queue)
                    if await jobs.get(no_ack=False, fail=False) is not None:
                        deliveries += 1
                await asyncio.sleep(0.05)
        finally:
            await queue.close()
            await store.close()

    # x-delivery-limit=N allows N redeliveries after the first delivery.
    assert run(scenario()) == topology.max_attempts + 1


def test_processing_bug_fails_the_job_and_dead_letters_it_once(topology):
    """
    RabbitMQ 4 does not count a requeue (nack with requeue) towards the
    delivery limit, so a message requeued on every failure would loop
    forever. A bug in processing dead-letters it on the first delivery.
    """

    class Malformed(ScriptedRunner):
        async def run(self, generation):
            self.calls += 1
            return Finished(result={}, duration=0.0, tokens_per_second=0.0)

    runner = Malformed()

    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(topology, store, runner=runner):
                job_id = await submit(store, queue)
                await wait_for_depth(topology.dead_letter_queue, 1)
                return await store.get(job_id)
        finally:
            await queue.close()
            await store.close()

    job = run(scenario())

    assert (job.status, job.error.status_code) == ("failed", 500)
    assert runner.calls == 1


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_lets_the_job_in_progress_finish(topology):
    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(
                topology, store, runner=ScriptedRunner(delay=1.0)
            ) as stop:
                job_id = await submit(store, queue)
                await wait_for_status(store, job_id, ("running",))
                stop.set()
            # The worker has returned: the job finished during the drain.
            job = await store.get(job_id)
            depth, consumers = await queue_state(topology.queue)
            return job, depth, consumers
        finally:
            await queue.close()
            await store.close()

    job, depth, consumers = run(scenario())

    assert job.status == "succeeded"
    assert (depth, consumers) == (0, 0)


def test_job_past_the_drain_deadline_is_returned_to_the_queue(topology):
    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            async with running_worker(
                topology, store, runner=ScriptedRunner(delay=30.0), drain=0.3
            ) as stop:
                job_id = await submit(store, queue)
                await wait_for_status(store, job_id, ("running",))
                stop.set()
            # Unacknowledged, so the broker puts it back for the next worker.
            await wait_for_depth(topology.queue, 1)
            return await store.get(job_id)
        finally:
            await queue.close()
            await store.close()

    job = run(scenario())

    # The record keeps the attempt that was interrupted.
    assert (job.status, job.attempts) == ("running", 1)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def test_publish_that_no_queue_would_take_is_an_error(topology):
    async def scenario():
        store, queue = new_store(), new_queue(topology)
        try:
            await submit(store, queue)  # declares the topology
            connection = await aio_pika.connect(RABBITMQ_URL)
            async with connection:
                channel = await connection.channel()
                await channel.queue_delete(topology.queue)
            await submit(store, queue)
        finally:
            await queue.close()
            await store.close()

    # Without mandatory publishing the broker would accept and drop it.
    with pytest.raises(JobQueueUnavailable):
        run(scenario())


def test_existing_queue_with_other_arguments_is_refused(topology):
    changed = JobTopology(
        topology.queue,
        max_attempts=topology.max_attempts + 1,
        retry_delay_seconds=topology.retry_delay_seconds,
    )

    async def scenario():
        store, original, conflicting = (
            new_store(),
            new_queue(topology),
            new_queue(changed),
        )
        try:
            await submit(store, original)
            await submit(store, conflicting)
        finally:
            await original.close()
            await conflicting.close()
            await store.close()

    with pytest.raises(JobQueueUnavailable, match="PRECONDITION_FAILED"):
        run(scenario())


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


def test_records_expire_and_ids_cannot_be_reused():
    async def scenario():
        store = RedisJobStore(REDIS_URL, ttl_seconds=60, timeout_seconds=5)
        try:
            job = Job(id=new_job_id(), url="/v1/completions", status="queued")
            await store.create(job)
            ttl = await store._redis.ttl(f"vortex:job:{job.id}")
            with pytest.raises(ValueError, match="already exists"):
                await store.create(job)
            await store.delete(job.id)
            return ttl
        finally:
            await store.close()

    assert 0 < run(scenario()) <= 60


def test_concurrent_submissions_with_one_key_create_one_job():
    async def scenario():
        store = new_store()
        try:
            key = f"it-{new_job_id()}"
            claims = await asyncio.gather(
                *(store.claim_idempotency_key(key, f"job_{n}") for n in range(20))
            )
            await store.release_idempotency_key(key)
            return claims
        finally:
            await store.close()

    claims = run(scenario())

    winners = [n for n, existing in enumerate(claims) if existing is None]
    assert len(winners) == 1
    # Every other request saw the winner's job.
    assert {c for c in claims if c is not None} == {f"job_{winners[0]}"}
