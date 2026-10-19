"""
Serialization guarantees around the non-thread-safe Llama object.

The core invariant: at most one worker thread is ever inside the model,
including when a request stops waiting for its thread (timeout or
cancellation) before that thread has exited. FakeLlama records the maximum
number of concurrent calls it observed, which must never exceed one.
"""

import asyncio

import app.main as main
from app.main import CompletionRequest, create_completion
from tests.helpers import metric

ACTIVE = "vortex_inference_active_requests"


def _run_with_app(scenario):
    """Runs an async scenario inside the real application lifespan."""

    async def runner():
        async with main.lifespan(main.app):
            return await scenario(main.llm._model)

    return asyncio.run(runner())


async def _wait_until_inside_model(fake) -> None:
    started = await asyncio.to_thread(fake.started.wait, 5)
    assert started, "worker thread never entered the model"


def test_concurrent_requests_are_serialized():
    async def scenario(fake):
        fake.steps, fake.step_delay = 5, 0.01
        requests = [create_completion(CompletionRequest(prompt="hi")) for _ in range(3)]
        await asyncio.gather(*requests)
        return fake

    fake = _run_with_app(scenario)

    assert len(fake.calls) == 3
    assert fake.max_active_calls == 1


def test_cancelled_request_keeps_permit_until_its_thread_exits():
    # Regression test: CancelledError used to bypass the timeout handler and
    # release the semaphore while the thread was still inside the model.
    async def scenario(fake):
        fake.steps, fake.step_delay = 200, 0.01

        first = asyncio.create_task(create_completion(CompletionRequest(prompt="a")))
        await _wait_until_inside_model(fake)
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass

        fake.steps = 1
        await create_completion(CompletionRequest(prompt="b"))
        return fake

    fake = _run_with_app(scenario)

    assert fake.max_active_calls == 1
    # The cancelled call observed the stop event instead of running to the end.
    assert fake.steps_run[0] < 200
    assert metric(ACTIVE) == 0


def test_shutdown_waits_for_abandoned_threads():
    async def scenario(fake):
        fake.steps, fake.step_delay = 200, 0.01
        task = asyncio.create_task(create_completion(CompletionRequest(prompt="a")))
        await _wait_until_inside_model(fake)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return fake

    # Leaving the lifespan context runs the shutdown path, which must not
    # return while a worker thread is still executing native code.
    fake = _run_with_app(scenario)

    assert fake.active_calls == 0
    assert main._drain_tasks == set()


def test_timeout_returns_504_and_frees_the_permit(client, fake_model, monkeypatch):
    monkeypatch.setattr(main.settings, "inference_timeout_seconds", 0.2)
    fake_model.steps, fake_model.step_delay = 500, 0.01
    timeouts_before = metric("vortex_inference_requests_total", status="timeout")

    response = client.post("/v1/completions", json={"prompt": "slow"})

    assert response.status_code == 504
    assert metric("vortex_inference_requests_total", status="timeout") == (
        timeouts_before + 1
    )

    # The next request queues behind the draining thread and then succeeds,
    # proving the permit was handed back rather than leaked.
    fake_model.steps = 1
    assert client.post("/v1/completions", json={"prompt": "fast"}).status_code == 200
    assert fake_model.steps_run[0] < 500
    assert fake_model.max_active_calls == 1
    assert metric(ACTIVE) == 0
