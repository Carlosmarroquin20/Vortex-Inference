"""The worker's metrics and probe endpoint (app/jobs/health.py)."""

import asyncio

import httpx2
import pytest

import app.jobs.health as health_module
from app.jobs.health import WorkerHealth, serve


class Backend:
    def __init__(self) -> None:
        self.busy = 0.0

    def busy_seconds(self) -> float:
        return self.busy


@pytest.fixture
def endpoint():
    backend = Backend()
    health = WorkerHealth(backend, timeout_seconds=10)
    server = serve(health, port=0, host="127.0.0.1")
    port = server.server_address[1]
    try:
        yield health, backend, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def get(url: str) -> tuple[int, str]:
    response = httpx2.get(url, timeout=5)
    return response.status_code, response.text


def test_ready_only_while_consuming(endpoint):
    health, _, base = endpoint

    assert get(f"{base}/health/ready")[0] == 503
    health.consuming = True
    assert get(f"{base}/health/ready")[0] == 200


def test_live_before_the_first_heartbeat(endpoint):
    _, _, base = endpoint

    assert get(f"{base}/health/live") == (200, "alive\n")


def test_stale_heartbeat_fails_liveness(endpoint, monkeypatch):
    health, _, base = endpoint
    health.beat()
    monkeypatch.setattr(health_module, "HEARTBEAT_STALE_SECONDS", -1.0)

    status, body = get(f"{base}/health/live")

    assert status == 503
    assert "event loop" in body


def test_wedged_inference_thread_fails_liveness(endpoint):
    health, backend, base = endpoint
    health.beat()
    backend.busy = 10 + health_module.STALL_GRACE_SECONDS + 1

    status, body = get(f"{base}/health/live")

    assert status == 503
    assert "stalled" in body


def test_slow_generation_is_still_alive(endpoint):
    health, backend, base = endpoint
    health.beat()
    # Past the deadline but within the grace for stopping at a token boundary.
    backend.busy = 10 + 5

    assert get(f"{base}/health/live")[0] == 200


def test_metrics_and_unknown_paths(endpoint):
    _, _, base = endpoint

    status, body = get(f"{base}/metrics")
    assert status == 200
    assert "vortex_worker_jobs_in_progress" in body
    assert get(f"{base}/other")[0] == 404


def test_heartbeat_task_beats_while_the_loop_runs():
    health = WorkerHealth(Backend(), timeout_seconds=10)

    async def scenario():
        task = asyncio.create_task(health.run_heartbeat())
        await asyncio.sleep(0)
        task.cancel()

    asyncio.run(scenario())

    assert health._heartbeat is not None
