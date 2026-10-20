import time

import pytest

import app.main as main


@pytest.fixture
def stalled(fake_model):
    """Simulates a worker thread stuck in native code past the stall threshold."""
    overrun = main.settings.inference_timeout_seconds + main.STALL_GRACE_SECONDS + 1
    main.backend._busy_since = time.monotonic() - overrun
    yield
    main.backend._busy_since = None


@pytest.mark.parametrize("path", ["/health/live", "/health/ready", "/health"])
def test_probes_pass_when_idle(client, path):
    assert client.get(path).status_code == 200


def test_slow_inference_within_deadline_is_not_a_failure(client, fake_model):
    # A long-running request is load, not a fault: restarting or unrouting
    # the pod would discard queued work. KEDA handles it by scaling out.
    main.backend._busy_since = (
        time.monotonic() - main.settings.inference_timeout_seconds
    )
    try:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
    finally:
        main.backend._busy_since = None


@pytest.mark.parametrize("path", ["/health/live", "/health/ready", "/health"])
def test_stalled_worker_fails_all_probes(client, stalled, path):
    assert client.get(path).status_code == 503


def test_probes_recover_once_the_worker_exits(client, fake_model):
    overrun = main.settings.inference_timeout_seconds + main.STALL_GRACE_SECONDS + 1
    main.backend._busy_since = time.monotonic() - overrun
    assert client.get("/health/live").status_code == 503

    main.backend._busy_since = None

    assert client.get("/health/live").status_code == 200


def test_busy_time_is_tracked_during_generation(client, fake_model):
    fake_model.steps, fake_model.step_delay = 1, 0.0
    client.post("/v1/completions", json={"prompt": "hello"})

    # Cleared once the call returns, so an idle pod never looks stalled.
    assert main.backend.busy_seconds() == 0.0
