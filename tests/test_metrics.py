import app.main as main
from tests.helpers import metric

DURATION_COUNT = "vortex_inference_duration_seconds_count"
REQUESTS = "vortex_inference_requests_total"


def test_success_is_counted_and_timed(client):
    requests_before = metric(REQUESTS, status="success")
    timed_before = metric(DURATION_COUNT, status="success")
    queued_before = metric("vortex_inference_queue_wait_seconds_count")

    client.post("/v1/completions", json={"prompt": "hello"})

    assert metric(REQUESTS, status="success") == requests_before + 1
    assert metric(DURATION_COUNT, status="success") == timed_before + 1
    assert metric("vortex_inference_queue_wait_seconds_count") == queued_before + 1


def test_timeouts_are_included_in_latency(client, fake_model, monkeypatch):
    # Excluding timeouts would hide the slowest requests from the latency SLO.
    monkeypatch.setattr(main.settings, "inference_timeout_seconds", 0.1)
    fake_model.steps, fake_model.step_delay = 500, 0.01
    before = metric(DURATION_COUNT, status="timeout")

    client.post("/v1/completions", json={"prompt": "slow"})

    assert metric(DURATION_COUNT, status="timeout") == before + 1


def test_client_errors_are_not_counted_as_service_errors(client):
    errors_before = metric(REQUESTS, status="error")
    rejected_before = metric(REQUESTS, status="rejected")

    client.post("/v1/completions", json={"prompt": "word " * 64})

    assert metric(REQUESTS, status="rejected") == rejected_before + 1
    assert metric(REQUESTS, status="error") == errors_before


def test_failures_are_counted_as_errors(client, fake_model):
    fake_model.error = RuntimeError("boom")
    before = metric(REQUESTS, status="error")

    client.post("/v1/completions", json={"prompt": "hello"})

    assert metric(REQUESTS, status="error") == before + 1


def test_token_counters_follow_reported_usage(client, fake_model):
    fake_model.completion_tokens = 11
    prompt_before = metric("vortex_inference_tokens_generated_total", type="prompt")
    completion_before = metric(
        "vortex_inference_tokens_generated_total", type="completion"
    )

    client.post("/v1/completions", json={"prompt": "one two three"})

    assert metric("vortex_inference_tokens_generated_total", type="prompt") == (
        prompt_before + 3
    )
    assert metric("vortex_inference_tokens_generated_total", type="completion") == (
        completion_before + 11
    )


def test_active_requests_gauge_returns_to_zero(client, fake_model):
    fake_model.error = RuntimeError("boom")
    client.post("/v1/completions", json={"prompt": "hello"})
    fake_model.error = None
    client.post("/v1/completions", json={"prompt": "hello"})

    assert metric("vortex_inference_active_requests") == 0


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "vortex_inference_active_requests" in response.text
