"""HTTP-level metrics and fault injection: what the canary analysis reads."""

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.logging_config import RequestContextMiddleware, metric_route
from app.main import FaultInjectionMiddleware
from tests.helpers import metric

RESPONSES = "vortex_http_responses_total"
DURATION_COUNT = "vortex_http_response_duration_seconds_count"


@pytest.mark.parametrize(
    ("path", "route"),
    [
        ("/v1/completions", "/v1/completions"),
        ("/v1/chat/completions", "/v1/chat/completions"),
        ("/v1/jobs", "/v1/jobs"),
        ("/v1/jobs/job_3f9c0a", "/v1/jobs/{job_id}"),
        ("/wp-login.php", "other"),
    ],
)
def test_routes_are_a_bounded_label(path, route):
    assert metric_route(path) == route


def test_a_completion_is_counted_by_route_and_class(client):
    before = metric(RESPONSES, route="/v1/completions", code="2xx")
    timed_before = metric(DURATION_COUNT, route="/v1/completions")

    client.post("/v1/completions", json={"prompt": "hello"})

    assert metric(RESPONSES, route="/v1/completions", code="2xx") == before + 1
    assert metric(DURATION_COUNT, route="/v1/completions") == timed_before + 1


def test_errors_are_counted_as_5xx(client, fake_model):
    fake_model.error = RuntimeError("boom")
    before = metric(RESPONSES, route="/v1/completions", code="5xx")

    client.post("/v1/completions", json={"prompt": "hello"})

    assert metric(RESPONSES, route="/v1/completions", code="5xx") == before + 1


def test_client_errors_are_counted_as_4xx_not_5xx(client):
    before_4xx = metric(RESPONSES, route="/v1/completions", code="4xx")
    before_5xx = metric(RESPONSES, route="/v1/completions", code="5xx")

    client.post("/v1/completions", json={"prompt": 42})

    assert metric(RESPONSES, route="/v1/completions", code="4xx") == before_4xx + 1
    assert metric(RESPONSES, route="/v1/completions", code="5xx") == before_5xx


def test_probes_and_scrapes_are_not_counted(client):
    before = metric(RESPONSES, route="other", code="2xx")

    client.get("/health/ready")
    client.get("/metrics")

    assert metric(RESPONSES, route="other", code="2xx") == before


def _app_with_faults(rate: float) -> TestClient:
    async def completions(request):
        return PlainTextResponse("ok")

    app = Starlette(
        routes=[
            Route("/v1/completions", completions, methods=["POST"]),
            Route("/v1/jobs", completions, methods=["POST"]),
        ]
    )
    # Same order as the server: faults inside the request middleware, so
    # they are counted.
    app.add_middleware(FaultInjectionMiddleware, rate=rate)
    app.add_middleware(RequestContextMiddleware)
    return TestClient(app)


def test_every_completion_fails_at_rate_one():
    client = _app_with_faults(1.0)
    before = metric(RESPONSES, route="/v1/completions", code="5xx")

    responses = [client.post("/v1/completions") for _ in range(5)]

    assert {r.status_code for r in responses} == {500}
    assert responses[0].json() == {"detail": "Injected fault (VORTEX_FAULT_RATE)."}
    assert metric(RESPONSES, route="/v1/completions", code="5xx") == before + 5


def test_no_completion_fails_at_rate_zero():
    client = _app_with_faults(0.0)

    assert {client.post("/v1/completions").status_code for _ in range(20)} == {200}


def test_only_completion_routes_are_affected():
    client = _app_with_faults(1.0)

    assert client.post("/v1/jobs").status_code == 200


def test_the_rate_is_honoured_statistically():
    middleware = FaultInjectionMiddleware(app=None, rate=0.3)
    middleware._random.seed(7)

    failures = sum(middleware._random.random() < middleware.rate for _ in range(10_000))

    assert 2_800 < failures < 3_200


def test_fault_rate_is_off_by_default_and_bounded():
    from app.config import Settings

    assert Settings().fault_rate == 0.0
    with pytest.raises(ValueError):
        Settings(fault_rate=1.5)
