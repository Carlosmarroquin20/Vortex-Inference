import pytest
from pydantic import SecretStr

import app.main as main

COMPLETION = {"prompt": "hello"}


@pytest.fixture
def api_key(monkeypatch) -> str:
    key = "test-api-key"
    monkeypatch.setattr(main.settings, "api_key", SecretStr(key))
    return key


def test_requests_are_accepted_without_a_configured_key(client):
    response = client.post("/v1/completions", json=COMPLETION)

    assert response.status_code == 200


def test_missing_token_is_rejected(client, api_key):
    response = client.post("/v1/completions", json=COMPLETION)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_token_is_rejected(client, api_key):
    response = client.post(
        "/v1/completions",
        json=COMPLETION,
        headers={"Authorization": "Bearer not-the-key"},
    )

    assert response.status_code == 401


def test_non_bearer_scheme_is_rejected(client, api_key):
    response = client.post(
        "/v1/completions",
        json=COMPLETION,
        headers={"Authorization": f"Basic {api_key}"},
    )

    assert response.status_code == 401


def test_correct_token_is_accepted(client, api_key):
    response = client.post(
        "/v1/completions",
        json=COMPLETION,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200


def test_rejected_requests_never_reach_the_model(client, fake_model, api_key):
    # Authentication runs before the request is queued, so unauthenticated
    # traffic cannot consume inference capacity or inflate the KEDA signal.
    client.post("/v1/completions", json=COMPLETION)

    assert fake_model.calls == []


@pytest.mark.parametrize(
    "path", ["/health", "/health/live", "/health/ready", "/metrics"]
)
def test_operational_endpoints_stay_open_with_auth_enabled(client, api_key, path):
    assert client.get(path).status_code == 200
