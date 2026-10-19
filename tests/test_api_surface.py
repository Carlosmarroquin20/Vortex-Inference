"""Information exposed by the API beyond the completion itself."""

import pytest


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_and_schema_are_disabled_by_default(client, path):
    assert client.get(path).status_code == 404


def test_internal_errors_do_not_leak_exception_details(client, fake_model):
    fake_model.error = RuntimeError("cannot mmap /models/secret-path.gguf")

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 500
    assert "secret-path" not in response.text
    assert response.json() == {"detail": "Inference failed due to an internal error."}


def test_health_reports_model_name_not_filesystem_path(client):
    body = client.get("/health/ready").json()

    assert body == {"status": "healthy", "model_loaded": True, "model": "test-model"}
    assert "/models/" not in str(body)
