"""POST /v1/jobs and GET /v1/jobs/{id} against in-memory store and queue."""

import re

import pytest
from pydantic import SecretStr

import app.main as main
from app.jobs.api import JobServices
from app.jobs.models import Job, JobError
from tests.fakes import FakeJobQueue, InMemoryJobStore
from tests.helpers import metric

COMPLETION_JOB = {
    "url": "/v1/completions",
    "body": {"prompt": "hello", "max_tokens": 8},
}
CHAT_JOB = {
    "url": "/v1/chat/completions",
    "body": {"messages": [{"role": "user", "content": "hi"}]},
}


@pytest.fixture
def jobs(client, monkeypatch) -> JobServices:
    services = JobServices(store=InMemoryJobStore(), queue=FakeJobQueue())
    monkeypatch.setattr(main.app.state, "jobs", services)
    return services


def submit(client, payload=COMPLETION_JOB, **headers):
    return client.post("/v1/jobs", json=payload, headers=headers)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def test_submission_is_queued_and_acknowledged(client, jobs):
    submitted = metric("vortex_jobs_submitted_total", url="/v1/completions")

    response = submit(client)

    assert response.status_code == 202
    job = response.json()
    assert re.fullmatch(r"job_[0-9a-f]{32}", job["id"])
    assert job["object"] == "job"
    assert job["status"] == "queued"
    assert response.headers["location"] == f"/v1/jobs/{job['id']}"
    # Fields without a value are omitted rather than sent as null.
    assert "result" not in job and "error" not in job

    assert jobs.store.jobs[job["id"]].status == "queued"
    (message,) = jobs.queue.messages
    assert message.job_id == job["id"]
    assert message.url == "/v1/completions"
    # Only what the client sent travels; the worker applies defaults when it
    # validates the body again.
    assert message.body == {"prompt": "hello", "max_tokens": 8}
    assert metric("vortex_jobs_submitted_total", url="/v1/completions") == submitted + 1


def test_chat_submission(client, jobs):
    response = submit(client, CHAT_JOB)

    assert response.status_code == 202
    assert jobs.queue.messages[-1].body == CHAT_JOB["body"]


@pytest.mark.parametrize(
    "payload",
    [
        {"url": "/v1/embeddings", "body": {"input": "x"}},
        {"url": "/v1/completions", "body": {"prompt": ""}},
        {"url": "/v1/completions", "body": {"prompt": "hi", "stream": True}},
        {"url": "/v1/chat/completions", "body": {"messages": []}},
        {"url": "/v1/completions", "body": {"prompt": "hi"}, "extra": 1},
        {"body": {"prompt": "hi"}},
    ],
    ids=[
        "unknown-endpoint",
        "invalid-body",
        "streaming",
        "invalid-chat-body",
        "unknown-field",
        "missing-url",
    ],
)
def test_invalid_submissions_are_rejected(client, jobs, payload):
    response = submit(client, payload)

    assert response.status_code == 422
    assert jobs.queue.messages == []
    assert jobs.store.jobs == {}


def test_jobs_require_the_api_key(client, jobs, monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", SecretStr("test-api-key"))

    assert submit(client).status_code == 401
    assert client.get("/v1/jobs/job_" + "0" * 32).status_code == 401
    assert submit(client, Authorization="Bearer test-api-key").status_code == 202


def test_jobs_disabled_answer_404(client):
    assert submit(client).status_code == 404
    assert client.get("/v1/jobs/job_" + "0" * 32).status_code == 404


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_repeated_idempotency_key_returns_the_first_job(client, jobs):
    first = submit(client, **{"Idempotency-Key": "order-42"})
    again = submit(client, **{"Idempotency-Key": "order-42"})
    other = submit(client, **{"Idempotency-Key": "order-43"})

    assert first.status_code == 202
    assert again.status_code == 200
    assert again.json()["id"] == first.json()["id"]
    assert other.json()["id"] != first.json()["id"]
    assert len(jobs.queue.messages) == 2


def test_key_of_an_expired_job_starts_a_new_one(client, jobs):
    first = submit(client, **{"Idempotency-Key": "k"}).json()
    del jobs.store.jobs[first["id"]]  # the record expired before the key

    again = submit(client, **{"Idempotency-Key": "k"})

    assert again.status_code == 202
    assert again.json()["id"] != first["id"]
    assert jobs.store.idempotency["k"] == again.json()["id"]


# ---------------------------------------------------------------------------
# Dependency failures
# ---------------------------------------------------------------------------


def test_store_outage_is_a_503_and_queues_nothing(client, jobs):
    jobs.store.unavailable = True
    failures = metric("vortex_job_submission_failures_total", component="store")

    response = submit(client)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert jobs.queue.messages == []
    assert (
        metric("vortex_job_submission_failures_total", component="store")
        == failures + 1
    )


def test_queue_outage_leaves_no_record_behind(client, jobs):
    jobs.queue.unavailable = True

    response = submit(client, **{"Idempotency-Key": "retry-me"})

    assert response.status_code == 503
    # Neither a job that would never run nor a key pointing at one remains,
    # so the client's retry with the same key creates the job.
    assert jobs.store.jobs == {}
    assert jobs.store.idempotency == {}

    jobs.queue.unavailable = False
    assert submit(client, **{"Idempotency-Key": "retry-me"}).status_code == 202


# ---------------------------------------------------------------------------
# Reading jobs
# ---------------------------------------------------------------------------


def test_get_returns_the_current_state(client, jobs):
    job_id = submit(client).json()["id"]

    response = client.get(f"/v1/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_finished_jobs_carry_their_result_or_error(client, jobs):
    done = Job(
        id="job_" + "a" * 32,
        url="/v1/completions",
        status="succeeded",
        attempts=1,
        result={"object": "text_completion", "choices": [{"text": "hi"}]},
    )
    failed = Job(
        id="job_" + "b" * 32,
        url="/v1/chat/completions",
        status="failed",
        attempts=1,
        error=JobError(status_code=400, message="The prompt does not fit."),
    )
    jobs.store.jobs = {done.id: done, failed.id: failed}

    assert client.get(f"/v1/jobs/{done.id}").json()["result"] == done.result
    assert client.get(f"/v1/jobs/{failed.id}").json()["error"] == {
        "status_code": 400,
        "message": "The prompt does not fit.",
    }


def test_unknown_or_malformed_ids(client, jobs):
    assert client.get("/v1/jobs/job_" + "f" * 32).status_code == 404
    # IDs are checked against their format before reaching Redis.
    assert client.get("/v1/jobs/not-a-job").status_code == 422


def test_store_outage_on_read_is_a_503(client, jobs):
    jobs.store.unavailable = True

    assert client.get("/v1/jobs/job_" + "0" * 32).status_code == 503
