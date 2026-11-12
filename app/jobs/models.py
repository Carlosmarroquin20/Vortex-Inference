"""
Asynchronous job records and submissions.

A submission names the endpoint and carries the same body that endpoint
accepts, as in OpenAI's Batch API, so the request is validated by the very
models the synchronous endpoints use and a job's result has exactly the shape
the endpoint would have returned.
"""

import secrets
import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.catalog import CatalogEnrichmentRequest
from app.schemas import ChatCompletionRequest, CompletionRequest

JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobUrl = Literal["/v1/completions", "/v1/chat/completions", "/v1/catalog/enrichments"]
CATALOG_URL = "/v1/catalog/enrichments"

# 128 random bits: IDs cannot be guessed or enumerated, which matters because
# every holder of the API key can read any job whose ID they know.
JOB_ID_PATTERN = r"^job_[0-9a-f]{32}$"


def new_job_id() -> str:
    return f"job_{secrets.token_hex(16)}"


class _NoStreaming(BaseModel):
    @model_validator(mode="after")
    def reject_streaming(self):
        if self.body.stream:
            raise ValueError(
                "stream is not supported for jobs; the result is read with "
                "GET /v1/jobs/{id} once the job has finished."
            )
        return self


class CompletionJobSubmission(_NoStreaming):
    url: Literal["/v1/completions"]
    body: CompletionRequest

    model_config = {"extra": "forbid"}


class ChatCompletionJobSubmission(_NoStreaming):
    url: Literal["/v1/chat/completions"]
    body: ChatCompletionRequest

    model_config = {"extra": "forbid"}


class CatalogEnrichmentJobSubmission(BaseModel):
    # Asynchronous only: there is no synchronous endpoint at this URL. It
    # names the job type the way the other URLs do (app/catalog).
    url: Literal["/v1/catalog/enrichments"]
    body: CatalogEnrichmentRequest

    model_config = {"extra": "forbid"}


JobSubmission = Annotated[
    CompletionJobSubmission
    | ChatCompletionJobSubmission
    | CatalogEnrichmentJobSubmission,
    Field(discriminator="url"),
]


class JobError(BaseModel):
    # The status and message the synchronous endpoint would have answered
    # with, so clients handle a failed job like a failed request.
    status_code: int
    message: str


class JobUsage(BaseModel):
    # Summed over every attempt, rejected ones included: what the job cost,
    # not just what its final answer cost. Kept for catalog enrichments,
    # whose attempts are retried on invalid answers.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_seconds: float = 0.0


class Job(BaseModel):
    """A job as stored in Redis and returned by the API."""

    id: str
    object: Literal["job"] = "job"
    url: JobUrl
    status: JobStatus
    created_at: int = Field(default_factory=lambda: int(time.time()))
    started_at: int | None = None
    finished_at: int | None = None
    # Attempts started so far; above 1 means a retry.
    attempts: int = 0
    # The endpoint's response body (a text_completion or chat.completion
    # object, or a catalog.enrichment record) once the job has succeeded.
    result: dict[str, Any] | None = None
    # Why the job failed or, while it is queued for a retry, why its last
    # attempt did.
    error: JobError | None = None
    usage: JobUsage | None = None


class JobMessage(BaseModel):
    """What travels through RabbitMQ: enough for a worker to run the job."""

    job_id: str
    url: JobUrl
    # The body as submitted (unset fields omitted), validated again by the
    # worker against the same model.
    body: dict[str, Any]
