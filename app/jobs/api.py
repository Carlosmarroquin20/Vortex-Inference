"""
HTTP endpoints for asynchronous jobs.

POST /v1/jobs answers 202 as soon as the job is durably queued; a worker runs
it later and GET /v1/jobs/{id} returns its state and, once finished, the
response the synchronous endpoint would have produced.

Submission order is chosen so a client is never told about a job that will
not run: the record is written to Redis first and the message published
second, and if publishing fails the record is removed before answering 503.
A crash between the two steps leaves a record that stays "queued" until its
TTL expires, which a client sees as a job that never started rather than a
job silently lost.
"""

import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Path, Request, Response

from app.jobs.models import (
    JOB_ID_PATTERN,
    Job,
    JobMessage,
    JobSubmission,
    new_job_id,
)
from app.jobs.queue import JobQueue, JobQueueUnavailable
from app.jobs.store import JobStore, JobStoreUnavailable
from app.metrics import JOB_SUBMISSION_FAILURES_TOTAL, JOBS_SUBMITTED_TOTAL

logger = logging.getLogger("vortex.jobs")

router = APIRouter(tags=["jobs"])


@dataclass(frozen=True, slots=True)
class JobServices:
    store: JobStore
    queue: JobQueue


def _services(request: Request) -> JobServices:
    services: JobServices | None = getattr(request.app.state, "jobs", None)
    if services is None:
        raise HTTPException(
            status_code=404, detail="Asynchronous jobs are not enabled on this server."
        )
    return services


def _unavailable(component: str) -> HTTPException:
    # Retry-After because these outages are usually short (a broker or
    # Redis restart) and the submission is safe to repeat, especially with
    # an Idempotency-Key.
    return HTTPException(
        status_code=503,
        detail=f"The job {component} is unavailable. Retry later.",
        headers={"Retry-After": "5"},
    )


@router.post(
    "/v1/jobs",
    status_code=202,
    response_model=Job,
    response_model_exclude_none=True,
    responses={
        200: {"description": "Replay of a submission with the same Idempotency-Key"}
    },
)
async def submit_job(
    submission: JobSubmission,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(min_length=1, max_length=255)] = None,
) -> Job:
    """
    Queues a completion or chat completion to be run by a worker.

    With an Idempotency-Key header, repeating a submission (for example
    after a timeout) returns the job the first one created, with status
    200, instead of running the request twice.
    """
    services = _services(request)
    job = Job(id=new_job_id(), url=submission.url, status="queued")

    try:
        if idempotency_key is not None:
            existing_id = await services.store.claim_idempotency_key(
                idempotency_key, job.id
            )
            if existing_id is not None:
                existing = await services.store.get(existing_id)
                if existing is not None:
                    response.status_code = 200
                    response.headers["Location"] = f"/v1/jobs/{existing.id}"
                    return existing
                # The job expired before its key did; let this submission
                # take the key over.
                await services.store.release_idempotency_key(idempotency_key)
                await services.store.claim_idempotency_key(idempotency_key, job.id)

        await services.store.create(job)
    except JobStoreUnavailable as exc:
        JOB_SUBMISSION_FAILURES_TOTAL.labels(component="store").inc()
        logger.error("Job store unavailable.", extra={"reason": str(exc)})
        raise _unavailable("store") from exc

    message = JobMessage(
        job_id=job.id,
        url=submission.url,
        body=submission.body.model_dump(mode="json", exclude_unset=True),
    )
    try:
        await services.queue.publish(message)
    except JobQueueUnavailable as exc:
        JOB_SUBMISSION_FAILURES_TOTAL.labels(component="queue").inc()
        logger.error("Job queue unavailable.", extra={"reason": str(exc)})
        await _discard(services, job.id, idempotency_key)
        raise _unavailable("queue") from exc

    JOBS_SUBMITTED_TOTAL.labels(url=job.url).inc()
    logger.info("Job queued.", extra={"job_id": job.id, "url": job.url})
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return job


@router.get("/v1/jobs/{job_id}", response_model=Job, response_model_exclude_none=True)
async def get_job(
    job_id: Annotated[str, Path(pattern=JOB_ID_PATTERN)], request: Request
) -> Job:
    """A job's state; includes the result once it has succeeded."""
    services = _services(request)
    try:
        job = await services.store.get(job_id)
    except JobStoreUnavailable as exc:
        logger.error("Job store unavailable.", extra={"reason": str(exc)})
        raise _unavailable("store") from exc
    if job is None:
        # Unknown and expired jobs look the same, by design: records are
        # kept for job_ttl_seconds and then forgotten.
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job


async def _discard(
    services: JobServices, job_id: str, idempotency_key: str | None
) -> None:
    """Best-effort removal of a job that could not be queued."""
    try:
        await services.store.delete(job_id)
        if idempotency_key is not None:
            await services.store.release_idempotency_key(idempotency_key)
    except JobStoreUnavailable:
        # The record then expires on its own; it stays "queued" meanwhile.
        logger.warning("Could not remove a job that failed to queue.", exc_info=True)
