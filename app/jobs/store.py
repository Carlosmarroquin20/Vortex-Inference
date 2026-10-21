"""
Job state and results in Redis.

Each job is one JSON document under vortex:job:{id} with a TTL that is
refreshed on every update, so a finished job's result stays readable for
job_ttl_seconds and nothing accumulates. An Idempotency-Key maps to the job
it created under vortex:idempotency:{sha256(key)}, with the same TTL.
"""

import hashlib
from typing import Protocol

import redis.asyncio as redis

from app.jobs.models import Job

_JOB_KEY = "vortex:job:{}"
_IDEMPOTENCY_KEY = "vortex:idempotency:{}"


class JobStoreUnavailable(RuntimeError):
    """Redis could not be reached or answered with an error."""


class JobStore(Protocol):
    async def create(self, job: Job) -> None:
        """Stores a new job; raises if the ID is already taken."""
        ...

    async def save(self, job: Job) -> None:
        """Replaces an existing job's record and refreshes its TTL."""
        ...

    async def get(self, job_id: str) -> Job | None: ...

    async def delete(self, job_id: str) -> None: ...

    async def claim_idempotency_key(self, key: str, job_id: str) -> str | None:
        """
        Binds key to job_id if it is free. Returns None when the claim
        succeeded, or the ID of the job that already holds the key.
        """
        ...

    async def release_idempotency_key(self, key: str) -> None: ...

    async def close(self) -> None: ...


class RedisJobStore:
    def __init__(
        self,
        url: str,
        ttl_seconds: int,
        timeout_seconds: float,
        client: redis.Redis | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        # Both timeouts are bounded so an unreachable Redis fails the request
        # promptly; redis-py would otherwise wait on the OS defaults. client
        # replaces the connection, for tests.
        self._redis = client or redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
        )

    async def create(self, job: Job) -> None:
        created = await self._call(
            self._redis.set(
                _JOB_KEY.format(job.id), job.model_dump_json(), nx=True, ex=self._ttl
            )
        )
        if not created:
            raise ValueError(f"job {job.id} already exists")

    async def save(self, job: Job) -> None:
        await self._call(
            self._redis.set(
                _JOB_KEY.format(job.id), job.model_dump_json(), ex=self._ttl
            )
        )

    async def get(self, job_id: str) -> Job | None:
        raw = await self._call(self._redis.get(_JOB_KEY.format(job_id)))
        return Job.model_validate_json(raw) if raw is not None else None

    async def delete(self, job_id: str) -> None:
        await self._call(self._redis.delete(_JOB_KEY.format(job_id)))

    async def claim_idempotency_key(self, key: str, job_id: str) -> str | None:
        name = _IDEMPOTENCY_KEY.format(_digest(key))
        if await self._call(self._redis.set(name, job_id, nx=True, ex=self._ttl)):
            return None
        return await self._call(self._redis.get(name))

    async def release_idempotency_key(self, key: str) -> None:
        await self._call(self._redis.delete(_IDEMPOTENCY_KEY.format(_digest(key))))

    async def close(self) -> None:
        await self._redis.aclose()

    @staticmethod
    async def _call(awaitable):
        try:
            return await awaitable
        except redis.RedisError as exc:
            raise JobStoreUnavailable(f"Redis request failed: {exc!r}") from exc


def _digest(key: str) -> str:
    # Hashed so client-chosen keys never become raw Redis key names: their
    # length and characters stay bounded whatever the client sends.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
