"""
Pieces shared by backends that call a remote model API.

Both remote backends translate upstream HTTP statuses the same way and report
busy_seconds() the same way; keeping that here means a provider-specific
backend only has to know how to build a request and read a response.
"""

import itertools
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from app.backends.base import BackendError, RequestRejectedError

logger = logging.getLogger("vortex.backends.upstream")

# Statuses that describe the request content (an oversized prompt, a
# moderation refusal, an out-of-range parameter). They are forwarded as 400
# so the client can act on them; every other failure is the server's problem.
_REJECTION_STATUSES = frozenset({400, 413, 422})
# 529 is Anthropic's "overloaded"; it calls for the same back-off as a 429.
_OVERLOAD_STATUSES = frozenset({429, 529})
_TIMEOUT_STATUSES = frozenset({408, 504})

# Upper bound on upstream error text forwarded to clients or logged.
MAX_ERROR_MESSAGE = 300


def status_error(
    status: int, message: str, retry_after: str | None = None
) -> RequestRejectedError | BackendError:
    """Exception to raise for a non-2xx upstream response."""
    message = message[:MAX_ERROR_MESSAGE]
    if status in _REJECTION_STATUSES:
        # Logged at WARNING because a server-side misconfiguration, such as a
        # parameter the model does not accept, produces the same status for
        # every request and would otherwise only show up as a rise in
        # "rejected" requests.
        logger.warning(
            "Upstream rejected the request.",
            extra={"upstream_status": status, "upstream_message": message},
        )
        return RequestRejectedError(
            f"The model provider rejected the request: {message}"
        )
    if status in _OVERLOAD_STATUSES:
        return BackendError(
            f"Upstream overloaded or rate limited (HTTP {status}): {message}",
            status_code=503,
            retry_after=retry_after,
        )
    if status in _TIMEOUT_STATUSES:
        return BackendError(
            f"Upstream timed out (HTTP {status}): {message}", status_code=504
        )
    # 401/403/404 land here too: a wrong key or model name is the operator's
    # to fix, not the client's.
    return BackendError(f"Upstream returned HTTP {status}: {message}")


class InflightTracker:
    """
    Start times of the calls currently waiting on the upstream API.

    Remote backends run several calls at once, so busy_seconds() reports the
    oldest one: that is the call the stall detection must notice if it never
    returns, regardless of how many newer calls complete around it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started: dict[int, float] = {}
        self._ids = itertools.count()

    @contextmanager
    def track(self) -> Iterator[None]:
        call_id = next(self._ids)
        with self._lock:
            self._started[call_id] = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                del self._started[call_id]

    def oldest_age(self) -> float:
        with self._lock:
            oldest = min(self._started.values(), default=None)
        return time.monotonic() - oldest if oldest is not None else 0.0
