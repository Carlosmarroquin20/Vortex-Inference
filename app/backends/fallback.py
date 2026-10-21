"""
A primary backend with a second one to fall back on.

The fallback takes a request only for conditions it can actually change:

    overloaded       the primary answered 503 (rate limited, over capacity)
    upstream_error   it failed upstream (502)
    timeout          its upstream did not answer in time (504)
    prompt_too_long  the prompt does not fit the primary's context; a model
                     with a larger window may still serve it
    circuit_open     the primary has failed repeatedly and is being skipped

Everything else goes back to the client unchanged. A request the client
abandoned is not retried, and a rejection that is about the request itself
(an invalid parameter, a moderation refusal) would be rejected by any model.

After fallback_failure_threshold consecutive primary failures the circuit
opens: requests go straight to the fallback for fallback_cooldown_seconds,
so a primary that is down costs its failure latency a few times rather than
on every request. The first request after the cooldown tries the primary
again, and its outcome closes or reopens the circuit.

A streamed request can only fall back before the primary has produced any
output; once text has reached the client, a failure ends the stream as it
would without a fallback.
"""

import contextlib
import logging
import threading
import time
from collections.abc import Iterator

from app.backends.base import (
    BackendError,
    GenerationResult,
    InferenceBackend,
    Prompt,
    PromptTooLongError,
    StreamEvent,
)
from app.metrics import BACKEND_FALLBACKS_TOTAL

logger = logging.getLogger("vortex.backends.fallback")

_REASONS = {503: "overloaded", 502: "upstream_error", 504: "timeout"}


def _fallback_reason(exc: Exception) -> str | None:
    """Why exc should go to the fallback, or None if it should not."""
    if isinstance(exc, PromptTooLongError):
        return "prompt_too_long"
    if isinstance(exc, BackendError):
        return _REASONS.get(exc.status_code, "upstream_error")
    return None


class _Circuit:
    """Consecutive-failure breaker; shared by concurrent requests."""

    def __init__(self, threshold: int, cooldown_seconds: float) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> bool:
        """Counts a failure; True when it opened the circuit."""
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._failures = 0
                self._open_until = time.monotonic() + self._cooldown
                return True
            return False


class FallbackBackend:
    def __init__(
        self,
        primary: InferenceBackend,
        fallback: InferenceBackend,
        fallback_model: str,
        failure_threshold: int,
        cooldown_seconds: float,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fallback_model = fallback_model
        self._circuit = _Circuit(failure_threshold, cooldown_seconds)

    def is_ready(self) -> bool:
        return self._primary.is_ready() or self._fallback.is_ready()

    def busy_seconds(self) -> float:
        return max(self._primary.busy_seconds(), self._fallback.busy_seconds())

    def close(self) -> None:
        try:
            self._primary.close()
        finally:
            self._fallback.close()

    def generate(
        self,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        stop_event: threading.Event | None = None,
    ) -> GenerationResult:
        args = (prompt, max_tokens, temperature, top_p, stop, stop_event)
        if self._circuit.is_open():
            return self._run_fallback("circuit_open", args)
        try:
            result = self._primary.generate(*args)
        except Exception as exc:
            reason = self._divert(exc, stop_event)
            return self._run_fallback(reason, args, cause=exc)
        self._circuit.record_success()
        return result

    def stream(
        self,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        args = (prompt, max_tokens, temperature, top_p, stop, stop_event)
        if self._circuit.is_open():
            BACKEND_FALLBACKS_TOTAL.labels(reason="circuit_open").inc()
            yield from self._fallback.stream(*args)
            return

        with contextlib.closing(self._primary.stream(*args)) as source:
            try:
                first = next(source)
            except StopIteration:
                return
            except Exception as exc:
                reason = self._divert(exc, stop_event)
                BACKEND_FALLBACKS_TOTAL.labels(reason=reason).inc()
                self._log(reason, exc)
                yield from self._fallback.stream(*args)
                return
            # Output has started: from here on the primary owns the request.
            self._circuit.record_success()
            yield first
            yield from source

    def _divert(self, exc: Exception, stop_event: threading.Event | None) -> str:
        """The fallback reason for exc; re-raises when it does not qualify."""
        reason = _fallback_reason(exc)
        if reason is None or (stop_event is not None and stop_event.is_set()):
            raise exc
        if reason != "prompt_too_long" and self._circuit.record_failure():
            logger.warning(
                "Primary backend failing repeatedly. Sending requests to the "
                "fallback for the cooldown.",
            )
        return reason

    def _run_fallback(
        self, reason: str, args: tuple, cause: Exception | None = None
    ) -> GenerationResult:
        BACKEND_FALLBACKS_TOTAL.labels(reason=reason).inc()
        if cause is not None:
            self._log(reason, cause)
        try:
            result = self._fallback.generate(*args)
        except Exception as exc:
            if cause is not None:
                # Both failed; the fallback's error is the one the client
                # gets, with the primary's attached for the log.
                raise exc from cause
            raise
        # The response names the model that produced it.
        return {**result, "model": self._fallback_model}

    def _log(self, reason: str, exc: Exception) -> None:
        logger.warning(
            "Primary backend could not serve the request. Using the fallback.",
            extra={"reason": reason, "error": str(exc)[:300]},
        )
