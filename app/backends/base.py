"""
Contract between the HTTP layer and the engine that produces completions.

main.py owns admission control, deadlines, metrics and the response schema;
a backend owns only how text is generated. Keeping that line here means a
new engine (a remote API, a different runtime) is one class plus a branch in
create_backend(), with the request path and its tests unchanged.
"""

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable


class GenerationResult(TypedDict):
    text: str
    # "stop" for a stop sequence or end of sequence, "length" for max_tokens.
    finish_reason: str | None
    tokens_prompt: int
    tokens_completion: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    """The next fragment of generated text."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """Final event of a stream that completed; carries what usage reports."""

    finish_reason: str | None
    tokens_prompt: int
    tokens_completion: int


StreamEvent = TextDelta | StreamEnd


class RequestRejectedError(ValueError):
    """
    The request cannot be served as submitted; the API answers 400.

    Raised for problems the client can fix by changing the request, so they
    are counted as "rejected" and kept out of the error-ratio SLO. The
    message is returned to the client and must not carry internal details.
    """


class PromptTooLongError(RequestRejectedError):
    """The prompt alone does not fit in the model's context window."""

    def __init__(self, prompt_tokens: int, n_ctx: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.n_ctx = n_ctx
        super().__init__(
            f"Prompt is {prompt_tokens} tokens; the context window is {n_ctx} "
            "tokens and must also leave room for the completion."
        )


class BackendError(RuntimeError):
    """
    The engine failed to produce a completion for a reason outside the client.

    status_code is the HTTP status the API answers with: 502 when an upstream
    service failed or refused the server's own credentials or configuration,
    503 when it is temporarily overloaded, 504 when it did not answer in time.
    The message is logged, never returned, since upstream error bodies can
    describe account and deployment details.
    """

    def __init__(
        self, message: str, status_code: int = 502, retry_after: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        # Upstream Retry-After value, forwarded on 503 so clients back off for
        # as long as the provider asked rather than retrying immediately.
        self.retry_after = retry_after


@runtime_checkable
class InferenceBackend(Protocol):
    """
    Synchronous generation engine driven from a worker thread.

    The HTTP layer calls generate() through run_in_executor and never enters
    a backend from two threads at once beyond max_concurrent_requests, so an
    implementation need not be thread-safe when that setting is 1.
    Construction may block (loading weights, opening connections); it runs
    once during application startup and should raise on any condition that
    makes serving impossible.
    """

    def is_ready(self) -> bool:
        """True once the backend can accept generate() calls."""
        ...

    def busy_seconds(self) -> float:
        """
        Seconds the oldest in-progress generate() call has been running, or 0.

        Feeds the stall detection behind the liveness and readiness probes, so
        it must keep growing while a call is stuck rather than reset.
        """
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        stop_event: threading.Event | None = None,
    ) -> GenerationResult:
        """
        Produces one completion for the prompt.

        stop_event is set by the HTTP layer when the request is abandoned
        (deadline or client cancellation). Implementations should return as
        soon as practical once it is set; the inference permit is held until
        they do. Raises RequestRejectedError (for example PromptTooLongError)
        for requests the client must change, and BackendError for failures
        the client cannot fix.
        """
        ...

    def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """
        Produces the completion incrementally: TextDelta events in order,
        then exactly one StreamEnd.

        Runs in a worker thread like generate() and raises the same errors.
        The HTTP layer waits for the first event before it commits to a
        streaming response, so a failure raised before any event still gets
        its proper status code; a failure after that can only end the stream.
        The consumer may stop iterating early (client disconnect, deadline);
        implementations must then release what they hold, which for a
        generator means cleaning up when it is closed, and should also stop
        once stop_event is set.
        """
        ...

    def close(self) -> None:
        """Releases resources at shutdown, after in-flight calls have drained."""
        ...
