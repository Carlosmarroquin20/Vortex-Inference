"""
Contract between the HTTP layer and the engine that produces completions.

main.py owns admission control, deadlines, metrics and the response schema;
a backend owns only how text is generated. Keeping that line here means a
new engine (a remote API, a different runtime) is one class plus a branch in
create_backend(), with the request path and its tests unchanged.
"""

import threading
from typing import Protocol, TypedDict, runtime_checkable


class GenerationResult(TypedDict):
    text: str
    # "stop" for a stop sequence or end of sequence, "length" for max_tokens.
    finish_reason: str | None
    tokens_prompt: int
    tokens_completion: int


class PromptTooLongError(ValueError):
    """The prompt alone does not fit in the model's context window."""

    def __init__(self, prompt_tokens: int, n_ctx: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.n_ctx = n_ctx
        super().__init__(
            f"Prompt is {prompt_tokens} tokens; the context window is {n_ctx} "
            "tokens and must also leave room for the completion."
        )


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
        they do. Raises PromptTooLongError when the prompt cannot fit in the
        context window, which the API reports as a client error.
        """
        ...
