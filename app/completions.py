"""
What the completion endpoints and the job worker have in common.

A request is reduced to a Generation, the backend runs it, and the result is
turned into the endpoint's response body. Failures are classified the same
way on both paths, so a job that fails records the status and message the
synchronous endpoint would have answered with.
"""

import logging
import time
from dataclasses import dataclass

from app.backends import (
    BackendError,
    GenerationResult,
    Prompt,
    PromptTooLongError,
    RequestRejectedError,
)
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionPerformance,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)

logger = logging.getLogger("vortex.inference")

# Client-facing text for BackendError, by status. The exception message holds
# the upstream details and goes to the log only.
_BACKEND_ERROR_DETAIL = {
    502: "The inference backend failed to produce a completion.",
    503: "The inference backend is overloaded. Retry later.",
    504: "The inference backend did not respond in time.",
}


@dataclass(frozen=True, slots=True)
class Generation:
    """What a request asks the backend for, independent of its endpoint."""

    prompt: Prompt
    max_tokens: int
    temperature: float
    top_p: float
    stop: list[str] | None

    @classmethod
    def for_completion(cls, request: CompletionRequest) -> "Generation":
        return cls(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

    @classmethod
    def for_chat(cls, request: ChatCompletionRequest) -> "Generation":
        return cls(
            prompt=request.conversation(),
            max_tokens=request.completion_limit(),
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences(),
        )


@dataclass(frozen=True, slots=True)
class Finished:
    result: GenerationResult
    duration: float
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class Failure:
    """A failed generation as the client sees it."""

    # Metric status: "rejected", "error" or "timeout".
    outcome: str
    status_code: int
    detail: str
    # Set for 503s whose upstream suggested a delay.
    retry_after: str | None = None


def timeout_detail(timeout_seconds: float) -> str:
    return (
        f"Inference exceeded {timeout_seconds}s timeout. "
        "Reduce max_tokens or retry when the service has scaled out."
    )


def classify_failure(exc: Exception) -> Failure:
    """Maps a backend failure to its metric outcome and client response, and logs it."""
    if isinstance(exc, RequestRejectedError):
        # A client error, not a service failure: counted separately so it
        # does not inflate the error-ratio SLO.
        details = (
            {"tokens_prompt": exc.prompt_tokens, "n_ctx": exc.n_ctx}
            if isinstance(exc, PromptTooLongError)
            else {}
        )
        logger.info("Request rejected.", extra={"reason": str(exc), **details})
        return Failure("rejected", 400, str(exc))

    if isinstance(exc, BackendError):
        # Upstream timeouts share the "timeout" status with local deadline
        # overruns so the latency SLO sees both.
        outcome = "timeout" if exc.status_code == 504 else "error"
        logger.error(
            "Inference backend failed.",
            extra={"reason": str(exc), "status_code": exc.status_code},
        )
        return Failure(
            outcome,
            exc.status_code,
            _BACKEND_ERROR_DETAIL.get(exc.status_code, _BACKEND_ERROR_DETAIL[502]),
            retry_after=exc.retry_after if exc.status_code == 503 else None,
        )

    # Full traceback goes to the server log only. Exception messages from
    # llama.cpp and the OS routinely include file paths and allocator state,
    # which should not be returned to API clients.
    logger.error("Inference failed.", exc_info=exc)
    return Failure("error", 500, "Inference failed due to an internal error.")


def _usage(result: GenerationResult) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=result["tokens_prompt"],
        completion_tokens=result["tokens_completion"],
        total_tokens=result["tokens_prompt"] + result["tokens_completion"],
    )


def _performance(done: Finished) -> CompletionPerformance:
    return CompletionPerformance(
        tokens_per_second=round(done.tokens_per_second, 3),
        duration_seconds=round(done.duration, 3),
    )


def completion_response(
    completion_id: str, model: str, done: Finished
) -> CompletionResponse:
    return CompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[
            CompletionChoice(
                index=0,
                text=done.result["text"],
                finish_reason=done.result["finish_reason"],
            )
        ],
        usage=_usage(done.result),
        performance=_performance(done),
    )


def chat_completion_response(
    completion_id: str, model: str, done: Finished
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(content=done.result["text"]),
                finish_reason=done.result["finish_reason"],
            )
        ],
        usage=_usage(done.result),
        performance=_performance(done),
    )
