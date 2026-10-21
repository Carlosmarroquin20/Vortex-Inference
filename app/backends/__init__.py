"""Inference backends and the factory that selects one from settings."""

from app.backends.base import (
    BackendError,
    ChatMessage,
    GenerationResult,
    InferenceBackend,
    Prompt,
    PromptTooLongError,
    RequestRejectedError,
    StreamEnd,
    StreamEvent,
    TextDelta,
)
from app.config import Settings

__all__ = [
    "BackendError",
    "ChatMessage",
    "GenerationResult",
    "InferenceBackend",
    "Prompt",
    "PromptTooLongError",
    "RequestRejectedError",
    "StreamEnd",
    "StreamEvent",
    "TextDelta",
    "create_backend",
]


def create_backend(settings: Settings) -> InferenceBackend:
    """
    Builds the backend named by settings.backend, wrapped with the fallback
    backend when one is configured (see app/backends/fallback.py).

    Implementations are imported here rather than at module level so a
    deployment only needs the dependencies of the backend it runs: importing
    the llama.cpp backend loads the native library, which an image serving
    a remote API does not need.
    """
    primary = _create(settings)
    fallback_settings = settings.fallback_settings()
    if fallback_settings is None:
        return primary

    from app.backends.fallback import FallbackBackend

    return FallbackBackend(
        primary,
        _create(fallback_settings),
        fallback_model=fallback_settings.model_name,
        failure_threshold=settings.fallback_failure_threshold,
        cooldown_seconds=settings.fallback_cooldown_seconds,
    )


def _create(settings: Settings) -> InferenceBackend:
    if settings.backend == "llama_cpp":
        from app.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(settings)

    if settings.backend == "openai_compatible":
        from app.backends.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(settings)

    if settings.backend == "anthropic":
        from app.backends.anthropic_api import AnthropicBackend

        return AnthropicBackend(settings)

    # Unreachable while Settings validates the name; kept so that adding a
    # value to the Literal without a branch here fails loudly at startup.
    raise ValueError(f"Unsupported inference backend: {settings.backend!r}")
