"""Inference backends and the factory that selects one from settings."""

from app.backends.base import (
    BackendError,
    GenerationResult,
    InferenceBackend,
    PromptTooLongError,
    RequestRejectedError,
)
from app.config import Settings

__all__ = [
    "BackendError",
    "GenerationResult",
    "InferenceBackend",
    "PromptTooLongError",
    "RequestRejectedError",
    "create_backend",
]


def create_backend(settings: Settings) -> InferenceBackend:
    """
    Builds the backend named by settings.backend.

    Implementations are imported here rather than at module level so a
    deployment only needs the dependencies of the backend it runs: importing
    the llama.cpp backend loads the native library, which an image serving
    a remote API does not need.
    """
    if settings.backend == "llama_cpp":
        from app.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(settings)

    if settings.backend == "openai_compatible":
        from app.backends.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(settings)

    # Unreachable while Settings validates the name; kept so that adding a
    # value to the Literal without a branch here fails loudly at startup.
    raise ValueError(f"Unsupported inference backend: {settings.backend!r}")
