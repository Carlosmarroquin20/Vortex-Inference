"""
LLM lifecycle wrapper for llama-cpp-python.

Isolation rationale: separating model I/O from the HTTP layer allows the
inference logic to be tested independently and swapped (e.g., switching from
llama-cpp-python to ctransformers or a remote vLLM endpoint) without touching
the API surface or the metrics instrumentation. This is the adapter pattern
applied to the ML inference layer.
"""

import logging

from llama_cpp import Llama

from app.config import Settings

logger = logging.getLogger("vortex.llm_loader")


class LLMLoader:
    """
    Stateful wrapper around a llama.cpp model instance.

    Thread-safety: llama-cpp-python's Llama object is NOT thread-safe.
    Concurrent access is prevented at the application layer via asyncio.Semaphore
    in main.py, not here. This class makes no synchronization guarantees.
    """

    def __init__(self, settings: Settings) -> None:
        self._ready = False

        logger.info(
            f"Loading GGUF model from disk. "
            f"path={settings.model_path} "
            f"n_ctx={settings.n_ctx} "
            f"n_threads={settings.n_threads} "
            f"n_gpu_layers={settings.n_gpu_layers}"
        )

        # verbose=False suppresses llama.cpp's per-token C++ stderr output.
        # In production, that output floods container logs and drowns out
        # structured application-level log lines. The model's behavior is
        # observable through Prometheus metrics, not stderr.
        self._model = Llama(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_gpu_layers=settings.n_gpu_layers,
            verbose=False,
        )

        self._ready = True
        logger.info("Model successfully loaded into memory and ready for inference.")

    def is_ready(self) -> bool:
        """
        Readiness check called by the /health endpoint.

        Returns False if the model is uninitialized or if the internal Llama
        object was garbage-collected — a scenario that indicates severe memory
        pressure on the node (e.g., kubelet OOM eviction of adjacent containers).
        """
        return self._ready and self._model is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> dict:
        """
        Synchronous inference call. Designed to be run in a ThreadPoolExecutor
        from the async event loop in main.py to avoid blocking the I/O thread.

        echo=False is critical: it prevents the model from including the prompt
        tokens in the output string, which would corrupt downstream parsing
        and inflate completion token counts.
        """
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )

        return {
            "text": output["choices"][0]["text"].strip(),
            "finish_reason": output["choices"][0]["finish_reason"],
            "tokens_prompt": output["usage"]["prompt_tokens"],
            "tokens_completion": output["usage"]["completion_tokens"],
        }
