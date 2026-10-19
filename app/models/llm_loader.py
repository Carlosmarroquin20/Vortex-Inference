"""
LLM lifecycle wrapper for llama-cpp-python.

Isolation rationale: separating model I/O from the HTTP layer allows the
inference logic to be tested independently and swapped (e.g., switching from
llama-cpp-python to ctransformers or a remote vLLM endpoint) without touching
the API surface or the metrics instrumentation. This is the adapter pattern
applied to the ML inference layer.
"""

import logging
import threading
import time

from llama_cpp import Llama

from app.config import Settings

logger = logging.getLogger("vortex.llm_loader")


class PromptTooLongError(ValueError):
    """The prompt alone does not fit in the model's context window."""

    def __init__(self, prompt_tokens: int, n_ctx: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.n_ctx = n_ctx
        super().__init__(
            f"Prompt is {prompt_tokens} tokens; the context window is {n_ctx} "
            "tokens and must also leave room for the completion."
        )


class LLMLoader:
    """
    Stateful wrapper around a llama.cpp model instance.

    Thread-safety: llama-cpp-python's Llama object is NOT thread-safe.
    Concurrent access is prevented at the application layer via asyncio.Semaphore
    in main.py, not here. This class makes no synchronization guarantees.
    """

    def __init__(self, settings: Settings) -> None:
        self._ready = False
        # Monotonic timestamp of the call currently inside llama.cpp, or None
        # when idle. Written by the worker thread and read by the event loop
        # for liveness; a single attribute store/load is atomic in CPython,
        # so no lock is needed for this advisory value.
        self._busy_since: float | None = None
        self._n_ctx = settings.n_ctx

        logger.info(
            "Loading GGUF model from disk.",
            extra={"model_path": settings.model_path},
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
        True once the model has been loaded into memory.

        The loader holds a strong reference to the Llama object for the life
        of the process, so this cannot regress after startup; a lost model
        means a dead process, which the kubelet detects on its own.
        """
        return self._ready

    def busy_seconds(self) -> float:
        """Seconds the current generate() call has spent in llama.cpp, or 0."""
        started = self._busy_since
        return time.monotonic() - started if started is not None else 0.0

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        stop_event: threading.Event | None = None,
    ) -> dict:
        """
        Synchronous inference call. Designed to be run in a ThreadPoolExecutor
        from the async event loop in main.py to avoid blocking the I/O thread.

        echo=False is critical: it prevents the model from including the prompt
        tokens in the output string, which would corrupt downstream parsing
        and inflate completion token counts.

        stop_event: when set by the async layer (e.g. on client timeout), the
        stopping_criteria callback tells llama.cpp to halt at the next token
        boundary. This lets the thread exit cleanly so the semaphore in main.py
        can be released without racing against an in-progress C++ call.
        """
        # Checked explicitly so the API can answer 400 with a precise message.
        # llama-cpp-python raises a bare ValueError for this case, which is
        # indistinguishable from internal failures. Tokenizing an extra time
        # is negligible next to generation.
        prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))
        if prompt_tokens >= self._n_ctx:
            raise PromptTooLongError(prompt_tokens, self._n_ctx)

        stopping_criteria = None
        if stop_event is not None:

            def stopping_criteria(tokens: list[int], scores: list[float]) -> bool:
                return stop_event.is_set()

        self._busy_since = time.monotonic()
        try:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False,
                stopping_criteria=stopping_criteria,
            )
        finally:
            self._busy_since = None

        # The text is returned verbatim. Completion clients concatenate prompt
        # and output, so trimming would drop meaningful leading whitespace
        # (for example the space between the last prompt word and the next).
        return {
            "text": output["choices"][0]["text"],
            "finish_reason": output["choices"][0]["finish_reason"],
            "tokens_prompt": output["usage"]["prompt_tokens"],
            "tokens_completion": output["usage"]["completion_tokens"],
        }
