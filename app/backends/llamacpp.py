"""
Local GGUF inference through llama-cpp-python.

The model runs in-process on the pod's CPUs (or GPU, in an image built with
a GPU backend), which is what the resource limits, thread caps and
single-permit concurrency in the deployment are sized for.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator

from llama_cpp import Llama

from app.backends.base import (
    GenerationResult,
    PromptTooLongError,
    StreamEnd,
    StreamEvent,
    TextDelta,
)
from app.config import Settings

logger = logging.getLogger("vortex.backends.llamacpp")


class LlamaCppBackend:
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
            extra={
                "model_path": settings.model_path,
                "n_ctx": settings.n_ctx,
                "n_threads": settings.n_threads,
                "n_gpu_layers": settings.n_gpu_layers,
            },
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

    def close(self) -> None:
        # Nothing to release ahead of process exit: the model's memory goes
        # with the process, and freeing it explicitly would race any worker
        # thread that outlived the shutdown deadline.
        pass

    def busy_seconds(self) -> float:
        """Seconds the current call has spent in llama.cpp, or 0."""
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
    ) -> GenerationResult:
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
        self._check_prompt(prompt)

        self._busy_since = time.monotonic()
        try:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False,
                stopping_criteria=self._stopping_criteria(stop_event),
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
        Streaming variant of generate(), built on llama-cpp-python's stream
        mode so stop sequences, end-of-generation tokens and max_tokens behave
        exactly as in a non-streamed call.

        The library does not report usage when streaming, and a chunk can
        span several tokens (it groups the tokens of a multi-byte character),
        so the completion count comes from re-tokenizing the returned text.
        It can differ by a token from the count of a non-streamed call, for
        example when a matched stop sequence consumed tokens that were never
        returned.
        """
        prompt_tokens = self._check_prompt(prompt)

        pieces: list[str] = []
        finish_reason: str | None = None
        self._busy_since = time.monotonic()
        try:
            chunks = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False,
                stopping_criteria=self._stopping_criteria(stop_event),
                stream=True,
            )
            try:
                for chunk in chunks:
                    choice = chunk["choices"][0]
                    if choice["text"]:
                        pieces.append(choice["text"])
                        yield TextDelta(choice["text"])
                    if choice["finish_reason"] is not None:
                        finish_reason = choice["finish_reason"]
            finally:
                # Also reached when the consumer closes this generator early:
                # closing the library's generator stops sampling at once.
                chunks.close()
        finally:
            # Cleared on every exit so the stall detection never sees a call
            # that is no longer running.
            self._busy_since = None

        text = "".join(pieces)
        completion_tokens = (
            len(self._model.tokenize(text.encode("utf-8"), add_bos=False))
            if text
            else 0
        )
        yield StreamEnd(finish_reason, prompt_tokens, completion_tokens)

    def _check_prompt(self, prompt: str) -> int:
        """
        Returns the prompt's token count, or raises if it cannot fit.

        Checked explicitly so the API can answer 400 with a precise message.
        llama-cpp-python raises a bare ValueError for this case, which is
        indistinguishable from internal failures. Tokenizing an extra time is
        negligible next to generation.
        """
        prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))
        if prompt_tokens >= self._n_ctx:
            raise PromptTooLongError(prompt_tokens, self._n_ctx)
        return prompt_tokens

    @staticmethod
    def _stopping_criteria(
        stop_event: threading.Event | None,
    ) -> Callable[[list[int], list[float]], bool] | None:
        if stop_event is None:
            return None

        def stopping_criteria(tokens: list[int], scores: list[float]) -> bool:
            return stop_event.is_set()

        return stopping_criteria
