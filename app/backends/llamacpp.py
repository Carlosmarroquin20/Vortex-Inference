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
from dataclasses import dataclass

from llama_cpp import Llama

from app.backends.base import (
    ChatMessage,
    GenerationResult,
    Prompt,
    PromptTooLongError,
    RequestRejectedError,
    StreamEnd,
    StreamEvent,
    TextDelta,
)
from app.config import Settings
from app.metrics import MODEL_WARMUP_DURATION_SECONDS

logger = logging.getLogger("vortex.backends.llamacpp")

StoppingCriteria = Callable[[list[int], list[float]], bool]


@dataclass(frozen=True, slots=True)
class _Prepared:
    """A prompt in the form llama-cpp-python is called with."""

    # Text for a completion; token IDs for a conversation, which is rendered
    # and tokenized here so its length can be checked before generation.
    model_input: str | list[int]
    prompt_tokens: int
    stop: list[str]
    stopping_criteria: StoppingCriteria | None


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
            # Prompt processing gets the same threads as generation. Left
            # unset, llama-cpp-python uses os.cpu_count(), which inside a
            # container is the node's CPU count rather than the pod's limit:
            # 12 threads on a 2-CPU quota, throttled by the kernel. Measured
            # with Qwen2.5-0.5B under --cpus 2: prompt processing went from
            # 11.5 to 115 tokens/s (scripts/bench_llama.py), streamed time to
            # first token from 1.18s to 0.16s at the median, and the first
            # request after startup from 35s to 1.5s; generation unchanged.
            n_threads_batch=settings.n_threads,
            n_gpu_layers=settings.n_gpu_layers,
            verbose=False,
        )
        self._chat_formatter = self._load_chat_formatter()
        if settings.warmup:
            self._warm_up()

        self._ready = True
        logger.info("Model successfully loaded into memory and ready for inference.")

    def _warm_up(self) -> None:
        """
        Generates one token before the backend reports ready.

        The weights are memory-mapped, so loading only maps the file: on a
        node whose page cache does not hold it, each page is read from disk
        the first time inference touches it, and the first request would pay
        for reading the whole model. llama.cpp's own tools run the same
        warm-up for that reason. One token runs every layer, which reads every
        weight; only the rows of the token embedding table that the warm-up
        prompt does not use stay unread, a small fraction of the file. With
        the file already cached it costs about 40ms (Qwen2.5-0.5B, 2 CPUs).
        """
        started = time.monotonic()
        self._model("Hello", max_tokens=1, temperature=0.0)
        # The warm-up prompt stays in llama-cpp-python's prefix cache
        # otherwise; requests should start from an empty context.
        self._model.reset()
        duration = time.monotonic() - started
        MODEL_WARMUP_DURATION_SECONDS.set(duration)
        logger.info("Model warmed up.", extra={"duration_seconds": round(duration, 2)})

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
        prompt: Prompt,
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
        prepared = self._prepare(prompt, stop, stop_event)

        self._busy_since = time.monotonic()
        try:
            output = self._model(
                prepared.model_input,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=prepared.stop,
                echo=False,
                stopping_criteria=prepared.stopping_criteria,
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
        prompt: Prompt,
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

        The library reports no usage when streaming, so completion tokens are
        counted as the chunks it returns: one per token, except that the
        tokens of a multi-byte character split across several tokens arrive
        as one chunk, and tokens of a matched stop sequence are not returned.
        Measured against a non-streamed call on the same prompt this matched
        exactly, where re-tokenizing the output text was off by one.
        """
        prepared = self._prepare(prompt, stop, stop_event)

        completion_tokens = 0
        finish_reason: str | None = None
        self._busy_since = time.monotonic()
        try:
            chunks = self._model(
                prepared.model_input,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=prepared.stop,
                echo=False,
                stopping_criteria=prepared.stopping_criteria,
                stream=True,
            )
            try:
                for chunk in chunks:
                    choice = chunk["choices"][0]
                    if choice["finish_reason"] is not None:
                        # The closing chunk carries no token.
                        finish_reason = choice["finish_reason"]
                        continue
                    completion_tokens += 1
                    # A token can decode to nothing on its own; it still
                    # counts, but there is nothing to forward.
                    if choice["text"]:
                        yield TextDelta(choice["text"])
            finally:
                # Also reached when the consumer closes this generator early:
                # closing the library's generator stops sampling at once.
                chunks.close()
        finally:
            # Cleared on every exit so the stall detection never sees a call
            # that is no longer running.
            self._busy_since = None

        yield StreamEnd(finish_reason, prepared.prompt_tokens, completion_tokens)

    def _prepare(
        self,
        prompt: Prompt,
        stop: list[str] | None,
        stop_event: threading.Event | None,
    ) -> _Prepared:
        if isinstance(prompt, str):
            prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))
            self._check_length(prompt_tokens)
            return _Prepared(
                model_input=prompt,
                prompt_tokens=prompt_tokens,
                stop=list(stop or []),
                stopping_criteria=self._stopping_criteria(stop_event, None),
            )
        return self._prepare_chat(prompt, stop, stop_event)

    def _prepare_chat(
        self,
        messages: list[ChatMessage],
        stop: list[str] | None,
        stop_event: threading.Event | None,
    ) -> _Prepared:
        """
        Renders a conversation with the model's own chat template.

        This is what llama-cpp-python's create_chat_completion does
        internally, done here instead so the result goes through the same
        completion call as a text prompt: create_chat_completion accepts no
        stopping_criteria, so a timed-out chat request could not be stopped
        until it reached max_tokens, and the prompt length could not be
        checked before generation. Tokenization flags and the template's stop
        conditions follow the library's handler.
        """
        if self._chat_formatter is None:
            raise RequestRejectedError(
                "The loaded model has no chat template. Use /v1/completions "
                "with a prompt in the model's own format."
            )
        try:
            formatted = self._chat_formatter(messages=messages)
        except Exception as exc:
            # Templates enforce their own rules with raise_exception(), for
            # example strict user/assistant alternation; rendering is a pure
            # function of the messages, so a failure is the conversation's.
            raise RequestRejectedError(
                f"The conversation does not fit the model's chat template: {exc}"
            ) from exc

        tokens = self._model.tokenize(
            formatted.prompt.encode("utf-8"),
            add_bos=not formatted.added_special,
            special=True,
        )
        self._check_length(len(tokens))

        template_stop = formatted.stop or []
        if isinstance(template_stop, str):
            template_stop = [template_stop]
        return _Prepared(
            model_input=tokens,
            prompt_tokens=len(tokens),
            stop=[*(stop or []), *template_stop],
            stopping_criteria=self._stopping_criteria(
                stop_event, formatted.stopping_criteria
            ),
        )

    def _load_chat_formatter(self):
        """
        Builds a formatter from the template embedded in the GGUF file, the
        same template the model was trained with and Hugging Face applies.
        """
        template = self._model.metadata.get("tokenizer.chat_template")
        if not template:
            logger.warning(
                "The model has no chat template in its metadata; "
                "/v1/chat/completions will reject requests."
            )
            return None

        # Imported here so environments that fake llama_cpp for tests only
        # need the submodule when a template is present.
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        eos_id = self._model.token_eos()
        bos_id = self._model.token_bos()
        logger.info("Chat template loaded from the model's metadata.")
        return Jinja2ChatFormatter(
            template=template,
            eos_token=self._token_text(eos_id),
            bos_token=self._token_text(bos_id),
            stop_token_ids=[eos_id],
        )

    def _token_text(self, token_id: int) -> str:
        if token_id == -1:
            return ""
        return self._model.detokenize([token_id], special=True).decode(
            "utf-8", errors="ignore"
        )

    def _check_length(self, prompt_tokens: int) -> None:
        """
        Raises if the prompt alone fills the context window.

        Checked explicitly so the API can answer 400 with a precise message.
        llama-cpp-python raises a bare ValueError for this case, which is
        indistinguishable from internal failures. Tokenizing an extra time is
        negligible next to generation.
        """
        if prompt_tokens >= self._n_ctx:
            raise PromptTooLongError(prompt_tokens, self._n_ctx)

    @staticmethod
    def _stopping_criteria(
        stop_event: threading.Event | None,
        template_criteria: StoppingCriteria | None,
    ) -> StoppingCriteria | None:
        if stop_event is None and template_criteria is None:
            return None

        def stopping_criteria(tokens: list[int], scores: list[float]) -> bool:
            if stop_event is not None and stop_event.is_set():
                return True
            return template_criteria is not None and bool(
                template_criteria(tokens, scores)
            )

        return stopping_criteria
