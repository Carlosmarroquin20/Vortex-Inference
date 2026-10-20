"""
In-process stand-in for llama_cpp.Llama.

Implements only the surface LlamaCppBackend uses (construction, metadata,
tokenize/detokenize, special token IDs, and the completion call with
stopping_criteria, streamed or not), plus instrumentation the tests rely on: a
record of calls, a high-water mark of concurrent calls to detect thread-safety
violations, and knobs to make generation slow or failing.

FakeChatFormatter stands in for llama_cpp.llama_chat_format.Jinja2ChatFormatter.
"""

import logging
import threading
import time
from types import SimpleNamespace

logger = logging.getLogger("tests.fake_llama")

BOS_ID, EOS_ID = 1, 2


class FakeLlama:
    # Set before the backend is built to give the model a chat template.
    chat_template: str | None = None

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.metadata = (
            {"tokenizer.chat_template": FakeLlama.chat_template}
            if FakeLlama.chat_template
            else {}
        )

        # Behaviour knobs, adjusted per test.
        self.text = " fake completion"
        self.finish_reason = "stop"
        self.completion_tokens = 3
        self.steps = 1  # token iterations per call
        self.step_delay = 0.0  # seconds per iteration
        self.error: Exception | None = None
        # Stream mode: the chunks to emit, one per step, and an error raised
        # after the first chunk to simulate a failure mid-stream.
        self.stream_pieces: list[str] | None = None
        self.stream_error: Exception | None = None

        # Instrumentation.
        self.calls: list[dict[str, object]] = []
        self.steps_run: list[int] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False):
        # One token per whitespace-separated word keeps context-window
        # arithmetic in tests easy to reason about.
        self.last_tokenize = {"text": text, "add_bos": add_bos, "special": special}
        return list(range(len(text.split())))

    def detokenize(self, tokens, prev_tokens=None, special: bool = False) -> bytes:
        names = {BOS_ID: b"<s>", EOS_ID: b"</s>"}
        return b"".join(names.get(token, b"?") for token in tokens)

    def token_bos(self) -> int:
        return BOS_ID

    def token_eos(self) -> int:
        return EOS_ID

    def _prompt_tokens(self, prompt) -> int:
        # A conversation arrives already tokenized; text is tokenized here,
        # as llama-cpp-python does.
        if isinstance(prompt, list):
            return len(prompt)
        return len(self.tokenize(prompt.encode()))

    def __call__(
        self, prompt: str, stopping_criteria=None, stream: bool = False, **kwargs
    ):
        if stream:
            return self._stream(prompt, stopping_criteria, kwargs)
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.calls.append({"prompt": prompt, **kwargs})
        self.started.set()
        logger.info("Fake generation started.")

        try:
            if self.error is not None:
                raise self.error

            completed = 0
            while completed < self.steps:
                completed += 1
                if self.step_delay:
                    time.sleep(self.step_delay)
                if stopping_criteria is not None and stopping_criteria([], []):
                    break
            self.steps_run.append(completed)

            return {
                "choices": [{"text": self.text, "finish_reason": self.finish_reason}],
                "usage": {
                    "prompt_tokens": self._prompt_tokens(prompt),
                    "completion_tokens": self.completion_tokens,
                },
            }
        finally:
            with self._lock:
                self.active_calls -= 1

    def _stream(self, prompt: str, stopping_criteria, kwargs: dict):
        # A generator, like llama-cpp-python's stream mode: nothing runs, and
        # the call does not count as active, until the first chunk is read.
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.calls.append({"prompt": prompt, "stream": True, **kwargs})
        self.started.set()

        try:
            if self.error is not None:
                raise self.error

            pieces = (
                self.stream_pieces if self.stream_pieces is not None else [self.text]
            )
            finish_reason = self.finish_reason
            completed = 0
            for piece in pieces:
                if self.step_delay:
                    time.sleep(self.step_delay)
                if stopping_criteria is not None and stopping_criteria([], []):
                    finish_reason = "stop"
                    break
                completed += 1
                yield {"choices": [{"text": piece, "finish_reason": None}]}
                if self.stream_error is not None:
                    raise self.stream_error
            self.steps_run.append(completed)
            yield {"choices": [{"text": "", "finish_reason": finish_reason}]}
        finally:
            with self._lock:
                self.active_calls -= 1


class FakeChatFormatter:
    """
    Renders "<|role|> content" per message plus an assistant cue, and records
    how it was built and called. render_error makes rendering fail the way a
    template's raise_exception() does.
    """

    instances: list["FakeChatFormatter"] = []
    render_error: Exception | None = None

    def __init__(self, template, eos_token, bos_token, stop_token_ids=None) -> None:
        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.stop_token_ids = stop_token_ids
        self.rendered: list[list[dict]] = []
        self.criteria_calls = 0
        FakeChatFormatter.instances.append(self)

    def __call__(self, *, messages, **kwargs):
        if FakeChatFormatter.render_error is not None:
            raise FakeChatFormatter.render_error
        self.rendered.append(messages)
        prompt = "".join(f"<|{m['role']}|> {m['content']} " for m in messages)

        def stopping_criteria(tokens, scores):
            self.criteria_calls += 1
            return False

        return SimpleNamespace(
            prompt=prompt + "<|assistant|>",
            stop=[self.eos_token],
            stopping_criteria=stopping_criteria,
            added_special=True,
        )
