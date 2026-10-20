"""
In-process stand-in for llama_cpp.Llama.

Implements only the surface LlamaCppBackend uses (construction, tokenize, and the
completion call with stopping_criteria, streamed or not), plus instrumentation
the tests rely on: a record of calls, a high-water mark of concurrent calls to
detect thread-safety violations, and knobs to make generation slow or failing.
"""

import logging
import threading
import time

logger = logging.getLogger("tests.fake_llama")


class FakeLlama:
    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs

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
        return list(range(len(text.split())))

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
                    "prompt_tokens": len(self.tokenize(prompt.encode())),
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
