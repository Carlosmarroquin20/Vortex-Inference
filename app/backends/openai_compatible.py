"""
Completions from an HTTP service that implements OpenAI's API.

Covers OpenAI itself, self-hosted servers exposing the same contract (vLLM,
Ollama, LM Studio, llama.cpp's llama-server) and routers such as OpenRouter.
Generation happens upstream, so this backend holds no model in memory and is
I/O-bound: a pod can serve many requests at once by raising
max_concurrent_requests, which the in-process backend cannot do.
"""

import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx2

from app.backends.base import (
    BackendError,
    GenerationResult,
    StreamEnd,
    StreamEvent,
    TextDelta,
)
from app.backends.upstream import MAX_ERROR_MESSAGE, InflightTracker, status_error
from app.config import Settings

logger = logging.getLogger("vortex.backends.openai_compatible")


def _payload_error(payload: object) -> str | None:
    """The error message in a JSON body or stream event, if it carries one."""
    # OpenAI and vLLM: {"error": {"message": ...}}; Ollama: {"error": "..."}.
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return None


def _error_message(response: httpx2.Response) -> str:
    """Best-effort extraction of the human-readable message from an error body."""
    try:
        return _payload_error(response.json()) or response.text
    except ValueError:
        return response.text


@contextmanager
def _transport_errors() -> Iterator[None]:
    """Maps failures below the HTTP layer to BackendError."""
    try:
        yield
    except httpx2.TimeoutException as exc:
        raise BackendError(
            f"Upstream did not respond in time: {exc!r}", status_code=504
        ) from exc
    except httpx2.TransportError as exc:
        # Connection refused, DNS failure, TLS error, reset mid-response.
        raise BackendError(f"Upstream unreachable: {exc!r}") from exc


class OpenAICompatibleBackend:
    """
    Forwards each completion to an OpenAI-compatible endpoint.

    Thread-safe: httpx2.Client may be shared across worker threads, and the
    in-flight bookkeeping behind busy_seconds() is guarded by a lock.
    """

    def __init__(
        self, settings: Settings, transport: httpx2.BaseTransport | None = None
    ) -> None:
        if settings.remote_base_url is None:
            raise ValueError("remote_base_url is required for this backend.")

        self._model = settings.model_name
        self._api_style = settings.remote_api_style
        self._max_tokens_field = settings.remote_max_tokens_field
        self._forward_sampling = settings.remote_forward_sampling

        headers = {}
        if settings.remote_api_key is not None:
            key = settings.remote_api_key.get_secret_value()
            headers["Authorization"] = f"Bearer {key}"

        # One client for the life of the process, so pooled connections skip
        # the TCP and TLS handshakes after the first request.
        #
        # The read timeout equals the request deadline. A non-streamed
        # response arrives in one piece once generation ends, so it bounds the
        # whole call: when the API answers 504, the abandoned worker thread
        # returns at about the same moment and its permit is released. That
        # is also what keeps busy_seconds() below the stall threshold. A
        # stream is bounded per chunk instead, and stops at the next chunk
        # once the HTTP layer gives up on it.
        #
        # Redirects are not followed: an API that redirects a POST is
        # misconfigured, and following it would resend the bearer token to
        # wherever the Location header points.
        self._client = httpx2.Client(
            base_url=settings.remote_base_url,
            headers=headers,
            timeout=httpx2.Timeout(
                settings.inference_timeout_seconds,
                connect=settings.remote_connect_timeout_seconds,
            ),
            follow_redirects=False,
            transport=transport,
        )

        self._inflight = InflightTracker()

        logger.info(
            "Using OpenAI-compatible backend.",
            extra={
                "base_url": settings.remote_base_url,
                "api_style": self._api_style,
                "model": self._model,
                "authenticated": settings.remote_api_key is not None,
            },
        )

    def is_ready(self) -> bool:
        """
        Always True once constructed.

        Startup deliberately makes no upstream call: a provider outage would
        otherwise crash-loop every replica, and restarting pods cannot fix a
        problem on the far side of the network. Failures surface per request
        as 5xx responses, which the error-ratio alerts already watch.
        """
        return True

    def busy_seconds(self) -> float:
        """Age of the oldest request currently waiting on the upstream API."""
        return self._inflight.oldest_age()

    def close(self) -> None:
        self._client.close()

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
        Sends one completion request and maps the reply to GenerationResult.

        stop_event cannot interrupt a request already on the wire; the read
        timeout bounds it instead. It is checked before sending, which covers
        a request abandoned while it waited for a worker thread.
        """
        if stop_event is not None and stop_event.is_set():
            return GenerationResult(
                text="", finish_reason=None, tokens_prompt=0, tokens_completion=0
            )

        path, body = self._request(prompt, max_tokens, temperature, top_p, stop)
        with _transport_errors(), self._inflight.track():
            response = self._client.post(path, json=body)

        if not response.is_success:
            raise status_error(
                response.status_code,
                _error_message(response),
                retry_after=response.headers.get("retry-after"),
            )
        return self._parse(response)

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
        Streams the completion from the upstream's server-sent events.

        Usage is requested with stream_options.include_usage, which OpenAI,
        vLLM, Ollama and llama-server honour; a server that ignores it leaves
        the counts at zero, as for a non-streamed reply without usage. When
        the consumer stops early, leaving the response context closes the
        connection, which is how OpenAI-compatible servers learn to stop
        generating.
        """
        if stop_event is not None and stop_event.is_set():
            yield StreamEnd(None, 0, 0)
            return

        path, body = self._request(prompt, max_tokens, temperature, top_p, stop)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        finish_reason: str | None = None
        usage: dict = {}
        with (
            _transport_errors(),
            self._inflight.track(),
            self._client.stream("POST", path, json=body) as response,
        ):
            if not response.is_success:
                response.read()
                raise status_error(
                    response.status_code,
                    _error_message(response),
                    retry_after=response.headers.get("retry-after"),
                )
            for line in response.iter_lines():
                if stop_event is not None and stop_event.is_set():
                    return
                # Other SSE fields (event:, id:, comments) carry nothing used
                # here; OpenAI-compatible servers put everything in data.
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = self._parse_event(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    text = self._delta_text(choice)
                    if text:
                        yield TextDelta(text)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        yield StreamEnd(
            finish_reason,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )

    def _request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> tuple[str, dict[str, object]]:
        body: dict[str, object] = {
            "model": self._model,
            self._max_tokens_field: max_tokens,
        }
        if self._forward_sampling:
            body["temperature"] = temperature
            body["top_p"] = top_p
        if stop:
            body["stop"] = stop

        if self._api_style == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
            return "/chat/completions", body
        body["prompt"] = prompt
        return "/completions", body

    def _parse(self, response: httpx2.Response) -> GenerationResult:
        try:
            payload = response.json()
            choice = payload["choices"][0]
            if self._api_style == "chat":
                # content is null when the model answers with a refusal or a
                # tool call instead of text; neither is requested here.
                text = choice["message"]["content"] or ""
            else:
                text = choice["text"]
            if not isinstance(text, str):
                raise TypeError(f"completion text is {type(text).__name__}")
            usage = payload.get("usage") or {}
            return GenerationResult(
                text=text,
                finish_reason=choice.get("finish_reason"),
                # Some compatible servers omit usage; zero keeps the token
                # counters monotonic instead of failing a valid completion.
                tokens_prompt=int(usage.get("prompt_tokens") or 0),
                tokens_completion=int(usage.get("completion_tokens") or 0),
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                "Upstream response is not a valid completion: "
                f"{response.text[:MAX_ERROR_MESSAGE]!r}"
            ) from exc

    @staticmethod
    def _parse_event(data: str) -> dict:
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise BackendError(
                f"Upstream stream event is not JSON: {data[:MAX_ERROR_MESSAGE]!r}"
            ) from exc
        if not isinstance(event, dict):
            raise BackendError(
                f"Upstream stream event is not an object: {data[:MAX_ERROR_MESSAGE]!r}"
            )
        # A failure after the stream opened arrives as an event, since the
        # status line has already been sent.
        message = _payload_error(event)
        if message is not None:
            raise BackendError(f"Upstream failed mid-stream: {message}")
        return event

    def _delta_text(self, choice: object) -> str | None:
        if not isinstance(choice, dict):
            raise BackendError(f"Upstream stream choice is malformed: {choice!r}")
        if self._api_style == "chat":
            delta = choice.get("delta") or {}
            text = delta.get("content") if isinstance(delta, dict) else None
        else:
            text = choice.get("text")
        if text is not None and not isinstance(text, str):
            raise BackendError(f"Upstream stream text is {type(text).__name__}")
        return text
