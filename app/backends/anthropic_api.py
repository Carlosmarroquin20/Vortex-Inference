"""
Completions from Claude models through Anthropic's Messages API.

Uses the official SDK rather than raw HTTP: it tracks the API's headers,
beta flags and typed errors as they change. The SDK is built on httpx2, the
same client the OpenAI-compatible backend uses.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import anthropic
import httpx2

from app.backends.base import (
    BackendError,
    GenerationResult,
    Prompt,
    StreamEnd,
    StreamEvent,
    TextDelta,
)
from app.backends.upstream import InflightTracker, status_error
from app.config import Settings

logger = logging.getLogger("vortex.backends.anthropic")

DEFAULT_BASE_URL = "https://api.anthropic.com"

# Beta that enables fallbacks="default" (server-side refusal fallback).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Messages API stop reasons mapped to OpenAI finish_reason values, which is
# what clients of /v1/completions understand. A refusal maps to
# content_filter, OpenAI's value for output withheld by a safety system.
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}

# HTTP status for each API error type, for errors that arrive as an event in
# an already-open stream and therefore carry no status of their own.
_ERROR_TYPE_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def _error_message(exc: anthropic.APIStatusError) -> str:
    # Error bodies are {"type": "error", "error": {"type": ..., "message": ...}}.
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return exc.message


class AnthropicBackend:
    """
    Sends each completion as a single-turn Messages API request.

    Thread-safe: the SDK client may be shared across worker threads, and the
    in-flight bookkeeping behind busy_seconds() is guarded by a lock.
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx2.Client | None = None,
    ) -> None:
        if settings.remote_api_key is None:
            raise ValueError("remote_api_key is required for this backend.")

        self._model = settings.model_name
        self._effort = settings.anthropic_effort
        self._refusal_fallback = settings.anthropic_refusal_fallback
        base_url = settings.remote_base_url or DEFAULT_BASE_URL

        # Everything the SDK would otherwise resolve on its own is passed
        # explicitly. Left to itself it reads ANTHROPIC_BASE_URL and a chain
        # of credential sources (environment variables, CLI profiles), so the
        # same configuration could reach a different endpoint or account
        # depending on the machine it runs on.
        #
        # max_retries=0: the SDK retries 429, 5xx and connection errors by
        # default, and each attempt gets the full timeout. Retrying inside
        # the request deadline would hold the inference permit past the point
        # where the API has already answered 504, and could push
        # busy_seconds() over the stall threshold that restarts the pod.
        # Clients retry instead, guided by Retry-After on 503.
        #
        # Redirects are not followed for the same reason as in the
        # OpenAI-compatible backend: they would resend the API key.
        self._client = anthropic.Anthropic(
            api_key=settings.remote_api_key.get_secret_value(),
            base_url=base_url,
            timeout=anthropic.Timeout(
                settings.inference_timeout_seconds,
                connect=settings.remote_connect_timeout_seconds,
            ),
            max_retries=0,
            http_client=http_client
            or anthropic.DefaultHttpxClient(follow_redirects=False),
        )
        self._inflight = InflightTracker()

        logger.info(
            "Using Anthropic backend.",
            extra={
                "base_url": base_url,
                "model": self._model,
                "effort": self._effort or "model default",
                "refusal_fallback": self._refusal_fallback,
            },
        )

    def is_ready(self) -> bool:
        # No upstream call at startup; see OpenAICompatibleBackend.is_ready().
        return True

    def busy_seconds(self) -> float:
        """Age of the oldest request currently waiting on the API."""
        return self._inflight.oldest_age()

    def close(self) -> None:
        self._client.close()

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
        Sends one Messages API request and maps the reply to GenerationResult.

        temperature and top_p are accepted for interface compatibility and
        not sent: current Claude models reject sampling parameters, and the
        SDK no longer exposes them. As in the OpenAI-compatible backend,
        stop_event is only checked before sending; the read timeout bounds a
        request already in flight.
        """
        if stop_event is not None and stop_event.is_set():
            return GenerationResult(
                text="", finish_reason=None, tokens_prompt=0, tokens_completion=0
            )

        with _api_errors(), self._inflight.track():
            message = self._create(self._params(prompt, max_tokens, stop))

        # Only text blocks are returned. With thinking enabled the response
        # also carries thinking blocks, which are internal to the model.
        text = "".join(block.text for block in message.content if block.type == "text")
        self._log_outcome(message.stop_reason, message.stop_details, message.model)
        return GenerationResult(
            text=text,
            finish_reason=_FINISH_REASONS.get(message.stop_reason, message.stop_reason),
            tokens_prompt=message.usage.input_tokens,
            tokens_completion=message.usage.output_tokens,
        )

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
        Streams the completion from the Messages API's server-sent events.

        Only text deltas are forwarded; thinking and the fallback marker block
        are internal. A refusal fallback continues on the same stream, so text
        streamed before a mid-stream decline stays valid and the fallback
        model's text follows it. If the decline is final the stream ends with
        finish_reason content_filter after the partial text, which the client
        should then discard; it cannot be withdrawn once sent.
        """
        if stop_event is not None and stop_event.is_set():
            yield StreamEnd(None, 0, 0)
            return

        tokens_prompt = tokens_completion = 0
        stop_reason = stop_details = served_by = None
        with (
            _api_errors(),
            self._inflight.track(),
            self._create(self._params(prompt, max_tokens, stop), stream=True) as events,
        ):
            for event in events:
                if stop_event is not None and stop_event.is_set():
                    # Leaving the context closes the connection, which ends
                    # generation (and billing) upstream.
                    return
                if event.type == "message_start":
                    served_by = event.message.model
                    tokens_prompt = event.message.usage.input_tokens
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta" and event.delta.text:
                        yield TextDelta(event.delta.text)
                elif event.type == "message_delta":
                    stop_reason = event.delta.stop_reason or stop_reason
                    stop_details = event.delta.stop_details or stop_details
                    # Cumulative counts. After a mid-stream fallback the final
                    # message_delta also restates the input count.
                    tokens_completion = event.usage.output_tokens
                    if event.usage.input_tokens:
                        tokens_prompt = event.usage.input_tokens

        self._log_outcome(stop_reason, stop_details, served_by)
        yield StreamEnd(
            _FINISH_REASONS.get(stop_reason, stop_reason),
            tokens_prompt,
            tokens_completion,
        )

    def _params(
        self, prompt: Prompt, max_tokens: int, stop: list[str] | None
    ) -> dict[str, object]:
        params: dict[str, object] = {"model": self._model, "max_tokens": max_tokens}
        if isinstance(prompt, str):
            params["messages"] = [{"role": "user", "content": prompt}]
        else:
            # The Messages API takes instructions in a top-level system field
            # rather than as a role. System messages are joined in order;
            # their position relative to the turns is not preserved. Turn
            # order rules (starting with a user turn, no assistant prefill on
            # current models) are the API's to enforce: a violation comes
            # back as a 400 the client can act on.
            system = [m["content"] for m in prompt if m["role"] == "system"]
            if system:
                params["system"] = "\n\n".join(system)
            params["messages"] = [
                {"role": m["role"], "content": m["content"]}
                for m in prompt
                if m["role"] != "system"
            ]
        if stop:
            params["stop_sequences"] = stop
        if self._effort is not None:
            params["output_config"] = {"effort": self._effort}
        return params

    def _create(self, params: dict[str, object], **options: object):
        if self._refusal_fallback:
            return self._client.beta.messages.create(
                betas=[_FALLBACK_BETA], fallbacks="default", **params, **options
            )
        return self._client.messages.create(**params, **options)

    def _log_outcome(
        self, stop_reason: str | None, stop_details: object, served_by: str | None
    ) -> None:
        if stop_reason == "refusal":
            # A refusal is an HTTP 200 and stays one here: the request was
            # valid and the model answered by declining. With the fallback
            # enabled this means the fallback model declined as well.
            logger.info(
                "Model declined the request.",
                extra={"category": getattr(stop_details, "category", None)},
            )
        if served_by is not None and served_by != self._model:
            # A refusal fallback served the request on another model.
            logger.info(
                "Completion served by a fallback model.",
                extra={"requested": self._model, "served": served_by},
            )


@contextmanager
def _api_errors() -> Iterator[None]:
    """Maps SDK exceptions to the backend error types."""
    try:
        yield
    except anthropic.APITimeoutError as exc:
        raise BackendError(
            f"Anthropic API did not respond in time: {exc!r}", status_code=504
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise BackendError(f"Anthropic API unreachable: {exc!r}") from exc
    except anthropic.APIStatusError as exc:
        # An error event inside a stream is raised with the stream's own
        # status, 200, so the status is recovered from the error type.
        status = exc.status_code
        if status < 400:
            status = _ERROR_TYPE_STATUS.get(exc.type or "", 500)
        # request_id identifies the call in Anthropic's own logs; it is what
        # support asks for when a failure needs investigating.
        raise status_error(
            status,
            f"{_error_message(exc)} (request_id={exc.request_id})",
            retry_after=exc.response.headers.get("retry-after"),
        ) from exc
