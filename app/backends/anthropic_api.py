"""
Completions from Claude models through Anthropic's Messages API.

Uses the official SDK rather than raw HTTP: it tracks the API's headers,
beta flags and typed errors as they change. The SDK is built on httpx2, the
same client the OpenAI-compatible backend uses.
"""

import logging
import threading

import anthropic
import httpx2
from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from app.backends.base import BackendError, GenerationResult
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
        prompt: str,
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

        params: dict[str, object] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stop:
            params["stop_sequences"] = stop
        if self._effort is not None:
            params["output_config"] = {"effort": self._effort}

        try:
            with self._inflight.track():
                if self._refusal_fallback:
                    message = self._client.beta.messages.create(
                        betas=[_FALLBACK_BETA], fallbacks="default", **params
                    )
                else:
                    message = self._client.messages.create(**params)
        except anthropic.APITimeoutError as exc:
            raise BackendError(
                f"Anthropic API did not respond in time: {exc!r}", status_code=504
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError(f"Anthropic API unreachable: {exc!r}") from exc
        except anthropic.APIStatusError as exc:
            # request_id identifies the call in Anthropic's own logs; it is
            # what support asks for when a failure needs investigating.
            message_text = f"{_error_message(exc)} (request_id={exc.request_id})"
            raise status_error(
                exc.status_code,
                message_text,
                retry_after=exc.response.headers.get("retry-after"),
            ) from exc

        return self._to_result(message)

    def _to_result(self, message: Message | BetaMessage) -> GenerationResult:
        # Only text blocks are returned. With thinking enabled the response
        # also carries thinking blocks, which are internal to the model.
        text = "".join(block.text for block in message.content if block.type == "text")
        stop_reason = message.stop_reason
        if stop_reason == "refusal":
            # A refusal is an HTTP 200 and stays one here: the request was
            # valid and the model answered by declining. With the fallback
            # enabled this means the fallback model declined as well.
            logger.info(
                "Model declined the request.",
                extra={"category": getattr(message.stop_details, "category", None)},
            )
        if message.model != self._model:
            # A refusal fallback served the request on another model.
            logger.info(
                "Completion served by a fallback model.",
                extra={"requested": self._model, "served": message.model},
            )

        return GenerationResult(
            text=text,
            finish_reason=_FINISH_REASONS.get(stop_reason, stop_reason),
            tokens_prompt=message.usage.input_tokens,
            tokens_completion=message.usage.output_tokens,
        )
