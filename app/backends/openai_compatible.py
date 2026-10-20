"""
Completions from an HTTP service that implements OpenAI's API.

Covers OpenAI itself, self-hosted servers exposing the same contract (vLLM,
Ollama, LM Studio, llama.cpp's llama-server) and routers such as OpenRouter.
Generation happens upstream, so this backend holds no model in memory and is
I/O-bound: a pod can serve many requests at once by raising
max_concurrent_requests, which the in-process backend cannot do.
"""

import logging
import threading

import httpx2

from app.backends.base import BackendError, GenerationResult
from app.backends.upstream import MAX_ERROR_MESSAGE, InflightTracker, status_error
from app.config import Settings

logger = logging.getLogger("vortex.backends.openai_compatible")


def _error_message(response: httpx2.Response) -> str:
    """Best-effort extraction of the human-readable message from an error body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text

    # OpenAI and vLLM: {"error": {"message": ...}}; Ollama: {"error": "..."}.
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return response.text


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
        # is also what keeps busy_seconds() below the stall threshold.
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
            path = "/chat/completions"
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            path = "/completions"
            body["prompt"] = prompt

        try:
            with self._inflight.track():
                response = self._client.post(path, json=body)
        except httpx2.TimeoutException as exc:
            raise BackendError(
                f"Upstream did not respond in time: {exc!r}", status_code=504
            ) from exc
        except httpx2.TransportError as exc:
            # Connection refused, DNS failure, TLS error, reset mid-response.
            raise BackendError(f"Upstream unreachable: {exc!r}") from exc

        if not response.is_success:
            raise status_error(
                response.status_code,
                _error_message(response),
                retry_after=response.headers.get("retry-after"),
            )
        return self._parse(response)

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
