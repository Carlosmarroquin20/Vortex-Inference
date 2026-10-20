"""
Request and response models of the HTTP API.

Field names and shapes follow OpenAI's API so its SDKs work unchanged;
server-specific additions are noted where they appear.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


class StreamOptions(BaseModel):
    # Sends one extra chunk before [DONE] with usage and no choices.
    include_usage: bool = False

    model_config = {"extra": "forbid"}


class CompletionRequest(BaseModel):
    # Accepted because the OpenAI SDKs always send it. The server hosts a
    # single model, so the value is not used for routing.
    model: str | None = Field(default=None, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=32768)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    # Each stop sequence is matched against the generated text after every
    # token, so both count and length are bounded. Empty strings are rejected:
    # they match immediately and would end generation before the first token.
    stop: (
        list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] | None
    ) = Field(default=None, max_length=4)
    # stream=true answers with server-sent events in OpenAI's chunk format.
    stream: bool = False
    stream_options: StreamOptions | None = None

    model_config = {"extra": "forbid"}
    # extra="forbid" prevents prompt injection via undocumented parameters
    # that could be processed by future versions of the model loader.

    @model_validator(mode="after")
    def stream_options_require_stream(self) -> "CompletionRequest":
        # Matches OpenAI, which rejects stream_options on a buffered request
        # instead of silently ignoring it.
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options is only allowed when stream is true.")
        return self


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: None = None
    finish_reason: str | None  # "stop" = stop sequence/EOS; "length" = max_tokens


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionPerformance(BaseModel):
    tokens_per_second: float
    duration_seconds: float


class CompletionResponse(BaseModel):
    """
    Mirrors OpenAI's text_completion object so the official SDKs and other
    OpenAI-compatible clients parse responses natively. `performance` is a
    server-specific extension; those clients ignore unknown fields.
    """

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage
    performance: CompletionPerformance


class HealthResponse(BaseModel):
    # Reports the logical model name only. The on-disk path is an internal
    # deployment detail and this endpoint is unauthenticated by design.
    status: str
    model_loaded: bool
    model: str


class LivenessResponse(BaseModel):
    status: str
