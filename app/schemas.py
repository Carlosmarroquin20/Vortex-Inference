"""
Request and response models of the HTTP API.

Field names and shapes follow OpenAI's API so its SDKs work unchanged;
server-specific additions are noted where they appear.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.backends.base import ChatMessage


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


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------
# Same bounds as /v1/completions where the fields overlap. A conversation can
# legitimately be longer than one prompt, so its total size has its own cap.
MAX_CHAT_MESSAGES = 256
MAX_MESSAGE_CHARS = 32768
MAX_CONVERSATION_CHARS = 65536

StopSequence = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class ChatContentPart(BaseModel):
    # Text only: image and audio parts fail validation instead of being
    # dropped, so a client never gets an answer to a question it did not ask.
    type: Literal["text"]
    text: str = Field(max_length=MAX_MESSAGE_CHARS)

    model_config = {"extra": "forbid"}


class ChatMessageIn(BaseModel):
    # "developer" is OpenAI's newer name for system instructions; every
    # backend receives it as "system".
    role: Literal["system", "developer", "user", "assistant"]
    content: (
        Annotated[str, StringConstraints(max_length=MAX_MESSAGE_CHARS)]
        | list[ChatContentPart]
    )

    model_config = {"extra": "forbid"}

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        # Parts are adjacent segments of one message, as OpenAI treats them.
        return "".join(part.text for part in self.content)

    def to_backend(self) -> ChatMessage:
        role = "system" if self.role == "developer" else self.role
        return ChatMessage(role=role, content=self.text())


class ChatCompletionRequest(BaseModel):
    # Accepted because the OpenAI SDKs always send it. The server hosts a
    # single model, so the value is not used for routing.
    model: str | None = Field(default=None, max_length=256)
    messages: list[ChatMessageIn] = Field(min_length=1, max_length=MAX_CHAT_MESSAGES)
    # max_completion_tokens is OpenAI's current name; max_tokens is still sent
    # by many clients. When both are present the current name wins.
    max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    # OpenAI accepts a single string or a list here.
    stop: StopSequence | list[StopSequence] | None = Field(default=None)
    stream: bool = False
    stream_options: StreamOptions | None = None

    # Unsupported OpenAI parameters (tools, n, response_format, logprobs...)
    # are rejected rather than ignored, so a client relying on one finds out.
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_request(self) -> "ChatCompletionRequest":
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options is only allowed when stream is true.")
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("stop accepts at most 4 sequences.")
        total = sum(len(message.text()) for message in self.messages)
        if total > MAX_CONVERSATION_CHARS:
            raise ValueError(
                f"messages total {total} characters; the limit is "
                f"{MAX_CONVERSATION_CHARS}."
            )
        return self

    def conversation(self) -> list[ChatMessage]:
        return [message.to_backend() for message in self.messages]

    def completion_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 256

    def stop_sequences(self) -> list[str] | None:
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    refusal: None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: None = None
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    """
    Mirrors OpenAI's chat.completion object. `performance` is the same
    server-specific extension as in CompletionResponse.
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
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
