"""
Configuration management via Pydantic Settings.

Design rationale: All tunables are environment-variable-driven to align with
the 12-factor app methodology. This allows the same container image to be
promoted across environments (dev → staging → prod) by changing only the
ConfigMap or Secret — not the image layer. No hardcoded paths or magic numbers.
"""

from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Runtime configuration for the inference server.

    Numeric fields carry range constraints so a malformed ConfigMap fails at
    process start with a clear validation error. Without them, values such as
    max_concurrent_requests=0 load successfully and only surface later as a
    semaphore that never admits a request.

    Values are resolved in this priority order:
      1. Environment variables (injected by Kubernetes ConfigMap/Secret)
      2. .env file (local development only, never committed)
      3. Defaults defined here (safe fallbacks for resource-constrained environments)
    """

    # --- Backend Selection ---
    # Engine that produces completions; see app/backends. The Literal makes an
    # unknown name a startup validation error rather than a failure on the
    # first request.
    #   llama_cpp:          the GGUF model at model_path, run in-process.
    #   openai_compatible:  an HTTP API that implements OpenAI's
    #                       /chat/completions or /completions (OpenAI, vLLM,
    #                       Ollama, LM Studio, OpenRouter, ...).
    backend: Literal["llama_cpp", "openai_compatible"] = "llama_cpp"

    # --- Model Configuration ---
    # model_name is reported in responses and, for remote backends, is the
    # model identifier sent upstream, so it must match the provider's name.
    model_path: str = "/models/phi-3-mini-4k-instruct-q4_k_m.gguf"
    model_name: str = "phi-3-mini-4k-instruct"

    # n_ctx defines the KV-cache allocation at startup. Setting this above the
    # model's actual training context (4096 for Phi-3-mini) wastes VRAM/RAM
    # without any quality benefit. In memory-constrained pods, this is the
    # first lever to pull if the process approaches its memory limit.
    n_ctx: int = Field(default=4096, ge=1)

    # n_threads should match the pod's CPU request, not its limit.
    # Using the limit value risks throttling during burst inference, which
    # causes latency spikes that look like application bugs in dashboards.
    # The relationship: n_threads=2 → ~1.5–2 tok/s on modern x86 with OpenBLAS.
    n_threads: int = Field(default=2, ge=1)

    # Number of layers to offload to the GPU; -1 offloads all of them. It only
    # has an effect when llama-cpp-python was compiled with a GPU backend.
    # The image in docker/Dockerfile is built for CPU (OpenBLAS), where any
    # value behaves like 0. Running on GPUs also requires an image built with
    # e.g. -DGGML_CUDA=ON and a GPU resource request in the Deployment.
    n_gpu_layers: int = Field(default=0, ge=-1)

    # max_concurrent_requests=1 is deliberate for single-pod CPU inference.
    # llama.cpp processes tokens sequentially in a single C++ thread pool.
    # Parallel requests don't increase throughput — they increase memory
    # pressure and produce degraded output for all in-flight requests.
    # KEDA handles concurrency by scaling horizontally, not vertically.
    max_concurrent_requests: int = Field(default=1, ge=1)

    # --- Remote Backend Configuration ---
    # Base URL including the API version prefix, e.g. https://api.openai.com/v1
    # or http://vllm.inference.svc:8000/v1. Required for remote backends and
    # deliberately without a default, so a misconfigured deployment fails at
    # startup instead of sending prompts to a provider nobody chose.
    remote_base_url: str | None = None

    # Sent as a bearer token. Optional because self-hosted servers such as
    # Ollama or a private vLLM often run without one. Distinct from api_key,
    # which authenticates clients of this server.
    remote_api_key: SecretStr | None = None

    # chat sends the prompt as a single user message to /chat/completions,
    # the only endpoint current hosted models support; the model's chat
    # template is applied upstream. completions forwards the raw prompt to
    # /completions, for self-hosted servers and base (non-chat) models.
    remote_api_style: Literal["chat", "completions"] = "chat"

    # OpenAI's reasoning models reject max_tokens on chat requests and require
    # max_completion_tokens; most compatible servers accept only the former.
    remote_max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = (
        "max_tokens"
    )

    # Seconds to establish the TCP/TLS connection. Kept short and separate
    # from inference_timeout_seconds, which bounds the whole generation, so
    # an unreachable provider fails fast instead of holding a permit.
    remote_connect_timeout_seconds: float = Field(default=10.0, gt=0)

    # --- Server Configuration ---
    port: int = Field(default=8080, ge=1, le=65535)

    # Inference timeout in seconds. Must be shorter than the Kubernetes
    # Service's session affinity timeout and the client's read timeout.
    # Tuned for worst-case Phi-3-mini on 2 CPU threads at 2048 max_tokens.
    inference_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Security ---
    # Shared bearer token for /v1/completions. Authentication is enforced only
    # when this is set, so local development runs without ceremony while the
    # cluster deployment always injects it from a Secret. SecretStr keeps the
    # value out of reprs and log lines if the settings object is ever printed.
    api_key: SecretStr | None = None

    # Interactive docs (/docs, /redoc) and the OpenAPI schema are off by
    # default: they publish the full request surface and validation limits to
    # anyone who can reach the service. Enable explicitly for local work.
    enable_docs: bool = False

    @field_validator("api_key", "remote_api_key", "remote_base_url", mode="before")
    @classmethod
    def blank_means_unset(cls, v: object) -> object:
        # An empty VORTEX_API_KEY= line in .env should mean "no auth", not
        # "every request must send an empty token"; the same applies to the
        # remote settings left blank in a template.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("remote_base_url")
    @classmethod
    def remote_base_url_must_be_http(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"remote_base_url must start with http:// or https://, got: {v!r}"
            )
        # Endpoint paths are appended with a leading slash.
        return v.rstrip("/")

    @model_validator(mode="after")
    def remote_backend_requires_base_url(self) -> "Settings":
        if self.backend != "llama_cpp" and self.remote_base_url is None:
            raise ValueError(
                f"VORTEX_REMOTE_BASE_URL is required when VORTEX_BACKEND is "
                f"{self.backend!r}."
            )
        return self

    @field_validator("model_path")
    @classmethod
    def model_path_must_be_gguf(cls, v: str) -> str:
        if not v.endswith(".gguf"):
            raise ValueError(
                f"model_path must point to a .gguf file, got: {v!r}. "
                "Set VORTEX_MODEL_PATH to the correct path."
            )
        return v

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="VORTEX_", case_sensitive=False
    )
