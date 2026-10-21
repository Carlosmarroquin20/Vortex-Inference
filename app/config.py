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
    #   anthropic:          Claude models through Anthropic's Messages API.
    backend: Literal["llama_cpp", "openai_compatible", "anthropic"] = "llama_cpp"

    # --- Model Configuration ---
    # model_name is reported in responses and, for remote backends, is the
    # model identifier sent upstream, so it must match the provider's name.
    # Remote backends require it to be set explicitly: the default names a
    # local GGUF model that no provider serves.
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
    # Prompt processing uses the same count (see app/backends/llamacpp.py).
    n_threads: int = Field(default=2, ge=1)

    # Generate one token at startup, before the pod reports ready, so the
    # first request does not pay for reading the memory-mapped weights from
    # disk. llama_cpp only.
    warmup: bool = True

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
    # Base URL of the upstream API. openai_compatible requires it, including
    # the version prefix (https://api.openai.com/v1, http://vllm:8000/v1), and
    # has deliberately no default, so a misconfigured deployment fails at
    # startup instead of sending prompts to a provider nobody chose.
    # anthropic defaults to https://api.anthropic.com (no version prefix).
    remote_base_url: str | None = None

    # Credential for the upstream API. Distinct from api_key, which
    # authenticates clients of this server. Optional for openai_compatible,
    # since self-hosted servers such as Ollama often run without one;
    # required for anthropic.
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

    # Whether openai_compatible sends temperature and top_p upstream. Most
    # compatible servers honour them; OpenAI's reasoning models reject
    # non-default values, so disable it for those. The anthropic backend never
    # sends them: current Claude models reject them and the SDK no longer
    # exposes them.
    remote_forward_sampling: bool = True

    # --- Anthropic Backend ---
    # output_config.effort, which sets how much the model thinks. Unset leaves
    # the model's default. Thinking tokens count against the request's
    # max_tokens, so short completions on models that always think may come
    # back truncated or empty unless effort is lowered; "low" suits most
    # completion traffic. Older models such as Haiku 4.5 reject the field.
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None

    # Server-side refusal fallback ("default" mode): when the model's safety
    # classifiers decline a request, the API re-runs it on the model Anthropic
    # recommends for that refusal category within the same call. Supported on
    # the first-party API for current Opus, Sonnet and Fable models; disable
    # for models or gateways that reject the parameter.
    anthropic_refusal_fallback: bool = True

    # --- Asynchronous Jobs ---
    # POST /v1/jobs queues a completion in RabbitMQ for a worker to run, and
    # GET /v1/jobs/{id} reads its state and result from Redis. Off by default:
    # the synchronous endpoints need neither service.
    jobs_enabled: bool = False

    # Connection URLs carry credentials, hence SecretStr: they never appear in
    # reprs or logs. amqp(s)://user:password@host:5672/vhost and
    # redis(s)://:password@host:6379/0.
    rabbitmq_url: SecretStr | None = None
    redis_url: SecretStr | None = None

    # Queue (and routing key) the jobs are published to. Its arguments are
    # fixed when RabbitMQ first creates it; changing job_max_attempts later
    # requires deleting the queue, or the declaration is refused.
    jobs_queue: str = Field(
        default="vortex.jobs", pattern=r"^[a-z0-9][a-z0-9._-]{0,99}$"
    )

    # How long a job's record, and with it its result, stays readable after
    # its last update. It also bounds how long an Idempotency-Key is honoured.
    job_ttl_seconds: int = Field(default=86400, ge=60, le=7 * 86400)

    # Attempts a job gets before it is marked failed. Transient upstream
    # failures (overload, rate limits, timeouts) are retried by the worker,
    # and RabbitMQ dead-letters a message delivered this many times without
    # an acknowledgement, so a job that keeps bringing workers down cannot
    # loop forever either.
    job_max_attempts: int = Field(default=3, ge=1, le=20)

    # Wait before the second attempt; it doubles for each attempt after
    # that, up to an hour. Each distinct delay is its own RabbitMQ queue
    # (see app/jobs/queue.py).
    job_retry_delay_seconds: int = Field(default=10, ge=1, le=3600)

    # Port of the worker's metrics and probe endpoint (python -m app.worker).
    worker_metrics_port: int = Field(default=9090, ge=1, le=65535)

    # Bound on each call to RabbitMQ or Redis from the API, so an unreachable
    # service turns into a prompt 503 instead of a hung request.
    jobs_service_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

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

    @field_validator(
        "api_key",
        "remote_api_key",
        "remote_base_url",
        "anthropic_effort",
        "rabbitmq_url",
        "redis_url",
        mode="before",
    )
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
    def remote_backend_requirements(self) -> "Settings":
        if self.backend == "llama_cpp":
            return self

        missing = []
        if "model_name" not in self.model_fields_set:
            missing.append("VORTEX_MODEL_NAME")
        if self.backend == "openai_compatible" and self.remote_base_url is None:
            missing.append("VORTEX_REMOTE_BASE_URL")
        if self.backend == "anthropic" and self.remote_api_key is None:
            missing.append("VORTEX_REMOTE_API_KEY")
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set when VORTEX_BACKEND is "
                f"{self.backend!r}."
            )
        return self

    @field_validator("rabbitmq_url")
    @classmethod
    def rabbitmq_url_must_be_amqp(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None and not v.get_secret_value().startswith(
            ("amqp://", "amqps://")
        ):
            raise ValueError("rabbitmq_url must start with amqp:// or amqps://")
        return v

    @field_validator("redis_url")
    @classmethod
    def redis_url_must_be_redis(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None and not v.get_secret_value().startswith(
            ("redis://", "rediss://")
        ):
            raise ValueError("redis_url must start with redis:// or rediss://")
        return v

    @model_validator(mode="after")
    def jobs_require_services(self) -> "Settings":
        if not self.jobs_enabled:
            return self
        missing = [
            name
            for name, value in (
                ("VORTEX_RABBITMQ_URL", self.rabbitmq_url),
                ("VORTEX_REDIS_URL", self.redis_url),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set when VORTEX_JOBS_ENABLED is true."
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
