# Vortex Inference

A production-grade MLOps platform for serving specialized Small Language Models (SLMs) on Kubernetes. Designed to demonstrate high-availability inference patterns in resource-constrained environments using event-driven autoscaling, native observability, and GGUF quantization.

---

## Architecture

```
                     ┌──────────────────────────────────────────────────┐
                     │              Minikube Cluster                     │
                     │                                                   │
  Client Request     │  ┌─────────────┐      ┌──────────────────────┐  │
 ────────────────────►  │ LoadBalancer │─────►│  vortex-inference    │  │
  (minikube tunnel)  │  │  Service    │      │  pod (FastAPI +       │  │
                     │  └─────────────┘      │  llama-cpp-python)    │  │
                     │                       │                       │  │
                     │                       │  ┌─────────────────┐  │  │
                     │                       │  │ /v1/*completions │  │  │
                     │                       │  │ /health          │  │  │
                     │                       │  │ /metrics ──────────────────►  Prometheus
                     │                       │  └─────────────────┘  │  │
                     │                       └──────────────────────┘  │
                     │                                 │               │
                     │              ┌──────────────────▼─────────────┐ │
                     │              │         KEDA ScaledObject       │ │
                     │              │  Prometheus trigger:            │ │
                     │              │  sum(active_requests) / 3       │ │
                     │              │  → scales Deployment 1..3       │ │
                     │              └────────────────────────────────┘ │
                     │                                                   │
                     │  ┌──────────────────────────────────────────┐   │
                     │  │  monitoring namespace                      │   │
                     │  │  Prometheus ◄── ServiceMonitor            │   │
                     │  │  Grafana                                   │   │
                     │  │  AlertManager ◄── PrometheusRules         │   │
                     │  └──────────────────────────────────────────┘   │
                     └──────────────────────────────────────────────────┘
```

**Data flow:** Client → LoadBalancer Service → FastAPI pod → llama-cpp-python (GGUF, CPU) → response. In parallel: Prometheus scrapes `/metrics` every 15s. KEDA queries Prometheus every 15s and adjusts `spec.replicas` on the Deployment based on `sum(vortex_inference_active_requests)`.

**Asynchronous jobs:** Client → `POST /v1/jobs` → API writes the job to Redis and publishes it to RabbitMQ → 202. KEDA reads the queue length from RabbitMQ's management API and scales the `vortex-worker` Deployment from zero; a worker runs the job on the same backend and stores the result in Redis, where `GET /v1/jobs/{id}` reads it. See [Asynchronous Jobs](#asynchronous-jobs).

---

## Stack

| Layer | Technology | Rationale |
|---|---|---|
| Inference runtime | `llama-cpp-python` + GGUF Q4_K_M | Best perplexity/size ratio for 4-bit CPU inference |
| API framework | FastAPI | Native async — health checks never block inference thread |
| Autoscaling | KEDA v2.21 (Prometheus trigger) | Scales on actual request load, not CPU utilization |
| Observability | Prometheus + kube-prometheus-stack | Native metrics pipeline — no sidecar required |
| Orchestration | Kubernetes (Minikube) | Single-node local cluster mirroring production topology |
| Model | Phi-3-mini-4k-instruct Q4_K_M | 2.4GB on disk, ~2.8GB RAM — fits within 4Gi pod limit |

---

## Key Design Decisions

### Why KEDA instead of HPA + CPU metrics?

CPU utilization is a lagging and misleading signal for LLM inference. A pod can be at 5% CPU while holding a queue of 8 requests waiting for the current inference call to complete (the model processes tokens serially). KEDA's Prometheus trigger scales on `vortex_inference_active_requests` — the gauge that reflects actual demand. The scaling formula is:

```
desiredReplicas = ceil( sum(active_requests across all pods) / threshold )
```

With `threshold: 3`, a queue of 7 concurrent requests produces `ceil(7/3) = 3 replicas`.

### Why `max_concurrent_requests: 1` per pod?

`llama-cpp-python`'s C++ backend processes tokens sequentially within a single thread pool. Two simultaneous inference calls do not double throughput — they double memory pressure and cause CFS CPU throttling that degrades both requests. Horizontal scaling (KEDA) is the correct mechanism for concurrency. Vertical parallelism is reserved for GPU deployments with tensor parallelism support.

### Why a 300s scale-down stabilization window?

Model loading takes 60–90s from pod creation to readiness probe passing. Scaling in too eagerly causes flapping: a pod is removed during a brief lull, then must be re-provisioned when the next burst arrives — paying the full startup cost again. The HPA that KEDA manages keeps the highest replica count recommended over the last five minutes (`scaleDown.stabilizationWindowSeconds: 300`) and then removes at most one pod per minute, at the cost of slightly higher idle resource usage.

KEDA's own `cooldownPeriod` is not used: with `minReplicaCount: 1` it only governs scaling to and from zero, which this workload never does. The job workers do scale to zero, and use it (see [Asynchronous Jobs](#asynchronous-jobs)).

### Thread counts follow the CPU limit, not the node

Inside a container, `os.cpu_count()` reports the node's CPUs, not the pod's quota. Every native thread pool sized from it oversubscribes the quota and gets throttled by the kernel: OpenBLAS and OpenMP (capped through `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` in the ConfigMap) and llama.cpp's prompt-processing threads, which llama-cpp-python defaults to `os.cpu_count()` when only `n_threads` is given. The backend passes `n_threads_batch=n_threads`. Measured with Qwen2.5-0.5B under a 2-CPU limit on a 12-CPU host:

| | `n_threads_batch` unset (12) | `n_threads_batch=2` |
|---|---|---|
| Prompt processing ([`scripts/bench_llama.py`](scripts/bench_llama.py)) | 11.5 tok/s | 115 tok/s |
| Generation | 40.1 tok/s | 38.6 tok/s |
| Streamed time to first token, P50 / P95 (`scripts/loadgen.py`) | 1.18s / 2.39s | 0.16s / 0.24s |
| First request after startup | 35.2s | 1.5s |

The same measurements show the BLAS backend, which ggml uses for prompt processing in batches of 32 tokens and more, gaining nothing for a model this small (135 tok/s with `n_batch=16`, which keeps it out). Re-run the benchmark against the deployed model before changing the build:

```bash
docker run --rm --cpus 2 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v ./models:/models:ro -v ./scripts:/bench:ro --entrypoint python vortex-inference:1.0.0 \
  /bench/bench_llama.py --model /models/phi-3-mini-4k-instruct-q4_k_m.gguf \
  --threads 2 --threads-batch 2,0 --batch 512,16
```

The server also generates one token at startup (`VORTEX_WARMUP`, on by default) before it reports ready, so a node that has not cached the memory-mapped weights reads them then rather than during the first request.

### Memory limit: 4Gi hard ceiling

The 4Gi limit simulates a dense multi-tenant node or an edge deployment where each tenant gets a guaranteed slice. It enforces engineering discipline: the model must fit within the budget or the quantization level must be increased. The `requests.memory: 2Gi` / `limits.memory: 4Gi` asymmetry allows burst allocation during model loading (which peaks higher than steady-state inference) without requiring the full 4Gi to be reserved on the node for scheduling.

---

## Prerequisites

- Docker Desktop
- `minikube` ≥ 1.39 (supports the pinned Kubernetes v1.35)
- `kubectl` ≥ 1.34 (within one minor version of the cluster)
- `helm` ≥ 3.14
- `curl` or `wget`, and `sha256sum` or `shasum` — the model download is verified against a pinned SHA-256
- 12GB free RAM, 8 CPU cores, 20GB disk

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd vortex-inference

# 2. Download the GGUF model (~2.4GB)
bash scripts/download-model.sh

# 3. Bootstrap Minikube + KEDA + kube-prometheus-stack
bash scripts/setup-minikube.sh

# 4. Copy model into Minikube node
minikube cp ./models/phi-3-mini-4k-instruct-q4_k_m.gguf /data/models/

# 5. Build the container image and load it into the minikube node
make build

# 6. Deploy the inference workload; waits for the startup probe (~90s for model loading)
make deploy

# 7. Expose the service
minikube tunnel  # Run in a separate terminal

# 8. Test inference
INFERENCE_IP=$(kubectl get svc vortex-inference-lb -n vortex-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
API_KEY=$(make --no-print-directory show-api-key)
curl -X POST "http://${INFERENCE_IP}/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"prompt": "Explain transformer attention in one paragraph:", "max_tokens": 200}'
```

The response follows OpenAI's `text_completion` schema, so the official SDKs work
unchanged. Server-side throughput is reported in an additional `performance`
object.

```python
from openai import OpenAI

client = OpenAI(base_url=f"http://{INFERENCE_IP}/v1", api_key=API_KEY)
completion = client.completions.create(
    model="phi-3-mini-4k-instruct",
    prompt="Explain transformer attention in one paragraph:",
    max_tokens=200,
)
print(completion.choices[0].text)
```

### Streaming

`"stream": true` returns server-sent events in OpenAI's chunk format, ending
with `data: [DONE]`; `"stream_options": {"include_usage": true}` adds a final
chunk with token usage. It works with every backend.

```bash
curl -N "http://${INFERENCE_IP}/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"prompt": "Explain transformer attention:", "max_tokens": 200, "stream": true}'
```

```python
for chunk in client.completions.create(
    model="phi-3-mini-4k-instruct",
    prompt="Explain transformer attention:",
    max_tokens=200,
    stream=True,
):
    print(chunk.choices[0].text, end="", flush=True)
```

- **Errors.** The response is committed only once the backend produces its
  first event, so a failure before any output (prompt too long, upstream rate
  limit, deadline) still returns the usual JSON error and status code. A
  failure after that ends the stream with an `error` event and no `[DONE]`,
  which the OpenAI SDKs raise as an `APIError`.
- **Deadline.** `VORTEX_INFERENCE_TIMEOUT_SECONDS` bounds the whole stream, as
  it bounds a buffered request.
- **Disconnects.** When the client goes away, generation stops at the next
  token (llama.cpp) or the upstream connection is closed (remote backends), and
  the inference permit is released as soon as the worker thread exits.
- **Refusals.** With the `anthropic` backend, a stream that ends with
  `finish_reason: "content_filter"` was declined after some text had been sent;
  that partial text should be discarded.
- **Token counts.** For `llama_cpp` the library reports no usage when
  streaming, so completion tokens are counted per returned chunk; that matches
  the buffered count except when a multi-byte character spans several tokens.
- **Streaming latency** is recorded as
  `vortex_inference_time_to_first_token_seconds`, from the request's arrival
  to its first chunk (what the client waits, queue included; the queue part
  alone is `vortex_inference_queue_wait_seconds`), and
  `vortex_inference_time_per_output_token_seconds`, the average gap between
  the tokens after the first.

### Chat completions

`/v1/chat/completions` takes a conversation in OpenAI's format and returns a
`chat.completion` object, or `chat.completion.chunk` events with
`"stream": true`. It shares the deadline, concurrency limit, metrics and error
handling of `/v1/completions`.

```python
reply = client.chat.completions.create(
    model="phi-3-mini-4k-instruct",
    messages=[
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=60,
)
print(reply.choices[0].message.content)
```

- **How the conversation is formatted.** `llama_cpp` applies the chat template
  embedded in the GGUF file, the one the model was trained with. On
  Qwen2.5-0.5B-Instruct the output and the prompt token count matched
  llama-cpp-python's own `create_chat_completion` exactly, buffered and
  streamed. A model without a template answers 400; use `/v1/completions` with
  a prompt in its own format. Remote backends pass the conversation to the
  provider's chat API; for `anthropic`, system messages become the Messages
  API's `system` field.
- **Accepted input.** Roles `system`, `developer` (treated as `system`), `user`
  and `assistant`; `content` as a string or a list of text parts. `stop` may be
  a string or a list; `max_completion_tokens` takes precedence over
  `max_tokens`. Tools, images, `n > 1`, `response_format` and other unsupported
  parameters are rejected with 422 rather than ignored.
- **Limits.** Up to 256 messages, 32,768 characters per message and 65,536 in
  total.

---

## Inference Backends

`VORTEX_BACKEND` selects where completions come from. The API, authentication,
metrics, probes and autoscaling signal are the same for every backend.

| Backend | Use it for | Model setting |
|---|---|---|
| `llama_cpp` (default) | Any GGUF model, run in-process on the pod's CPUs | `VORTEX_MODEL_PATH` (file on the PVC) |
| `openai_compatible` | OpenAI, or a model served elsewhere by vLLM, Ollama, LM Studio, `llama-server` or a router such as OpenRouter | `VORTEX_MODEL_NAME` (the upstream model ID) |
| `anthropic` | Claude models through Anthropic's Messages API | `VORTEX_MODEL_NAME` (e.g. `claude-opus-5-5`) |

**Another local model.** Any GGUF file works without code changes: download it
with `HF_REPO=... HF_FILENAME=... HF_REVISION=... MODEL_SHA256=... MODEL_SIZE=...
bash scripts/download-model.sh`, point `VORTEX_MODEL_PATH`/`VORTEX_MODEL_NAME`/
`VORTEX_N_CTX` at it, and resize the pod's memory limit to fit.

**A model behind an API.** Larger models can run on a GPU machine under vLLM
or Ollama, or at a hosted provider, with Vortex in front of them:

```bash
VORTEX_BACKEND=openai_compatible
VORTEX_REMOTE_BASE_URL=https://api.openai.com/v1   # or http://<vllm-host>:8000/v1
VORTEX_REMOTE_API_KEY=sk-...                       # omit for unauthenticated servers
VORTEX_MODEL_NAME=<provider-model-id>              # sent upstream as "model"
VORTEX_MAX_CONCURRENT_REQUESTS=16                  # generation is upstream, not on the pod
```

- `VORTEX_REMOTE_API_STYLE=chat` (default) sends the prompt as a single user
  message to `/chat/completions`, the only endpoint current hosted models offer;
  the model's chat template is applied upstream. `completions` forwards the raw
  prompt to `/completions`, for base models on self-hosted servers.
- OpenAI reasoning models need `VORTEX_REMOTE_MAX_TOKENS_FIELD=max_completion_tokens`
  and `VORTEX_REMOTE_FORWARD_SAMPLING=false`, since they reject non-default
  `temperature`/`top_p`.

**Claude models.** The `anthropic` backend calls the Messages API through the
official SDK:

```bash
VORTEX_BACKEND=anthropic
VORTEX_REMOTE_API_KEY=sk-ant-...
VORTEX_MODEL_NAME=claude-opus-5-5
VORTEX_ANTHROPIC_EFFORT=low            # optional; see below
VORTEX_MAX_CONCURRENT_REQUESTS=16
```

- The prompt is sent as a single user turn. `temperature` and `top_p` are
  accepted by the API but not forwarded: current Claude models reject them.
- Current models think before answering, and thinking tokens count against the
  request's `max_tokens`. With small `max_tokens` values a completion can come
  back truncated (`finish_reason: "length"`) or empty; `VORTEX_ANTHROPIC_EFFORT=low`
  keeps thinking short for typical completion traffic.
- Server-side refusal fallback is on (`VORTEX_ANTHROPIC_REFUSAL_FALLBACK=true`):
  if the model's safety classifiers decline a request, the API re-runs it on
  Anthropic's recommended fallback model within the same call. A request that
  is still declined returns 200 with empty text and `finish_reason: "content_filter"`.
  Turn it off for models or gateways that reject the parameter.
- The SDK's automatic retries are disabled so a request never outlives its
  deadline; Anthropic's 529 (overloaded) is returned as 503 with `Retry-After`,
  like a 429. `ANTHROPIC_*` environment variables are ignored: the endpoint and
  key always come from the `VORTEX_*` settings.

**Behaviour shared by the remote backends:**

- Upstream 400/413/422 responses are returned to the client as 400 with the
  provider's message and counted as `rejected`. Other upstream failures become
  502 (503 with `Retry-After` on rate limiting, 504 on timeouts) without
  exposing upstream details; the cause is in the server log. A configuration
  error that the provider reports as 400, such as an unsupported parameter,
  therefore shows up as a rise in `rejected` requests and a WARNING per
  request, not in the error ratio.
- In Kubernetes, remote backends run with the `remote` deployment profile
  below; the default manifests block all egress from the inference pods.

### Deployment profiles

The manifests are a Kustomize base plus overlays, selected with `PROFILE`:

| Profile | Directory | What it deploys |
|---|---|---|
| `local` (default) | `k8s/base` | `llama_cpp` with the GGUF model on a PersistentVolume, 2 CPU / 2–4Gi pods, no egress |
| `remote` | `k8s/overlays/remote` | A remote backend (Claude by default), no model volume, 250m CPU / 128–512Mi pods, 16 concurrent requests per pod, egress limited to DNS and HTTPS |

```bash
make build                       # same image for both profiles
make provider-key                # prompts for the provider's API key (no echo)
make deploy PROFILE=remote
```

The provider is chosen in `k8s/overlays/remote/configmap.yaml`, which has
examples for OpenAI and for a vLLM server inside the cluster. The egress policy
allows TCP 443 to public addresses only: private ranges, and with them the cloud
metadata endpoint, stay unreachable. Standard NetworkPolicy cannot match
hostnames; restricting egress to the provider's domain needs a CNI extension
such as Cilium's `toFQDNs`. A provider inside the cluster needs its own rule,
sketched in `k8s/overlays/remote/networkpolicy-egress.yaml`.

`make deploy` with the `local` profile removes the remote egress policy, since
`kubectl apply` never deletes objects on its own. Switching from `local` to
`remote` leaves the model's PersistentVolume and claim in place, unused; delete
them with `kubectl delete -f k8s/base/model-pv.yaml` if they are no longer needed.

### Fallback backend

A second backend can take the requests the primary cannot serve
(`app/backends/fallback.py`): the primary is overloaded or rate limited
(503), fails or times out upstream (502, 504), or the prompt does not fit its
context, as when a 4K-context local model meets a long document that a
provider's larger window can take. Client errors still go back unchanged, and
abandoned requests are not retried. After `VORTEX_FALLBACK_FAILURE_THRESHOLD`
(3) consecutive failures the primary is skipped for
`VORTEX_FALLBACK_COOLDOWN_SECONDS` (30), then tried again.

```bash
VORTEX_FALLBACK_BACKEND=anthropic
VORTEX_FALLBACK_MODEL_NAME=claude-opus-5-5
VORTEX_FALLBACK_REMOTE_API_KEY=...        # or VORTEX_FALLBACK_REMOTE_BASE_URL for openai_compatible
```

The fallback shares every other setting with the primary. A buffered response
names the model that produced it in `model`; a stream can only fall back
before its first token, and keeps the primary's name in its chunks.
`vortex_backend_fallbacks_total{reason}` counts every request that moved.
In the cluster, a remote fallback needs egress to its provider: the remote
profile's policy, or a rule for a provider inside the cluster.

Verified with two instances of this server, the second serving as the
first's OpenAI-compatible fallback: a 738-token prompt sent to a primary
with a 256-token context came back from the fallback, buffered and streamed;
with the fallback stopped, the client got its 502 and short prompts kept
working on the primary.

---

## Asynchronous Jobs

For work that does not need an immediate answer (offline evaluation,
summarizing a backlog, anything a client would otherwise hold a connection
open for), `POST /v1/jobs` queues a request in RabbitMQ and answers at once;
a worker runs it and `GET /v1/jobs/{id}` returns the result from Redis. The
submission names the endpoint and carries the body it accepts, as in OpenAI's
Batch API, so a job's result has exactly the shape that endpoint would return.

```bash
curl -s -X POST "http://${INFERENCE_IP}/v1/jobs" \
  -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
  -H "Idempotency-Key: report-2026-10-21" \
  -d '{"url": "/v1/chat/completions",
       "body": {"messages": [{"role": "user", "content": "Summarize..."}]}}'
# 202 {"id": "job_3f9c...", "object": "job", "status": "queued", ...}

curl -s "http://${INFERENCE_IP}/v1/jobs/job_3f9c..." -H "Authorization: Bearer ${API_KEY}"
# {"status": "succeeded", "result": {"object": "chat.completion", ...}, ...}
```

- **Durability.** The 202 is sent only after RabbitMQ confirms it has stored
  the message (publisher confirms, persistent messages, quorum queues). If
  RabbitMQ or Redis is unreachable the API answers 503 with `Retry-After`;
  the synchronous endpoints keep working, since connections are opened on
  first use and not at startup.
- **Idempotency.** Repeating a submission with the same `Idempotency-Key`
  returns the original job (200) instead of queueing it again, so a client can
  retry after a timeout without running the request twice.
- **Lifetime.** Records expire `VORTEX_JOB_TTL_SECONDS` (24h) after their last
  update; an unknown and an expired job both answer 404.
- **Retries.** Transient upstream failures (overload, rate limits, timeouts,
  unreachable provider) are retried up to `VORTEX_JOB_MAX_ATTEMPTS` (3) in
  total, after `VORTEX_JOB_RETRY_DELAY_SECONDS` (10s) and doubling. The job
  shows `queued` with the last attempt's `error` while it waits. Rejections,
  internal errors and local deadline overruns fail at once: they would fail
  the same way again.
- **Poison messages.** The quorum queue counts deliveries: a job delivered
  `VORTEX_JOB_MAX_ATTEMPTS` times without completing (its worker crashed each
  time) is marked failed and moves to the `vortex.jobs.dead` queue instead of
  looping, as does a message the worker cannot parse. A bug in processing
  fails the job and dead-letters it on the first delivery: RabbitMQ 4 does
  not count requeues towards the delivery limit, so requeueing would loop.
- **Status values**: `queued`, `running`, `succeeded` (with `result`) and
  `failed` (with `error`: the status code and message the endpoint would have
  answered with).

Jobs are off unless `VORTEX_JOBS_ENABLED=true`, with `VORTEX_RABBITMQ_URL` and
`VORTEX_REDIS_URL` set.

### Worker

`python -m app.worker` runs the jobs. It is the same image and configuration
as the API, so it loads the same backend, and takes up to
`VORTEX_MAX_CONCURRENT_REQUESTS` jobs at a time (its RabbitMQ prefetch).

- **At-least-once.** A delivery is acknowledged only after its outcome is in
  Redis. If Redis is unreachable the job is put back through the retry queue
  without using an attempt and runs once Redis returns; a redelivered job that
  had already finished is skipped.
- **Retry queues.** A retry is published to `vortex.jobs.retry.<delay>s`, a
  queue with no consumers whose TTL dead-letters the job back to the jobs
  exchange. There is one queue per delay because RabbitMQ only expires
  messages at the head of a queue.
- **Shutdown.** On SIGTERM the worker stops consuming, lets jobs in progress
  finish (up to `VORTEX_INFERENCE_TIMEOUT_SECONDS` plus 15s) and exits; jobs
  still running at the deadline are stopped and redelivered elsewhere.
- **Probes.** The same port serves `/health/ready` (model loaded and
  consuming) and `/health/live`, which fails when the event loop stops
  running or an inference thread is wedged past its deadline, the two
  conditions a restart fixes.
- **Metrics** on `VORTEX_WORKER_METRICS_PORT` (9090):
  `vortex_worker_jobs_total{outcome}`, `vortex_worker_job_duration_seconds`,
  `vortex_worker_job_wait_seconds` (submission to first attempt) and
  `vortex_worker_jobs_in_progress`. Its log lines carry the job ID in
  `request_id`.

### In the cluster

`make deploy` runs the whole pipeline in the `vortex-inference` namespace:
RabbitMQ and Redis as single-replica StatefulSets (`k8s/base/rabbitmq.yaml`,
`k8s/base/redis.yaml`, the same pinned images the integration tests use), and
the workers (`k8s/base/worker.yaml`). `make jobs-secrets`, which `deploy`
runs first, generates their credentials once into the `vortex-inference-jobs`
Secret.

- **Scale to zero.** KEDA's RabbitMQ scaler adds one worker per
  `VORTEX_MAX_CONCURRENT_REQUESTS` messages in `vortex.jobs` and removes the
  last one after 5 idle minutes. It counts unacknowledged messages too (the
  management API rather than AMQP), so a worker busy with the last job is not
  scaled away. Measured on minikube: a job queued on an idle cluster had a
  worker 12s later and its result 63s after submission, model load included.
- **Batch yields to interactive.** Workers run under the `vortex-batch`
  PriorityClass, below everything else, so on a full node the scheduler
  preempts a worker for an API replica; the worker drains on SIGTERM and its
  job is redelivered.
- **Network.** RabbitMQ and Redis accept connections only from the API and the
  workers, the management API only from KEDA, and the metrics ports only from
  the monitoring namespace (`k8s/base/networkpolicy-jobs.yaml`). Neither has
  any egress.
- **Persistence.** RabbitMQ keeps its queues and Redis its append-only file on
  PersistentVolumeClaims; Redis refuses writes rather than evicting job
  records when it reaches `maxmemory`.
- **Alerts** for the dead-letter queue, jobs nobody consumes, slow starts and
  failing submissions, with runbooks in [docs/runbooks.md](docs/runbooks.md),
  and an Asynchronous Jobs row in the Grafana dashboard.

## Evaluation

`scripts/run_evals.py` runs a dataset of conversations against
`/v1/chat/completions` and grades each reply with deterministic checks
(contains, regex, valid JSON with given keys, word and line counts). Replies
are generated with temperature 0, so a run is reproducible, and the same
dataset compares backends on equal terms: a local GGUF model, a vLLM server
or a provider's API behind the remote profile.

[`evals/datasets/core.jsonl`](evals/datasets/core.jsonl) holds 20 cases tagged
by what they exercise: reasoning, knowledge, instruction following,
structured output, multi-turn chat, extraction, classification and
translation. It is a regression gate rather than a benchmark: the cases are
easy enough for a small model to pass most of, so a drop means something in
the serving path (chat templating, stop handling, sampling defaults) or the
model changed.

```bash
python scripts/run_evals.py --url http://127.0.0.1:8080 --json report.json
make eval EVAL_ARGS="--min-pass-rate 0.9"   # against the deployed model
```

CI runs the dataset on every push against the image it has just built, with
Qwen2.5-0.5B pinned by revision and SHA-256, and fails below 85%. The
[baseline](evals/baselines/qwen2.5-0.5b-instruct-q2_k.json) for that model is
18/20, identical across runs; its two failures are the model's own (it names
`NodeSelector` as the component that assigns pods to nodes, and wraps an
extracted address in a sentence).

## Load Testing

`scripts/loadgen.py` measures latency from the client side: end to end, and for
streamed requests time to first token and time per output token as chunks
arrive. It needs only the Python standard library and can write a JSON report
to compare runs.

```bash
make loadgen                                   # through the LoadBalancer (make tunnel)
make loadgen LOADGEN_ARGS="--endpoint chat --stream --concurrency 8 --requests 40 --json run.json"
VORTEX_API_KEY=... python3 scripts/loadgen.py --url http://127.0.0.1:8080 --stream
```

Workers run a closed loop (each sends its next request when the previous one
ends), so results describe latency at a given concurrency. A measured example,
Qwen2.5-0.5B-Instruct Q2_K on llama.cpp with 2 CPUs, streamed chat, 48 tokens:

| Concurrency | TTFT p50 | TPOT p50 | Output tokens/s |
|---|---|---|---|
| 1 | 3.1 s | 30 ms | 6.2 |
| 4 | 16.7 s | 31 ms | 10.4 |

Decode speed holds under load while time to first token absorbs the queue: one
permit per pod serializes requests, which is the cue for KEDA to add replicas.
The first request after startup took 28 s at concurrency 1, paying the page
faults of the memory-mapped weights. `scripts/load-test.sh` remains the
KEDA scale-out demonstration.

## Observability

```bash
# Prometheus UI
kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n monitoring

# Grafana (user admin; the password is generated at install time)
make grafana-password
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring

# KEDA scaling activity
kubectl describe scaledobject vortex-inference-scaler -n vortex-inference

# Live pod metrics
kubectl top pods -n vortex-inference
```

The **Vortex Inference** dashboard (`k8s/base/monitoring/grafana-dashboard.json`) is
provisioned automatically by `make deploy` through the Grafana sidecar; run
`make dashboard` to push changes to the JSON without redeploying. It can also be
imported manually, in which case pick the Prometheus data source from the
dashboard's *Data source* selector.

**Key Prometheus queries for the Grafana dashboard:**

```promql
# Tokens per second (5m rate)
rate(vortex_inference_tokens_generated_total{type="completion"}[5m])

# P95 inference latency (completed and timed-out requests)
histogram_quantile(0.95, sum(rate(vortex_inference_duration_seconds_bucket{status=~"success|timeout"}[5m])) by (le))

# P95 queue wait before inference starts
histogram_quantile(0.95, sum(rate(vortex_inference_queue_wait_seconds_bucket[5m])) by (le))

# Active request queue depth (KEDA scaling signal)
sum(vortex_inference_active_requests{namespace="vortex-inference"})

# Error ratio
sum(rate(vortex_inference_requests_total{status=~"error|timeout"}[5m])) / sum(rate(vortex_inference_requests_total[5m]))
```

---

## SLOs

| SLO | Target | Alert Severity |
|---|---|---|
| P95 inference latency | < 10s | Warning |
| P99 inference latency | < 30s | Critical |
| P95 time to first token (streamed) | < 10s | Warning |
| Error ratio (5m window) | < 1% | Warning |
| Error ratio (5m window) | < 5% | Critical |
| Ready pod count | ≥ 1 | Critical |

---

## Directory Structure

```
vortex-inference/
├── app/
│   ├── main.py               # FastAPI server — lifecycle, both endpoints, request execution
│   ├── worker.py             # Job worker entry point (python -m app.worker)
│   ├── completions.py        # Request-to-response pieces shared by the API and the worker
│   ├── schemas.py            # OpenAI-compatible request and response models
│   ├── metrics.py            # Prometheus metric definitions
│   ├── config.py             # Pydantic Settings — all tunables via env vars
│   ├── logging_config.py     # logfmt formatter + request ID middleware
│   ├── jobs/                 # Asynchronous jobs: API, RabbitMQ topology, Redis store, processor
│   └── backends/
│       ├── __init__.py       # create_backend(): selects the engine from VORTEX_BACKEND
│       ├── base.py           # InferenceBackend protocol shared by every engine
│       ├── llamacpp.py       # Local GGUF inference via llama-cpp-python
│       ├── openai_compatible.py # Remote inference over an OpenAI-compatible API
│       ├── anthropic_api.py  # Claude models via the Anthropic SDK
│       ├── fallback.py       # Primary + fallback backend with a circuit breaker
│       └── upstream.py       # Status mapping shared by the remote backends
├── tests/                    # pytest suite against an in-process llama_cpp fake; integration/ needs RabbitMQ and Redis
├── docs/
│   └── runbooks.md           # One runbook per alert, linked via runbook_url
├── docker/
│   └── Dockerfile            # Multi-stage build: OpenBLAS compilation + slim runtime
├── k8s/
│   ├── base/                 # Default (local) profile and base of the overlays
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml    # Namespace + ResourceQuota + LimitRange
│   │   ├── networkpolicy.yaml # Default deny; HTTP ingress, no egress
│   │   ├── networkpolicy-jobs.yaml # RabbitMQ, Redis and worker traffic
│   │   ├── configmap.yaml    # Runtime configuration (non-sensitive)
│   │   ├── model-pv.yaml     # PersistentVolume + PVC for GGUF model storage
│   │   ├── deployment.yaml   # Deployment with probes, anti-affinity, resource limits
│   │   ├── service.yaml      # ClusterIP + LoadBalancer services
│   │   ├── pdb.yaml          # PodDisruptionBudget
│   │   ├── keda-autoscaler.yaml # ScaledObject with Prometheus trigger
│   │   ├── rabbitmq.yaml     # Job broker: StatefulSet + Service
│   │   ├── redis.yaml        # Job store: StatefulSet + Service + config
│   │   ├── worker.yaml       # Job workers, KEDA RabbitMQ scaler, PriorityClass
│   │   └── monitoring/
│   │       ├── grafana-dashboard.json # Dashboard, provisioned via the Grafana sidecar
│   │       ├── prometheus-rules.yaml  # PrometheusRule CRD — SLO alerts + recording rules
│   │       └── servicemonitor.yaml    # ServiceMonitor/PodMonitor CRDs — scrape config
│   └── overlays/
│       └── remote/           # Remote-backend profile: patches over base/
├── evals/                    # Evaluation datasets and recorded baselines
├── scripts/
│   ├── setup-minikube.sh     # Cluster bootstrap: Minikube + KEDA + kube-prometheus-stack
│   ├── download-model.sh     # Pinned-revision GGUF download, SHA-256 verified, resumable
│   ├── load-test.sh          # KEDA scale-out demonstration
│   └── loadgen.py            # Client-side latency measurement (TTFT, TPOT, JSON reports)
├── requirements.txt          # Runtime dependencies (baked into the image)
├── requirements-dev.txt      # Test and lint tooling
├── pyproject.toml            # ruff, pytest and coverage configuration
├── .dockerignore             # Must live at the build context root
├── .env.example
└── README.md
```

---

## Running Tests

The suite replaces `llama_cpp` with an in-process fake, so it needs neither the
native library nor model weights and runs in a few seconds:

```bash
grep -v '^llama-cpp-python' requirements.txt > /tmp/requirements-test.txt
pip install -r /tmp/requirements-test.txt -r requirements-dev.txt
pytest --cov
ruff check . && ruff format . --check
```

The job tests in `tests/integration` need a real RabbitMQ and Redis, and are
reported as skipped without them. `make test-integration` starts both from
`tests/integration/compose.yaml` (pinned images, loopback-only ports 5673 and
6380), runs the tests and removes the containers; CI runs them on every push.
They cover what the in-memory fakes can only assume: publisher confirms and
mandatory routing, the retry queues' TTL dead-lettering, the quorum queue's
delivery limit, draining on shutdown, and atomic idempotency claims.

```bash
make test-integration
make test-integration PYTEST_ARGS="-k retry -x"
```

The alerting rules have their own unit tests, run with
[promtool](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/)
(shipped with Prometheus) and [yq](https://github.com/mikefarah/yq):

```bash
yq '.spec' k8s/base/monitoring/prometheus-rules.yaml > tests/monitoring/rendered-rules.yaml
promtool test rules tests/monitoring/prometheus-rules.test.yaml
```
