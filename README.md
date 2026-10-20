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

KEDA's own `cooldownPeriod` is not used: with `minReplicaCount: 1` it only governs scaling to and from zero, which this workload never does.

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

---

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
│   ├── schemas.py            # OpenAI-compatible request and response models
│   ├── metrics.py            # Prometheus metric definitions
│   ├── config.py             # Pydantic Settings — all tunables via env vars
│   ├── logging_config.py     # logfmt formatter + request ID middleware
│   └── backends/
│       ├── __init__.py       # create_backend(): selects the engine from VORTEX_BACKEND
│       ├── base.py           # InferenceBackend protocol shared by every engine
│       ├── llamacpp.py       # Local GGUF inference via llama-cpp-python
│       ├── openai_compatible.py # Remote inference over an OpenAI-compatible API
│       ├── anthropic_api.py  # Claude models via the Anthropic SDK
│       └── upstream.py       # Status mapping shared by the remote backends
├── tests/                    # pytest suite against an in-process llama_cpp fake
├── docs/
│   └── runbooks.md           # One runbook per alert, linked via runbook_url
├── docker/
│   └── Dockerfile            # Multi-stage build: OpenBLAS compilation + slim runtime
├── k8s/
│   ├── base/                 # Default (local) profile and base of the overlays
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml    # Namespace + ResourceQuota + LimitRange
│   │   ├── networkpolicy.yaml # Default deny; HTTP ingress, no egress
│   │   ├── configmap.yaml    # Runtime configuration (non-sensitive)
│   │   ├── model-pv.yaml     # PersistentVolume + PVC for GGUF model storage
│   │   ├── deployment.yaml   # Deployment with probes, anti-affinity, resource limits
│   │   ├── service.yaml      # ClusterIP + LoadBalancer services
│   │   ├── pdb.yaml          # PodDisruptionBudget
│   │   ├── keda-autoscaler.yaml # ScaledObject with Prometheus trigger
│   │   └── monitoring/
│   │       ├── grafana-dashboard.json # Dashboard, provisioned via the Grafana sidecar
│   │       ├── prometheus-rules.yaml  # PrometheusRule CRD — SLO alerts + recording rules
│   │       └── servicemonitor.yaml    # ServiceMonitor CRD — Prometheus scrape config
│   └── overlays/
│       └── remote/           # Remote-backend profile: patches over base/
├── scripts/
│   ├── setup-minikube.sh     # Cluster bootstrap: Minikube + KEDA + kube-prometheus-stack
│   └── download-model.sh     # Pinned-revision GGUF download, SHA-256 verified, resumable
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

The alerting rules have their own unit tests, run with
[promtool](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/)
(shipped with Prometheus) and [yq](https://github.com/mikefarah/yq):

```bash
yq '.spec' k8s/base/monitoring/prometheus-rules.yaml > tests/monitoring/rendered-rules.yaml
promtool test rules tests/monitoring/prometheus-rules.test.yaml
```
