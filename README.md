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
                     │                       │  │ /v1/completions  │  │  │
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
| Autoscaling | KEDA v2.14 (Prometheus trigger) | Scales on actual request load, not CPU utilization |
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

### Why `cooldownPeriod: 300s`?

Model loading takes 60–90s from pod creation to readiness probe passing. A short cooldown period causes scale-in/scale-out flapping: the scaler removes a pod during a brief lull, then must re-provision it when the next burst arrives — paying the full startup cost again. Five minutes of stabilization eliminates this pattern at the cost of slightly higher idle resource usage.

### Memory limit: 4Gi hard ceiling

The 4Gi limit simulates a dense multi-tenant node or an edge deployment where each tenant gets a guaranteed slice. It enforces engineering discipline: the model must fit within the budget or the quantization level must be increased. The `requests.memory: 2Gi` / `limits.memory: 4Gi` asymmetry allows burst allocation during model loading (which peaks higher than steady-state inference) without requiring the full 4Gi to be reserved on the node for scheduling.

---

## Prerequisites

- Docker Desktop
- `minikube` ≥ 1.32
- `kubectl` ≥ 1.28
- `helm` ≥ 3.14
- `huggingface-cli` (`pip install huggingface_hub`) — for verified model download
- 12GB free RAM, 6 CPU cores, 20GB disk

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

# 5. Build the container image inside Minikube's Docker daemon
eval $(minikube docker-env)
docker build -f docker/Dockerfile -t vortex-inference:1.0.0 .

# 6. Deploy the inference workload
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/keda-autoscaler.yaml

# 7. Wait for the pod to pass its startup probe (~90s for model loading)
kubectl rollout status deployment/vortex-inference -n vortex-inference

# 8. Expose the service
minikube tunnel  # Run in a separate terminal

# 9. Test inference
INFERENCE_IP=$(kubectl get svc vortex-inference-lb -n vortex-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST "http://${INFERENCE_IP}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformer attention in one paragraph:", "max_tokens": 200}'
```

---

## Observability

```bash
# Prometheus UI
kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n monitoring

# Grafana (admin / vortex-admin)
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring

# KEDA scaling activity
kubectl describe scaledobject vortex-inference-scaler -n vortex-inference

# Live pod metrics
kubectl top pods -n vortex-inference
```

**Key Prometheus queries for the Grafana dashboard:**

```promql
# Tokens per second (5m rate)
rate(vortex_inference_tokens_generated_total{type="completion"}[5m])

# P95 inference latency
histogram_quantile(0.95, sum(rate(vortex_inference_duration_seconds_bucket[5m])) by (le))

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
| Error ratio (5m window) | < 1% | Warning |
| Error ratio (5m window) | < 5% | Critical |
| Ready pod count | ≥ 1 | Critical |

---

## Directory Structure

```
vortex-inference/
├── app/
│   ├── main.py               # FastAPI server — lifecycle, endpoints, Prometheus metrics
│   ├── config.py             # Pydantic Settings — all tunables via env vars
│   └── models/
│       └── llm_loader.py     # llama-cpp-python adapter — isolated from HTTP layer
├── docker/
│   ├── Dockerfile            # Multi-stage build: OpenBLAS compilation + slim runtime
│   └── .dockerignore
├── k8s/
│   ├── namespace.yaml        # Namespace + ResourceQuota + LimitRange
│   ├── configmap.yaml        # Runtime configuration (non-sensitive)
│   ├── model-pv.yaml         # PersistentVolume + PVC for GGUF model storage
│   ├── deployment.yaml       # Deployment with probes, anti-affinity, resource limits
│   ├── service.yaml          # ClusterIP + LoadBalancer services
│   ├── keda-autoscaler.yaml  # ScaledObject with Prometheus trigger
│   └── monitoring/
│       ├── prometheus-rules.yaml   # PrometheusRule CRD — SLO alerts + recording rules
│       └── servicemonitor.yaml     # ServiceMonitor CRD — Prometheus scrape config
├── scripts/
│   ├── setup-minikube.sh     # Cluster bootstrap: Minikube + KEDA + kube-prometheus-stack
│   └── download-model.sh     # Phi-3-mini GGUF download with integrity verification
├── requirements.txt
├── .env.example
└── README.md
```
