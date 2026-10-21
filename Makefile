# =============================================================================
# Vortex Inference — Makefile
#
# Centralizes all operational commands into a single interface.
# Rationale: ad-hoc shell commands are not reproducible and do not document
# the intended operational workflow. A Makefile makes the critical path
# explicit and executable, reducing human error during demos and onboarding.
#
# Usage: make <target>
# =============================================================================

# --- Shell configuration ---
# bash is required for pipefail and process substitution used in some recipes.
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

# --- Project configuration ---
# These variables can be overridden at invocation: make build IMAGE_TAG=2.0.0
NAMESPACE        := vortex-inference
IMAGE_NAME       := vortex-inference
IMAGE_TAG        := 1.0.0
API_KEY_SECRET   := vortex-inference-api-key
PROVIDER_KEY_SECRET := vortex-inference-provider-key
JOBS_SECRET      := vortex-inference-jobs
MONITORING_NAMESPACE := monitoring

# Deployment profile, a Kustomize directory under k8s/:
#   local   the llama_cpp backend with the GGUF model on a PersistentVolume
#   remote  a model API (Anthropic, OpenAI-compatible); see k8s/overlays/remote
# Select with: make deploy PROFILE=remote
PROFILE ?= local
ifeq ($(PROFILE),local)
KUSTOMIZE_DIR := k8s/base
else ifeq ($(PROFILE),remote)
KUSTOMIZE_DIR := k8s/overlays/remote
else
$(error PROFILE must be "local" or "remote", got "$(PROFILE)")
endif
# Cluster sizing and component versions (Kubernetes, KEDA,
# kube-prometheus-stack) live in scripts/setup-minikube.sh, which is what
# `make setup` runs.

# --- Color codes for output readability ---
GREEN  := \033[0;32m
YELLOW := \033[1;33m
RED    := \033[0;31m
CYAN   := \033[0;36m
NC     := \033[0m

# --- Computed values ---
FULL_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

# .PHONY declares targets that are not files. Without this, make would skip
# a target if a file with the same name existed in the directory.
.PHONY: help setup model build api-key show-api-key provider-key jobs-secrets require-provider-key grafana-password dashboard deploy undeploy status logs metrics test test-integration load-test loadgen eval tunnel clean

# =============================================================================
# Default target — always show help first
# Rationale: an undocumented Makefile is as bad as no Makefile. The default
# target enforces that every operator knows the available commands.
# =============================================================================
help:
	@echo ""
	@printf "  $(CYAN)Vortex Inference — Operational Commands$(NC)\n"
	@echo ""
	@printf "  $(GREEN)Bootstrap$(NC)\n"
	@printf "    make setup       Start Minikube, install KEDA + kube-prometheus-stack\n"
	@printf "    make model       Download Phi-3-mini GGUF model (~2.4GB)\n"
	@printf "    make build       Build the image and load it into minikube\n"
	@printf "    make api-key     Create the API key Secret if it does not exist\n"
	@printf "    make provider-key  Store the model provider's API key (remote profile)\n"
	@printf "    make jobs-secrets  Create the RabbitMQ and Redis credentials if missing\n"
	@printf "    make dashboard   Provision the Grafana dashboard via the sidecar\n"
	@printf "    make deploy      Apply the manifests (PROFILE=local|remote, default local)\n"
	@echo ""
	@printf "  $(GREEN)Operations$(NC)\n"
	@printf "    make status      Show pods, ScaledObject, and HPA state\n"
	@printf "    make logs        Tail inference server logs (live)\n"
	@printf "    make metrics     Dump raw Prometheus metrics from a pod\n"
	@printf "    make tunnel      Start minikube tunnel (run in a separate terminal)\n"
	@echo ""
	@printf "  $(GREEN)Testing$(NC)\n"
	@printf "    make test        Send a single inference request to verify the stack\n"
	@printf "    make load-test   Run concurrent requests to trigger KEDA scale-out\n"
	@printf "    make loadgen     Measure client-side latency (TTFT, TPOT); see LOADGEN_ARGS\n"
	@printf "    make eval        Run the evaluation dataset against the deployed model\n"
	@printf "    make test-integration  Run the jobs tests against RabbitMQ and Redis in Docker\n"
	@echo ""
	@printf "  $(GREEN)Teardown$(NC)\n"
	@printf "    make undeploy    Remove inference workload (keeps monitoring stack)\n"
	@printf "    make clean       Delete Minikube cluster entirely\n"
	@echo ""

# =============================================================================
# Bootstrap
# =============================================================================

setup:
	@printf "$(GREEN)[setup]$(NC) Bootstrapping local Kubernetes environment...\n"
	@bash scripts/setup-minikube.sh

model:
	@printf "$(GREEN)[model]$(NC) Downloading Phi-3-mini GGUF model...\n"
	@bash scripts/download-model.sh

# Builds with the host's Docker daemon, then copies the image into the
# cluster's container runtime. `minikube docker-env` is not used: it only
# works when the node runs the Docker runtime, and minikube 1.39 defaults to
# containerd. `minikube image load` works with either runtime and is what
# makes imagePullPolicy: Never in the Deployment resolve without a registry.
build:
	@printf "$(GREEN)[build]$(NC) Building $(FULL_IMAGE)...\n"
	@docker build \
		-f docker/Dockerfile \
		-t $(FULL_IMAGE) \
		--label "git.commit=$$(git rev-parse --short HEAD)" \
		--label "build.timestamp=$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		.
	@printf "$(GREEN)[build]$(NC) Loading $(FULL_IMAGE) into the minikube node...\n"
	@minikube image load $(FULL_IMAGE)
	@printf "$(GREEN)[build]$(NC) Image $(FULL_IMAGE) available in the cluster.\n"

api-key:
	# Generates the API key Secret only when it is missing, so re-running deploy
	# never rotates a key that clients already hold. The key is kept out of
	# version control entirely; rotate it with:
	#   kubectl delete secret $(API_KEY_SECRET) -n $(NAMESPACE) && make api-key
	@kubectl apply -f k8s/base/namespace.yaml
	@if kubectl get secret $(API_KEY_SECRET) -n $(NAMESPACE) >/dev/null 2>&1; then \
		printf "$(GREEN)[api-key]$(NC) Secret $(API_KEY_SECRET) already exists. Keeping current key.\n"; \
	else \
		kubectl create secret generic $(API_KEY_SECRET) \
			-n $(NAMESPACE) \
			--from-literal=api-key="$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"; \
		printf "$(GREEN)[api-key]$(NC) Secret $(API_KEY_SECRET) created.\n"; \
	fi
	@printf "$(GREEN)[api-key]$(NC) Read it with: make show-api-key\n"

show-api-key:
	@kubectl get secret $(API_KEY_SECRET) -n $(NAMESPACE) \
		-o jsonpath='{.data.api-key}' | base64 -d; echo

# Stores the model provider's API key for PROFILE=remote, creating or
# replacing the Secret. The key is prompted for without echo, or read from
# PROVIDER_API_KEY for automation. It reaches kubectl inside a manifest on
# stdin, never as a command-line argument, so it stays out of shell history
# and the process list. create/replace rather than apply: apply would copy
# the key into the last-applied-configuration annotation, which
# `kubectl describe secret` prints.
# Running pods read the key at startup; after a rotation run
#   kubectl rollout restart deployment/$(IMAGE_NAME) -n $(NAMESPACE)
provider-key:
	@kubectl apply -f k8s/base/namespace.yaml
	@key="$${PROVIDER_API_KEY:-}"; \
	if [ -z "$$key" ]; then \
		read -rsp "Provider API key: " key; echo; \
	fi; \
	if [ -z "$$key" ]; then \
		printf "$(RED)[provider-key]$(NC) No key given; nothing stored.\n"; exit 1; \
	fi; \
	if kubectl get secret $(PROVIDER_KEY_SECRET) -n $(NAMESPACE) >/dev/null 2>&1; then \
		verb=replace; \
	else \
		verb=create; \
	fi; \
	printf 'apiVersion: v1\nkind: Secret\nmetadata:\n  name: %s\n  namespace: %s\ntype: Opaque\ndata:\n  api-key: %s\n' \
		"$(PROVIDER_KEY_SECRET)" "$(NAMESPACE)" "$$(printf '%s' "$$key" | base64 | tr -d '\n')" \
		| kubectl "$$verb" -f -

# Generates the RabbitMQ and Redis credentials, and the connection URLs built
# from them, when the Secret is missing. Never rotated by a re-run: RabbitMQ
# applies its password only on first boot, so a new one here would lock the
# API and the workers out of a broker that still has the old one. The
# manifest goes to kubectl on stdin, keeping the passwords out of the process
# list; create rather than apply keeps them out of annotations.
jobs-secrets:
	@kubectl apply -f k8s/base/namespace.yaml
	@if kubectl get secret $(JOBS_SECRET) -n $(NAMESPACE) >/dev/null 2>&1; then \
		printf "$(GREEN)[jobs-secrets]$(NC) Secret $(JOBS_SECRET) already exists. Keeping current credentials.\n"; \
	else \
		rabbit=$$(python3 -c 'import secrets; print(secrets.token_hex(24))'); \
		redis=$$(python3 -c 'import secrets; print(secrets.token_hex(24))'); \
		printf 'apiVersion: v1\nkind: Secret\nmetadata:\n  name: %s\n  namespace: %s\ntype: Opaque\nstringData:\n  rabbitmq-username: "vortex"\n  rabbitmq-password: "%s"\n  rabbitmq-url: "amqp://vortex:%s@vortex-rabbitmq:5672/"\n  rabbitmq-management-url: "http://vortex:%s@vortex-rabbitmq.%s.svc:15672"\n  redis-password: "%s"\n  redis-url: "redis://:%s@vortex-redis:6379/0"\n  redis-auth.conf: "requirepass %s"\n' \
			"$(JOBS_SECRET)" "$(NAMESPACE)" "$$rabbit" "$$rabbit" "$$rabbit" "$(NAMESPACE)" "$$redis" "$$redis" "$$redis" \
			| kubectl create -f -; \
		printf "$(GREEN)[jobs-secrets]$(NC) Secret $(JOBS_SECRET) created.\n"; \
	fi

require-provider-key:
	@kubectl get secret $(PROVIDER_KEY_SECRET) -n $(NAMESPACE) >/dev/null 2>&1 || { \
		printf "$(RED)[deploy]$(NC) Secret $(PROVIDER_KEY_SECRET) not found. Run: make provider-key\n"; \
		exit 1; \
	}

# Generated by the Grafana chart at install time; see setup-minikube.sh.
# Kept outside the recipe: make echoes recipe comment lines, which would mix
# with the password when the output is captured, e.g. PW=$(make grafana-password).
grafana-password:
	@kubectl get secret kube-prometheus-stack-grafana -n $(MONITORING_NAMESPACE) \
		-o jsonpath='{.data.admin-password}' | base64 -d; echo

dashboard:
	# kube-prometheus-stack's Grafana sidecar loads any ConfigMap labelled
	# grafana_dashboard=1 in its namespace, so the dashboard appears without a
	# manual import and stays in sync with the JSON in this repository.
	# Rendered client-side and applied, so re-running updates it in place.
	@printf "$(GREEN)[dashboard]$(NC) Provisioning Grafana dashboard in $(MONITORING_NAMESPACE)...\n"
	@kubectl create configmap vortex-inference-dashboard \
		-n $(MONITORING_NAMESPACE) \
		--from-file=vortex-inference.json=k8s/base/monitoring/grafana-dashboard.json \
		--dry-run=client -o yaml \
		| kubectl label --local -f - grafana_dashboard=1 -o yaml \
		| kubectl apply -f -

# The provider key check comes first so a remote deploy that cannot succeed
# fails before creating anything.
#
# kubectl apply never deletes objects that are absent from its input, so the
# local profile removes the remote profile's egress policy explicitly; left
# behind, it would keep egress open for pods that should have none. It is
# removed before the apply so the local pods never start with it in place.
deploy: $(if $(filter remote,$(PROFILE)),require-provider-key) api-key jobs-secrets dashboard
	@printf "$(GREEN)[deploy]$(NC) Applying profile $(PROFILE) ($(KUSTOMIZE_DIR)) to namespace $(NAMESPACE)...\n"
ifeq ($(PROFILE),local)
	@kubectl delete networkpolicy vortex-inference-allow-provider-egress \
		-n $(NAMESPACE) --ignore-not-found
endif
	@kubectl apply -k $(KUSTOMIZE_DIR)
	@printf "$(GREEN)[deploy]$(NC) Waiting for rollout (a local model load may take up to 90s)...\n"
	@kubectl rollout status statefulset/vortex-rabbitmq -n $(NAMESPACE) --timeout=300s
	@kubectl rollout status statefulset/vortex-redis -n $(NAMESPACE) --timeout=120s
	@kubectl rollout status deployment/$(IMAGE_NAME) -n $(NAMESPACE) --timeout=300s
	@printf "$(GREEN)[deploy]$(NC) Deployment ready. Job workers start when jobs are queued.\n"
	@$(MAKE) status

undeploy:
	# Removes the inference workload but preserves the monitoring stack (Prometheus,
	# Grafana, KEDA). This allows re-deploying with a new image without losing
	# historical metrics or dashboard state. RabbitMQ and Redis stay too, with
	# the jobs still queued and the results not yet collected.
	@printf "$(YELLOW)[undeploy]$(NC) Removing inference workload from $(NAMESPACE)...\n"
	@kubectl delete -f k8s/base/keda-autoscaler.yaml --ignore-not-found
	@kubectl delete -f k8s/base/worker.yaml --ignore-not-found
	@kubectl delete -f k8s/base/pdb.yaml --ignore-not-found
	@kubectl delete -f k8s/base/deployment.yaml --ignore-not-found
	@kubectl delete -f k8s/base/service.yaml --ignore-not-found
	@printf "$(YELLOW)[undeploy]$(NC) Workload removed. Monitoring stack intact.\n"

# =============================================================================
# Operations
# =============================================================================

status:
	@printf "\n$(CYAN)--- Pods ---$(NC)\n"
	@kubectl get pods -n $(NAMESPACE) -o wide
	@printf "\n$(CYAN)--- ScaledObject (KEDA) ---$(NC)\n"
	@kubectl get scaledobject -n $(NAMESPACE) 2>/dev/null || echo "No ScaledObject found."
	@printf "\n$(CYAN)--- HPA (managed by KEDA) ---$(NC)\n"
	@kubectl get hpa -n $(NAMESPACE) 2>/dev/null || echo "No HPA found."
	@printf "\n$(CYAN)--- Services ---$(NC)\n"
	@kubectl get svc -n $(NAMESPACE)
	@echo ""

logs:
	# Follows logs from a single pod. For multi-pod log aggregation,
	# use Grafana/Loki or: kubectl logs -l app.kubernetes.io/name=vortex-inference -n $(NAMESPACE) --prefix
	@printf "$(GREEN)[logs]$(NC) Tailing inference server logs (Ctrl+C to exit)...\n"
	@kubectl logs \
		-l app.kubernetes.io/name=$(IMAGE_NAME) \
		-n $(NAMESPACE) \
		--follow \
		--tail=50 \
		--prefix

metrics:
	# Reads raw Prometheus metrics from a single pod through the API server's
	# pod proxy. Useful for verifying metric names and label values before
	# checking whether the ServiceMonitor is scraping them. The proxy is used
	# instead of kubectl exec because the slim image ships no HTTP client.
	@printf "$(GREEN)[metrics]$(NC) Fetching Prometheus metrics from inference pod...\n"
	@POD=$$(kubectl get pod -n $(NAMESPACE) -l app.kubernetes.io/name=$(IMAGE_NAME) \
		-o jsonpath='{.items[0].metadata.name}') && \
	kubectl get --raw "/api/v1/namespaces/$(NAMESPACE)/pods/$$POD:8080/proxy/metrics" \
		| grep -E "^vortex_"

tunnel:
	@printf "$(YELLOW)[tunnel]$(NC) Starting minikube tunnel — keep this terminal open.\n"
	@printf "$(YELLOW)[tunnel]$(NC) This provisions a LoadBalancer IP for the inference service.\n"
	@minikube tunnel

# =============================================================================
# Testing
# =============================================================================

test:
	@printf "$(GREEN)[test]$(NC) Sending single inference request...\n"
	@INFERENCE_IP=$$(kubectl get svc vortex-inference-lb \
		-n $(NAMESPACE) \
		-o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null); \
	if [ -z "$$INFERENCE_IP" ]; then \
		printf "$(RED)[test]$(NC) LoadBalancer IP not assigned. Is 'make tunnel' running?\n"; \
		exit 1; \
	fi; \
	API_KEY=$$(kubectl get secret $(API_KEY_SECRET) -n $(NAMESPACE) \
		-o jsonpath='{.data.api-key}' | base64 -d); \
	printf "$(GREEN)[test]$(NC) Endpoint: http://$$INFERENCE_IP/v1/completions\n"; \
	curl -s -X POST "http://$$INFERENCE_IP/v1/completions" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $$API_KEY" \
		-d '{"prompt": "In one sentence, what is Kubernetes?", "max_tokens": 60}' \
		| python3 -m json.tool

# Integration tests for the jobs against RabbitMQ and Redis in Docker; the
# same compose file the CI job uses. The containers are removed afterwards
# whether the tests pass or not. Pass pytest options through PYTEST_ARGS.
INTEGRATION_COMPOSE := tests/integration/compose.yaml
PYTEST_ARGS ?=
test-integration:
	@printf "$(GREEN)[test-integration]$(NC) Starting RabbitMQ and Redis...\n"
	@docker compose -f $(INTEGRATION_COMPOSE) up -d --wait
	@VORTEX_IT_RABBITMQ_URL=amqp://vortex:integration-only@127.0.0.1:5673/ \
	VORTEX_IT_REDIS_URL=redis://:integration-only@127.0.0.1:6380/0 \
	python -m pytest tests/integration $(PYTEST_ARGS); \
	status=$$?; \
	docker compose -f $(INTEGRATION_COMPOSE) down -v; \
	exit $$status

load-test:
	@printf "$(GREEN)[load-test]$(NC) Starting load test to trigger KEDA scale-out...\n"
	@bash scripts/load-test.sh

# Client-side latency (end to end, time to first token, time per output token)
# through the LoadBalancer. Options pass through scripts/loadgen.py, e.g.
#   make loadgen LOADGEN_ARGS="--endpoint chat --stream --concurrency 8 --json run.json"
# The API key reaches the script through its environment, not its arguments.
LOADGEN_ARGS ?= --endpoint chat --stream --concurrency 4 --requests 20
loadgen:
	@INFERENCE_IP=$$(kubectl get svc vortex-inference-lb -n $(NAMESPACE) \
		-o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null); \
	if [ -z "$$INFERENCE_IP" ]; then \
		printf "$(RED)[loadgen]$(NC) LoadBalancer IP not assigned. Is 'make tunnel' running?\n"; \
		exit 1; \
	fi; \
	VORTEX_API_KEY=$$(kubectl get secret $(API_KEY_SECRET) -n $(NAMESPACE) \
		-o jsonpath='{.data.api-key}' | base64 -d) \
		python3 scripts/loadgen.py --url "http://$$INFERENCE_IP" $(LOADGEN_ARGS)

# Runs the evaluation dataset against the deployed model through the
# LoadBalancer. Options pass through scripts/run_evals.py, e.g.
#   make eval EVAL_ARGS="--min-pass-rate 0.9 --json report.json"
EVAL_ARGS ?=
eval:
	@INFERENCE_IP=$$(kubectl get svc vortex-inference-lb -n $(NAMESPACE) \
		-o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null); \
	if [ -z "$$INFERENCE_IP" ]; then \
		printf "$(RED)[eval]$(NC) LoadBalancer IP not assigned. Is 'make tunnel' running?\n"; \
		exit 1; \
	fi; \
	VORTEX_API_KEY=$$(kubectl get secret $(API_KEY_SECRET) -n $(NAMESPACE) \
		-o jsonpath='{.data.api-key}' | base64 -d) \
		python3 scripts/run_evals.py --url "http://$$INFERENCE_IP" $(EVAL_ARGS)
