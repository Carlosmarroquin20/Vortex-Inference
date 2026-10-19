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
MONITORING_NAMESPACE := monitoring
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
.PHONY: help setup model build api-key show-api-key grafana-password dashboard deploy undeploy status logs metrics test load-test tunnel clean

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
	@printf "    make dashboard   Provision the Grafana dashboard via the sidecar\n"
	@printf "    make deploy      Apply all Kubernetes manifests\n"
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
	@kubectl apply -f k8s/namespace.yaml
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
		--from-file=vortex-inference.json=k8s/monitoring/grafana-dashboard.json \
		--dry-run=client -o yaml \
		| kubectl label --local -f - grafana_dashboard=1 -o yaml \
		| kubectl apply -f -

deploy: api-key dashboard
	@printf "$(GREEN)[deploy]$(NC) Applying Kubernetes manifests to namespace $(NAMESPACE)...\n"
	@kubectl apply -f k8s/networkpolicy.yaml
	@kubectl apply -f k8s/configmap.yaml
	@kubectl apply -f k8s/model-pv.yaml
	@kubectl apply -f k8s/service.yaml
	@kubectl apply -f k8s/monitoring/prometheus-rules.yaml
	@kubectl apply -f k8s/monitoring/servicemonitor.yaml
	@kubectl apply -f k8s/deployment.yaml
	@kubectl apply -f k8s/pdb.yaml
	@kubectl apply -f k8s/keda-autoscaler.yaml
	@printf "$(GREEN)[deploy]$(NC) Waiting for rollout (model loading may take up to 90s)...\n"
	@kubectl rollout status deployment/$(IMAGE_NAME) -n $(NAMESPACE) --timeout=300s
	@printf "$(GREEN)[deploy]$(NC) Deployment ready.\n"
	@$(MAKE) status

undeploy:
	# Removes the inference workload but preserves the monitoring stack (Prometheus,
	# Grafana, KEDA). This allows re-deploying with a new image without losing
	# historical metrics or dashboard state.
	@printf "$(YELLOW)[undeploy]$(NC) Removing inference workload from $(NAMESPACE)...\n"
	@kubectl delete -f k8s/keda-autoscaler.yaml --ignore-not-found
	@kubectl delete -f k8s/pdb.yaml --ignore-not-found
	@kubectl delete -f k8s/deployment.yaml --ignore-not-found
	@kubectl delete -f k8s/service.yaml --ignore-not-found
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

load-test:
	@printf "$(GREEN)[load-test]$(NC) Starting load test to trigger KEDA scale-out...\n"
	@bash scripts/load-test.sh
