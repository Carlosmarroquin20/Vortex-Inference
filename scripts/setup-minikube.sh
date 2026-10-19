#!/usr/bin/env bash
# =============================================================================
# Vortex Inference — Minikube cluster bootstrap
#
# Provisions a local Kubernetes environment that mirrors a production edge
# node configuration: constrained resources, full observability stack,
# and event-driven autoscaling via KEDA.
#
# Prerequisites: minikube, kubectl, helm (v3+), docker
# =============================================================================

set -euo pipefail

# --- Color output for readability ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Configuration ---
MINIKUBE_MEMORY="10240"     # 10GB: leaves 2GB for the host OS during demos
MINIKUBE_CPUS="8"         # 3 replicas x 2 CPU requests + system and monitoring pods
MINIKUBE_DISK="20g"

# Versions are pinned and chosen together. KEDA 2.21 is tested against
# Kubernetes 1.34-1.36, so the cluster runs 1.35 rather than minikube's
# floating "stable" (1.37 at the time of writing, outside KEDA's tested
# window). The kubeconform step in CI validates against the same version;
# bump all three together.
KUBERNETES_VERSION="v1.35.9"
KEDA_VERSION="2.21.0"
PROMETHEUS_STACK_VERSION="91.9.0"
NAMESPACE="vortex-inference"
MONITORING_NAMESPACE="monitoring"

# =============================================================================
# 1. Start Minikube
# =============================================================================
# --cni=calico: minikube's default CNI does not enforce NetworkPolicy, which
# would leave k8s/networkpolicy.yaml accepted by the API server but inert.
log_info "Starting Minikube node (memory=${MINIKUBE_MEMORY}MB, cpus=${MINIKUBE_CPUS})"
minikube start \
  --memory="${MINIKUBE_MEMORY}" \
  --cpus="${MINIKUBE_CPUS}" \
  --disk-size="${MINIKUBE_DISK}" \
  --driver=docker \
  --kubernetes-version="${KUBERNETES_VERSION}" \
  --cni=calico \
  --addons=metrics-server

log_info "Minikube started. Node status:"
kubectl get nodes -o wide

# =============================================================================
# 2. Install kube-prometheus-stack (Prometheus + Grafana + AlertManager)
# =============================================================================
log_info "Installing kube-prometheus-stack v${PROMETHEUS_STACK_VERSION}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace "${MONITORING_NAMESPACE}" \
  --create-namespace \
  --version "${PROMETHEUS_STACK_VERSION}" \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
  --set grafana.enabled=true \
  --wait --timeout=300s

# No admin password is passed: the Grafana chart then generates a random one,
# stores it in the kube-prometheus-stack-grafana Secret, and keeps it across
# `helm upgrade`. A literal in this script would be the same well-known
# credential on every cluster built from this repository.
log_info "kube-prometheus-stack installed."
log_warn "Grafana: kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n ${MONITORING_NAMESPACE}"
log_warn "Grafana credentials: user admin, password via 'make grafana-password'"

# =============================================================================
# 3. Install KEDA
# =============================================================================
# Installed from the versioned Helm chart rather than by applying a release
# manifest URL: helm records the installed version, upgrades in place, and
# --wait blocks until the operator and metrics server are ready, so the
# ScaledObject applied later never races the CRD registration.
log_info "Installing KEDA v${KEDA_VERSION}"
helm repo add kedacore https://kedacore.github.io/charts
helm repo update kedacore

helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --version "${KEDA_VERSION}" \
  --wait --timeout=300s

log_info "KEDA v${KEDA_VERSION} installed and ready."

# =============================================================================
# 4. Create model storage directory in Minikube node
# =============================================================================
log_info "Creating /data/models directory inside Minikube node"
minikube ssh "sudo mkdir -p /data/models && sudo chmod 755 /data/models"

log_warn "Model file not yet present. Run scripts/download-model.sh next."
log_warn "Then copy to Minikube with:"
log_warn "  minikube cp ./models/phi-3-mini-4k-instruct-q4_k_m.gguf /data/models/"

# =============================================================================
# 5. Apply Kubernetes manifests
# =============================================================================
log_info "Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/networkpolicy.yaml
kubectl apply -f k8s/configmap.yaml

# The Deployment requires this Secret to start. It is created only when
# missing so that re-running the bootstrap never rotates a key clients use.
if ! kubectl get secret vortex-inference-api-key -n "${NAMESPACE}" &>/dev/null; then
  kubectl create secret generic vortex-inference-api-key \
    -n "${NAMESPACE}" \
    --from-literal=api-key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  log_info "API key Secret created. Read it with: make show-api-key"
fi
kubectl apply -f k8s/model-pv.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/monitoring/prometheus-rules.yaml
kubectl apply -f k8s/monitoring/servicemonitor.yaml

log_warn "Skipping the workload (Deployment, PDB, ScaledObject) until the model is loaded."
log_warn "After the model is available, run: make deploy"

# =============================================================================
# 6. Container image
# =============================================================================
# Not built here: `make build` builds with the host Docker daemon and loads
# the image into the node with `minikube image load`, which works with the
# containerd runtime that minikube 1.39 uses by default.
log_warn "Build and load the inference image with: make build"

# =============================================================================
# Summary
# =============================================================================
echo ""
log_info "=== Bootstrap complete ==="
echo ""
echo "  Next steps:"
echo "  1. scripts/download-model.sh"
echo "  2. minikube cp ./models/phi-3-mini-4k-instruct-q4_k_m.gguf /data/models/"
echo "  3. make build     (build the image and load it into the node)"
echo "  4. make deploy    (Deployment, PodDisruptionBudget, ScaledObject)"
echo "  5. minikube tunnel  (in a separate terminal)"
echo ""
