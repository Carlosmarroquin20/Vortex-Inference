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
MINIKUBE_CPUS="6"
MINIKUBE_DISK="20g"
KEDA_VERSION="2.14.0"
PROMETHEUS_STACK_VERSION="58.4.0"
NAMESPACE="vortex-inference"
MONITORING_NAMESPACE="monitoring"

# =============================================================================
# 1. Start Minikube
# =============================================================================
log_info "Starting Minikube node (memory=${MINIKUBE_MEMORY}MB, cpus=${MINIKUBE_CPUS})"
minikube start \
  --memory="${MINIKUBE_MEMORY}" \
  --cpus="${MINIKUBE_CPUS}" \
  --disk-size="${MINIKUBE_DISK}" \
  --driver=docker \
  --kubernetes-version=stable \
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
  --set grafana.adminPassword=vortex-admin \
  --wait --timeout=300s

log_info "kube-prometheus-stack installed."
log_warn "Grafana: kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n ${MONITORING_NAMESPACE}"
log_warn "Grafana credentials: admin / vortex-admin"

# =============================================================================
# 3. Install KEDA
# =============================================================================
log_info "Installing KEDA v${KEDA_VERSION}"
kubectl apply -f "https://github.com/kedacore/keda/releases/download/v${KEDA_VERSION}/keda-${KEDA_VERSION}.yaml"

# Wait for KEDA controller to be ready before applying ScaledObjects
log_info "Waiting for KEDA operator to become ready..."
kubectl wait --for=condition=ready pod \
  -l app=keda-operator \
  -n keda \
  --timeout=120s

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
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/model-pv.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/monitoring/prometheus-rules.yaml
kubectl apply -f k8s/monitoring/servicemonitor.yaml

log_warn "Skipping deployment.yaml and keda-autoscaler.yaml until model is loaded."
log_warn "After model is available, run:"
log_warn "  kubectl apply -f k8s/deployment.yaml"
log_warn "  kubectl apply -f k8s/keda-autoscaler.yaml"

# =============================================================================
# 6. Build the Docker image inside Minikube's Docker daemon
# =============================================================================
log_info "Building inference server image inside Minikube Docker daemon..."
log_warn "This step runs 'eval \$(minikube docker-env)' which affects the current shell."
log_warn "Run manually: eval \$(minikube docker-env) && docker build -f docker/Dockerfile -t vortex-inference:1.0.0 ."

# =============================================================================
# Summary
# =============================================================================
echo ""
log_info "=== Bootstrap complete ==="
echo ""
echo "  Next steps:"
echo "  1. scripts/download-model.sh"
echo "  2. minikube cp ./models/phi-3-mini-4k-instruct-q4_k_m.gguf /data/models/"
echo "  3. eval \$(minikube docker-env)"
echo "  4. docker build -f docker/Dockerfile -t vortex-inference:1.0.0 ."
echo "  5. kubectl apply -f k8s/deployment.yaml"
echo "  6. kubectl apply -f k8s/keda-autoscaler.yaml"
echo "  7. minikube tunnel  (in a separate terminal)"
echo ""
