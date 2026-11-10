#!/usr/bin/env bash
# =============================================================================
# Vortex Inference — GitOps bootstrap
#
# The only imperative step of the GitOps setup (ADR 0002). On the cluster of
# the current kubectl context it:
#
#   1. copies the evaluation-gated model onto the node, where the model
#      PersistentVolumes in gitops/cluster point;
#   2. installs the Sealed Secrets key, so the controller can open the
#      SealedSecrets committed under k8s/overlays/;
#   3. installs Argo CD from gitops/bootstrap/argocd;
#   4. applies the root Application, which brings up everything else from
#      git: Argo CD itself, Sealed Secrets, KEDA, Argo Rollouts, Prometheus,
#      and the staging and prod environments.
#
# After this, changes reach the cluster only through commits. Re-running is
# safe: every step converges.
#
# Works with any single-node cluster whose node is a Docker container on
# this host (kind, minikube with the docker driver); that is how the model
# gets onto the node without a registry or a privileged pod.
#
# Configuration (environment):
#   GITOPS_REPO              repository Argo CD reads (default: this repo on GitHub)
#   GITOPS_REVISION          branch or commit to track (default: main)
#   SEALED_SECRETS_KEY       private key PEM, as stored in the repository secret
#   SEALED_SECRETS_KEY_FILE  or a file holding it (default: .secrets/sealed-secrets.key)
#   GITOPS_WAIT              wait for every Application to be healthy (default: true)
#   GITOPS_WAIT_TIMEOUT      seconds to wait (default: 900)
#
# Prerequisites: kubectl, docker, curl
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[bootstrap-gitops]${NC} $*"; }
log_error() { echo -e "${RED}[bootstrap-gitops]${NC} $*" >&2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GITOPS_REPO="${GITOPS_REPO:-https://github.com/Carlosmarroquin20/Vortex-Inference.git}"
GITOPS_REVISION="${GITOPS_REVISION:-main}"
SEALED_SECRETS_KEY_FILE="${SEALED_SECRETS_KEY_FILE:-.secrets/sealed-secrets.key}"
GITOPS_WAIT="${GITOPS_WAIT:-true}"
GITOPS_WAIT_TIMEOUT="${GITOPS_WAIT_TIMEOUT:-900}"

# The model the environments serve (k8s/components/gitops-environment):
# the one the evaluation gate scores, pinned to a revision and a checksum.
export MODEL_FILENAME="qwen2.5-0.5b-instruct-q2_k.gguf"
export HF_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
export HF_FILENAME="qwen2.5-0.5b-instruct-q2_k.gguf"
export HF_REVISION="9217f5db79a29953eb74d5343926648285ec7e67"
export MODEL_SHA256="9ee36184e616dfc76df4f5dd66f908dbde6979524ae36e6cefb67f532f798cb8"
export MODEL_SIZE="415182688"

kubectl cluster-info > /dev/null || { log_error "No reachable cluster in the current kubectl context."; exit 1; }
log_info "Cluster: $(kubectl config current-context); tracking ${GITOPS_REPO} at ${GITOPS_REVISION}"

# --- 1. Model -----------------------------------------------------------------
nodes=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
if [ "$(wc -w <<< "${nodes}")" -ne 1 ]; then
  log_error "Expected a single-node cluster, found: ${nodes}"
  exit 1
fi
node="${nodes}"
bash scripts/download-model.sh
remote_size=$(docker exec "${node}" sh -c "stat -c %s /data/models/${MODEL_FILENAME} 2>/dev/null || echo 0")
if [ "${remote_size}" != "${MODEL_SIZE}" ]; then
  log_info "Copying ${MODEL_FILENAME} onto node ${node}..."
  # Written under a temporary name and renamed, so a pod never maps a
  # partially copied file.
  docker exec -i "${node}" sh -c "mkdir -p /data/models \
    && cat > /data/models/.${MODEL_FILENAME}.partial \
    && mv /data/models/.${MODEL_FILENAME}.partial /data/models/${MODEL_FILENAME}" \
    < "models/${MODEL_FILENAME}"
fi

# --- 2. Sealing key -----------------------------------------------------------
key_file="$(mktemp)"
trap 'rm -f "${key_file}"' EXIT
if [ -n "${SEALED_SECRETS_KEY:-}" ]; then
  printf '%s\n' "${SEALED_SECRETS_KEY}" > "${key_file}"
elif [ -f "${SEALED_SECRETS_KEY_FILE}" ]; then
  cp "${SEALED_SECRETS_KEY_FILE}" "${key_file}"
else
  log_error "No Sealed Secrets key: set SEALED_SECRETS_KEY or create ${SEALED_SECRETS_KEY_FILE}."
  log_error "Without it the controller cannot open the committed SealedSecrets."
  exit 1
fi
# The controller adopts any TLS secret in its namespace carrying this label
# as a decryption key, so it must exist before the controller first starts.
# Piped through stdin: the key never appears in a command line or in a
# last-applied-configuration annotation (create, not apply).
if ! kubectl get secret -n kube-system sealed-secrets-key-gitops > /dev/null 2>&1; then
  log_info "Installing the Sealed Secrets key..."
  kubectl create secret tls sealed-secrets-key-gitops -n kube-system \
    --cert gitops/sealed-secrets/sealing-key.crt --key "${key_file}" \
    --dry-run=client -o yaml \
    | kubectl label --local -f - sealedsecrets.bitnami.com/sealed-secrets-key=active -o yaml \
    | kubectl create -f -
fi

# --- 3. Argo CD ---------------------------------------------------------------
log_info "Installing Argo CD..."
# Server-side: the Argo CD CRDs are too large for the annotation client-side
# apply keeps.
kubectl apply --server-side --force-conflicts -k gitops/bootstrap/argocd > /dev/null
kubectl rollout status -n argocd deployment/argocd-repo-server --timeout=300s
kubectl rollout status -n argocd deployment/argocd-server --timeout=300s
kubectl rollout status -n argocd statefulset/argocd-application-controller --timeout=300s

# --- 4. Root Application ------------------------------------------------------
log_info "Applying the root Application..."
sed -e "s|\${GITOPS_REPO}|${GITOPS_REPO}|g" -e "s|\${GITOPS_REVISION}|${GITOPS_REVISION}|g" \
  gitops/bootstrap/root-app.yaml | kubectl apply -f -

if [ "${GITOPS_WAIT}" = "true" ]; then
  bash scripts/gitops-wait.sh "${GITOPS_WAIT_TIMEOUT}"
fi
log_info "Done. From here on, changes reach the cluster through git."
