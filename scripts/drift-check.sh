#!/usr/bin/env bash
# =============================================================================
# Proves that Argo CD reverts a manual change to an environment.
#
# Usage: scripts/drift-check.sh [NAMESPACE] [TIMEOUT_SECONDS]
#
# Lowers the API's replica floor (the ScaledObject's minReplicaCount) by
# hand, the way an operator might during an incident, and waits for Argo
# CD's self-heal to put back the value from git.
#
# The floor is changed rather than the Deployment's replica count: KEDA owns
# that count, and the manifests leave it out so Argo CD does not fight the
# autoscaler (k8s/components/gitops-environment). A scale command would be
# undone by KEDA, which proves nothing about GitOps.
# =============================================================================

set -euo pipefail

namespace="${1:-vortex-prod}"
timeout="${2:-180}"
scaler=vortex-inference-scaler

desired=$(kubectl get scaledobject -n "${namespace}" "${scaler}" -o jsonpath='{.spec.minReplicaCount}')
drifted=$((desired > 1 ? desired - 1 : desired + 1))

echo "[drift-check] ${namespace}: minReplicaCount is ${desired} in git; setting it to ${drifted} by hand"
kubectl patch scaledobject -n "${namespace}" "${scaler}" --type merge \
  -p "{\"spec\":{\"minReplicaCount\":${drifted}}}" > /dev/null
start=${SECONDS}

while [ $((SECONDS - start)) -lt "${timeout}" ]; do
  current=$(kubectl get scaledobject -n "${namespace}" "${scaler}" -o jsonpath='{.spec.minReplicaCount}')
  if [ "${current}" = "${desired}" ]; then
    echo "[drift-check] ${namespace}: Argo CD restored minReplicaCount=${desired} after $((SECONDS - start))s"
    kubectl get events -n argocd --field-selector involvedObject.name="${namespace}" \
      --sort-by=.lastTimestamp 2>/dev/null | tail -3 || true
    exit 0
  fi
  sleep 2
done
echo "[drift-check] ${namespace}: still ${current} after ${timeout}s; self-heal did not revert the change" >&2
exit 1
