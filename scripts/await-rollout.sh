#!/usr/bin/env bash
# =============================================================================
# Waits for an environment's canary to finish, logging each step.
#
# Usage: scripts/await-rollout.sh NAMESPACE COMMIT [TIMEOUT_SECONDS]
#
# First waits for Argo CD to have synced COMMIT into the environment, then
# follows the API's Rollout until it is Healthy with the commit's pod
# template as the stable revision, which means every analysis passed.
# Fails as soon as the Rollout is Degraded (an analysis failed or the canary
# never became ready) and prints the failing AnalysisRun.
# =============================================================================

set -euo pipefail

namespace="${1:?usage: await-rollout.sh NAMESPACE COMMIT [TIMEOUT_SECONDS]}"
commit="${2:?usage: await-rollout.sh NAMESPACE COMMIT [TIMEOUT_SECONDS]}"
timeout="${3:-1200}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
status() { python3 "${root}/scripts/rollout_status.py" "${namespace}" "$@"; }

kubectl annotate applications.argoproj.io -n argocd "${namespace}" \
  argocd.argoproj.io/refresh=normal --overwrite > /dev/null
deadline=$((SECONDS + timeout))
until [ "$(kubectl get applications.argoproj.io -n argocd "${namespace}" \
  -o jsonpath='{.status.sync.revision}')" = "${commit}" ]; do
  [ "${SECONDS}" -lt "${deadline}" ] || { echo "[await-rollout] ${commit:0:7} never synced" >&2; exit 1; }
  sleep 5
done
echo "[await-rollout] ${namespace}: Argo CD synced ${commit:0:7}"

last=""
while [ "${SECONDS}" -lt "${deadline}" ]; do
  now=$(status)
  if [ "${now}" != "${last}" ]; then echo "[await-rollout] ${now}"; last="${now}"; fi
  rollout=$(kubectl get rollout -n "${namespace}" vortex-inference \
    -o jsonpath='{.status.phase}/{.status.stableRS}/{.status.currentPodHash}/{.status.observedGeneration}/{.metadata.generation}')
  IFS=/ read -r phase stable current observed generation <<< "${rollout}"
  if [ "${phase}" = "Degraded" ]; then
    echo "[await-rollout] ${namespace}: rollout aborted: $(status message)" >&2
    kubectl get analysisruns -n "${namespace}" -l "rollouts-pod-template-hash=${current}" -o yaml >&2 || true
    exit 1
  fi
  if [ "${phase}" = "Healthy" ] && [ "${stable}" = "${current}" ] && [ "${observed}" = "${generation}" ]; then
    echo "[await-rollout] ${namespace}: ${stable} is stable at 100%"
    exit 0
  fi
  sleep 5
done
echo "[await-rollout] ${namespace}: timed out after ${timeout}s" >&2
exit 1
