#!/usr/bin/env bash
# =============================================================================
# Smoke test of one GitOps environment, through its Service.
#
# Usage: scripts/smoke-test.sh NAMESPACE [EXPECTED_IMAGE]
#
# Passes when:
#   - every inference pod runs EXPECTED_IMAGE (tag@digest), when given, so a
#     pass proves the promoted image is the one serving;
#   - /health/ready answers 200;
#   - an authenticated completion returns generated text;
#   - an unauthenticated one is refused with 401.
#
# The API key is read from the environment's Secret, which the Sealed
# Secrets controller decrypted from git: a pass also proves the sealing key
# and the SealedSecrets agree.
# =============================================================================

set -euo pipefail

namespace="${1:?usage: smoke-test.sh NAMESPACE [EXPECTED_IMAGE]}"
expected_image="${2:-}"
port="${SMOKE_PORT:-18080}"

fail() { echo "[smoke-test] ${namespace}: $*" >&2; exit 1; }

serving_images() {
  # Pods being deleted are left out: after a rollout the old pods spend
  # their grace period terminating, still Running, but no longer behind the
  # Service.
  kubectl get pods -n "${namespace}" -l app.kubernetes.io/name=vortex-inference -o json \
    | python3 -c '
import json, sys
pods = json.load(sys.stdin)["items"]
print("\n".join(sorted({
    p["spec"]["containers"][0]["image"]
    for p in pods
    if p["status"].get("phase") == "Running" and not p["metadata"].get("deletionTimestamp")
})))'
}

if [ -n "${expected_image}" ]; then
  for _ in $(seq 60); do
    images=$(serving_images)
    [ "${images}" = "${expected_image}" ] && break
    sleep 2
  done
  [ "${images}" = "${expected_image}" ] \
    || fail "pods run [${images//$'\n'/, }], expected ${expected_image}"
  echo "[smoke-test] ${namespace}: every inference pod runs ${expected_image}"
fi

api_key=$(kubectl get secret -n "${namespace}" vortex-inference-api-key -o jsonpath='{.data.api-key}' | base64 -d)
[ -n "${api_key}" ] || fail "the API key Secret is empty"

kubectl port-forward -n "${namespace}" svc/vortex-inference "${port}:80" > /dev/null 2>&1 &
forward_pid=$!
trap 'kill ${forward_pid} 2>/dev/null || true' EXIT
for _ in $(seq 30); do
  curl -sf "http://127.0.0.1:${port}/health/ready" > /dev/null && break
  sleep 1
done
curl -sf "http://127.0.0.1:${port}/health/ready" > /dev/null || fail "/health/ready did not answer 200"

response=$(curl -s -w '\n%{http_code}' "http://127.0.0.1:${port}/v1/completions" \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}')
status="${response##*$'\n'}"
body="${response%$'\n'*}"
[ "${status}" = "200" ] || fail "completion returned ${status}: ${body}"
text=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["choices"][0]["text"])' "${body}")
[ -n "${text// /}" ] || fail "completion returned no text: ${body}"
echo "[smoke-test] ${namespace}: completion -> ${text}"

status=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/v1/completions" \
  -H "Content-Type: application/json" -d '{"prompt": "x", "max_tokens": 1}')
[ "${status}" = "401" ] || fail "an unauthenticated request returned ${status}, expected 401"

echo "[smoke-test] ${namespace}: passed"
