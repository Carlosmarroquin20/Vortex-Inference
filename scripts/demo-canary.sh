#!/usr/bin/env bash
# =============================================================================
# Canary demonstration: a good release is promoted, a bad one is stopped by
# the analysis, and git revert restores the environment.
#
# Usage: DEMO_BRANCH=<branch> scripts/demo-canary.sh
#
# Everything goes through git, as in production. The cluster's Argo CD must
# track DEMO_BRANCH (make bootstrap-gitops GITOPS_REVISION=<branch>), and
# this script must be able to push to it. The script commits each release
# to the prod overlay on that branch, in a temporary worktree so the
# current checkout is left alone, and Argo CD applies it:
#
#   1. good release  a new pod template (RELEASE_ID=good): the canary goes
#                    10% -> analysis -> 50% -> analysis -> 100%
#   2. bad release   the same plus VORTEX_FAULT_RATE=0.5, half the
#                    completions fail: the first analysis fails and Argo
#                    Rollouts aborts, scaling the canary away; the stable
#                    pods served throughout
#   3. git revert    of the bad commit: the desired state is the stable
#                    release again and the Rollout returns to Healthy
#
# Client traffic runs in the cluster for the whole demonstration
# (scripts/traffic.sh), because the analysis refuses to judge a canary
# without requests.
#
# Exits non-zero if any step does not end as described, so CI can use it as
# an end-to-end test. Writes a timestamped transcript to DEMO_LOG.
# =============================================================================

set -euo pipefail

branch="${DEMO_BRANCH:?DEMO_BRANCH must name the branch the cluster tracks}"
namespace="${DEMO_NAMESPACE:-vortex-prod}"
overlay="k8s/overlays/${namespace#vortex-}"
log="${DEMO_LOG:-demo-canary.log}"
timeout="${DEMO_STEP_TIMEOUT:-900}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: > "${log}"
start=${SECONDS}
say() {
  local line
  line=$(printf '[%4ds] %s' "$((SECONDS - start))" "$*")
  echo "${line}"
  printf '%s\t%s\n' "$((SECONDS - start))" "$*" >> "${log}"
}
banner() { say ""; say "=== $* ==="; }

# --- Rollout state -------------------------------------------------------------
status() { python3 "${root}/scripts/rollout_status.py" "${namespace}" "$@"; }

# Polls until the Rollout reaches the wanted phase, logging each change.
await_phase() {
  local want="$1" deadline=$((SECONDS + timeout)) last=""
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    local now
    now=$(status)
    if [ "${now}" != "${last}" ]; then say "${now}"; last="${now}"; fi
    [ "$(status phase)" = "${want}" ] && return 0
    sleep 5
  done
  say "timed out waiting for ${want}"
  return 1
}

# --- Git ------------------------------------------------------------------------
worktree="$(mktemp -d)"
cleanup() {
  bash "${root}/scripts/traffic.sh" stop "${namespace}" > /dev/null 2>&1 || true
  git -C "${root}" worktree remove --force "${worktree}" > /dev/null 2>&1 || true
}
trap cleanup EXIT
git -C "${root}" fetch --quiet origin "${branch}"
git -C "${root}" worktree add --quiet --detach "${worktree}" "origin/${branch}"

publish() {
  # GitHub occasionally answers a push with a transient 5xx; a release must
  # not fail on that, so the push is retried with a short backoff. Nothing
  # else writes to this branch, so a retry never needs a rebase.
  local attempt
  for attempt in 1 2 3 4; do
    git -C "${worktree}" push --quiet origin "HEAD:${branch}" && break
    [ "${attempt}" -lt 4 ] || { say "FAIL: could not push to ${branch}"; return 1; }
    say "push rejected (attempt ${attempt}); retrying"
    sleep $((attempt * 5))
  done
  kubectl annotate applications.argoproj.io -n argocd "${namespace}" \
    argocd.argoproj.io/refresh=normal --overwrite > /dev/null
  say "pushed $(git -C "${worktree}" log -1 --format='%h %s') to ${branch}; Argo CD syncing"
}

# Commits a release to the overlay: environment entries added to the
# inference container, which change the pod template and so start a canary.
# A JSON patch against the Rollout, because in the overlay the component has
# already turned the Deployment into one.
release() {
  local name="$1"
  shift
  {
    for entry in "RELEASE_ID=${name}" "$@"; do
      printf -- '- op: add\n  path: /spec/template/spec/containers/0/env/-\n'
      printf -- '  value:\n    name: %s\n    value: "%s"\n' "${entry%%=*}" "${entry#*=}"
    done
  } > "${worktree}/${overlay}/demo-release.yaml"
  if ! grep -q demo-release.yaml "${worktree}/${overlay}/kustomization.yaml"; then
    (cd "${worktree}/${overlay}" \
      && kustomize edit add patch --path demo-release.yaml --kind Rollout --name vortex-inference)
  fi
  git -C "${worktree}" add "${overlay}"
  git -C "${worktree}" commit --quiet -m "demo(${namespace#vortex-}): release ${name}"
  publish
}

stable_hash() { status stable; }

# --- Demonstration --------------------------------------------------------------
banner "Starting point"
await_phase Healthy
initial=$(stable_hash)
say "stable revision ${initial}"
bash "${root}/scripts/traffic.sh" start "${namespace}" 3600 2 > /dev/null
say "client traffic started: 2 concurrent clients, short completions"

banner "1. Good release"
release good
sleep 10
await_phase Healthy
good=$(stable_hash)
[ "${good}" != "${initial}" ] || { say "FAIL: the good release did not become stable"; exit 1; }
say "PASS: good release ${good} promoted to 100% after two passing analyses"

banner "2. Bad release (VORTEX_FAULT_RATE=0.5)"
release bad VORTEX_FAULT_RATE=0.5
sleep 10
await_phase Degraded
[ "$(stable_hash)" = "${good}" ] || { say "FAIL: the bad release replaced the stable one"; exit 1; }
say "controller: $(status message)"
say "PASS: analysis failed, rollout aborted; stable ${good} kept serving"

banner "3. git revert"
git -C "${worktree}" revert --no-edit HEAD > /dev/null
publish
sleep 10
await_phase Healthy
[ "$(stable_hash)" = "${good}" ] || { say "FAIL: revert did not restore the good release"; exit 1; }
say "PASS: desired state is the good release again; Rollout Healthy"

bash "${root}/scripts/traffic.sh" stop "${namespace}" > /dev/null
say "demonstration complete in $((SECONDS - start))s"
