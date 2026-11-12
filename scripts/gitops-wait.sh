#!/usr/bin/env bash
# =============================================================================
# Waits until Argo CD reports Applications Synced and Healthy.
#
# Usage: scripts/gitops-wait.sh [TIMEOUT_SECONDS] [REVISION] [APP...]
#
#   TIMEOUT_SECONDS  give up after this long (default 900)
#   REVISION         also require each app to be synced to this commit SHA;
#                    "-" or empty accepts any revision
#   APP...           Applications to wait for (default: every Application)
#
# Asks Argo CD to refresh first, so a commit pushed a moment ago is picked
# up now rather than at the next reconciliation. On timeout it prints the
# state of every Application and the events of unhealthy resources, which
# is what a failed CI run needs to be diagnosed from its log alone.
# =============================================================================

set -euo pipefail

timeout="${1:-900}"
revision="${2:-}"
[ "${revision}" = "-" ] && revision=""
shift 2 2>/dev/null || shift $#
apps=("$@")

ns=argocd
deadline=$((SECONDS + timeout))

list_apps() {
  if [ ${#apps[@]} -gt 0 ]; then
    printf '%s\n' "${apps[@]}"
  else
    kubectl get applications.argoproj.io -n "${ns}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
  fi
}

state() {
  kubectl get applications.argoproj.io -n "${ns}" "$1" \
    -o jsonpath='{.status.sync.status}/{.status.health.status}/{.status.sync.revision}' 2>/dev/null || echo "missing//"
}

for app in $(list_apps); do
  kubectl annotate applications.argoproj.io -n "${ns}" "${app}" \
    argocd.argoproj.io/refresh=normal --overwrite > /dev/null 2>&1 || true
done

while :; do
  pending=()
  for app in $(list_apps); do
    IFS=/ read -r sync health synced_revision <<< "$(state "${app}")"
    if [ "${sync}" != "Synced" ] || [ "${health}" != "Healthy" ] \
      || { [ -n "${revision}" ] && [ "${synced_revision}" != "${revision}" ]; }; then
      pending+=("${app}=${sync:-?}/${health:-?}@${synced_revision:0:7}")
    fi
  done
  # The app list itself can be empty right after bootstrap, before the root
  # app has created its children.
  if [ ${#pending[@]} -eq 0 ] && [ -n "$(list_apps)" ]; then
    echo "[gitops-wait] all Applications Synced and Healthy${revision:+ at ${revision:0:7}}"
    exit 0
  fi
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    echo "[gitops-wait] timed out after ${timeout}s; still pending: ${pending[*]}" >&2
    kubectl get applications.argoproj.io -n "${ns}" -o wide >&2 || true
    # Every resource that is not Healthy, with Argo CD's reason, for each
    # Application that is not (resource health is persisted in the status,
    # see gitops/bootstrap/argocd/argocd-cmd-params-cm.yaml).
    kubectl get applications.argoproj.io -n "${ns}" -o json | python3 -c '
import json, sys
for app in json.load(sys.stdin)["items"]:
    status = app.get("status", {})
    if status.get("health", {}).get("status") == "Healthy":
        continue
    print(app["metadata"]["name"] + ":", status.get("health", {}))
    for condition in status.get("conditions", []):
        print("  condition", condition.get("type"), condition.get("message"))
    for resource in status.get("resources", []):
        health = resource.get("health")
        if health and health.get("status") != "Healthy":
            where = resource.get("namespace", "") + "/" + resource["name"]
            print("  ", resource["kind"], where, health.get("status"), health.get("message", ""))
' >&2 || true
    kubectl get events -A --field-selector type=Warning --sort-by=.lastTimestamp 2>/dev/null | tail -40 >&2 || true
    exit 1
  fi
  echo "[gitops-wait] waiting: ${pending[*]:-for the root Application to create its children}"
  sleep 15
done
