#!/usr/bin/env bash
# =============================================================================
# Points one GitOps environment at an image, as a commit on main.
#
# Usage: scripts/promote.sh ENVIRONMENT IMAGE_REF
#   ENVIRONMENT  staging | prod (a directory under k8s/overlays/)
#   IMAGE_REF    registry/name:tag@sha256:digest
#
# Runs `kustomize edit set image` in the environment's overlay, commits the
# change and pushes it. The commit is the deployment: Argo CD applies it.
# Prints the SHA of the commit that carries the change, or of HEAD when the
# environment already ran this image (a re-run is a no-op, not an error).
#
# Pushes race with other merges to main, so a rejected push is rebased onto
# the new main and retried. A conflict cannot happen unless someone edited
# the same image lines by hand, and then failing loudly is the right outcome.
#
# Requires kustomize and a checkout whose credentials can push to main.
# =============================================================================

set -euo pipefail

environment="${1:?usage: promote.sh ENVIRONMENT IMAGE_REF}"
image_ref="${2:?usage: promote.sh ENVIRONMENT IMAGE_REF}"
overlay="k8s/overlays/${environment}"

[ -f "${overlay}/kustomization.yaml" ] || { echo "no overlay at ${overlay}" >&2; exit 1; }
[[ "${image_ref}" == *:*@sha256:* ]] || { echo "IMAGE_REF must be name:tag@sha256:digest" >&2; exit 1; }

tag="${image_ref%@*}"
tag="${tag##*:}"

(cd "${overlay}" && kustomize edit set image "vortex-inference=${image_ref}")

if git diff --quiet -- "${overlay}"; then
  echo "${environment} already runs ${image_ref}" >&2
  git rev-parse HEAD
  exit 0
fi

git add -- "${overlay}/kustomization.yaml"
git commit --quiet -m "deploy(${environment}): ${tag}" -m "Image: ${image_ref}"

for attempt in 1 2 3 4 5; do
  if git push --quiet origin HEAD:main; then
    git rev-parse HEAD
    exit 0
  fi
  echo "push rejected (attempt ${attempt}); rebasing onto origin/main" >&2
  git pull --quiet --rebase origin main
  sleep $((attempt * 2))
done
echo "could not push the ${environment} promotion" >&2
exit 1
