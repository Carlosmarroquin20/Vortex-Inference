# 0001. Publish signed multi-architecture images to GHCR

- Status: accepted
- Date: 2026-11-09

## Context

CI built the image and scanned it, but never published it. `make deploy`
loads `vortex-inference:1.0.0` into minikube with `imagePullPolicy: Never`, so
nothing outside one laptop could run a given commit, and nothing could tell
which commit a running image came from. Continuous deployment needs an
artifact that:

- lives in a registry the clusters can pull from without credentials;
- is immutable and addressable by content, so the image that passed staging
  is the image prod runs;
- carries proof of origin that a cluster or an auditor can check without
  trusting the registry;
- runs on both x86 and ARM nodes. ARM instances (Graviton, Cobalt, Ampere)
  are priced below comparable x86 instances per vCPU, and CPU time is this
  service's main cost.

## Decision

`.github/workflows/release.yml` publishes `ghcr.io/carlosmarroquin20/vortex-inference`.

**Build.** One native build per architecture: `ubuntu-24.04` for amd64 and
`ubuntu-24.04-arm` for arm64. Each is pushed by digest, untagged. A final job
joins the two into one multi-architecture index and tags only the index.
llama.cpp compiles for a fixed CPU baseline per architecture, set in the
Dockerfile: x86-64-v3 for amd64, ARMv8.2-A with dot product and FP16 for
arm64. The baseline never comes from the build machine.

**Tags.** A push to main produces `sha-<7 chars>` and `edge`. A `v*` tag does
not rebuild. It adds `X.Y.Z`, `X.Y` and `latest` to the index main already
built for that commit, and checks that the new tags resolve to the same
digest. Environments pin `tag@digest`. The tag is for humans, and the
digest is what the kubelet pulls.

**Gate.** Trivy scans each architecture before anything is pushed. A
CRITICAL finding that has a fix fails the release. Fixable HIGH findings
are only listed. They still block pull requests in `ci.yml`, where they can
be fixed before merge. A CVE published after a merge should not strand a
commit that already passed review. Unfixed findings are excluded
everywhere, because there is nothing to upgrade to.

**Evidence.** The pattern follows Aegis-RAG's release workflow:

- `cosign sign --recursive` signs keylessly, through GitHub OIDC and
  Fulcio, with a Rekor transparency log entry. It signs the index and each
  platform manifest.
- `actions/attest-build-provenance` attests SLSA build provenance for the
  index.
- Syft produces an SPDX SBOM per platform. It is signed with
  `cosign attest --type spdxjson`, and BuildKit also attaches its own SBOM
  and provenance.

A `verify` job runs on a clean runner of each architecture. It checks the
signature, the SBOM attestation and the provenance against this workflow's
identity (`release.yml` on `refs/heads/main` or `refs/tags/v*`). It then
starts the image on that architecture. The release is only green if
verification passes.

## Alternatives considered

- **QEMU emulation for arm64 (`docker/setup-qemu-action`).** This is the usual
  approach, but here it compiles llama.cpp and OpenBLAS bindings inside an
  emulator. That takes hours instead of about 8 minutes, past any
  reasonable job timeout. Native ARM runners are free for public
  repositories.
- **Cross-compiling arm64 on the x86 runner.** This is fast, but it needs a
  cross toolchain and a cross-built OpenBLAS, and `pip wheel` for a native
  extension does not cross-compile without a lot of glue. The build-time
  BLAS check in the Dockerfile also has to run on the target CPU, so it
  would be lost.
- **amd64 only.** This is simpler, and it is what Aegis-RAG does because
  PyTorch makes ARM builds expensive there. It rules out the cheaper ARM
  node pools for a service whose main cost is CPU time.
- **`GGML_NATIVE=ON` or `GGML_CPU_ALL_VARIANTS`.** Native builds produce an
  image that crashes with SIGILL on an older node. Building all variants and
  loading backends dynamically gives the best kernel on every CPU. However,
  it multiplies build time and image size, and it changes how llama-cpp-python
  loads its libraries. The fixed baselines cover every CPU this would
  realistically run on.
- **Signing with a long-lived key (`cosign generate-key-pair`).** This works
  offline and without Rekor, but the key has to be stored as a secret,
  rotated and protected. Keyless signing ties the signature to the workflow
  identity, so there is nothing to leak. Verification then pins the
  repository and workflow path instead of a public key.
- **Notation (Notary v2).** It does the same job, but its verification
  ecosystem (Ratify, Kyverno support) is narrower than Sigstore's. It also
  needs a certificate authority or a key.
- **Rebuilding on every version tag.** This is common, but the tag would
  then point to a digest that never went through staging and prod. BuildKit
  output is not bit-for-bit reproducible, so even the same commit gives a
  new digest. Retagging releases exactly what was tested.
- **Docker Hub or a cloud registry (ECR, Artifact Registry).** These need a
  separate account and credentials stored as secrets. GHCR authenticates
  with the workflow's own `GITHUB_TOKEN` and is free for public images.

## Consequences

- Any commit on main can be pulled by digest, and its signature, SBOM and
  provenance can be checked with the commands in the README.
- A release costs two runners for about 10 minutes when the cache is cold,
  and about 2 minutes when it is warm.
- Pull requests that touch the image build both architectures without
  publishing. An arm64 compile error then shows up before merge, not on
  main.
- The arm64 baseline excludes ARMv8.0 hardware such as Raspberry Pi 4 and
  Graviton1. Those CPUs would need `GGML_CPU_ARM_ARCH=armv8-a` and would run
  noticeably slower.
