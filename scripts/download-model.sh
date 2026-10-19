#!/usr/bin/env bash
# =============================================================================
# Vortex Inference — GGUF model download
#
# Downloads the Phi-3-mini-4k-instruct Q4_K_M quantized model from
# Hugging Face Hub. The Q4_K_M quantization scheme offers the best
# perplexity-to-size ratio for 4-bit formats: the K-quant variant applies
# higher precision to attention and feed-forward weight matrices that have
# the greatest impact on output quality.
#
# Target: ~2.4GB download, ~2.8GB RAM footprint after mmap loading.
#
# Integrity: the file is fetched from a pinned repository commit, not from
# the moving `main` branch, and its SHA-256 is checked against the value
# published for that commit before it is accepted. The download goes to a
# .part file that only replaces the final path once the hash matches, so an
# interrupted or tampered download can never be mistaken for a valid model.
# An existing file is re-verified instead of being trusted by name.
#
# Every value below can be overridden through the environment, e.g. to fetch
# a different quantization: HF_FILENAME=... MODEL_SHA256=... bash scripts/download-model.sh
# Set HF_TOKEN to authenticate if anonymous downloads are rate-limited.
# =============================================================================

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
MODEL_FILENAME="${MODEL_FILENAME:-phi-3-mini-4k-instruct-q4_k_m.gguf}"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILENAME}"

# The microsoft/ repository is the official source; community re-uploads
# carry a real risk of tampered weights.
HF_REPO="${HF_REPO:-microsoft/Phi-3-mini-4k-instruct-gguf}"
HF_FILENAME="${HF_FILENAME:-Phi-3-mini-4k-instruct-q4.gguf}"

# Repository commit, plus the file's SHA-256 (its Git LFS object id) and size
# in bytes at that commit, all taken from the Hugging Face API. Update the
# three together.
HF_REVISION="${HF_REVISION:-a64113399c2f6b8ad3e11c394733a2ddadaa7f33}"
MODEL_SHA256="${MODEL_SHA256:-8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef}"
MODEL_SIZE="${MODEL_SIZE:-2393231072}"

HF_URL="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/${HF_FILENAME}"

sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        echo "[ERROR] Neither sha256sum nor shasum is available." >&2
        exit 1
    fi
}

verify() {
    local actual
    echo "[INFO] Verifying SHA-256 of $1 ..."
    actual=$(sha256_of "$1")
    if [ "${actual}" != "${MODEL_SHA256}" ]; then
        echo "[ERROR] Checksum mismatch for $1" >&2
        echo "[ERROR]   expected ${MODEL_SHA256}" >&2
        echo "[ERROR]   actual   ${actual}" >&2
        return 1
    fi
    echo "[INFO] Checksum OK."
}

mkdir -p "${MODELS_DIR}"

if [ -f "${MODEL_PATH}" ]; then
    echo "[INFO] Model already present at ${MODEL_PATH}."
    if verify "${MODEL_PATH}"; then
        exit 0
    fi
    echo "[ERROR] The existing file does not match the pinned checksum." >&2
    echo "[ERROR] Delete it and re-run this script to download a fresh copy." >&2
    exit 1
fi

PART_PATH="${MODEL_PATH}.part"
AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
    AUTH_HEADER=(--header "Authorization: Bearer ${HF_TOKEN}")
fi

part_size() {
    if [ -f "${PART_PATH}" ]; then wc -c < "${PART_PATH}" | tr -d ' '; else echo 0; fi
}

# A .part file that already has the full size was downloaded by a previous
# run that stopped before verification. Asking the server to resume past the
# end returns 416, so go straight to verification instead.
if [ "$(part_size)" = "${MODEL_SIZE}" ]; then
    echo "[INFO] Found a complete download from a previous run at ${PART_PATH}."
else
    echo "[INFO] Downloading ${HF_FILENAME}"
    echo "[INFO]   repository: ${HF_REPO}"
    echo "[INFO]   revision:   ${HF_REVISION}"
    echo ""
    # Both clients resume a partial .part file from an interrupted run. On
    # failure the partial file is kept so the next run can resume it.
    # ${AUTH_HEADER[@]+...} expands to nothing for an empty array without
    # tripping `set -u` on bash 3.2 (the macOS system shell).
    if command -v curl &>/dev/null; then
        curl -fL --retry 3 --continue-at - --progress-bar \
            ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} -o "${PART_PATH}" "${HF_URL}"
    elif command -v wget &>/dev/null; then
        wget --continue --tries=3 --show-progress \
            ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} -O "${PART_PATH}" "${HF_URL}"
    else
        echo "[ERROR] Neither curl nor wget is available. Install one and retry." >&2
        exit 1
    fi
fi

if ! verify "${PART_PATH}"; then
    # A resumed download appended to a corrupt prefix cannot be repaired by
    # resuming again, so the partial file is discarded.
    rm -f "${PART_PATH}"
    echo "[ERROR] Downloaded file discarded. Re-run to download again." >&2
    exit 1
fi

mv "${PART_PATH}" "${MODEL_PATH}"

echo ""
echo "[INFO] Download complete."
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] File size:  $(du -sh "${MODEL_PATH}" | cut -f1)"
echo ""
echo "[INFO] Next step: copy to Minikube node:"
echo "  minikube cp ${MODEL_PATH} /data/models/${MODEL_FILENAME}"
