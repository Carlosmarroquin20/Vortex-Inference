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
# =============================================================================

set -euo pipefail

MODELS_DIR="./models"
MODEL_FILENAME="phi-3-mini-4k-instruct-q4_k_m.gguf"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILENAME}"

# Hugging Face repository for Phi-3-mini GGUF variants.
# The microsoft/Phi-3-mini-4k-instruct-gguf repo is the official source —
# using community re-uploads risks weight tampering.
HF_REPO="microsoft/Phi-3-mini-4k-instruct-gguf"
HF_FILENAME="Phi-3-mini-4k-instruct-q4.gguf"

mkdir -p "${MODELS_DIR}"

if [ -f "${MODEL_PATH}" ]; then
  echo "[INFO] Model already exists at ${MODEL_PATH}. Skipping download."
  echo "[INFO] File size: $(du -sh "${MODEL_PATH}" | cut -f1)"
  exit 0
fi

echo "[INFO] Downloading ${MODEL_FILENAME} from Hugging Face..."
echo "[INFO] Repository: ${HF_REPO}"
echo "[INFO] Expected size: ~2.4GB"
echo ""

# Prefer huggingface_hub CLI if available — handles authentication tokens,
# resume on partial downloads, and SHA256 verification automatically.
if command -v huggingface-cli &>/dev/null; then
  echo "[INFO] Using huggingface-cli for download with integrity verification."
  huggingface-cli download \
    "${HF_REPO}" \
    "${HF_FILENAME}" \
    --local-dir "${MODELS_DIR}" \
    --local-dir-use-symlinks False

  # Rename to the canonical filename expected by the ConfigMap
  if [ -f "${MODELS_DIR}/${HF_FILENAME}" ] && [ "${HF_FILENAME}" != "${MODEL_FILENAME}" ]; then
    mv "${MODELS_DIR}/${HF_FILENAME}" "${MODEL_PATH}"
  fi

else
  # Fallback to wget/curl for environments without huggingface-cli.
  # Note: this method does not verify file integrity — run a manual SHA256
  # check against the hash published in the Hugging Face repository.
  echo "[WARN] huggingface-cli not found. Falling back to wget."
  echo "[WARN] Install with: pip install huggingface_hub for integrity verification."

  HF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILENAME}"

  if command -v wget &>/dev/null; then
    wget --show-progress -O "${MODEL_PATH}" "${HF_URL}"
  elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${MODEL_PATH}" "${HF_URL}"
  else
    echo "[ERROR] Neither wget nor curl is available. Install one and retry."
    exit 1
  fi
fi

echo ""
echo "[INFO] Download complete."
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] File size:  $(du -sh "${MODEL_PATH}" | cut -f1)"
echo ""
echo "[INFO] Next step: copy to Minikube node:"
echo "  minikube cp ${MODEL_PATH} /data/models/${MODEL_FILENAME}"
