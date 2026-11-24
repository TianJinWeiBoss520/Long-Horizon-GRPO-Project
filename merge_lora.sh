#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-pretrained/Qwen/Qwen3-4B-Instruct-2507}"
: "${SFT_ADAPTER_DIR:?Set SFT_ADAPTER_DIR to the saved LoRA checkpoint directory}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${PROJECT_ROOT}/saved/output/qwen3_4b_sft_merged}"

swift export \
  --model "${BASE_MODEL}" \
  --adapters "${SFT_ADAPTER_DIR}" \
  --output_dir "${MERGED_MODEL_DIR}" \
  --merge_lora true \
  --safe_serialization true
