#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-datasets1}"
GPU_ID="${GPU_ID:-0}"
CHECKPOINT_NAME="datasets1_ODinMJ"
CHECKPOINT_DIR="${BUNDLE_DIR}/checkpoints"
PAIRED_DIR="${BUNDLE_DIR}/outputs/dataset1_v77"
METRIC_DIR="${BUNDLE_DIR}/outputs/metrics"
SOURCE_DIR="${BUNDLE_DIR}"
if [[ "${DATASET_ROOT}" != /* ]]; then
  DATASET_ROOT="${BUNDLE_DIR}/${DATASET_ROOT}"
fi
USER_IQA_ROOT="${IQA_ROOT:-${BUNDLE_DIR}/metrics}"

if [[ "${PYTHON_BIN}" == */* && ! -x "${PYTHON_BIN}" ]] || \
   [[ "${PYTHON_BIN}" != */* && -z "$(command -v "${PYTHON_BIN}" || true)" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}/testA" ]]; then
  echo "dataset1 testA not found: ${DATASET_ROOT}/testA" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/${CHECKPOINT_NAME}/train_opt.json" ]]; then
  echo "Checkpoint options not found under ${CHECKPOINT_DIR}/${CHECKPOINT_NAME}" >&2
  exit 1
fi

mkdir -p "${PAIRED_DIR}" "${METRIC_DIR}"

echo "=== Dataset1 V77 paired full-size inference ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${SOURCE_DIR}" \
  "${PYTHON_BIN}" "${SOURCE_DIR}/scripts/save_paired_test_outputs.py" \
  --checkpoints_dir "${CHECKPOINT_DIR}" \
  --name "${CHECKPOINT_NAME}" \
  --dataroot "${DATASET_ROOT}" \
  --phase test \
  --epoch best_val \
  --results_dir "${PAIRED_DIR}" \
  --gpu_ids 0 \
  --num_threads 0 \
  --pad_multiple 16 \
  | tee "${METRIC_DIR}/save_paired_test_outputs.log"

echo "=== Dataset1 V77 PSNR/SSIM with IQA我自己的 ==="
"${PYTHON_BIN}" "${BUNDLE_DIR}/evaluate_with_user_iqa.py" \
  --iqa_root "${USER_IQA_ROOT}" \
  --reference_dir "${DATASET_ROOT}/testGT" \
  --result_dir "${PAIRED_DIR}" \
  --label "historical_source_paired_fullsize_best_val" \
  --crop_border 4 \
  --output_json "${METRIC_DIR}/user_iqa_historical_source_paired_fullsize.json" \
  --output_csv "${METRIC_DIR}/user_iqa_historical_source_paired_fullsize.csv" \
  | tee "${METRIC_DIR}/user_iqa_historical_source_paired_fullsize.log"

echo "Finished. Results are under ${BUNDLE_DIR}/outputs."
