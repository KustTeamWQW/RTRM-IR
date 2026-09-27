#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-datasets1}"
GPU_ID="${GPU_ID:-0}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-reproduced_training/checkpoints}"
NAME="${NAME:-datasets1_ODinMJ}"

if [[ "${PYTHON_BIN}" == */* && ! -x "${PYTHON_BIN}" ]] || \
   [[ "${PYTHON_BIN}" != */* && -z "$(command -v "${PYTHON_BIN}" || true)" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${DATASET_ROOT}" != /* ]]; then
  DATASET_ROOT="${BUNDLE_DIR}/${DATASET_ROOT}"
fi
if [[ "${CHECKPOINTS_DIR}" != /* ]]; then
  CHECKPOINTS_DIR="${BUNDLE_DIR}/${CHECKPOINTS_DIR}"
fi
if [[ ! -d "${DATASET_ROOT}/trainA" ]]; then
  echo "dataset1 trainA not found: ${DATASET_ROOT}/trainA" >&2
  exit 1
fi

mkdir -p "${CHECKPOINTS_DIR}"
cd "${BUNDLE_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${BUNDLE_DIR}" \
  "${PYTHON_BIN}" "${BUNDLE_DIR}/train_from_bundle.py" \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --name "${NAME}" \
  --gpu-id 0
