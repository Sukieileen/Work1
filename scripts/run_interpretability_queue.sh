#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research/MetaLog"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${ROOT}/outputs/experiments/interpretability"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate work1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${ROOT}"

FULL_H2D_RUN="hdfs_to_hpc_sr065_parserfree_na_v1_bimamba_do0p1_st96_cv4_ex4_ps0p5_plw0p1_20260517"
FULL_H2D_CKPT="${ROOT}/outputs/models/clean/HPC_sr065_parser_free/model/${FULL_H2D_RUN}_phaseC_best.pt"

FULL_H2H_RUN="hpc_to_hdfs_parserfree_na_v1_bimamba_do0p1_st96_cv4_ex4_ps0p5_plw0p1_sr0p65_20260512"
FULL_H2H_CKPT="${ROOT}/outputs/models/clean/HDFS_parser_free/model/${FULL_H2H_RUN}_phaseC_best.pt"

{
  echo "[E2] routing analysis start: ${DATE_TAG}"
  python scripts/analyze_moe_routing.py \
    --direction hdfs_to_hpc_sr065 \
    --parser parser_free \
    --checkpoint "${FULL_H2D_CKPT}" \
    --splits target_train target_test \
    --use-moe --use-normality-anchor --router-use-distance \
    --prototype-scale 0.5 --prototype-loss-weight 0.1 --prototype-sep-weight 1e-3 \
    --mamba_state 96 --mamba_conv 4 --mamba_expand 4 --dropout 0.1 \
    --output_dir "${OUT_ROOT}" 2>&1

  python scripts/analyze_moe_routing.py \
    --direction hpc_to_hdfs \
    --parser parser_free \
    --checkpoint "${FULL_H2H_CKPT}" \
    --source_dataset_override HPC \
    --splits target_train target_test \
    --use-moe --use-normality-anchor --router-use-distance \
    --prototype-scale 0.5 --prototype-loss-weight 0.1 --prototype-sep-weight 1e-3 \
    --mamba_state 96 --mamba_conv 4 --mamba_expand 4 --dropout 0.1 \
    --output_dir "${OUT_ROOT}" 2>&1

  echo "[E3] normality distance analysis start: ${DATE_TAG}"
  python scripts/analyze_normality_distance.py \
    --direction hdfs_to_hpc_sr065 \
    --parser parser_free \
    --checkpoint "${FULL_H2D_CKPT}" \
    --splits target_test \
    --use-moe --use-normality-anchor --router-use-distance \
    --prototype-scale 0.5 --prototype-loss-weight 0.1 --prototype-sep-weight 1e-3 \
    --mamba_state 96 --mamba_conv 4 --mamba_expand 4 --dropout 0.1 \
    --output_dir "${OUT_ROOT}" 2>&1

  python scripts/analyze_normality_distance.py \
    --direction hpc_to_hdfs \
    --parser parser_free \
    --checkpoint "${FULL_H2H_CKPT}" \
    --source_dataset_override HPC \
    --splits target_test \
    --use-moe --use-normality-anchor --router-use-distance \
    --prototype-scale 0.5 --prototype-loss-weight 0.1 --prototype-sep-weight 1e-3 \
    --mamba_state 96 --mamba_conv 4 --mamba_expand 4 --dropout 0.1 \
    --output_dir "${OUT_ROOT}" 2>&1
} | tee "${LOG_DIR}/interpretability_${DATE_TAG}.log"

echo "Interpretability finished. Log: ${LOG_DIR}/interpretability_${DATE_TAG}.log"
