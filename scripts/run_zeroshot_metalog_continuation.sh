#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${ROOT}/MetaLog/outputs/experiments/zeroshot/results_zeroshot_knownmix_v3_raw.csv}"
LOG_DIR="${LOG_DIR:-${ROOT}/MetaLog/outputs/experiments/zeroshot/logs_knownmix_v3}"
mkdir -p "${LOG_DIR}"

SPIRIT_MAX_LINES="${SPIRIT_MAX_LINES:-4700000}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
cd "${ROOT}/MetaLog"

run_cmd() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${DATE_TAG}_${name}.log"
  echo "[$(date '+%F %T')] START ${name}" | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE ${name}" | tee -a "${log_file}"
}

RUN_NAME="${RUN_NAME:-hdfs30_hpc065_known_mix_pf_na_zeroshot_20260519_195514}"
MODEL_DIR="${ROOT}/MetaLog/outputs/models/clean/HDFS30+HPC_sr065_parser_free/model"
PHASEB_LAST="${MODEL_DIR}/${RUN_NAME}_phaseB_last.pt"
PHASEB_BEST="${MODEL_DIR}/${RUN_NAME}_phaseB_best.pt"
PHASEB_TH="${MODEL_DIR}/${RUN_NAME}_phaseB_best.threshold.txt"

if [ ! -s "${PHASEB_LAST}" ] || [ ! -s "${PHASEB_BEST}" ] || [ ! -s "${PHASEB_TH}" ]; then
  echo "Missing phaseB artifacts under ${MODEL_DIR}" >&2
  exit 1
fi

run_cmd prepare_datasets \
  conda run -n work1 python "${ROOT}/MetaLog/scripts/prepare_zeroshot_datasets.py" \
    --datasets OpenStack SPIRIT \
    --spirit_max_lines "${SPIRIT_MAX_LINES}"

run_cmd resume_phaseC \
  conda run -n work1 python "${ROOT}/MetaLog/approaches/MetaLog.py" \
    --mode train \
    --direction hdfs30_hpc065_known_mix \
    --parser parser_free \
    --plm_model bert-base-uncased \
    --plm_max_length 64 \
    --plm_batch_size 64 \
    --plm_pooling mean \
    --protocol clean \
    --source_train_ratio 0.30 \
    --target_normal_ratio 1.0 \
    --target_anomaly_ratio 1.0 \
    --known_dev_ratio 0.10 \
    --backbone bimamba \
    --dropout 0.1 \
    --mamba_state 96 \
    --mamba_conv 4 \
    --mamba_expand 4 \
    --use-moe \
    --use-normality-anchor \
    --prototype-scale 0.5 \
    --prototype-loss-weight 0.1 \
    --prototype-sep-weight 1e-3 \
    --prototype-margin-global 1.0 \
    --prototype-margin-expert 1.0 \
    --prototype-target-normal-only \
    --router-use-distance \
    --warmup_epochs 0 \
    --joint_epochs 0 \
    --calibration_epochs 3 \
    --checkpoint "${PHASEB_LAST}" \
    --mode train \
    --run_name "${RUN_NAME}" \
    --zero_epoch_metrics_file "${ROOT}/MetaLog/outputs/experiments/zeroshot/results_zeroshot_knownmix_v3_epoch_raw.csv" \
    --zero_epoch_targets OpenStack,SPIRIT \
    --zero_method_name MetaLog \
    --spirit_max_lines "${SPIRIT_MAX_LINES}"

for target in OpenStack SPIRIT; do
  run_cmd "metalog_${target}" \
    conda run -n work1 python "${ROOT}/MetaLog/scripts/run_zeroshot_eval.py" \
      --known_direction hdfs30_hpc065_known_mix \
      --zero_target "${target}" \
      --checkpoint "${MODEL_DIR}/${RUN_NAME}_phaseC_best.pt" \
      --threshold_file "${MODEL_DIR}/${RUN_NAME}_phaseC_best.threshold.txt" \
      --method_name "MetaLog" \
      --threshold_source "known-mix-dev selection" \
      --output "${OUT}" \
      --use-moe \
      --use-normality-anchor \
      --router-use-distance \
      --prototype-scale 0.5 \
      --prototype-loss-weight 0.1 \
      --prototype-sep-weight 1e-3 \
      --mamba_state 96 \
      --mamba_conv 4 \
      --mamba_expand 4 \
      --dropout 0.1 \
      --lambda_target 4.0 \
      --spirit_max_lines "${SPIRIT_MAX_LINES}"
done

run_cmd summarize \
  conda run -n work1 python "${ROOT}/MetaLog/scripts/summarize_zeroshot_results.py" \
    --input "${OUT}" \
    --output_dir "${ROOT}/MetaLog/outputs/experiments/zeroshot" \
    --plot_direction hdfs30_hpc065_known_mix
