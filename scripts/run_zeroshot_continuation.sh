#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/MetaLog/outputs/experiments/zeroshot/results_zeroshot_raw.csv"
LOG_DIR="${ROOT}/MetaLog/outputs/experiments/zeroshot/logs"
mkdir -p "${LOG_DIR}"

BASE_ENV="${BASE_ENV:-work1}"
DEVICE="${DEVICE:-auto}"
SPIRIT_MAX_LINES="${SPIRIT_MAX_LINES:-4700000}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

run_cmd() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${DATE_TAG}_${name}.log"
  echo "[$(date '+%F %T')] START ${name}" | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE ${name}" | tee -a "${log_file}"
}

source ~/miniconda3/etc/profile.d/conda.sh

cd "${ROOT}/MetaLog_base/MetaLog"

BASE_NAME="${BASE_NAME:-hdfs30_hpc065_known_mix_metalog_base_zeroshot_${DATE_TAG}}"
BASE_RUN="${ROOT}/MetaLog_base/MetaLog/outputs/metalog_runs/${BASE_NAME}"
if [ ! -s "${BASE_RUN}/metrics.json" ]; then
  run_cmd train_metalog_base_hdfs30_hpc065_known_mix \
    conda run -n "${BASE_ENV}" python "${ROOT}/MetaLog_base/MetaLog/metalog_base_run.py" \
      --direction hdfs30_hpc065_known_mix \
      --source_train_ratio 0.30 \
      --target_normal_ratio 1.0 \
      --target_anomaly_ratio 1.0 \
      --target_dev_ratio 0.10 \
      --use_target_dev_in_training 1 \
      --epochs 10 \
      --batch_size 100 \
      --test_batch_size 1024 \
      --hidden_size 100 \
      --num_layers 2 \
      --lr 0.002 \
      --inner_lr 0.002 \
      --output_root "${ROOT}/MetaLog_base/MetaLog/outputs/metalog_runs" \
      --run_name "${BASE_NAME}"
fi

for target in OpenStack SPIRIT; do
  run_cmd "metalog_base_${target}" \
    conda run -n "${BASE_ENV}" python "${ROOT}/MetaLog_base/MetaLog/metalog_base_zeroshot_eval.py" \
      --run_dir "${BASE_RUN}" \
      --zero_target "${target}" \
      --output "${OUT}"
done

run_cmd summarize_after_base \
  conda run -n "${BASE_ENV}" python "${ROOT}/MetaLog/scripts/summarize_zeroshot_results.py" \
    --input "${OUT}" \
    --output_dir "${ROOT}/MetaLog/outputs/experiments/zeroshot" \
    --plot_direction hdfs30_hpc065_known_mix

echo "Zero-shot continuation finished. Results: ${OUT}"
