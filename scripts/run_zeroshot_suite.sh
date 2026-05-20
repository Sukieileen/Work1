#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${ROOT}/MetaLog/outputs/experiments/zeroshot/results_zeroshot_spirit_only_raw.csv}"
LOG_DIR="${LOG_DIR:-${ROOT}/MetaLog/outputs/experiments/zeroshot/logs_spirit_only}"
mkdir -p "${LOG_DIR}"

META_ENV="${META_ENV:-work1}"
BASE_ENV="${BASE_ENV:-work1}"
LOGROBUST_ENV="${LOGROBUST_ENV:-logrobust}"
LOGACTION_ENV="${LOGACTION_ENV:-metalog}"
DEVICE="${DEVICE:-auto}"
SPIRIT_MAX_LINES="${SPIRIT_MAX_LINES:-4700000}"
METALOG_SOURCE_CSV="${METALOG_SOURCE_CSV:-${ROOT}/MetaLog/outputs/experiments/zeroshot/results_zeroshot_knownmix_v3_raw.csv}"
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

run_cmd prepare_datasets \
  conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/prepare_zeroshot_datasets.py" \
    --datasets SPIRIT \
    --spirit_max_lines "${SPIRIT_MAX_LINES}"

METALOG_RUN="${METALOG_RUN:-hdfs30_hpc065_known_mix_pf_na_zeroshot_20260519_195514}"
METALOG_CKPT="${ROOT}/MetaLog/outputs/models/clean/HDFS30+HPC_sr065_parser_free/model/${METALOG_RUN}_phaseC_best.pt"
METALOG_TH="${ROOT}/MetaLog/outputs/models/clean/HDFS30+HPC_sr065_parser_free/model/${METALOG_RUN}_phaseC_best.threshold.txt"
if [ ! -s "${METALOG_CKPT}" ] || [ ! -s "${METALOG_TH}" ]; then
  echo "Missing MetaLog checkpoint or threshold: ${METALOG_CKPT}" >&2
  exit 1
fi

run_cmd seed_metalog_spirit_row \
  conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/seed_zeroshot_spirit_csv.py" \
    --source "${METALOG_SOURCE_CSV}" \
    --output "${OUT}" \
    --known_direction hdfs30_hpc065_known_mix \
    --zero_target SPIRIT \
    --method MetaLog

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
run_cmd metalog_base_SPIRIT \
  conda run -n "${BASE_ENV}" python "${ROOT}/MetaLog_base/MetaLog/metalog_base_zeroshot_eval.py" \
    --run_dir "${BASE_RUN}" \
    --zero_target SPIRIT \
    --output "${OUT}"

LOGROBUST_NAME="${LOGROBUST_NAME:-hdfs30_hpc065_known_mix_logrobust_zeroshot_${DATE_TAG}}"
LOGROBUST_RUN="${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/outputs/metalog_runs/${LOGROBUST_NAME}"
if [ ! -s "${LOGROBUST_RUN}/metrics.json" ]; then
  run_cmd train_logrobust_hdfs30_hpc065_known_mix \
    conda run -n "${LOGROBUST_ENV}" python "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/metalog_logrobust_run.py" \
      --direction hdfs30_hpc065_known_mix \
      --source_train_ratio 0.30 \
      --target_normal_ratio 1.0 \
      --target_anomaly_ratio 1.0 \
      --known_mix 1 \
      --use_target_dev_in_training 1 \
      --history_size 10 \
      --batch_size 1024 \
      --max_epoch 10 \
      --dropout 0.2 \
      --hidden_size 128 \
      --num_layers 2 \
      --lr 0.001 \
      --device "${DEVICE}" \
      --output_root "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/outputs/metalog_runs" \
      --run_name "${LOGROBUST_NAME}"
fi
for target in SPIRIT; do
  run_cmd "logrobust_${target}" \
    conda run -n "${LOGROBUST_ENV}" python "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/metalog_logrobust_zeroshot_eval.py" \
      --run_dir "${LOGROBUST_RUN}" \
      --zero_target "${target}" \
      --device "${DEVICE}" \
      --output "${OUT}"
done

LOGACTION_NAME="${LOGACTION_NAME:-hdfs30_hpc065_known_mix_logaction_zeroshot_${DATE_TAG}}"
LOGACTION_RUN="${ROOT}/LogAction/LogAction/outputs/metalog_runs/${LOGACTION_NAME}"
if [ ! -s "${LOGACTION_RUN}/metrics.json" ]; then
  run_cmd train_logaction_hdfs30_hpc065_known_mix \
    conda run -n "${LOGACTION_ENV}" python "${ROOT}/LogAction/LogAction/metalog_logaction_run.py" \
      --direction hdfs30_hpc065_known_mix \
      --source_train_ratio 0.30 \
      --target_normal_ratio 1.0 \
      --target_anomaly_ratio 1.0 \
      --known_mix 1 \
      --source_full_train 0 \
      --max_seq_len 128 \
      --target_dev_ratio 0.1 \
      --use_target_dev_in_training 1 \
      --epoch 60 \
      --batch_size 512 \
      --hidden_size 128 \
      --num_layers 2 \
      --lr 0.001 \
      --weight_decay 5e-6 \
      --active_epochs 5,10,20,40 \
      --active_ratio 0.05 \
      --first_sample_ratio 0.2 \
      --device "${DEVICE}" \
      --output_root "${ROOT}/LogAction/LogAction/outputs/metalog_runs" \
      --run_name "${LOGACTION_NAME}"
fi
for target in SPIRIT; do
  run_cmd "logaction_${target}" \
    conda run -n "${LOGACTION_ENV}" python "${ROOT}/LogAction/LogAction/metalog_logaction_zeroshot_eval.py" \
      --run_dir "${LOGACTION_RUN}" \
      --zero_target "${target}" \
      --device "${DEVICE}" \
      --output "${OUT}"
done

run_cmd summarize \
  conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/summarize_zeroshot_results.py" \
    --input "${OUT}" \
    --output_dir "${ROOT}/MetaLog/outputs/experiments/zeroshot" \
    --plot_direction hdfs30_hpc065_known_mix

echo "Zero-shot suite submitted work finished. Results: ${OUT}"
