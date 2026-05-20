#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
ZERO_ROOT="${ROOT}/MetaLog/outputs/experiments/zeroshot"
LOG_DIR="${ZERO_ROOT}/logs_repair"
RAW_SUMMARY_DIR="${ZERO_ROOT}/summary_raw"
SPIRIT_SUMMARY_DIR="${ZERO_ROOT}/summary_spirit_only"
OUT_RAW="${OUT_RAW:-${ZERO_ROOT}/results_zeroshot_raw.csv}"
OUT_SPIRIT="${OUT_SPIRIT:-${ZERO_ROOT}/results_zeroshot_spirit_only_raw.csv}"
QUEUE_LOG="${LOG_DIR}/${DATE_TAG}_zeroshot_repair.log"

META_ENV="${META_ENV:-work1}"
BASE_ENV="${BASE_ENV:-work1}"
LOGROBUST_ENV="${LOGROBUST_ENV:-logrobust}"
LOGACTION_ENV="${LOGACTION_ENV:-metalog}"
DEVICE="${DEVICE:-auto}"
SPIRIT_MAX_LINES="${SPIRIT_MAX_LINES:-4700000}"

METALOG_H2D_RUN="${METALOG_H2D_RUN:-hdfs10_to_hpc065_pf_na_zeroshot_20260519_140730}"
METALOG_H2D_DIR="${ROOT}/MetaLog/outputs/models/clean/HPC_sr065_parser_free/model"
METALOG_H2D_CKPT="${METALOG_H2D_DIR}/${METALOG_H2D_RUN}_phaseC_best.pt"
METALOG_H2D_TH="${METALOG_H2D_DIR}/${METALOG_H2D_RUN}_phaseC_best.threshold.txt"

METALOG_KNOWNMIX_CSV="${ROOT}/MetaLog/outputs/experiments/zeroshot/results_zeroshot_knownmix_v3_raw.csv"
BASE_H2D_RUN="${BASE_H2D_RUN:-hdfs10_to_hpc065_metalog_base_zeroshot_20260519_152842}"
BASE_H2D_DIR="${ROOT}/MetaLog_base/MetaLog/outputs/metalog_runs/${BASE_H2D_RUN}"
LOGROBUST_H2D_RUN="${LOGROBUST_H2D_RUN:-hdfs10_to_hpc065_logrobust_zeroshot_20260519_140730}"
LOGROBUST_H2D_DIR="${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/outputs/metalog_runs/${LOGROBUST_H2D_RUN}"

BASE_KNOWNMIX_RUN="${BASE_KNOWNMIX_RUN:-hdfs30_hpc065_known_mix_metalog_base_zeroshot_20260519_215355}"
BASE_KNOWNMIX_DIR="${ROOT}/MetaLog_base/MetaLog/outputs/metalog_runs/${BASE_KNOWNMIX_RUN}"
LOGROBUST_KNOWNMIX_RUN="${LOGROBUST_KNOWNMIX_RUN:-hdfs30_hpc065_known_mix_logrobust_zeroshot_20260519_215355}"
LOGROBUST_KNOWNMIX_DIR="${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/outputs/metalog_runs/${LOGROBUST_KNOWNMIX_RUN}"
LOGACTION_KNOWNMIX_RUN="${LOGACTION_KNOWNMIX_RUN:-hdfs30_hpc065_known_mix_logaction_zeroshot_${DATE_TAG}}"
LOGACTION_KNOWNMIX_DIR="${ROOT}/LogAction/LogAction/outputs/metalog_runs/${LOGACTION_KNOWNMIX_RUN}"

mkdir -p "${LOG_DIR}" "${RAW_SUMMARY_DIR}" "${SPIRIT_SUMMARY_DIR}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

run_cmd() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${DATE_TAG}_${name}.log"
  log "START ${name}"
  "$@" 2>&1 | tee -a "${log_file}"
  log "DONE ${name}"
}

run_in_dir() {
  local name="$1"
  local dir="$2"
  shift 2
  local log_file="${LOG_DIR}/${DATE_TAG}_${name}.log"
  log "START ${name}"
  (
    cd "${dir}"
    "$@"
  ) 2>&1 | tee -a "${log_file}"
  log "DONE ${name}"
}

backup_if_exists() {
  local path="$1"
  if [ -f "${path}" ]; then
    local backup="${path%.csv}_backup_${DATE_TAG}.csv"
    mv "${path}" "${backup}"
    log "Backed up existing $(basename "${path}") to $(basename "${backup}")"
  fi
}

require_file() {
  local path="$1"
  if [ ! -s "${path}" ]; then
    echo "Required file missing: ${path}" >&2
    exit 1
  fi
}

prepare_zero_datasets() {
  run_in_dir prepare_datasets "${ROOT}/MetaLog" \
    conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/prepare_zeroshot_datasets.py" \
      --datasets OpenStack SPIRIT \
      --spirit_max_lines "${SPIRIT_MAX_LINES}"
}

rebuild_raw_results() {
  require_file "${METALOG_H2D_CKPT}"
  require_file "${METALOG_H2D_TH}"
  require_file "${BASE_H2D_DIR}/metrics.json"
  require_file "${LOGROBUST_H2D_DIR}/metrics.json"

  backup_if_exists "${OUT_RAW}"

  for target in OpenStack SPIRIT; do
    run_in_dir "raw_metalog_${target}" "${ROOT}/MetaLog" \
      conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/run_zeroshot_eval.py" \
        --known_direction hdfs_to_hpc_sr065 \
        --zero_target "${target}" \
        --checkpoint "${METALOG_H2D_CKPT}" \
        --threshold_file "${METALOG_H2D_TH}" \
        --method_name MetaLog \
        --threshold_source "HPC selection" \
        --output "${OUT_RAW}" \
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

  for target in OpenStack SPIRIT; do
    run_in_dir "raw_metalog_base_${target}" "${ROOT}/MetaLog_base/MetaLog" \
      conda run -n "${BASE_ENV}" python "${ROOT}/MetaLog_base/MetaLog/metalog_base_zeroshot_eval.py" \
        --run_dir "${BASE_H2D_DIR}" \
        --zero_target "${target}" \
        --output "${OUT_RAW}"
  done

  for target in OpenStack SPIRIT; do
    run_in_dir "raw_logrobust_${target}" "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical" \
      conda run -n "${LOGROBUST_ENV}" python "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/metalog_logrobust_zeroshot_eval.py" \
        --run_dir "${LOGROBUST_H2D_DIR}" \
        --zero_target "${target}" \
        --device "${DEVICE}" \
        --output "${OUT_RAW}"
  done

  run_in_dir summarize_raw "${ROOT}/MetaLog" \
    conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/summarize_zeroshot_results.py" \
      --input "${OUT_RAW}" \
      --output_dir "${RAW_SUMMARY_DIR}" \
      --plot_direction hdfs_to_hpc_sr065
}

resume_spirit_suite() {
  require_file "${METALOG_KNOWNMIX_CSV}"
  require_file "${BASE_KNOWNMIX_DIR}/metrics.json"
  require_file "${LOGROBUST_KNOWNMIX_DIR}/metrics.json"

  backup_if_exists "${OUT_SPIRIT}"

  run_in_dir seed_knownmix_spirit "${ROOT}/MetaLog" \
    conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/seed_zeroshot_spirit_csv.py" \
      --source "${METALOG_KNOWNMIX_CSV}" \
      --output "${OUT_SPIRIT}" \
      --known_direction hdfs30_hpc065_known_mix \
      --zero_target SPIRIT \
      --method MetaLog

  run_in_dir spirit_metalog_base "${ROOT}/MetaLog_base/MetaLog" \
    conda run -n "${BASE_ENV}" python "${ROOT}/MetaLog_base/MetaLog/metalog_base_zeroshot_eval.py" \
      --run_dir "${BASE_KNOWNMIX_DIR}" \
      --zero_target SPIRIT \
      --output "${OUT_SPIRIT}"

  run_in_dir spirit_logrobust "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical" \
    conda run -n "${LOGROBUST_ENV}" python "${ROOT}/LogADE-empirical-LogRobust/LogADEmpirical/metalog_logrobust_zeroshot_eval.py" \
      --run_dir "${LOGROBUST_KNOWNMIX_DIR}" \
      --zero_target SPIRIT \
      --device "${DEVICE}" \
      --output "${OUT_SPIRIT}"

  if [ ! -s "${LOGACTION_KNOWNMIX_DIR}/metrics.json" ]; then
    run_in_dir train_spirit_logaction "${ROOT}/LogAction/LogAction" \
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
        --run_name "${LOGACTION_KNOWNMIX_RUN}"
  else
    log "Reusing existing LogAction run ${LOGACTION_KNOWNMIX_RUN}"
  fi

  run_in_dir spirit_logaction "${ROOT}/LogAction/LogAction" \
    conda run -n "${LOGACTION_ENV}" python "${ROOT}/LogAction/LogAction/metalog_logaction_zeroshot_eval.py" \
      --run_dir "${LOGACTION_KNOWNMIX_DIR}" \
      --zero_target SPIRIT \
      --device "${DEVICE}" \
      --output "${OUT_SPIRIT}"

  run_in_dir summarize_spirit "${ROOT}/MetaLog" \
    conda run -n "${META_ENV}" python "${ROOT}/MetaLog/scripts/summarize_zeroshot_results.py" \
      --input "${OUT_SPIRIT}" \
      --output_dir "${SPIRIT_SUMMARY_DIR}" \
      --plot_direction hdfs30_hpc065_known_mix
}

main() {
  prepare_zero_datasets
  rebuild_raw_results
  resume_spirit_suite
  log "Zero-shot repair finished."
  log "Raw results: ${OUT_RAW}"
  log "SPIRIT-only results: ${OUT_SPIRIT}"
}

main "$@"
