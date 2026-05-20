#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research/MetaLog"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="${ROOT}/outputs/experiments/full_remaining_queue_${DATE_TAG}.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

wait_for_screen() {
  local name="$1"
  while screen -ls | grep -q "${name}"; do
    log "Waiting for screen ${name} to finish..."
    sleep 300
  done
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate work1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

wait_for_screen "zeroshot_metalog_20260519"

log "Running zero-shot continuation for MetaLog_base baseline."
bash "${ROOT}/scripts/run_zeroshot_continuation.sh" 2>&1 | tee -a "${QUEUE_LOG}"

log "Running remaining E1-E4 experiment queue."
bash "${ROOT}/scripts/run_remaining_experiments_queue.sh" 2>&1 | tee -a "${QUEUE_LOG}"

log "Full remaining queue finished."
