#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research/MetaLog"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${ROOT}/outputs/experiments/hparams"
LOG_DIR="${OUT_ROOT}/logs"
METRICS_DIR="${OUT_ROOT}/epoch_metrics"
RAW="${OUT_ROOT}/hparam_sensitivity_raw.csv"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate work1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${ROOT}"

echo "direction,hyperparam_name,hyperparam_value,precision,recall,f1,threshold,checkpoint,metrics_file" > "${RAW}"

run_trial() {
  local direction="$1"
  local hp_name="$2"
  local hp_value="$3"
  shift 3
  local value_tag="${hp_value//./p}"
  value_tag="${value_tag//-/m}"
  local run_name="${direction}_hparam_${hp_name}_${value_tag}_${DATE_TAG}"
  local metrics_file="${METRICS_DIR}/${run_name}.csv"
  local log_file="${LOG_DIR}/${run_name}.log"
  local target_dir="HDFS_parser_free"
  local checkpoint="${ROOT}/outputs/models/clean/${target_dir}/model/${run_name}_phaseC_best.pt"

  echo "RUN ${run_name}"
  python -u approaches/MetaLog.py \
    --mode train \
    --direction "${direction}" \
    --parser parser_free \
    --protocol clean \
    --source_train_ratio 0.65 \
    --target_normal_ratio 0.10 \
    --target_anomaly_ratio 0.01 \
    --backbone bimamba \
    --dropout 0.1 \
    --mamba_state 96 \
    --mamba_conv 4 \
    --mamba_expand 4 \
    --use-moe \
    --use-normality-anchor \
    --router-use-distance \
    --prototype-scale 1.0 \
    --prototype-loss-weight 0.1 \
    --prototype-sep-weight 1e-3 \
    --warmup_epochs 5 \
    --joint_epochs 8 \
    --calibration_epochs 3 \
    --warmup_lr 2e-4 \
    --joint_backbone_lr 2e-4 \
    --joint_gate_lr 2e-4 \
    --joint_expert_lr 2e-4 \
    --calibration_gate_lr 2e-4 \
    --calibration_expert_lr 2e-4 \
    --weight_decay 1e-2 \
    --lambda_target 4.0 \
    --target_positive_fraction 0.5 \
    --auto_threshold \
    --threshold_min 0.50 \
    --threshold_max 0.95 \
    --threshold_step 0.005 \
    --threshold 0.5 \
    --run_name "${run_name}" \
    --epoch_metrics_file "${metrics_file}" \
    "$@" \
    > "${log_file}" 2>&1

  python - "$direction" "$hp_name" "$hp_value" "$checkpoint" "$metrics_file" "$RAW" <<'PY'
import csv
import sys

direction, hp_name, hp_value, checkpoint, metrics_file, raw_path = sys.argv[1:]
with open(metrics_file, newline='', encoding='utf-8') as reader:
    rows = list(csv.DictReader(reader))
selected = [row for row in rows if row.get('phase') == 'phase_c' and row.get('selected_for_best') == '1']
if not selected:
    selected = [row for row in rows if row.get('phase') == 'phase_c']
row = selected[-1] if selected else rows[-1]
with open(raw_path, 'a', newline='', encoding='utf-8') as writer:
    csv.writer(writer).writerow([
        direction,
        hp_name,
        hp_value,
        row['test_precision'],
        row['test_recall'],
        row['test_f1'],
        row['selected_threshold'],
        checkpoint,
        metrics_file,
    ])
PY
}

for value in 2 4 6 8; do
  run_trial hpc_to_hdfs num_experts "${value}" --moe_num_experts "${value}" --moe_top_k 2
done

for value in 1 2 3; do
  run_trial hpc_to_hdfs topk "${value}" --moe_num_experts 4 --moe_top_k "${value}"
done

for value in 0 0.01 0.05 0.1 0.2; do
  run_trial hpc_to_hdfs proto_weight "${value}" --prototype-loss-weight "${value}"
done

python scripts/summarize_hparam_sensitivity.py --input "${RAW}" --output_dir "${OUT_ROOT}"
echo "Hyperparameter sensitivity finished. Raw: ${RAW}"
