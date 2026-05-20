#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbhh/Research/MetaLog"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${ROOT}/outputs/experiments/ablation"
LOG_DIR="${OUT_ROOT}/logs"
METRICS_DIR="${OUT_ROOT}/epoch_metrics"
MANIFEST="${OUT_ROOT}/component_ablation_manifest_${DATE_TAG}.csv"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"

echo "direction,variant_id,variant_name,use_moe,use_normality_anchor,router_use_distance,prototype_scale,prototype_loss_weight,prototype_sep_weight,metrics_file,checkpoint" > "${MANIFEST}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate work1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${ROOT}"

ABLATION_DIRECTIONS="${ABLATION_DIRECTIONS:-hpc_to_hdfs hdfs_to_hpc_sr065}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-A0 A1 A2 A3}"
ABLATION_INCLUDE_A4="${ABLATION_INCLUDE_A4:-1}"
ABLATION_A2_PROTO_SCALE="${ABLATION_A2_PROTO_SCALE:-0.5}"
ABLATION_A3_PROTO_SCALE="${ABLATION_A3_PROTO_SCALE:-0.5}"

direction_enabled() {
  case " ${ABLATION_DIRECTIONS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

variant_enabled() {
  case " ${ABLATION_VARIANTS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

run_variant() {
  local direction="$1"
  local variant_id="$2"
  local variant_name="$3"
  local use_moe="$4"
  local use_na="$5"
  local router_distance="$6"
  local proto_scale="$7"
  local proto_loss="$8"
  local proto_sep="$9"
  shift 9

  local run_name="${direction}_${variant_id}_$(echo "${variant_name}" | tr ' +' '__' | tr -cd '[:alnum:]_')_${DATE_TAG}"
  local metrics_file="${METRICS_DIR}/${run_name}.csv"
  local log_file="${LOG_DIR}/${run_name}.log"
  local target_dir="HDFS_parser_free"
  if [[ "${direction}" == hdfs_to_hpc* ]]; then
    target_dir="HPC_sr065_parser_free"
  fi
  local checkpoint="${ROOT}/outputs/models/clean/${target_dir}/model/${run_name}_phaseC_best.pt"

  if [ -s "${metrics_file}" ] && [ -s "${checkpoint}" ]; then
    echo "SKIP ${run_name}"
  else
    echo "RUN ${run_name}"
    python -u approaches/MetaLog.py \
      --mode train \
      --direction "${direction}" \
      --parser parser_free \
      --protocol clean \
      --source_train_ratio "$([[ "${direction}" == hpc_to_hdfs ]] && echo 0.65 || echo 0.30)" \
      --target_normal_ratio "$([[ "${direction}" == hpc_to_hdfs ]] && echo 0.10 || echo 0.30)" \
      --target_anomaly_ratio 0.01 \
      --backbone bimamba \
      --dropout 0.1 \
      --mamba_state 96 \
      --mamba_conv 4 \
      --mamba_expand 4 \
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
  fi

  echo "${direction},${variant_id},${variant_name},${use_moe},${use_na},${router_distance},${proto_scale},${proto_loss},${proto_sep},${metrics_file},${checkpoint}" >> "${MANIFEST}"
}

append_existing_variant() {
  local direction="$1"
  local variant_id="$2"
  local variant_name="$3"
  local use_moe="$4"
  local use_na="$5"
  local router_distance="$6"
  local proto_scale="$7"
  local proto_loss="$8"
  local proto_sep="$9"
  local metrics_file="$10"
  local checkpoint="$11"
  echo "${direction},${variant_id},${variant_name},${use_moe},${use_na},${router_distance},${proto_scale},${proto_loss},${proto_sep},${metrics_file},${checkpoint}" >> "${MANIFEST}"
}

for direction in hpc_to_hdfs hdfs_to_hpc_sr065; do
  if ! direction_enabled "${direction}"; then
    continue
  fi
  if variant_enabled A0; then
    run_variant "${direction}" A0 "BiMamba+LinearAttn+Dense" 0 0 0 0 0 0 \
      --no-use-moe --no-use-normality-anchor --no-router-use-distance \
      --prototype-scale 0 --prototype-loss-weight 0 --prototype-sep-weight 0
  fi
  if variant_enabled A1; then
    run_variant "${direction}" A1 "Vanilla-MoE" 1 0 0 0 0 0 \
      --use-moe --no-use-normality-anchor --no-router-use-distance \
      --prototype-scale 0 --prototype-loss-weight 0 --prototype-sep-weight 0
  fi
  if variant_enabled A2; then
    run_variant "${direction}" A2 "Prototype-Logit" 1 1 0 "${ABLATION_A2_PROTO_SCALE}" 0 0 \
      --use-moe --use-normality-anchor --no-router-use-distance \
      --prototype-scale "${ABLATION_A2_PROTO_SCALE}" --prototype-loss-weight 0 --prototype-sep-weight 0
  fi
  if variant_enabled A3; then
    run_variant "${direction}" A3 "Prototype-Loss" 1 1 0 "${ABLATION_A3_PROTO_SCALE}" 0.1 1e-3 \
      --use-moe --use-normality-anchor --no-router-use-distance \
      --prototype-scale "${ABLATION_A3_PROTO_SCALE}" --prototype-loss-weight 0.1 --prototype-sep-weight 1e-3
  fi
done

if [[ "${ABLATION_INCLUDE_A4}" == "1" ]]; then
  if direction_enabled hpc_to_hdfs; then
    append_existing_variant \
      hpc_to_hdfs A4 "Full-Model" 1 1 1 0.5 0.1 1e-3 \
      "${ROOT}/outputs/results/supervised_protocol/hpc_to_hdfs_parserfree_na_v1_bimamba_do0p1_st96_cv4_ex4_ps0p5_plw0p1_sr0p65_20260512.csv" \
      "${ROOT}/outputs/models/clean/HDFS_parser_free/model/hpc_to_hdfs_parserfree_na_v1_bimamba_do0p1_st96_cv4_ex4_ps0p5_plw0p1_sr0p65_20260512_phaseC_best.pt"
  fi
  if direction_enabled hdfs_to_hpc_sr065; then
    append_existing_variant \
      hdfs_to_hpc_sr065 A4 "Full-Model" 1 1 1 0.5 0.1 1e-3 \
      "${ROOT}/outputs/results/supervised_protocol/hdfs_to_hpc_sr065_parserfree_na_v1_bimamba_do0p1_st96_cv4_ex4_ps0p5_plw0p1_20260517.csv" \
      "${ROOT}/outputs/models/clean/HPC_sr065_parser_free/model/hdfs_to_hpc_sr065_parserfree_na_v1_bimamba_do0p1_st96_cv4_ex4_ps0p5_plw0p1_20260517_phaseC_best.pt"
  fi
fi

python scripts/summarize_epoch_metrics.py --manifest "${MANIFEST}" --output_dir "${OUT_ROOT}"
echo "Component ablation finished. Manifest: ${MANIFEST}"
