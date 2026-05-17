#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate work1
cd /home/lbhh/Research/MetaLog || exit 1

SEARCH_ROOT=/home/lbhh/Research/MetaLog/outputs/results/supervised_protocol_grid/h2d_pf_na_grid12_v1_20260513
mkdir -p "$SEARCH_ROOT"/epoch_metrics "$SEARCH_ROOT"/trial_logs

STATUS_FILE="$SEARCH_ROOT/grid_status.csv"
echo "trial,run_name,return_code,metrics_file,log_file" > "$STATUS_FILE"

MODEL_LRS=(2e-4 3e-4)
DROPOUTS=(0.0 0.1)
MAMBA_STATES=(64 96 128)

tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

trial=0
total=12
for lr in "${MODEL_LRS[@]}"; do
  for dropout in "${DROPOUTS[@]}"; do
    for state in "${MAMBA_STATES[@]}"; do
      trial=$((trial + 1))

      lr_tag=$(tag_value "$lr")
      dropout_tag=$(tag_value "$dropout")

      run_name="hpc_to_hdfs_pf_na_grid12_v1_t${trial}_lr${lr_tag}_do${dropout_tag}_st${state}_cv4_ex4_ps0p5_plw0p1_lt4p0_tp0p5_20260513"
      metrics_file="$SEARCH_ROOT/epoch_metrics/${run_name}.csv"
      log_file="$SEARCH_ROOT/trial_logs/${run_name}.log"

      if [ -s "$metrics_file" ]; then
        echo "===== Trial $trial/$total SKIP | $run_name ====="
        echo "$trial,$run_name,SKIP,$metrics_file,$log_file" >> "$STATUS_FILE"
        continue
      fi

      echo "===== Trial $trial/$total RUN | $run_name ====="
      {
        echo "RUN_NAME: $run_name"
        echo "LR: $lr"
        echo "DROPOUT: $dropout"
        echo "MAMBA_STATE: $state"
        echo
      } > "$log_file"

      python -u approaches/MetaLog_BH.py \
        --mode train \
        --parser parser_free \
        --plm_model bert-base-uncased \
        --plm_max_length 64 \
        --plm_batch_size 64 \
        --plm_pooling mean \
        --protocol clean \
        --backbone bimamba \
        --dropout "$dropout" \
        --mamba_state "$state" \
        --mamba_conv 4 \
        --mamba_expand 4 \
        --use-normality-anchor \
        --prototype-scale 0.5 \
        --prototype-loss-weight 0.1 \
        --prototype-sep-weight 1e-3 \
        --prototype-margin-global 1.0 \
        --prototype-margin-expert 1.0 \
        --prototype-target-normal-only \
        --router-use-distance \
        --warmup_epochs 5 \
        --joint_epochs 8 \
        --calibration_epochs 3 \
        --warmup_lr "$lr" \
        --joint_backbone_lr "$lr" \
        --joint_gate_lr "$lr" \
        --joint_expert_lr "$lr" \
        --calibration_gate_lr "$lr" \
        --calibration_expert_lr "$lr" \
        --weight_decay 1e-2 \
        --lambda_target 4.0 \
        --target_positive_fraction 0.5 \
        --auto_threshold \
        --threshold_min 0.50 \
        --threshold_max 0.95 \
        --threshold_step 0.005 \
        --threshold 0.5 \
        --run_name "$run_name" \
        --epoch_metrics_file "$metrics_file" \
        >> "$log_file" 2>&1

      rc=$?
      echo "$trial,$run_name,$rc,$metrics_file,$log_file" >> "$STATUS_FILE"
      echo "===== Trial $trial/$total DONE rc=$rc | $run_name ====="
    done
  done
done

echo "Grid search finished. Status: $STATUS_FILE"
