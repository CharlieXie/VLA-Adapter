#!/usr/bin/env bash
set -euo pipefail

cd /workspace/VLA-Adapter

mkdir -p logs outputs

current_time="$(date +%Y%m%d_%H%M%S)"
data_name="calvin_abc"
run_note="VLA-Adapter--calvin_abc--4gpu-val-wandb--${current_time}"
log_file="logs/${run_note}.log"
config_file_path="${CONFIG_FILE_PATH:-pretrained_models/configs}"
resume_args=()

if [[ -n "${RESUME_CHECKPOINT_DIR:-}" ]]; then
  if [[ -z "${RESUME_STEP:-}" ]]; then
    echo "RESUME_STEP must be set when RESUME_CHECKPOINT_DIR is set" >&2
    exit 1
  fi
  config_file_path="${RESUME_CHECKPOINT_DIR}"
  resume_args=(
    --resume True
    --resume_step "${RESUME_STEP}"
    --resum_vla_path "${RESUME_CHECKPOINT_DIR}"
  )
fi

echo "${log_file}" > logs/latest_calvin_4gpu_val.log

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-${data_name}}"

/opt/miniforge3/bin/conda run --no-capture-output -n vla-calvin \
  torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path "${config_file_path}" \
    --data_root_dir /workspace/data \
    --dataset_name "${data_name}" \
    --run_root_dir outputs \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay 200000 \
    --max_steps 200005 \
    --use_val_set True \
    --val_freq 5000 \
    --val_time_limit 180 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_project "${WANDB_PROJECT}" \
    --run_id_note "${run_note}" \
    "${resume_args[@]}" \
  2>&1 | tee "${log_file}"
