#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATASETS=("RoG-webqsp" "RoG-cwq")
TRAIN_DATASET_NAME="webqsp_cwq"

DATA_ROOT=${DATA_ROOT:-datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/fortrain}
MODEL_ROOT=${MODEL_ROOT:-models}

ONTOLOGY_PATH=${ONTOLOGY_PATH:-${DATA_ROOT}/ontology_graph_freebase.json}
BASE_MODEL=${BASE_MODEL:-${MODEL_ROOT}/Llama-2-7b-chat-hf}
PROMPT_PATH=${PROMPT_PATH:-prompts/llama2.txt}
TRAIN_CONFIG=${TRAIN_CONFIG:-config/deepspeed_zero3.yml}

TAIL_RAW_ROOT="${OUTPUT_ROOT}/tail_type_raw"
FINETUNE_ROOT="${OUTPUT_ROOT}/finetune_tail_types"

COMBINED_FINETUNE_DIR="${FINETUNE_ROOT}/${TRAIN_DATASET_NAME}"
COMBINED_FINETUNE_PATH="${COMBINED_FINETUNE_DIR}/finetune_question2tailtypes_train.jsonl"
TAIL_MODEL_DIR=${TAIL_MODEL_DIR:-${MODEL_ROOT}/OPI_tail_types_llama2}

BATCH_SIZE=${BATCH_SIZE:-4}
EPOCH=${EPOCH:-3}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
USE_PEFT=${USE_PEFT:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
AUTO_FIND_BATCH_SIZE=${AUTO_FIND_BATCH_SIZE:-False}
ATTN_IMP=${ATTN_IMP:-flash_attention_2}
RESPONSE_TEMPLATE=${RESPONSE_TEMPLATE:-"[/INST]"}
N_PROC=${N_PROC:-1}

mkdir -p "${TAIL_RAW_ROOT}" "${FINETUNE_ROOT}" "${COMBINED_FINETUNE_DIR}" "${TAIL_MODEL_DIR}"

echo "[1/3] Building tail-type supervision data for WebQSP and CWQ..."

for DATASET_NAME in "${DATASETS[@]}"; do
  echo "[INFO] Building tail-type supervision data for ${DATASET_NAME}..."

  python src/joint_training/build_tail_types_dataset_reconstruct.py \
    --data_path "${DATA_ROOT}" \
    --d "${DATASET_NAME}" \
    --split train \
    --ontology_path "${ONTOLOGY_PATH}" \
    --output_path "${TAIL_RAW_ROOT}" \
    --save_name "tail_types_dataset_train.jsonl" \
    --n "${N_PROC}" \
    --remove_duplicate
done

echo "[2/3] Building and merging instruction-tuning data..."

FINETUNE_PATHS=()

for DATASET_NAME in "${DATASETS[@]}"; do
  TAIL_RAW_PATH="${TAIL_RAW_ROOT}/${DATASET_NAME}/tail_types_dataset_train.jsonl"
  FINETUNE_DIR="${FINETUNE_ROOT}/${DATASET_NAME}"
  FINETUNE_PATH="${FINETUNE_DIR}/finetune_question2tailtypes_train.jsonl"

  mkdir -p "${FINETUNE_DIR}"

  echo "[INFO] Building instruction-tuning data for ${DATASET_NAME}..."

  python src/joint_training/build_finetune_tailtypes.py \
    --dataset_path "${DATA_ROOT}/${DATASET_NAME}" \
    --tail_types_path "${TAIL_RAW_PATH}" \
    --prompt_path "${PROMPT_PATH}" \
    --model_path "${BASE_MODEL}" \
    --output_path "${FINETUNE_PATH}"

  FINETUNE_PATHS+=("${FINETUNE_PATH}")
done

cat "${FINETUNE_PATHS[@]}" > "${COMBINED_FINETUNE_PATH}"

echo "[INFO] Combined instruction-tuning data saved to: ${COMBINED_FINETUNE_PATH}"

echo "[3/3] Fine-tuning tail-type prediction model..."
accelerate launch --config_file "${TRAIN_CONFIG}" src/joint_training/joint_finetuning.py \
  --data_path_list "${COMBINED_FINETUNE_PATH}" \
  --model_name_or_path "${BASE_MODEL}" \
  --output_dir "${TAIL_MODEL_DIR}" \
  --max_seq_length 128 \
  --use_peft "${USE_PEFT}" \
  --bf16 True \
  --num_train_epochs "${EPOCH}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --load_in_8bit False \
  --load_in_4bit False \
  --save_merged False \
  --report_to "none" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --auto_find_batch_size "${AUTO_FIND_BATCH_SIZE}" \
  --attn_implementation "${ATTN_IMP}" \
  --run_name "tail_type_model_${TRAIN_DATASET_NAME}" \
  --response_template "${RESPONSE_TEMPLATE}"

echo "[DONE] Training finished. Model saved to: ${TAIL_MODEL_DIR}"