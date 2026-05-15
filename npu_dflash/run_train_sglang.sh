#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$BUNDLE_DIR")

export SPECFORGE_DEVICE=${SPECFORGE_DEVICE:-npu}
export SGLANG_DEVICE=${SGLANG_DEVICE:-npu}
export PYTHONPATH="$BUNDLE_DIR/runtime:$ROOT_DIR:${PYTHONPATH:-}"

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export NPU_ASD_ENABLE=${NPU_ASD_ENABLE:-0}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1536}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-1}

TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:-$BUNDLE_DIR/cache/tiny_dflash_train.jsonl}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-dflash.json}
OUTPUT_DIR=${OUTPUT_DIR:-$BUNDLE_DIR/outputs/qwen3-8b-dflash-sglang}

NUM_NPUS=${NUM_NPUS:-1}
TP_SIZE=${TP_SIZE:-$NUM_NPUS}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
MAX_LENGTH=${MAX_LENGTH:-512}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-6e-4}
NUM_ANCHORS=${NUM_ANCHORS:-16}
BLOCK_SIZE=${BLOCK_SIZE:-16}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-ascend}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.4}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-}

LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-999999}
REPORT_TO=${REPORT_TO:-none}
MASTER_ADDR_LOCAL=${MASTER_ADDR_LOCAL:-127.0.0.1}
MASTER_PORT_LOCAL=${MASTER_PORT_LOCAL:-29534}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}

[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]] || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }
if (( NUM_NPUS % TP_SIZE != 0 )); then
    echo "ERROR: NUM_NPUS ($NUM_NPUS) must be divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR" "$BUNDLE_DIR/logs" "$BUNDLE_DIR/cache"

trust_args=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    trust_args+=(--trust-remote-code)
fi

context_args=()
if [[ -n "$SGLANG_CONTEXT_LENGTH" ]]; then
    context_args+=(--sglang-context-length "$SGLANG_CONTEXT_LENGTH")
fi

cat <<EOF
============= SpecForge DFlash NPU training: embedded SGLang =============
ROOT_DIR                  : $ROOT_DIR
TARGET_MODEL              : $TARGET_MODEL
DRAFT_CONFIG              : $DRAFT_CONFIG
TRAIN_DATA                : $TRAIN_DATA
OUTPUT_DIR                : $OUTPUT_DIR
NUM_NPUS / TP_SIZE        : $NUM_NPUS / $TP_SIZE
SGLANG_ATTENTION_BACKEND  : $SGLANG_ATTENTION_BACKEND
ASCEND_RT_VISIBLE_DEVICES : $ASCEND_RT_VISIBLE_DEVICES
==========================================================================
EOF

cd "$ROOT_DIR"

ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES \
torchrun \
    --nproc_per_node "$NUM_NPUS" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr "$MASTER_ADDR_LOCAL" \
    --master_port "$MASTER_PORT_LOCAL" \
    "$BUNDLE_DIR/train_dflash_npu.py" \
    --target-model-path "$TARGET_MODEL" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$BUNDLE_DIR/cache" \
    --target-model-backend sglang \
    --tp-size "$TP_SIZE" \
    --attention-backend sdpa \
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    "${context_args[@]}" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --max-length "$MAX_LENGTH" \
    --learning-rate "$LR" \
    --num-anchors "$NUM_ANCHORS" \
    --block-size "$BLOCK_SIZE" \
    --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --warmup-ratio "$WARMUP_RATIO" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --chat-template qwen \
    --report-to "$REPORT_TO" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    "${trust_args[@]}"
