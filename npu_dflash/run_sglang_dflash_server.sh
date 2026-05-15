#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$BUNDLE_DIR")

export SPECFORGE_DEVICE=${SPECFORGE_DEVICE:-npu}
export SGLANG_DEVICE=${SGLANG_DEVICE:-npu}
export PYTHONPATH="$BUNDLE_DIR/runtime:$ROOT_DIR:${PYTHONPATH:-}"

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1536}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export SGLANG_SET_CPU_AFFINITY=${SGLANG_SET_CPU_AFFINITY:-1}
export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-0}
export SGLANG_ENABLE_DFLASH_SPEC_V2=${SGLANG_ENABLE_DFLASH_SPEC_V2:-0}
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-0}
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost,::1}
export no_proxy=${no_proxy:-$NO_PROXY}

TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-ascend}
SGLANG_DRAFT_ATTENTION_BACKEND=${SGLANG_DRAFT_ATTENTION_BACKEND:-$SGLANG_ATTENTION_BACKEND}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.75}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-16}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-$SPECULATIVE_NUM_DRAFT_TOKENS}
SPECULATIVE_ALGORITHM=${SPECULATIVE_ALGORITHM:-DFLASH}
DTYPE=${DTYPE:-bfloat16}

extra_args=()
if [[ -n "${EXTRA_SGLANG_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$EXTRA_SGLANG_ARGS"
fi

mkdir -p "$BUNDLE_DIR/logs"

cat <<EOF
================ SGLang DFlash server on NPU ================
ROOT_DIR                       : $ROOT_DIR
TARGET_MODEL                   : $TARGET_MODEL
DRAFT_MODEL                    : $DRAFT_MODEL
HOST:PORT                      : $HOST:$PORT
TP_SIZE                        : $TP_SIZE
ASCEND_RT_VISIBLE_DEVICES      : $ASCEND_RT_VISIBLE_DEVICES
SGLANG_ATTENTION_BACKEND       : $SGLANG_ATTENTION_BACKEND
SGLANG_DRAFT_ATTENTION_BACKEND : $SGLANG_DRAFT_ATTENTION_BACKEND
SPECULATIVE_ALGORITHM          : $SPECULATIVE_ALGORITHM
SPECULATIVE_NUM_DRAFT_TOKENS   : $SPECULATIVE_NUM_DRAFT_TOKENS
DFLASH_BLOCK_SIZE              : $DFLASH_BLOCK_SIZE
=============================================================
EOF

cd "$ROOT_DIR"

server_args=(
    --device "$SGLANG_DEVICE"
    --model-path "$TARGET_MODEL"
    --host "$HOST"
    --port "$PORT"
    --tp-size "$TP_SIZE"
    --attention-backend "$SGLANG_ATTENTION_BACKEND"
    --speculative-draft-attention-backend "$SGLANG_DRAFT_ATTENTION_BACKEND"
    --speculative-algorithm "$SPECULATIVE_ALGORITHM"
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    --dtype "$DTYPE"
    --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
    --trust-remote-code
)

case "$SPECULATIVE_ALGORITHM" in
    DFLASH)
        python "$BUNDLE_DIR/check_sglang_dflash_support.py" \
            --draft-backend "$SGLANG_DRAFT_ATTENTION_BACKEND"
        server_args+=(--speculative-dflash-block-size "$DFLASH_BLOCK_SIZE")
        if [[ -n "${SPECULATIVE_DFLASH_DRAFT_WINDOW_SIZE:-}" ]]; then
            server_args+=(--speculative-dflash-draft-window-size "$SPECULATIVE_DFLASH_DRAFT_WINDOW_SIZE")
        fi
        ;;
    EAGLE3)
        server_args+=(--speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}")
        server_args+=(--speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK:-1}")
        ;;
    *)
        if [[ -n "${SPECULATIVE_NUM_STEPS:-}" ]]; then
            server_args+=(--speculative-num-steps "$SPECULATIVE_NUM_STEPS")
        fi
        if [[ -n "${SPECULATIVE_EAGLE_TOPK:-}" ]]; then
            server_args+=(--speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK")
        fi
        ;;
esac

ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES \
python -m sglang.launch_server "${server_args[@]}" "${extra_args[@]}"
