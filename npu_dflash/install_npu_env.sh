#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$BUNDLE_DIR")
THIRD_PARTY_DIR=${THIRD_PARTY_DIR:-$BUNDLE_DIR/third_party}
ENV_PATH=${ENV_PATH:-$BUNDLE_DIR/conda/specforge_npu}

PYTHON_VERSION=${PYTHON_VERSION:-3.11}
HUAWEI_INDEX=${HUAWEI_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple/}
PIP_INDEX_ARGS=(-i "$HUAWEI_INDEX" --trusted-host mirrors.huaweicloud.com)

SGLANG_REPO=${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}
SGLANG_REF=${SGLANG_REF:-v0.5.9}
SGLANG_DIR=${SGLANG_DIR:-$THIRD_PARTY_DIR/sglang}
SGLANG_INSTALL_NO_DEPS=${SGLANG_INSTALL_NO_DEPS:-0}

SGL_KERNEL_REPO=${SGL_KERNEL_REPO:-https://github.com/Sawyer117/sgl-kernel-npu.git}
SGL_KERNEL_REF=${SGL_KERNEL_REF:-npu-install-stable}
SGL_KERNEL_DIR=${SGL_KERNEL_DIR:-$THIRD_PARTY_DIR/sgl-kernel-npu}

SKIP_SGLANG_INSTALL=${SKIP_SGLANG_INSTALL:-0}
SKIP_SGL_KERNEL_NPU=${SKIP_SGL_KERNEL_NPU:-0}

if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${CANN_HOME:-}" ]]; then
    cat >&2 <<'EOF'
ERROR: CANN is not sourced.

Run this first, then re-run install_npu_env.sh:
  export CANN_HOME=/path/to/CANN/8.5.0.x
  source "$CANN_HOME/ascend-toolkit/set_env.sh"
  [ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
  [ -f "$CANN_HOME/nnal/atb/set_env.sh" ] && source "$CANN_HOME/nnal/atb/set_env.sh"
EOF
    exit 1
fi

mkdir -p "$THIRD_PARTY_DIR" "$BUNDLE_DIR/cache" "$BUNDLE_DIR/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    conda create -p "$ENV_PATH" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_PATH"

python -m pip install -r "$BUNDLE_DIR/requirements-ascend.txt"

if [[ "$SKIP_SGLANG_INSTALL" != "1" ]]; then
    if [[ ! -d "$SGLANG_DIR/.git" ]]; then
        git clone "$SGLANG_REPO" "$SGLANG_DIR"
    fi
    git -C "$SGLANG_DIR" fetch --tags
    git -C "$SGLANG_DIR" checkout "$SGLANG_REF"

    cp "$SGLANG_DIR/python/pyproject.toml" "$SGLANG_DIR/python/pyproject.toml.bak.npu_dflash"
    cp "$SGLANG_DIR/python/pyproject_npu.toml" "$SGLANG_DIR/python/pyproject.toml"

    if [[ "$SGLANG_INSTALL_NO_DEPS" == "1" ]]; then
        python -m pip install -e "$SGLANG_DIR/python[srt_npu]" --no-deps "${PIP_INDEX_ARGS[@]}"
    else
        python -m pip install -e "$SGLANG_DIR/python[srt_npu]" "${PIP_INDEX_ARGS[@]}"
    fi
fi

python -m pip install triton-ascend "${PIP_INDEX_ARGS[@]}"

if [[ "$SKIP_SGL_KERNEL_NPU" != "1" ]]; then
    if [[ ! -d "$SGL_KERNEL_DIR/.git" ]]; then
        git clone -b "$SGL_KERNEL_REF" "$SGL_KERNEL_REPO" "$SGL_KERNEL_DIR"
    fi
    git -C "$SGL_KERNEL_DIR" fetch --tags origin
    git -C "$SGL_KERNEL_DIR" checkout "$SGL_KERNEL_REF"
    (
        cd "$SGL_KERNEL_DIR"
        bash build.sh -a kernels
        python -m pip install output/sgl_kernel_npu*.whl
    )
fi

(
    cd "$ROOT_DIR"
    python -m pip install -e . --no-deps
)

cat <<EOF

NPU DFlash environment installed.

Activate it with:
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_PATH"

Then verify:
  python "$BUNDLE_DIR/verify_env.py"
EOF
