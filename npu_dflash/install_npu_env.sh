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
# Official DFlash depends on SGLang's native DFLASH speculative worker. The
# older v0.5.9 NPU tag can train through SpecForge, but it cannot serve DFlash
# checkpoints because it only exposes EAGLE-style speculative algorithms.
SGLANG_REF=${SGLANG_REF:-refs/pull/23000/head}
SGLANG_DIR=${SGLANG_DIR:-$THIRD_PARTY_DIR/sglang}
SGLANG_INSTALL_NO_DEPS=${SGLANG_INSTALL_NO_DEPS:-1}
SGLANG_RESET_THIRD_PARTY=${SGLANG_RESET_THIRD_PARTY:-1}

SGL_KERNEL_REPO=${SGL_KERNEL_REPO:-https://github.com/Sawyer117/sgl-kernel-npu.git}
SGL_KERNEL_REF=${SGL_KERNEL_REF:-npu-install-stable}
SGL_KERNEL_DIR=${SGL_KERNEL_DIR:-$THIRD_PARTY_DIR/sgl-kernel-npu}

SKIP_SGLANG_INSTALL=${SKIP_SGLANG_INSTALL:-0}
SKIP_SGL_KERNEL_NPU=${SKIP_SGL_KERNEL_NPU:-0}
SKIP_PYTHON_DEPS=${SKIP_PYTHON_DEPS:-0}

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

checkout_git_ref() {
    local repo_dir=$1
    local ref=$2

    if [[ "$SGLANG_RESET_THIRD_PARTY" == "1" ]]; then
        git -C "$repo_dir" reset --hard
    fi

    git -C "$repo_dir" fetch --tags origin
    if git -C "$repo_dir" checkout "$ref"; then
        return
    fi

    git -C "$repo_dir" fetch origin "$ref"
    git -C "$repo_dir" checkout FETCH_HEAD
}

source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    conda create -p "$ENV_PATH" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_PATH"

if [[ "$SKIP_PYTHON_DEPS" != "1" ]]; then
    python -m pip install -r "$BUNDLE_DIR/requirements-ascend.txt"
    python -m pip install triton-ascend "${PIP_INDEX_ARGS[@]}"
else
    echo "Skipping Python dependency install because SKIP_PYTHON_DEPS=1."
fi

if [[ "$SKIP_SGLANG_INSTALL" != "1" ]]; then
    if [[ ! -d "$SGLANG_DIR/.git" ]]; then
        git clone "$SGLANG_REPO" "$SGLANG_DIR"
    fi
    git -C "$SGLANG_DIR" remote set-url origin "$SGLANG_REPO"
    checkout_git_ref "$SGLANG_DIR" "$SGLANG_REF"

    if [[ ! -f "$SGLANG_DIR/python/pyproject_npu.toml" ]]; then
        cat >&2 <<EOF
ERROR: SGLang ref '$SGLANG_REF' does not contain python/pyproject_npu.toml.

Use a ref with both Ascend NPU packaging and native DFLASH support, for example:
  SGLANG_REF=refs/pull/23000/head bash npu_dflash/install_npu_env.sh
EOF
        exit 1
    fi

    cp "$SGLANG_DIR/python/pyproject.toml" "$SGLANG_DIR/python/pyproject.toml.bak.npu_dflash"
    cp "$SGLANG_DIR/python/pyproject_npu.toml" "$SGLANG_DIR/python/pyproject.toml"

    if [[ "$SGLANG_INSTALL_NO_DEPS" == "1" ]]; then
        python -m pip install -e "$SGLANG_DIR/python[srt_npu]" --no-deps "${PIP_INDEX_ARGS[@]}"
    else
        python -m pip install -e "$SGLANG_DIR/python[srt_npu]" "${PIP_INDEX_ARGS[@]}"
    fi

    python "$BUNDLE_DIR/patch_installed_sglang.py"
    python "$BUNDLE_DIR/check_sglang_dflash_support.py" --quiet --draft-backend ascend
fi

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
