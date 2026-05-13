# NPU DFlash Bundle for SpecForge

This folder is a self-contained Ascend NPU workflow for Qwen3-8B DFlash in
SpecForge. It is designed to be committed as a single folder, pulled on the NPU
server, installed there, and run there.

The parent SpecForge source tree is used as-is. This bundle does not require
editing tracked SpecForge files on the NPU server.

## What This Folder Contains

- `requirements-ascend.txt`: pinned Ascend-compatible Python dependencies.
- `install_npu_env.sh`: creates `npu_dflash/conda/specforge_npu`, installs deps,
  installs SGLang NPU, builds `sgl_kernel_npu`, and editable-installs SpecForge.
- `runtime/sitecustomize.py`: opt-in `torch_npu.contrib.transfer_to_npu` shim.
- `runtime/patches.py`: runtime-only compatibility patches.
- `train_dflash_npu.py`: wrapper around `scripts/train_dflash.py`.
- `run_train_hf.sh`: DFlash training on NPU with HF target backend.
- `run_train_sglang.sh`: DFlash training on NPU with embedded SGLang target backend.
- `infer_hf.py`: HF target plus DFlash drafter inference on NPU.
- `run_sglang_dflash_server.sh`: OpenAI-compatible SGLang DFlash server on NPU.
- `chat_client.py`: small OpenAI-compatible client for the SGLang server.
- `make_tiny_dataset.py`: creates a tiny JSONL smoke-test dataset.
- `verify_env.py`: verifies imports and NPU availability.

Generated files stay under ignored paths in this folder:

- `npu_dflash/conda/`
- `npu_dflash/third_party/`
- `npu_dflash/outputs/`
- `npu_dflash/cache/`
- `npu_dflash/logs/`

## Runtime Strategy

The existing SpecForge training code is CUDA-shaped. On Ascend NPU, this bundle
activates `torch_npu.contrib.transfer_to_npu` by setting:

```bash
export SPECFORGE_DEVICE=npu
export PYTHONPATH="$PWD/npu_dflash/runtime:$PWD:${PYTHONPATH:-}"
```

The runtime patches do four things:

- map CUDA-shaped PyTorch calls onto NPU via `transfer_to_npu`;
- filter embedded SGLang `ServerArgs` kwargs against the installed SGLang version;
- pass `device=npu` to embedded SGLang when supported;
- make HF DFlash generation NPU-safe by casting the acceptance bool tensor before
  `cumprod()`.

The default SGLang attention backend is `ascend`. Do not use CUDA-only backends
such as `fa3` or `flashinfer` on NPU.

## Install on the NPU Server

Start inside your SpecForge checkout:

```bash
cd /path/to/SpecForge
```

Source CANN first:

```bash
export CANN_HOME=/path/to/CANN/8.5.0.x
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ] && source "$CANN_HOME/nnal/atb/set_env.sh"
```

Install the environment:

```bash
bash npu_dflash/install_npu_env.sh
```

Activate it:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./npu_dflash/conda/specforge_npu
```

Verify:

```bash
python npu_dflash/verify_env.py
python npu_dflash/make_tiny_dataset.py
```

`verify_env.py` should show successful imports for `torch_npu`, `sglang`,
`sgl_kernel_npu`, `triton`, and `specforge`, plus:

```text
torch.npu.is_available() : True
torch.npu.device_count() : <positive number>
```

## Dataset Regimes

Use these regimes in order. The tiny dataset proves the code path works; the
PerfectBlend and CodeAlpaca samples are small real training sets; regenerated
PerfectBlend is the most representative DFlash training regime because the
assistant answers come from the same Qwen3-8B target model you train against.

### Regime 0: Tiny Smoke Dataset

This is already created during verification:

```bash
python npu_dflash/make_tiny_dataset.py
```

Use it only for install/runtime smoke tests:

```text
npu_dflash/cache/tiny_dflash_train.jsonl
```

### Regime 1: PerfectBlend 2K Quick Training

This is the best first small real dataset. It uses original PerfectBlend
assistant answers, not Qwen3-regenerated answers.

```bash
python scripts/prepare_data.py \
  --dataset perfectblend \
  --sample-size 2048 \
  --output-path npu_dflash/cache/dataset/perfectblend_2k
```

Train with:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/dataset/perfectblend_2k/perfectblend_train.jsonl \
NUM_NPUS=1 ASCEND_RT_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 MAX_LENGTH=1024 NUM_ANCHORS=64 NUM_EPOCHS=1 \
REPORT_TO=none \
bash npu_dflash/run_train_hf.sh
```

### Regime 2: PerfectBlend 10K Short Real Run

Use this after 2K works. It is still small, but it is large enough to give a
more meaningful drafter checkpoint than the smoke datasets.

```bash
python scripts/prepare_data.py \
  --dataset perfectblend \
  --sample-size 10000 \
  --output-path npu_dflash/cache/dataset/perfectblend_10k
```

Train with:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/dataset/perfectblend_10k/perfectblend_train.jsonl \
NUM_NPUS=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 ACCUMULATION_STEPS=1 MAX_LENGTH=2048 NUM_ANCHORS=256 NUM_EPOCHS=1 \
REPORT_TO=tensorboard \
bash npu_dflash/run_train_hf.sh
```

### Regime 3: CodeAlpaca 5K Coding Run

Use this if you want a small code-heavy drafter experiment.

```bash
python scripts/prepare_data.py \
  --dataset codealpaca-20k \
  --sample-size 5000 \
  --output-path npu_dflash/cache/dataset/codealpaca_5k
```

Train with:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/dataset/codealpaca_5k/codealpaca-20k_train.jsonl \
NUM_NPUS=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 ACCUMULATION_STEPS=1 MAX_LENGTH=2048 NUM_ANCHORS=256 NUM_EPOCHS=1 \
REPORT_TO=tensorboard \
bash npu_dflash/run_train_hf.sh
```

### Regime 4: PerfectBlend With Qwen3-8B-Regenerated Answers

This is the recommended meaningful DFlash training regime. First prepare seed
prompts, then run a Qwen3-8B SGLang target server, then regenerate assistant
answers with that server.

Prepare the seed file:

```bash
python scripts/prepare_data.py \
  --dataset perfectblend \
  --sample-size 10000 \
  --output-path npu_dflash/cache/dataset/perfectblend_10k_seed
```

Start a plain Qwen3-8B SGLang server in shell 1:

```bash
export TARGET_MODEL=/path/to/Qwen3-8B

ASCEND_RT_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --device npu \
  --model-path "$TARGET_MODEL" \
  --served-model-name qwen3-8b \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 1 \
  --attention-backend ascend \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

For an 8-NPU regeneration server, use:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m sglang.launch_server \
  --device npu \
  --model-path "$TARGET_MODEL" \
  --served-model-name qwen3-8b \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 8 \
  --attention-backend ascend \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

Check the server from shell 2:

```bash
curl http://127.0.0.1:30000/health
```

Regenerate answers from shell 2:

```bash
python scripts/regenerate_train_data.py \
  --model qwen3-8b \
  --server-address 127.0.0.1:30000 \
  --input-file-path ./npu_dflash/cache/dataset/perfectblend_10k_seed/perfectblend_train.jsonl \
  --output-file-path ./npu_dflash/cache/dataset/perfectblend_10k_qwen3_regen.jsonl \
  --num-samples 10000 \
  --concurrency 32 \
  --max-tokens 1024 \
  --temperature 0.7 \
  --resume
```

Train on the regenerated file:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/dataset/perfectblend_10k_qwen3_regen.jsonl \
NUM_NPUS=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 ACCUMULATION_STEPS=1 MAX_LENGTH=3072 NUM_ANCHORS=512 NUM_EPOCHS=3 \
REPORT_TO=tensorboard \
bash npu_dflash/run_train_hf.sh
```

After HF regenerated training works, run the same regenerated dataset through
the embedded SGLang training backend:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/dataset/perfectblend_10k_qwen3_regen.jsonl \
NUM_NPUS=8 TP_SIZE=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 MAX_LENGTH=3072 NUM_ANCHORS=512 NUM_EPOCHS=3 \
REPORT_TO=tensorboard \
bash npu_dflash/run_train_sglang.sh
```

## HF Backend Training Smoke Test

Use one NPU first:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/tiny_dflash_train.jsonl \
NUM_NPUS=1 ASCEND_RT_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 MAX_LENGTH=512 NUM_ANCHORS=16 NUM_EPOCHS=1 \
REPORT_TO=none \
bash npu_dflash/run_train_hf.sh
```

The checkpoint should appear under:

```text
npu_dflash/outputs/qwen3-8b-dflash-hf/
```

For a larger HF run, point `TRAIN_DATA` at your regenerated dataset and increase
`NUM_NPUS`, `ASCEND_RT_VISIBLE_DEVICES`, `MAX_LENGTH`, `NUM_ANCHORS`, and
`NUM_EPOCHS` as needed:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=/path/to/perfectblend_qwen3-8b_regen.jsonl \
NUM_NPUS=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 ACCUMULATION_STEPS=1 MAX_LENGTH=3072 NUM_ANCHORS=512 NUM_EPOCHS=6 \
REPORT_TO=tensorboard \
bash npu_dflash/run_train_hf.sh
```

## HF Backend Inference

Use a trained checkpoint, or omit `DRAFT_MODEL` to try the official DFlash draft
from Hugging Face:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
DRAFT_MODEL=./npu_dflash/outputs/qwen3-8b-dflash-hf/epoch_1_step_* \
python npu_dflash/infer_hf.py "Explain speculative decoding simply."
```

If `DRAFT_MODEL` contains a wildcard, `infer_hf.py` picks the newest matching
checkpoint.

## Embedded SGLang Backend Training Smoke Test

After the HF smoke path works, try embedded SGLang:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=./npu_dflash/cache/tiny_dflash_train.jsonl \
NUM_NPUS=1 TP_SIZE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 MAX_LENGTH=512 NUM_ANCHORS=16 NUM_EPOCHS=1 \
REPORT_TO=none \
bash npu_dflash/run_train_sglang.sh
```

Defaults:

- `SGLANG_ATTENTION_BACKEND=ascend`
- `SGLANG_MEM_FRACTION_STATIC=0.4`
- `SGLANG_DEVICE=npu`

For a full single-node 8-NPU run:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA=/path/to/perfectblend_qwen3-8b_regen.jsonl \
NUM_NPUS=8 TP_SIZE=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 MAX_LENGTH=3072 NUM_ANCHORS=512 NUM_EPOCHS=6 \
REPORT_TO=tensorboard \
bash npu_dflash/run_train_sglang.sh
```

If SGLang OOMs, lower `SGLANG_MEM_FRACTION_STATIC`. If the installed SGLang build
does not recognize `ascend`, try:

```bash
SGLANG_ATTENTION_BACKEND=torch_native bash npu_dflash/run_train_sglang.sh
```

## SGLang DFlash Serving

Start the server:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
DRAFT_MODEL=/path/to/dflash/checkpoint \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash npu_dflash/run_sglang_dflash_server.sh
```

Optional server settings:

```bash
PORT=30000
TP_SIZE=1
SGLANG_ATTENTION_BACKEND=ascend
SGLANG_DRAFT_ATTENTION_BACKEND=ascend
SGLANG_MEM_FRACTION_STATIC=0.75
EXTRA_SGLANG_ARGS="--some-extra-flag value"
```

In a second shell:

```bash
curl http://127.0.0.1:30000/health
python npu_dflash/chat_client.py "Explain speculative decoding simply."
```

## Common Failures

### `CANN is not sourced`

Source the CANN scripts before running `install_npu_env.sh` or any NPU script.
`ASCEND_HOME_PATH` should be non-empty after sourcing.

### `torch_npu==2.9.0` not found

Try:

```bash
python -m pip install torch_npu -i https://mirrors.huaweicloud.com/repository/pypi/simple/ --trusted-host mirrors.huaweicloud.com
```

If pip installs a post-release such as `2.9.0.post1`, update
`requirements-ascend.txt` accordingly.

### SGLang install stalls on CUDA packages

Re-run with dependency resolution disabled and then install missing import-time
packages one by one:

```bash
SGLANG_INSTALL_NO_DEPS=1 bash npu_dflash/install_npu_env.sh
python -c "import sglang.srt.managers.mm_utils"
```

Do not install CUDA-only packages such as `flashinfer-python`, `sgl-kernel`, or
`vllm-flash-attn` on NPU.

### `sgl_kernel_npu` build uses the wrong CANN path

This bundle installs from `Sawyer117/sgl-kernel-npu` branch
`npu-install-stable`, which is meant to respect the sourced CANN path. Confirm:

```bash
which msopgen
echo "$ASCEND_HOME_PATH"
```

Both should point at your intended CANN installation.

## Local Laptop Checks

On a non-NPU machine, do not expect `verify_env.py` to pass. The useful local
checks are syntax and shell dry inspection:

```bash
python -m py_compile npu_dflash/*.py npu_dflash/runtime/*.py
bash -n npu_dflash/*.sh
```

The actual training and inference checks must run on the NPU server.
