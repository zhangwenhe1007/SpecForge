#!/usr/bin/env python3
"""Create a tiny conversational JSONL dataset for NPU smoke tests."""

from __future__ import annotations

import json
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
OUT_PATH = BUNDLE_DIR / "cache" / "tiny_dflash_train.jsonl"


EXAMPLES = [
    (
        "Explain speculative decoding in simple terms.",
        "Speculative decoding is a way to make a large language model answer faster. "
        "A small draft model proposes several future tokens, and the larger target "
        "model checks those proposals in one pass. Accepted draft tokens are kept, "
        "and rejected positions are replaced by the target model, so the answer keeps "
        "the target model's behavior while reducing the number of slow target steps.",
    ),
    (
        "Give three practical tips for debugging a distributed training run.",
        "First, verify that every rank sees the same code, model path, and dataset path. "
        "Second, run a one-device smoke test before increasing the world size. Third, "
        "log the rank, local rank, device id, batch shape, and first error traceback so "
        "communication bugs can be separated from model or data bugs.",
    ),
    (
        "What is the difference between a target model and a draft model?",
        "The target model is the main model whose output quality and probability "
        "distribution we want to preserve. The draft model is a smaller or cheaper "
        "model that predicts candidate tokens. In speculative decoding, the target "
        "model validates those candidates, so the draft model can improve speed "
        "without becoming the final authority.",
    ),
    (
        "Describe why a tiny smoke dataset is useful.",
        "A tiny smoke dataset is useful because it tests the complete training path "
        "quickly. It checks tokenization, chat templates, dataloading, distributed "
        "initialization, target hidden-state capture, drafter loss computation, "
        "backpropagation, optimizer updates, and checkpoint saving before spending "
        "hours on the full dataset.",
    ),
    (
        "Explain the role of CANN in an Ascend NPU workflow.",
        "CANN provides the runtime, compiler, communication libraries, and operator "
        "support needed by Ascend NPU software. Before running PyTorch NPU programs, "
        "the shell sources CANN environment scripts so commands and libraries such as "
        "HCCL, compiler tools, and runtime paths are discoverable by Python packages.",
    ),
    (
        "Why should DFlash use SDPA on NPU for the draft model?",
        "SDPA is a PyTorch attention backend that is more portable across device "
        "types than CUDA-specific flash attention kernels. On Ascend NPU, CUDA-only "
        "kernels such as FlashAttention or FA3 are not available, so using SDPA keeps "
        "the DFlash draft path simple and compatible for training smoke tests.",
    ),
    (
        "Write a short checklist for running an NPU training smoke test.",
        "Source CANN first. Activate the conda environment. Verify torch_npu imports "
        "and torch.npu reports available devices. Generate the tiny dataset. Use one "
        "visible NPU, batch size one, a short max length, and report_to none. Confirm "
        "that the run logs at least one step and writes a checkpoint directory.",
    ),
    (
        "What should you check after starting an SGLang server?",
        "After starting an SGLang server, check the health endpoint, confirm the port "
        "is listening, send one small chat request, and inspect the server log for "
        "device selection, attention backend, model loading, and speculative decoding "
        "messages. A successful small request is more useful than assuming startup "
        "alone proves the serving path works.",
    ),
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for idx, (prompt, answer) in enumerate(EXAMPLES):
            row = {
                "id": f"tiny-{idx:04d}",
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(EXAMPLES)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
