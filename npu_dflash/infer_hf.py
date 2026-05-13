#!/usr/bin/env python3
"""HF target-model + DFlash drafter inference on Ascend NPU."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BUNDLE_DIR.parent
RUNTIME_DIR = BUNDLE_DIR / "runtime"

os.environ.setdefault("SPECFORGE_DEVICE", "npu")
os.environ.setdefault("SGLANG_DEVICE", "npu")

for path in (str(RUNTIME_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import patches  # noqa: E402


def _resolve_pattern(value: str) -> str:
    matches = glob.glob(value)
    if not matches:
        return value
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def _sync(torch, device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.synchronize()


def _format_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template is None:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain speculative decoding in simple terms.",
    )
    parser.add_argument(
        "--target-model",
        default=os.environ.get("TARGET_MODEL", "/share/canada_group_folder/ckpt/Qwen3-8B"),
    )
    parser.add_argument(
        "--draft-model",
        default=os.environ.get("DRAFT_MODEL", "z-lab/Qwen3-8B-DFlash-b16"),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "npu:0"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patches.apply_all()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from specforge.modeling.draft.dflash import DFlashDraftModel

    target_model = _resolve_pattern(args.target_model)
    draft_model = _resolve_pattern(args.draft_model)

    print("=== Loading models ===")
    print(f"target : {target_model}")
    print(f"draft  : {draft_model}")
    print(f"device : {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention_backend,
        trust_remote_code=True,
    ).to(args.device).eval()
    draft = DFlashDraftModel.from_pretrained(
        draft_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention_backend,
        trust_remote_code=True,
    ).to(args.device).eval()

    text = _format_prompt(tokenizer, args.prompt, args.enable_thinking)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    _sync(torch, args.device)
    start = time.perf_counter()
    output_ids = draft.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids,
        temperature=args.temperature,
    )
    _sync(torch, args.device)
    elapsed = time.perf_counter() - start

    new_ids = output_ids[0, input_ids.shape[1] :]
    generated = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
    acceptance = getattr(draft, "_last_acceptance_lengths", [])
    tok_s = len(new_ids) / elapsed if elapsed > 0 else 0.0

    print()
    print("=== Output ===")
    print(generated)
    print()
    print("=== Stats ===")
    print(f"new tokens       : {len(new_ids)}")
    print(f"elapsed seconds  : {elapsed:.3f}")
    print(f"throughput tok/s : {tok_s:.2f}")
    if acceptance:
        print(f"accept blocks    : {len(acceptance)}")
        print(f"avg accept len   : {sum(acceptance) / len(acceptance):.2f}")
        print(f"min/max accept   : {min(acceptance)}/{max(acceptance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
