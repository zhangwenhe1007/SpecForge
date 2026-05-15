import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer


def npu_sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def main():
    model_path = os.environ.get("TARGET_MODEL", "/home/f00518697/models/Qwen3-8B")
    device = os.environ.get("DEVICE", "npu:0")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))

    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = "Explain speculative decoding simply."

    print("=== Loading base model ===")
    print(f"target : {model_path}")
    print(f"device : {device}")
    print(f"max_new_tokens : {max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    model = model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    npu_sync()
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    npu_sync()
    elapsed = time.time() - start

    new_tokens = output_ids.shape[-1] - input_ids.shape[-1]
    throughput = new_tokens / elapsed if elapsed > 0 else float("nan")

    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    print("\n=== Output ===")
    print(generated)

    print("\n=== Stats ===")
    print(f"new tokens       : {new_tokens}")
    print(f"elapsed seconds  : {elapsed:.3f}")
    print(f"throughput tok/s : {throughput:.2f}")


if __name__ == "__main__":
    main()
