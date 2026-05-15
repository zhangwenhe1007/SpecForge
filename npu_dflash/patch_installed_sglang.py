#!/usr/bin/env python3
"""Apply small source patches to the editable SGLang NPU checkout.

The patches are intentionally limited to generated third-party code under
``npu_dflash/third_party`` or the active editable SGLang source tree. They avoid
tracked SpecForge edits while keeping the NPU serving path reproducible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


def _sglang_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None:
        raise RuntimeError("sglang is not importable")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin is None:
        raise RuntimeError("could not resolve sglang source path")
    return Path(spec.origin).resolve().parent


def _write_if_changed(path: Path, text: str) -> bool:
    original = path.read_text(encoding="utf-8")
    if text == original:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def patch_deepgemm(root: Path) -> str:
    path = root / "srt/layers/deep_gemm_wrapper/configurer.py"
    if not path.exists():
        return "missing"

    text = path.read_text(encoding="utf-8")
    patched = re.sub(
        r"^ENABLE_JIT_DEEPGEMM\s*=.*$",
        "ENABLE_JIT_DEEPGEMM = False  # patched by npu_dflash for Ascend NPU",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return "patched" if _write_if_changed(path, patched) else "already"


def patch_dflash_worker(root: Path) -> str:
    path = root / "srt/speculative/dflash_worker.py"
    if not path.exists():
        return "missing"

    text = path.read_text(encoding="utf-8")
    patched = text.replace(
        'supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton")',
        'supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton", "ascend")',
        1,
    )
    return "patched" if _write_if_changed(path, patched) else "already"


def main() -> int:
    try:
        root = _sglang_root()
    except Exception as exc:
        print(f"[npu_dflash] Could not locate SGLang: {exc}", file=sys.stderr)
        return 1

    results = {
        "deep_gemm": patch_deepgemm(root),
        "dflash_worker": patch_dflash_worker(root),
    }
    for name, result in results.items():
        print(f"[npu_dflash] SGLang patch {name}: {result}")

    if results["dflash_worker"] == "missing":
        print(
            "[npu_dflash] WARNING: this SGLang install has no native DFlash worker.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
