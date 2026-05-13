#!/usr/bin/env python3
"""Verify the NPU DFlash runtime environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BUNDLE_DIR.parent
RUNTIME_DIR = BUNDLE_DIR / "runtime"

os.environ.setdefault("SPECFORGE_DEVICE", "npu")
os.environ.setdefault("SGLANG_DEVICE", "npu")

for path in (str(RUNTIME_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _import(name: str):
    print(f"import {name:<20} ... ", end="", flush=True)
    module = importlib.import_module(name)
    print("OK")
    return module


def main() -> int:
    print("=== npu_dflash environment verification ===")
    print(f"ROOT_DIR   : {ROOT_DIR}")
    print(f"BUNDLE_DIR : {BUNDLE_DIR}")
    print(f"SPECFORGE_DEVICE : {os.environ.get('SPECFORGE_DEVICE')}")
    print()

    try:
        import patches

        patches.apply_all()
    except Exception as exc:
        print(f"ERROR: failed to apply runtime patches: {exc}", file=sys.stderr)
        return 1

    try:
        torch = _import("torch")
        torch_npu = _import("torch_npu")
        transformers = _import("transformers")
        sglang = _import("sglang")
        sgl_kernel_npu = _import("sgl_kernel_npu")
        triton = _import("triton")
        specforge = _import("specforge")
    except Exception as exc:
        print(f"\nERROR: import verification failed: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"torch                    : {getattr(torch, '__version__', 'unknown')}")
    print(f"torch_npu                : {getattr(torch_npu, '__version__', _version('torch-npu'))}")
    print(f"transformers             : {getattr(transformers, '__version__', 'unknown')}")
    print(f"sglang                   : {getattr(sglang, '__version__', _version('sglang'))}")
    print(f"sgl_kernel_npu path      : {getattr(sgl_kernel_npu, '__path__', 'unknown')}")
    print(f"triton                   : {getattr(triton, '__version__', _version('triton-ascend'))}")
    print(f"triton file              : {getattr(triton, '__file__', 'unknown')}")
    print(f"specforge file           : {getattr(specforge, '__file__', 'unknown')}")

    print()
    has_npu_attr = hasattr(torch, "npu")
    print(f"torch has npu attr       : {has_npu_attr}")
    if not has_npu_attr:
        print("ERROR: torch.npu is missing", file=sys.stderr)
        return 1

    available = torch.npu.is_available()
    device_count = torch.npu.device_count()
    print(f"torch.npu.is_available() : {available}")
    print(f"torch.npu.device_count() : {device_count}")
    if not available or device_count <= 0:
        print("ERROR: no available NPU was detected", file=sys.stderr)
        return 1

    print()
    print("Environment looks ready for NPU DFlash smoke tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
