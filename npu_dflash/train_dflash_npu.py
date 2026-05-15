#!/usr/bin/env python3
"""NPU wrapper for SpecForge's scripts/train_dflash.py."""

from __future__ import annotations

import importlib.util
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

import patches  # noqa: E402


def main() -> None:
    patches.apply_all()

    train_path = ROOT_DIR / "scripts" / "train_dflash.py"
    spec = importlib.util.spec_from_file_location("specforge_train_dflash", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["specforge_train_dflash"] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
