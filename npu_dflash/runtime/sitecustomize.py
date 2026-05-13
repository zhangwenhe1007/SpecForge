"""Opt-in Ascend NPU compatibility shim for SpecForge.

Python imports this file automatically when this directory is on PYTHONPATH.
The shim is intentionally inert unless SPECFORGE_DEVICE=npu.
"""

import os
import sys


def _enabled() -> bool:
    return os.environ.get("SPECFORGE_DEVICE", "").lower() in {"npu", "ascend"}


if _enabled():
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - only meaningful on NPU hosts
        message = f"[npu_dflash] Failed to activate transfer_to_npu: {exc}\n"
        if os.environ.get("SPECFORGE_NPU_STRICT", "0") == "1":
            raise
        sys.stderr.write(message)
