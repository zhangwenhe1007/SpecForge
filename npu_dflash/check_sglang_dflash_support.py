#!/usr/bin/env python3
"""Check that the active SGLang install can serve native DFlash drafts."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path
import sys


def _ok(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"OK: {message}")


def _failures_hint() -> str:
    return """

Install or repair the DFlash-capable SGLang source with:

  SGLANG_REF=refs/pull/23000/head \\
  SGLANG_INSTALL_NO_DEPS=1 \\
  SKIP_SGL_KERNEL_NPU=1 \\
  bash npu_dflash/install_npu_env.sh

DFlash checkpoints should be served with --speculative-algorithm DFLASH.
Using EAGLE3 makes SGLang load the DFlash checkpoint as a generic HF draft
model and commonly fails at get_input_embeddings().
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-backend", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []

    try:
        import sglang

        sglang_path = Path(inspect.getfile(sglang)).resolve()
        version = getattr(sglang, "__version__", "unknown")
        _ok(f"sglang import path: {sglang_path}", quiet=args.quiet)
        _ok(f"sglang version: {version}", quiet=args.quiet)
    except Exception as exc:
        print(f"ERROR: failed to import sglang: {exc}", file=sys.stderr)
        print(_failures_hint(), file=sys.stderr)
        return 1

    try:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if "DFLASH" not in SpeculativeAlgorithm.__members__:
            errors.append("SpeculativeAlgorithm does not contain DFLASH.")
        else:
            _ok("SpeculativeAlgorithm contains DFLASH", quiet=args.quiet)
    except Exception as exc:
        errors.append(f"Could not inspect SpeculativeAlgorithm: {exc}")

    try:
        from sglang.srt.server_args import ServerArgs

        if not hasattr(ServerArgs, "speculative_dflash_block_size"):
            errors.append("ServerArgs has no speculative_dflash_block_size field.")
        else:
            _ok("ServerArgs contains DFlash block-size field", quiet=args.quiet)
    except Exception as exc:
        errors.append(f"Could not inspect ServerArgs: {exc}")

    if importlib.util.find_spec("sglang.srt.models.dflash") is None:
        errors.append("sglang.srt.models.dflash is not importable.")
    else:
        _ok("native SGLang DFlash model module is importable", quiet=args.quiet)

    if args.draft_backend == "ascend":
        try:
            from sglang.srt.speculative.dflash_worker import DFlashWorker

            worker_path = Path(inspect.getsourcefile(DFlashWorker) or "").resolve()
            worker_text = worker_path.read_text(encoding="utf-8")
            if "supported_draft_backends" in worker_text and "ascend" not in worker_text:
                errors.append(
                    "DFlashWorker does not allow the ascend draft attention backend. "
                    "Run npu_dflash/patch_installed_sglang.py."
                )
            else:
                _ok("DFlashWorker allows ascend draft backend", quiet=args.quiet)
        except Exception as exc:
            errors.append(f"Could not inspect DFlashWorker backend support: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(_failures_hint(), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
