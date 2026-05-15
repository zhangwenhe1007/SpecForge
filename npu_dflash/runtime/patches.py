"""Runtime-only compatibility patches for the NPU DFlash bundle.

This module deliberately avoids editing SpecForge source files. The wrappers in
../*.sh and ../*.py import this module before running training or inference.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any


def _npu_enabled() -> bool:
    return os.environ.get("SPECFORGE_DEVICE", "").lower() in {"npu", "ascend"}


def _log(message: str) -> None:
    if os.environ.get("NPU_DFLASH_VERBOSE_PATCHES", "0") == "1":
        print(f"[npu_dflash.patch] {message}", file=sys.stderr)


def _add_choices(choices: Any, names: list[str]) -> Any:
    if choices is None:
        return choices
    if isinstance(choices, tuple):
        out = list(choices)
        for name in names:
            if name not in out:
                out.append(name)
        return tuple(out)
    if isinstance(choices, list):
        for name in names:
            if name not in choices:
                choices.append(name)
        return choices
    try:
        out = list(choices)
    except TypeError:
        return choices
    for name in names:
        if name not in out:
            out.append(name)
    return out


def activate_transfer_to_npu() -> None:
    """Activate torch_npu's CUDA-shaped compatibility layer in this process."""
    if not _npu_enabled():
        return
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    _log("transfer_to_npu activated")


def patch_sglang_attention_choices() -> None:
    """Allow NPU SGLang attention backend names through SpecForge argparse."""
    backend_names = ["ascend", "ascend_attn", "torch_native"]

    try:
        import sglang.srt.server_args as server_args

        server_args.ATTENTION_BACKEND_CHOICES = _add_choices(
            getattr(server_args, "ATTENTION_BACKEND_CHOICES", None), backend_names
        )
        _log("patched sglang.srt.server_args.ATTENTION_BACKEND_CHOICES")
    except Exception as exc:
        _log(f"skipped sglang attention choices patch: {exc}")

    try:
        import specforge.args as specforge_args

        specforge_args.ATTENTION_BACKEND_CHOICES = _add_choices(
            getattr(specforge_args, "ATTENTION_BACKEND_CHOICES", None), backend_names
        )
        _log("patched specforge.args.ATTENTION_BACKEND_CHOICES")
    except Exception as exc:
        _log(f"skipped specforge attention choices patch: {exc}")


def patch_sglang_backend_args() -> None:
    """Make SpecForge pass `device=npu` to embedded SGLang when requested."""
    try:
        import specforge.args as specforge_args
    except Exception as exc:
        _log(f"skipped SGLangBackendArgs patch: {exc}")
        return

    cls = specforge_args.SGLangBackendArgs
    if getattr(cls, "_npu_dflash_to_kwargs_patched", False):
        return

    original_to_kwargs = cls.to_kwargs

    def to_kwargs(self):
        kwargs = original_to_kwargs(self)
        if _npu_enabled():
            kwargs.setdefault("device", os.environ.get("SGLANG_DEVICE", "npu"))
        return {key: value for key, value in kwargs.items() if value is not None}

    cls.to_kwargs = to_kwargs
    cls._npu_dflash_to_kwargs_patched = True
    _log("patched SGLangBackendArgs.to_kwargs")


def _make_filtered_server_args(real_server_args):
    if getattr(real_server_args, "_npu_dflash_filtered", False):
        return real_server_args

    try:
        signature = inspect.signature(real_server_args)
        allowed = set(signature.parameters)
    except Exception:
        allowed = None

    def filtered_server_args(*args, **kwargs):
        if _npu_enabled():
            kwargs.setdefault("device", os.environ.get("SGLANG_DEVICE", "npu"))
        if allowed is not None:
            dropped = sorted(k for k in kwargs if k not in allowed)
            if dropped:
                _log(f"dropping unsupported ServerArgs kwargs: {', '.join(dropped)}")
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in allowed and value is not None
            }
        return real_server_args(*args, **kwargs)

    filtered_server_args._npu_dflash_filtered = True
    return filtered_server_args


def patch_embedded_sglang_server_args() -> None:
    """Filter ServerArgs kwargs in SpecForge embedded target backends."""
    module_names = [
        "specforge.modeling.target.dflash_target_model",
        "specforge.modeling.target.eagle3_target_model",
    ]
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["ServerArgs"])
            module.ServerArgs = _make_filtered_server_args(module.ServerArgs)
            _log(f"patched {module_name}.ServerArgs")
        except Exception as exc:
            _log(f"skipped {module_name}.ServerArgs patch: {exc}")


def patch_dflash_spec_generate() -> None:
    """Patch DFlash HF generation for NPU bool.cumprod compatibility."""
    try:
        import torch
        from transformers import DynamicCache

        from specforge.modeling.draft.dflash import (
            DFlashDraftModel,
            extract_context_feature,
            sample,
        )
    except Exception as exc:
        _log(f"skipped DFlash generation patch: {exc}")
        return

    if getattr(DFlashDraftModel.spec_generate, "_npu_dflash_safe", False):
        return

    @torch.inference_mode()
    def spec_generate(
        self,
        target,
        input_ids,
        max_new_tokens: int,
        stop_token_ids,
        temperature: float,
    ):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        device = input_ids.device

        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .to(torch.int32)
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        if self.mask_token_id is not None:
            output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids_tensor
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        self._last_acceptance_lengths = acceptance_lengths
        return output_ids

    spec_generate._npu_dflash_safe = True
    DFlashDraftModel.spec_generate = spec_generate
    _log("patched DFlashDraftModel.spec_generate")


def apply_all() -> None:
    activate_transfer_to_npu()
    patch_sglang_attention_choices()
    patch_sglang_backend_args()
    patch_embedded_sglang_server_args()
    patch_dflash_spec_generate()
