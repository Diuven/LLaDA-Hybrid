"""Fuse the two training deltas into the single LLaDA-Hybrid adapter file.

Training produced two separate artefacts:

  linear_attn_delta.pt   hedgehog feature map + alpha for each hybrid layer
                         (stage 1: attention distillation)
  lora_delta.pt          PEFT LoRA factors on query_key_value / dense
                         (stage 2: LoRA recovery)

Neither contains any base weight. Every other parameter of the served model is
bit-identical to the public ``inclusionAI/LLaDA2.1-mini`` checkpoint, which is
why the released adapter is ~13 MB rather than a full 16B copy.

This script merges the two into ``adapter.safetensors`` and rewrites the keys
into the names the SGLang model actually looks up, so nothing has to be remapped
at load time:

  hedgehog.model.layers.{i}.attention.hybrid_core.hedgehog_weights
  hedgehog.model.layers.{i}.attention.hybrid_core.alpha
  lora.model.layers.{i}.attention.{query_key_value,dense}.lora_{A,B}

The LoRA factors are kept factored (A and B, fp32, unscaled). They are merged as
``W += (alpha / r) * B @ A`` at load, in fp32, which is exactly the arithmetic
the reported results were measured with; pre-multiplying the scaling or
pre-materialising the product would change the rounding.

Usage:
    python scripts/build_adapter.py \
        --distill  path/to/step_15000/linear_attn_delta.pt \
        --lora     path/to/step_300/lora_delta.pt \
        --out      checkpoints/llada-hybrid-6l/adapter.safetensors
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file

# HF training module name -> SGLang module name for the hybrid attention core.
_HF_TO_SGLANG = (".linear_attention.", ".hybrid_core.")

# PEFT wraps every key; strip the wrapper and the adapter-name infix.
_PEFT_PREFIX = "base_model.model."
_PEFT_SUFFIXES = {".lora_A.default.weight": ".lora_A", ".lora_B.default.weight": ".lora_B"}


def convert_hedgehog(path: str) -> "OrderedDict[str, torch.Tensor]":
    state = torch.load(path, map_location="cpu", weights_only=True)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in state.items():
        name = key.replace(*_HF_TO_SGLANG)
        if name.endswith(".alpha"):
            # Stored during training as [1, H, 1, 1]; the model parameter is [H].
            # Squeeze here so the checkpoint matches the parameter it loads into.
            tensor = tensor.reshape(-1)
        out[f"hedgehog.{name}"] = tensor.contiguous()
    return out


def convert_lora(path: str) -> "OrderedDict[str, torch.Tensor]":
    state = torch.load(path, map_location="cpu", weights_only=True)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in state.items():
        if not key.startswith(_PEFT_PREFIX):
            raise ValueError(f"unexpected LoRA key (no PEFT prefix): {key}")
        name = key[len(_PEFT_PREFIX) :]
        for suffix, short in _PEFT_SUFFIXES.items():
            if name.endswith(suffix):
                name = name[: -len(suffix)] + short
                break
        else:
            raise ValueError(f"unexpected LoRA key (no lora_A/lora_B suffix): {key}")
        out[f"lora.{name}"] = tensor.contiguous()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distill", required=True, help="linear_attn_delta.pt from stage 1")
    ap.add_argument("--lora", required=True, help="lora_delta.pt from stage 2")
    ap.add_argument("--out", required=True, help="output adapter.safetensors")
    args = ap.parse_args()

    tensors = convert_hedgehog(args.distill)
    n_hedgehog = len(tensors)
    tensors.update(convert_lora(args.lora))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out)

    size = os.path.getsize(args.out)
    print(f"wrote {args.out}")
    print(f"  hedgehog tensors : {n_hedgehog}")
    print(f"  lora tensors     : {len(tensors) - n_hedgehog}")
    print(f"  file size        : {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
