"""Materialise a runnable LLaDA-Hybrid model directory.

The released checkpoint is an adapter, not a full model: every weight outside the
hybrid attention layers is bit-identical to ``inclusionAI/LLaDA2.1-mini``. This
script joins the two into a directory SGLang can serve directly:

  config.json          base config + the hybrid layer plan + adapter path
  *.safetensors        symlinks to the base weight shards (nothing is copied)
  tokenizer files      symlinks to the base tokenizer

Nothing is duplicated on disk, so the directory costs a few kilobytes.

Usage:
    python scripts/prepare_model.py --out models/llada-hybrid-6l
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ADAPTER_DIR = os.path.join(REPO_ROOT, "checkpoints", "llada-hybrid-6l")


def resolve_base(base: str) -> str:
    """Local directory as given, or an HF repo id downloaded to the local cache."""
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download

    return snapshot_download(base)


def build_config(base_dir: str, adapter_dir: str) -> dict:
    from transformers import AutoConfig

    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        spec = json.load(f)

    cfg = json.loads(
        AutoConfig.from_pretrained(base_dir, trust_remote_code=True).to_json_string()
    )

    n_layers = int(cfg["num_hidden_layers"])
    layer_types = ["attention"] * n_layers
    for i in spec["linear_layers"]:
        layer_types[i] = "hybrid"

    cfg["architectures"] = [spec["architecture"]]
    cfg["layers_block_type"] = layer_types
    cfg["use_linear_attention"] = True
    cfg["use_block_softmax_hybrid"] = True
    cfg["use_qk_norm"] = True
    cfg["feature_dim"] = spec["feature_dim"]
    cfg["share_hedgehog_kv_groups"] = spec["share_hedgehog_kv_groups"]
    cfg["block_size"] = spec["block_size"]
    cfg["llada_hybrid_adapter"] = os.path.join(adapter_dir, "adapter.safetensors")
    cfg["llada_hybrid_lora_rank"] = spec["lora"]["r"]
    cfg["llada_hybrid_lora_alpha"] = spec["lora"]["alpha"]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="model directory to create")
    ap.add_argument(
        "--adapter-dir",
        default=DEFAULT_ADAPTER_DIR,
        help="directory holding adapter.safetensors and adapter_config.json",
    )
    ap.add_argument(
        "--base",
        default=None,
        help="base model: local path or HF repo id "
        "(default: the base_model recorded in adapter_config.json)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    adapter_dir = os.path.abspath(args.adapter_dir)
    adapter_path = os.path.join(adapter_dir, "adapter.safetensors")
    if not os.path.exists(adapter_path):
        raise SystemExit(f"adapter not found: {adapter_path}")

    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        spec = json.load(f)
    base_dir = resolve_base(args.base or spec["base_model"])
    print(f"base model: {base_dir}")

    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        if not args.overwrite:
            raise SystemExit(f"{out} already exists (pass --overwrite to replace it)")
        shutil.rmtree(out)
    os.makedirs(out)

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(build_config(base_dir, adapter_dir), f, indent=2)

    # Symlink everything else from the base snapshot: weight shards, tokenizer,
    # chat template. config.json is ours and must not be overwritten.
    for name in sorted(os.listdir(base_dir)):
        if name == "config.json":
            continue
        os.symlink(os.path.join(base_dir, name), os.path.join(out, name))

    print(f"model directory: {out}")
    print(f"adapter        : {adapter_path}")


if __name__ == "__main__":
    main()
