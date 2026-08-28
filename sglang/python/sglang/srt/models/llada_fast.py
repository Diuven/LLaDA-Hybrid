"""
LLaDA Fast — LLaDA 2.0 with selected attention layers replaced by
block-wise softmax + linear (hedgehog) hybrid attention.

SGLang integration follows the Mamba hybrid pattern exactly:
  - HybridReqToTokenPool  allocated by model_runner (via mambaish_config)
  - MambaPool             holds S_state / Z_state per (layer, slot)
  - LLaDAFastAttnBackend  translates req_pool_indices → mamba_cache_indices
  - LLaDAFastHybridAttention  replaces selected layers at __init__ time
  - No global state, no monkey-patching, no lazy init in forward()
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.llada_fast import LLADA_FAST_HYBRID_TYPES
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llada2 import LLaDA2MoeModelLM
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import add_prefix
from sglang.srt.layers.attention.llada_fast_hybrid_kernel import BlockSoftmaxLinearHybrid

logger = logging.getLogger(__name__)


class LLaDAFastHybridAttention(nn.Module):
    """
    Drop-in replacement for LLaDA2MoeAttention on hybrid layers.

    Forward path:
      hidden_states → QKV proj → optional QK norm → RoPE
        → attn_backend.linear_attn_backend.forward_qkv(q, k, v, ...)
        → dense proj
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id  = layer_id
        self.alt_stream = alt_stream

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        total_heads    = int(config.num_attention_heads)
        total_kv_heads = int(config.num_key_value_heads)
        head_dim = int(
            getattr(config, "head_dim", None) or (config.hidden_size // total_heads)
        )

        self.num_heads    = total_heads    // attn_tp_size
        self.num_kv_heads = max(1, total_kv_heads // attn_tp_size)
        self.head_dim     = head_dim
        self.q_size       = head_dim * self.num_heads
        self.kv_size      = head_dim * self.num_kv_heads
        self.use_qk_norm  = bool(getattr(config, "use_qk_norm", True))

        self.query_key_value = QKVParallelLinear(
            config.hidden_size,
            head_dim,
            total_heads,
            total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.key_layernorm   = RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            total_heads * head_dim,
            config.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        rot_dim = (
            int(head_dim * config.partial_rotary_factor)
            if hasattr(config, "partial_rotary_factor")
            else getattr(config, "rotary_dim", head_dim)
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=rot_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.hybrid_core = BlockSoftmaxLinearHybrid(
            config=config,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.query_layernorm,
                k_norm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )

        q, k = self.rotary_emb(positions, q, k)

        backend = forward_batch.attn_backend.linear_attn_backend
        T = q.shape[0]
        q3 = q.view(T, self.num_heads, self.head_dim)
        k3 = k.view(T, self.num_kv_heads, self.head_dim)
        v3 = v.view(T, self.num_kv_heads, self.head_dim)
        attn_output = backend.forward_qkv(
            mixer=self.hybrid_core,
            q=q3,
            k=k3,
            v=v3,
            layer_id=self.layer_id,
        )

        attn_output, _ = self.dense(attn_output)
        return attn_output


class LLaDAFastForCausalLM(LLaDA2MoeModelLM):
    """
    LLaDA 2.0 with hybrid attention layers.

    Layer swapping happens at __init__ — standard SGLang model lifecycle,
    no forward-time pool creation, no global state.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)

        block_types = getattr(
            config, "layers_block_type", ["attention"] * config.num_hidden_layers
        )
        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        for i in range(self.model.start_layer, self.model.end_layer):
            if str(block_types[i]) in LLADA_FAST_HYBRID_TYPES:
                layer_prefix = add_prefix(
                    f"layers.{i}.attention",
                    add_prefix("model", prefix),
                )
                self.model.layers[i].attention = LLaDAFastHybridAttention(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=layer_prefix,
                    alt_stream=alt_stream,
                )
                sys.stderr.write(f"[LLaDAFast] layer {i} → hybrid attention\n")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load the stock base weights, then apply the LLaDA-Hybrid adapter.

        Everything that differs from the public ``inclusionAI/LLaDA2.1-mini``
        checkpoint lives in one safetensors file, keyed by role:

          ``hedgehog.<sglang param name>``
              Feature map and alpha for each hybrid layer.  These are new
              parameters (the base checkpoint has nothing to load into them),
              so they are *copied* in.

          ``lora.<module>.lora_{A,B}``
              LoRA factors for query_key_value / dense.  These are *merged*
              into the base weights here: ``W += (alpha / r) * B @ A``.

        Merging at load is what makes the served network architecturally
        identical to the distilled model — same kernels, same FLOPs, same KV
        footprint, and no adapter cost at inference time.
        """
        super().load_weights(weights)

        adapter_path = getattr(self.config, "llada_hybrid_adapter", None)
        if not adapter_path:
            raise RuntimeError(
                "[LLaDAFast] config.llada_hybrid_adapter is not set. The hybrid "
                "layers have no trained weights without it, so refusing to run "
                "with a randomly initialised feature map."
            )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"[LLaDAFast] adapter not found: {adapter_path}"
            )

        from safetensors.torch import load_file

        sys.stderr.write(f"[LLaDAFast] Loading adapter from {adapter_path}\n")
        adapter = load_file(adapter_path)

        hedgehog = {k[len("hedgehog.") :]: v for k, v in adapter.items() if k.startswith("hedgehog.")}
        lora = {k[len("lora.") :]: v for k, v in adapter.items() if k.startswith("lora.")}
        unknown = [k for k in adapter if not k.startswith(("hedgehog.", "lora."))]
        if unknown:
            raise RuntimeError(
                f"[LLaDAFast] adapter has {len(unknown)} keys with no known role "
                f"prefix (expected 'hedgehog.' or 'lora.'): {unknown[:5]}"
            )

        params = dict(self.named_parameters())
        tp_rank = get_attention_tp_rank()
        tp_size = get_attention_tp_size()

        self._load_hedgehog(hedgehog, params, tp_rank, tp_size)
        self._merge_lora(lora, params, tp_rank, tp_size)

    def _load_hedgehog(
        self,
        hedgehog: dict,
        params: dict,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Copy the distilled feature-map parameters into the hybrid layers."""
        loaded, skipped = 0, []
        for name, tensor in hedgehog.items():
            param = params.get(name)
            if param is None:
                skipped.append(name)
                continue
            # TP-shard tensors stored whole (head dim larger than the local param).
            if tensor.shape != param.shape and tp_size > 1:
                tensor = self._tp_shard_delta(name, tensor, tp_rank, tp_size)
            # Squeeze to match the param layout (e.g. alpha [1, H, 1, 1] -> [H]).
            # Raise explicitly rather than silently corrupting weights.
            if tensor.shape != param.shape:
                try:
                    tensor = tensor.reshape(param.shape)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"[LLaDAFast] hedgehog shape mismatch for '{name}': "
                        f"adapter {tuple(tensor.shape)} cannot be reshaped to "
                        f"param {tuple(param.shape)}"
                    ) from e
            param.data.copy_(tensor.to(param.device, dtype=param.dtype))
            loaded += 1

        if loaded == 0:
            raise RuntimeError(
                "[LLaDAFast] adapter carried no hedgehog parameters that match this "
                f"model. Adapter keys: {list(hedgehog)[:5]}"
            )
        sys.stderr.write(
            f"[LLaDAFast] hedgehog: {loaded} tensors loaded"
            + (f", {len(skipped)} skipped: {skipped[:5]}" if skipped else "")
            + "\n"
        )
        sys.stderr.flush()

    def _merge_lora(
        self,
        lora: dict,
        params: dict,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Pre-merge the LoRA factors into the base weights.

        ``W_merged = W_base + (alpha / r) * B @ A``, computed in fp32 and
        accumulated into the parameter's own dtype.

        For TP > 1:
          - query_key_value (column parallel) -> shard the delta on dim 0
          - dense           (row parallel)    -> shard the delta on dim 1
        """
        if not lora:
            return

        rank = int(getattr(self.config, "llada_hybrid_lora_rank", 16))
        alpha = int(getattr(self.config, "llada_hybrid_lora_alpha", 8))
        scaling = alpha / rank

        sys.stderr.write(
            f"[LLaDAFast] LoRA merge (r={rank}, alpha={alpha}, scaling={scaling})\n"
        )

        # Group lora_A / lora_B by the base module they modify.
        pairs: dict = {}
        for key, value in lora.items():
            if key.endswith(".lora_A"):
                pairs.setdefault(key[: -len(".lora_A")] + ".weight", {})["A"] = value
            elif key.endswith(".lora_B"):
                pairs.setdefault(key[: -len(".lora_B")] + ".weight", {})["B"] = value
            else:
                raise RuntimeError(f"[LLaDAFast] unexpected LoRA key: {key}")

        merged, missed = 0, []
        for param_name, ab in pairs.items():
            if "A" not in ab or "B" not in ab or param_name not in params:
                missed.append(param_name)
                continue

            A = ab["A"].float()  # [r, in_features]
            B = ab["B"].float()  # [out_features, r]
            delta = scaling * (B @ A)  # [out_features, in_features]
            param = params[param_name]

            if delta.shape != param.shape and tp_size > 1:
                if "query_key_value" in param_name:
                    chunk = delta.shape[0] // tp_size
                    delta = delta[tp_rank * chunk : (tp_rank + 1) * chunk]
                elif "dense" in param_name:
                    chunk = delta.shape[1] // tp_size
                    delta = delta[:, tp_rank * chunk : (tp_rank + 1) * chunk]

            # A mismatch here means the shard math is wrong (wrong tp_size, wrong
            # dim). Report it rather than letting PyTorch raise an opaque broadcast
            # error from inside add_.
            if delta.shape != param.shape:
                missed.append(
                    f"{param_name}: delta {tuple(delta.shape)} vs param {tuple(param.shape)}"
                )
                continue

            param.data.add_(delta.to(param.device, dtype=param.dtype))
            merged += 1

        sys.stderr.write(
            f"[LLaDAFast] LoRA merged: {merged} modules"
            + (f", {len(missed)} missed: {missed[:5]}" if missed else "")
            + "\n"
        )
        sys.stderr.flush()

        # The adapter carried LoRA factors, so merging nothing means the key
        # mapping is broken. Fail loudly rather than silently serving base weights.
        if merged == 0:
            raise RuntimeError(
                f"[LLaDAFast] adapter carried {len(pairs)} LoRA module(s) but none "
                f"were merged. Key mapping failed.\n"
                f"  Adapter modules: {list(pairs)[:5]}\n"
                f"  Model params look like: "
                f"{[k for k in params if 'attention' in k and 'weight' in k][:3]}"
            )

    @staticmethod
    def _tp_shard_delta(
        name: str, tensor: torch.Tensor, tp_rank: int, tp_size: int
    ) -> torch.Tensor:
        """Slice a full (non-sharded) delta tensor to the local TP rank.

        hedgehog_weights : [H_total, D, F]   → shard dim 0
        alpha            : [H_total]          → shard dim 0  (new checkpoints)
                         : [1, H_total, 1, 1] → shard dim 1  (old checkpoints)
        """
        if "hedgehog_weights" in name:
            h_total = tensor.shape[0]
            h_local = h_total // tp_size
            return tensor[tp_rank * h_local : (tp_rank + 1) * h_local].contiguous()
        elif "alpha" in name:
            if tensor.ndim == 1:
                # New checkpoint format: [H]
                h_total = tensor.shape[0]
                h_local = h_total // tp_size
                return tensor[tp_rank * h_local : (tp_rank + 1) * h_local].contiguous()
            else:
                # Old checkpoint format: [1, H, 1, 1]
                h_total = tensor.shape[1]
                h_local = h_total // tp_size
                return tensor[:, tp_rank * h_local : (tp_rank + 1) * h_local].contiguous()
        else:
            logger.warning(
                f"[LLaDAFast] Cannot TP-shard unknown delta param {name} "
                f"(shape {tensor.shape}); loading as-is."
            )
            return tensor


EntryClass = LLaDAFastForCausalLM
