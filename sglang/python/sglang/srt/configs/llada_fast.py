# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config adapter for LLaDA Fast hybrid (softmax + linear) attention models.

LLaDA Fast uses trust_remote_code config (LLaDA2MoeConfig) which cannot be made
to inherit from SGLang config classes.  LLaDAFastConfigAdapter wraps the raw HF
config and exposes the three properties required by mambaish_config:

  full_attention_layer_ids    – pure softmax-attention layer indices
  hybrid_attention_layer_ids  – hybrid (softmax + linear) layer indices
  mamba2_cache_params         – state shape for MambaPool allocation

State layout stored in MambaPool per (layer, slot):
  S_state : temporal shape (2, H/tp, 2F, D)  — dim 0: [committed=0, working=1]
  Z_state : conv[0] = committed (H/tp*2F, 1), conv[1] = working (H/tp*2F, 1)
"""

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)

# Layer type strings that indicate a hybrid (softmax + linear) attention layer.
LLADA_FAST_HYBRID_TYPES = frozenset(
    {"hybrid", "hybrid_linear", "block_softmax_linear_hybrid"}
)


def _is_llada_fast_config(hf_config) -> bool:
    """Return True if hf_config is a LLaDAFast (block-softmax hybrid) config."""
    archs = getattr(hf_config, "architectures", []) or []
    if not getattr(hf_config, "use_block_softmax_hybrid", False):
        return False
    return "LLaDAFastForCausalLM" in archs


class LLaDAFastConfigAdapter:
    """
    Thin wrapper around the raw HF config that adds the mambaish_config interface.
    All attribute access falls through to the underlying config.
    """

    def __init__(self, hf_config):
        # Use object.__setattr__ to avoid infinite recursion in __getattr__
        object.__setattr__(self, "_cfg", hf_config)

    # Delegate everything to the underlying config
    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_cfg"), name)

    # ------------------------------------------------------------------ #
    # Required by mambaish_config / HybridReqToTokenPool / HybridLinearKVPool
    # ------------------------------------------------------------------ #

    @property
    def full_attention_layer_ids(self) -> list[int]:
        cfg = object.__getattribute__(self, "_cfg")
        block_types = getattr(
            cfg, "layers_block_type", ["attention"] * cfg.num_hidden_layers
        )
        return [
            i for i, t in enumerate(block_types) if str(t) not in LLADA_FAST_HYBRID_TYPES
        ]

    @property
    def kv_full_attention_layer_ids(self) -> list[int]:
        """Layers that need Radix KV slots in HybridLinearKVPool.

        Only the softmax (non-hybrid) layers write KV; the hybrid layers keep an
        O(1) recurrent state in MambaPool instead, which is where the KV-footprint
        saving comes from.
        """
        return self.full_attention_layer_ids

    @property
    def hybrid_attention_layer_ids(self) -> list[int]:
        cfg = object.__getattribute__(self, "_cfg")
        block_types = getattr(
            cfg, "layers_block_type", ["attention"] * cfg.num_hidden_layers
        )
        return [
            i for i, t in enumerate(block_types) if str(t) in LLADA_FAST_HYBRID_TYPES
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        cfg = object.__getattribute__(self, "_cfg")
        tp = get_attention_tp_size()
        num_heads = int(cfg.num_attention_heads)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
        head_dim = int(
            getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
        )
        feature_dim = int(getattr(cfg, "feature_dim"))

        # When share_hedgehog_kv_groups=True, state is per-kv-head not per-q-head.
        share_kv = bool(getattr(cfg, "share_hedgehog_kv_groups", False))
        state_heads = num_kv_heads if share_kv else num_heads
        h_local = state_heads // tp

        # S_state : temporal (2, h_local, 2F, D) — committed=0, working=1
        # Z_state : conv[0]=committed, conv[1]=working
        z_shape = (h_local * 2 * feature_dim, 1)
        shape = Mamba2StateShape(
            conv=[z_shape, z_shape],
            temporal=(2, h_local, 2 * feature_dim, head_dim),
            intermediate_size=0,
            conv_dim=h_local * 2 * feature_dim,
            ssm_state_size=head_dim,
            num_heads=h_local,
            head_dim=2 * feature_dim,
            state_size=head_dim,
            conv_kernel=2,
        )

        # Z_state (stored in conv slot) is the denominator accumulator and is
        # more calibration-sensitive, so keep it in fp32. S_state dominates
        # state bandwidth and is accumulated in fp32 inside the Triton kernels,
        # so store it as bf16.
        return Mamba2CacheParams(
            shape=shape,
            layers=self.hybrid_attention_layer_ids,
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.bfloat16),
        )
