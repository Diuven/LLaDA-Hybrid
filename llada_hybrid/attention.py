"""Attention dispatch layer for LLaDA2 MoE.

Routes each layer to hybrid (block-softmax + Hedgehog linear) or standard softmax
attention, according to the config.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from .configuration_llada2_moe import LLaDA2MoeConfig
from .norm import LLaDA2MoeRMSNorm
from .rotary import apply_rotary_pos_emb
from .linear_attention import OrderInvariantKernelLinearAttention
from .hybrid_attention import BlockSoftmaxLinearHybrid

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class LLaDA2MoeAttention(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                "Instantiating attention without layer_idx is not recommended if you use caching."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False
        self.sliding_window = getattr(config, "sliding_window", None)

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        if self.config.use_qk_norm:
            self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

        use_linear = getattr(config, "use_linear_attention", False)
        if use_linear:
            block_size = getattr(config, "block_size", 32)
            if getattr(config, "use_block_softmax_hybrid", False):
                self.linear_attention = BlockSoftmaxLinearHybrid(config, block_size=block_size)
            else:
                self.linear_attention = OrderInvariantKernelLinearAttention(config, block_size=block_size)

        linear_layers = config.linear_attention_layers
        if linear_layers is not None:
            self.is_linear_active = self.layer_idx in linear_layers
        else:
            self.is_linear_active = use_linear

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        half_len: Optional[int] = None,
        prefix_cache=None,
        cache_mode: Optional[str] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ── BlockDiffusionCache path ──────────────────────────────────────────
        if prefix_cache is not None and cache_mode is not None:
            return self._forward_with_prefix_cache(
                query_states, key_states, value_states,
                input_shape, prefix_cache, cache_mode,
                attention_mask, key_padding_mask, **kwargs,
            )

        # ── Original path (DynamicCache / no cache) ──────────────────────────
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Caching requires layer_idx set on attention.")
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if output_attentions:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        elif getattr(self, "is_linear_active", False) and hasattr(self, "linear_attention"):
            key_states_exp   = repeat_kv(key_states, self.num_key_value_groups)
            value_states_exp = repeat_kv(value_states, self.num_key_value_groups)

            if half_len is not None and hasattr(self.linear_attention, "forward_bd3lm"):
                linear_out = self.linear_attention.forward_bd3lm(
                    query_states, key_states_exp, value_states_exp,
                    half_len=half_len,
                    key_padding_mask=key_padding_mask,
                ).transpose(1, 2)
            else:
                linear_out = self.linear_attention(
                    query_states, key_states_exp, value_states_exp,
                    attention_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                ).transpose(1, 2)

            if isinstance(self.linear_attention, BlockSoftmaxLinearHybrid):
                attn_output = linear_out
            else:
                anchor_ratio = getattr(self, "softmax_anchor_ratio", 0.0)
                if anchor_ratio > 0.0:
                    softmax_out, _ = attention_interface(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0,
                        scaling=self.scaling,
                        sliding_window=self.sliding_window,
                        **kwargs,
                    )
                    attn_output = (1.0 - anchor_ratio) * linear_out + anchor_ratio * softmax_out
                else:
                    attn_output = linear_out

            attn_weights = None
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights, past_key_value

    def _forward_with_prefix_cache(
        self,
        query_states: torch.Tensor,    # (B, H, q_len, D) — current block only, RoPE applied
        key_states: torch.Tensor,      # (B, H, q_len, D) — current block only, RoPE applied
        value_states: torch.Tensor,    # (B, H, q_len, D)
        input_shape: torch.Size,
        prefix_cache,                  # BlockDiffusionCache
        cache_mode: str,               # "read" or "commit"
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Attention with prefix cache for block-diffusion KV cache generation."""
        li = self.layer_idx

        if getattr(self, "is_linear_active", False) and hasattr(self, "linear_attention"):
            # ── Linear / hybrid layer: use recurrent state cache ──────────
            if cache_mode == "read":
                state = prefix_cache.get_recurrent_state_snapshot(li)
            else:  # "commit"
                state = prefix_cache.get_recurrent_state(li)

            key_states_exp = repeat_kv(key_states, self.num_key_value_groups)
            value_states_exp = repeat_kv(value_states, self.num_key_value_groups)

            # Hedgehog hybrid keeps two states: the numerator S and the
            # normaliser Z.
            init_S = state[0] if state is not None else None
            init_Z = state[1] if state is not None else None
            result = self.linear_attention(
                query_states, key_states_exp, value_states_exp,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                initial_S_state=init_S,
                initial_Z_state=init_Z,
                return_final_state=(cache_mode == "commit"),
            )
            if cache_mode == "commit":
                linear_out, S_final, Z_final = result
                prefix_cache.update_recurrent_state(li, S_final, Z_final)
            else:
                linear_out = result

            attn_output = linear_out.transpose(1, 2)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.dense(attn_output)
            return attn_output, None, None

        else:
            # ── Softmax layer: use KV cache ──────────────────────────────
            cached_kv = prefix_cache.get_kv(li)
            if cached_kv is not None:
                past_k, past_v = cached_kv
                k_all = torch.cat([past_k, key_states], dim=2)
                v_all = torch.cat([past_v, value_states], dim=2)
            else:
                k_all = key_states
                v_all = value_states

            if cache_mode == "commit":
                prefix_cache.update_kv(li, key_states, value_states)

            # Build attention mask: current block queries attend to all prefix + current block
            # All prefix positions are visible (causal: they're committed past blocks)
            # Current block is bidirectional within itself
            prefix_len = k_all.shape[2] - query_states.shape[2]
            q_len = query_states.shape[2]
            total_kv_len = k_all.shape[2]

            # (1, 1, q_len, total_kv_len): 0 = attend, -inf = mask
            cache_mask = torch.zeros(
                1, 1, q_len, total_kv_len,
                device=query_states.device, dtype=query_states.dtype,
            )
            # All positions visible (prefix is causal, current block is bidirectional)

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                k_all,
                v_all,
                cache_mask,
                dropout=0.0,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.dense(attn_output)
            return attn_output, None, None
