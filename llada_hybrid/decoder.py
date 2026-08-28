"""Decoder layer for LLaDA2 MoE: attention + MoE/MLP + residual connections."""

from typing import Optional, Tuple

import torch
from torch import nn

from .configuration_llada2_moe import LLaDA2MoeConfig
from .norm import LLaDA2MoeRMSNorm
from .moe import LLaDA2MoeMLP, LLaDA2MoeSparseMoeBlock
from .attention import LLaDA2MoeAttention


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = LLaDA2MoeAttention(config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        half_len: Optional[int] = None,
        prefix_cache=None,
        cache_mode: Optional[str] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            key_padding_mask=key_padding_mask,
            block_attention_mask=block_attention_mask,
            half_len=half_len,
            prefix_cache=prefix_cache,
            cache_mode=cache_mode,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs
