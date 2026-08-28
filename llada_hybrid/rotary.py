"""Rotary position embedding for LLaDA2 MoE."""

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
try:
    from transformers.modeling_rope_utils import dynamic_rope_update
except ImportError:
    def dynamic_rope_update(fn):  # no-op for transformers < 4.51
        return fn

from .configuration_llada2_moe import LLaDA2MoeConfig


def _compute_default_rope_parameters(config, device=None, seq_len=None, **rope_kwargs):
    """Standard un-scaled (NeoX-style) RoPE inverse frequencies.

    Local copy of transformers 4.x ``_compute_default_rope_parameters``, which
    transformers >= 5.0 removed from ``ROPE_INIT_FUNCTIONS`` (it dropped the
    "default" key). This reproduces that function's config-based call path
    byte-for-byte — same ``dim`` derivation, same ``int64``→``float`` arange,
    same ``attention_factor = 1.0`` — so it is numerically identical to the
    RoPE the model was trained/distilled with under transformers 4.57.x.

    Returns ``(inv_freq, attention_scaling)``.
    """
    base = config.rope_theta
    partial_rotary_factor = (
        config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    )
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # un-scaled RoPE has no attention scaling
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, attention_factor


class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            rope_params = config.rope_parameters
        else:
            rope_params = getattr(config, "rope_scaling", {}) or {}

        self.rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if rope_init_fn is None:
            # transformers >= 5.0 removed the "default" entry from
            # ROPE_INIT_FUNCTIONS. Fall back to the local un-scaled RoPE init,
            # which is numerically identical to transformers 4.x's "default"
            # (and to what SGLang computes for inference).
            rope_init_fn = _compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
