"""Unified prefix cache for block-diffusion generation.

Stores per-layer state that depends on the layer type:
  - softmax layers: standard KV cache (past keys and values)
  - linear/hybrid layers: recurrent state (S_state, Z_state)

Used by `LLaDA2MoeModelLM._generate_with_kv_cache()` to avoid
recomputing committed prefix blocks during iterative refinement.
"""

from typing import Optional, Tuple

import torch


class BlockDiffusionCache:
    """Per-layer prefix cache for block-diffusion generation.

    After each block is finalized, its contribution is committed:
      - softmax layers: K/V tensors appended
      - linear/hybrid layers: recurrent state updated

    During refinement of the active block, the cache is read-only.
    """

    def __init__(self, num_layers: int, device: torch.device):
        self.num_layers = num_layers
        self.device = device

        # Per-layer type: "softmax" or "linear" (hybrid uses "linear" — same state)
        self.layer_type: list[str] = ["softmax"] * num_layers

        # Softmax layers: (key, value) each (B, H, cached_len, D)
        self.kv_cache: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

        # Linear/hybrid layers: (S_state, Z_state)
        #   S_state: (B, H, 2F, D) fp32
        #   Z_state: (B, H, 2F)    fp32
        self.recurrent_state: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

    # ── Softmax KV cache ──────────────────────────────────────────────────────

    def get_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.kv_cache[layer_idx]

    def update_kv(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Append new block's K/V to the cached prefix."""
        existing = self.kv_cache[layer_idx]
        if existing is None:
            self.kv_cache[layer_idx] = (new_k, new_v)
        else:
            old_k, old_v = existing
            self.kv_cache[layer_idx] = (
                torch.cat([old_k, new_k], dim=2),
                torch.cat([old_v, new_v], dim=2),
            )

    # ── Linear/hybrid recurrent state ─────────────────────────────────────────

    def get_recurrent_state(
        self, layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.recurrent_state[layer_idx]

    def get_recurrent_state_snapshot(
        self, layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a detached clone for read-only use during refinement."""
        state = self.recurrent_state[layer_idx]
        if state is None:
            return None
        S, Z = state
        return (S.clone(), Z.clone())

    def update_recurrent_state(
        self, layer_idx: int, S_state: torch.Tensor, Z_state: torch.Tensor
    ):
        self.recurrent_state[layer_idx] = (S_state, Z_state)

    def init_recurrent_state(
        self,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        feature_dim_2x: int,
        head_dim: int,
    ):
        """Initialize zero recurrent state for a linear/hybrid layer."""
        S = torch.zeros(
            batch_size, num_heads, feature_dim_2x, head_dim,
            dtype=torch.float32, device=self.device,
        )
        Z = torch.zeros(
            batch_size, num_heads, feature_dim_2x,
            dtype=torch.float32, device=self.device,
        )
        self.recurrent_state[layer_idx] = (S, Z)
