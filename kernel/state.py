"""Shared containers for the hybrid attention state and the per-request block plan.

Extracted so the reference implementation and the production kernel wrapper can
agree on the layout without either depending on the other.
"""

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class HybridStateView:
    """
    Recurrent linear attention state for one pool of request slots.

    S_state : [num_slots, H_kv, 2F, D]
        Outer-product accumulator:
          S_pos = S[:, :, :F, :]   ← sum_t phi_k_pos(t)^T v(t)
          S_neg = S[:, :, F:, :]   ← sum_t phi_k_neg(t)^T v(t)

    Z_state : [num_slots, H_kv, 2F]
        Key accumulator:
          Z_pos = Z[:, :, :F]   ← sum_t phi_k_pos(t)
          Z_neg = Z[:, :, F:]   ← sum_t phi_k_neg(t)
    """
    S_state: torch.Tensor   # [slots, H_kv, 2F, D] fp32
    Z_state: torch.Tensor   # [slots, H_kv, 2F]    fp32


# ---------------------------------------------------------------------------
# Block plan
# ---------------------------------------------------------------------------

@dataclass
class RequestBlockPlan:
    req_indices: torch.Tensor
    seq_lens: torch.Tensor
    start_offsets: torch.Tensor
    block_size: int


def build_request_block_plan(seq_lens, req_indices, block_size, device):
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
    req_indices_t = req_indices.to(device=device, dtype=torch.long)

    if seq_lens_t.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return RequestBlockPlan(
            req_indices=empty,
            seq_lens=empty,
            start_offsets=empty,
            block_size=block_size,
        )

    if torch.any(seq_lens_t <= 0):
        raise ValueError(f"All seq_lens must be > 0, got {seq_lens}")
    if torch.any(seq_lens_t > block_size):
        raise ValueError(
            f"Extend-only kernel supports one block/request with len <= {block_size}, "
            f"got seq_lens={seq_lens}"
        )

    start_offsets = torch.zeros_like(seq_lens_t)
    if seq_lens_t.numel() > 1:
        start_offsets[1:] = torch.cumsum(seq_lens_t[:-1], dim=0)

    return RequestBlockPlan(
        req_indices=req_indices_t,
        seq_lens=seq_lens_t,
        start_offsets=start_offsets,
        block_size=block_size,
    )
