"""Thin wrapper around the hybrid attention kernels SGLang actually serves.

The kernels are imported from ``sglang.srt.layers.attention.llada_fast_hybrid_kernel``
rather than copied here, so this can never drift from what runs in production.
The wrapper only reshapes inputs and picks a launch configuration; all the
arithmetic lives in the imported kernels.

Two paths are exposed, both checked by ``test_kernel.py``:

  ``forward``        state update, then a streaming output kernel.
  ``forward_fused``  output and state update in one launch.

Only the per-q-head layout is supported (each query head has its own feature
map), which is what the released checkpoint uses. ``tl.dot`` runs with
``allow_tf32=False`` throughout.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

# The kernels under test are the ones SGLang actually serves, imported from the
# installed (patched) SGLang rather than copied here -- a copy would drift.
from sglang.srt.layers.attention.llada_fast_hybrid_kernel import (
    _hybrid_update_state_kernel_rw,
    _hybrid_output_kernel_streaming,
    _hybrid_output_update_kernel_fused_perq,
)

from state import HybridStateView, RequestBlockPlan


class HedgehogHybridAttnProd(nn.Module):
    """SGLang-production-equivalent hybrid attention (per-q-head, no KV share)."""

    def __init__(
        self,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        feature_dim: int = 64,
        block_size: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.H = int(num_heads)
        self.Hkv = int(num_kv_heads)
        self.D = int(head_dim)
        self.F = int(feature_dim)
        self.S = int(block_size)
        self.eps = float(eps)
        self.k_scale = 1.0
        self.v_scale = 1.0
        self._num_kv_groups = self.H // self.Hkv

        if self.H % self.Hkv != 0:
            raise ValueError(
                f"num_heads ({self.H}) must be divisible by num_kv_heads ({self.Hkv})"
            )

        # Same init as the local test wrapper (identity-ish W, alpha=0).
        W = torch.eye(self.D, self.F).unsqueeze(0).expand(self.H, -1, -1).contiguous()
        self.hedgehog_weights = nn.Parameter(W.float())
        self.alpha = nn.Parameter(torch.zeros(self.H, dtype=torch.float32))

    def make_state(
        self,
        num_slots: int,
        device,
        dtype=torch.float32,
        *,
        s_dtype: Optional[torch.dtype] = None,
        z_dtype: Optional[torch.dtype] = None,
    ) -> HybridStateView:
        s_dtype = dtype if s_dtype is None else s_dtype
        z_dtype = dtype if z_dtype is None else z_dtype
        S = torch.zeros(num_slots, self.H, 2 * self.F, self.D, device=device, dtype=s_dtype)
        Z = torch.zeros(num_slots, self.H, 2 * self.F, device=device, dtype=z_dtype)
        return HybridStateView(S_state=S, Z_state=Z)

    def forward(
        self,
        q: torch.Tensor,  # [T, H_q, D]
        k: torch.Tensor,  # [T, H_kv, D]
        v: torch.Tensor,  # [T, H_kv, D]
        plan: RequestBlockPlan,
        read_state: HybridStateView,
        write_state: Optional[HybridStateView] = None,
        use_updated_state_for_output: bool = False,
    ) -> torch.Tensor:
        """Two-launch baseline: update_state → output_streaming. Matches the
        production dispatcher's `streaming + state update` fast path on L40/Ada
        (the Hopper bf16-fused fast path is disabled in the dispatcher)."""
        if write_state is None:
            write_state = read_state

        out = torch.zeros_like(q)

        if plan.req_indices.numel() == 0:
            return out

        R = plan.req_indices.numel()
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()

        W = self.hedgehog_weights
        if not W.is_contiguous():
            W = W.contiguous()

        # Production update kernel — grid (R, H_q) for the non-kv-shared layout.
        # UPDATE_USES_KV_HEADS=False so each q-head writes its own state slot
        # (per-q-head W and per-q-head state shape S[slots,H_q,2F,D]).
        _hybrid_update_state_kernel_rw[(R, self.H)](
            k_c, v_c,
            plan.req_indices,
            plan.start_offsets, plan.seq_lens,
            read_state.S_state, read_state.Z_state,
            write_state.S_state, write_state.Z_state,
            W,
            self.k_scale, self.v_scale,
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            read_state.S_state.stride(0), read_state.S_state.stride(1),
            read_state.S_state.stride(2), read_state.S_state.stride(3),
            read_state.Z_state.stride(0), read_state.Z_state.stride(1),
            read_state.Z_state.stride(2),
            write_state.S_state.stride(0), write_state.S_state.stride(1),
            write_state.S_state.stride(2), write_state.S_state.stride(3),
            write_state.Z_state.stride(0), write_state.Z_state.stride(1),
            write_state.Z_state.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            self._num_kv_groups,
            UPDATE_USES_KV_HEADS=False,
            eps=self.eps,
            BLOCK_S=self.S,
            D=self.D,
            F=self.F,
            num_warps=4,
            num_stages=1,
        )

        state_for_output = write_state if use_updated_state_for_output else read_state

        # Production passes sigmoid(alpha) as alpha_gate (the kernel does NOT
        # apply sigmoid internally — see BlockSoftmaxLinearHybrid._alpha_gate_cache
        # in llada_fast_hybrid_kernel.py). The local test kernel applies sigmoid
        # inside, so the calling conventions differ. Match production here.
        alpha_gate = torch.sigmoid(self.alpha.detach().float()).contiguous()

        # Production streaming output kernel — grid (R, H_q).
        # SHARE_KV_GROUPS=False so state_h == q_h (per-q-head state).
        # BLOCK_F=F since F=64 fits in a single tile here.
        _hybrid_output_kernel_streaming[(R, self.H)](
            q_c, k_c, v_c, out,
            plan.req_indices,
            plan.start_offsets, plan.seq_lens,
            state_for_output.S_state, state_for_output.Z_state,
            W, alpha_gate,
            self.k_scale, self.v_scale,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            state_for_output.S_state.stride(0), state_for_output.S_state.stride(1),
            state_for_output.S_state.stride(2), state_for_output.S_state.stride(3),
            state_for_output.Z_state.stride(0), state_for_output.Z_state.stride(1),
            state_for_output.Z_state.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            self._num_kv_groups,
            SHARE_KV_GROUPS=False,
            eps=self.eps,
            scale=self.D ** -0.5,
            BLOCK_S=self.S,
            D=self.D,
            F=self.F,
            BLOCK_F=self.F,
            num_warps=4,
            num_stages=1,
        )
        return out

    def forward_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: RequestBlockPlan,
        read_state: HybridStateView,
        write_state: Optional[HybridStateView] = None,
        use_updated_state_for_output: bool = False,  # unused: fused kernel writes state
    ) -> torch.Tensor:
        """BF16-WGMMA fused fast path: single kernel call computes output AND writes
        state in one launch (vs forward() which uses two). Matches the production
        dispatcher path that fires for `plan.max_blocks == 1 and not share_hedgehog_kv_groups`.
        Operands are downcast to bf16 inside the kernel for WGMMA throughput on Hopper.
        """
        if write_state is None:
            write_state = read_state

        out = torch.zeros_like(q)
        if plan.req_indices.numel() == 0:
            return out

        R = plan.req_indices.numel()
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()
        W = self.hedgehog_weights
        if not W.is_contiguous():
            W = W.contiguous()
        alpha_gate = torch.sigmoid(self.alpha.detach().float()).contiguous()

        _hybrid_output_update_kernel_fused_perq[(R, self.H)](
            q_c, k_c, v_c, out,
            plan.req_indices, plan.start_offsets, plan.seq_lens,
            read_state.S_state, read_state.Z_state,
            write_state.S_state, write_state.Z_state,
            W, alpha_gate,
            self.k_scale, self.v_scale,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            read_state.S_state.stride(0), read_state.S_state.stride(1),
            read_state.S_state.stride(2), read_state.S_state.stride(3),
            read_state.Z_state.stride(0), read_state.Z_state.stride(1),
            read_state.Z_state.stride(2),
            write_state.S_state.stride(0), write_state.S_state.stride(1),
            write_state.S_state.stride(2), write_state.S_state.stride(3),
            write_state.Z_state.stride(0), write_state.Z_state.stride(1),
            write_state.Z_state.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            self._num_kv_groups,
            eps=self.eps,
            scale=self.D ** -0.5,
            BLOCK_S=self.S,
            D=self.D,
            F=self.F,
            num_warps=8,
            num_stages=1,
        )
        return out


__all__ = ["HedgehogHybridAttnProd"]
