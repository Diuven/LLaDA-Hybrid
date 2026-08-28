"""
Pure-PyTorch reference implementation of _hybrid_extend_kernel_rw.

dtype handling matches src/llada_fast/modeling/hybrid_attention.py
BlockSoftmaxLinearHybrid exactly:
  - hedgehog feature map einsum runs in q.dtype (e.g. bf16/fp16),
    softmax runs in fp32, phi is cast back to q.dtype (round-trip).
  - QK matmul, exp, softmax-branch numerator/denominator run in fp32
    via explicit `.float()` casts at the matmul boundary.
  - Linear branch reads fp32 state; phi is cast to fp32 at the matmul.
  - Final output cast to q.dtype.

Structure differences from hybrid_attention.py BlockSoftmaxLinearHybrid
(kept the same as before):
  - Flat [T, H, D] token layout, not (B, H, L, D).
  - Single block per request (extend-only); no multi-block recurrence loop.
  - Explicit committed-read / working-write state split.
  - GQA at the state level: state is [slots, H_kv, 2F, D] (per kv-head),
    shared across q-heads in the group. q-head h reads kv_h = h // groups.

  For per-q-head W/state (SGLang / HedgehogHybridAttnPerQ), see
  `ref_hybrid_extend_per_q` in this module.

Tensor layouts (identical to the Triton kernel):
  q   : [T, H_q,  D]
  k   : [T, H_kv, D]
  v   : [T, H_kv, D]
  out : [T, H_q,  D]

State layouts (identical to HybridStateView):
  S : [num_slots, H_kv, 2F, D]   fp32
  Z : [num_slots, H_kv, 2F]      fp32
  Positive branch: [:, :, :F, :] / [:, :, :F]
  Negative branch: [:, :, F:, :] / [:, :, F:]
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

# Re-use the state container from the Triton wrapper
from state import HybridStateView


EPS = 1e-6


def _hybrid_feature_map(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Hedgehog feature map matching hybrid_attention.py._feature_map exactly.

    x : [S, D]    in x.dtype (e.g. bf16/fp16/fp32)
    W : [D, F]    cast to x.dtype before einsum
    Returns phi : [S, 2F] in x.dtype, with
        phi[:, :F] = softmax( x @ W, dim=-1)  (fp32 softmax, cast back to x.dtype)
        phi[:, F:] = softmax(-x @ W, dim=-1)
    """
    u = x @ W.to(dtype=x.dtype)              # [S, F] in x.dtype
    u_f32 = u.float()
    return torch.cat(
        [F.softmax(u_f32, dim=-1), F.softmax(-u_f32, dim=-1)], dim=-1
    ).to(dtype=x.dtype)                      # [S, 2F] in x.dtype


def ref_hybrid_extend(
    q:           torch.Tensor,        # [T, H_q,  D]  (any dtype; ops in fp32)
    k:           torch.Tensor,        # [T, H_kv, D]
    v:           torch.Tensor,        # [T, H_kv, D]
    seq_lens:    List[int],           # [R] tokens per request, each <= block_size
    req_indices: torch.Tensor,        # [R] slot→state index
    read_state:  HybridStateView,     # committed S/Z (read only)
    write_state: HybridStateView,     # working   S/Z (written in-place)
    W:           torch.Tensor,        # [H_kv, D, F]  hedgehog projection (per kv-head)
    alpha:       torch.Tensor,        # [H_q]        blend gate logits
    scale:       float  = None,       # 1/sqrt(D); inferred if None
    eps:         float  = EPS,
) -> torch.Tensor:
    """
    Pure PyTorch reference for _hybrid_extend_kernel_rw.

    Processes each (request r, q-head h) independently in Python loops,
    mirroring the (pid_r, pid_h) program-id grid of the Triton kernel.

    Returns out : [T, H_q, D]  same dtype as q.
    """
    T, H_q,  D   = q.shape
    _,  H_kv, _  = k.shape
    num_kv_groups = H_q // H_kv
    F_dim         = W.shape[-1]          # feature dim (state is 2F wide)
    R             = len(seq_lens)
    scale         = scale or D ** -0.5
    out_dtype     = q.dtype

    out = torch.zeros_like(q)            # [T, H_q, D]

    # Build token start offsets from seq_lens (identical to build_request_block_plan)
    start_offsets = [0] * R
    for r in range(1, R):
        start_offsets[r] = start_offsets[r - 1] + seq_lens[r - 1]

    # One Python iteration = one Triton program (pid_r, pid_h)
    for r in range(R):
        slot      = int(req_indices[r].item())
        start     = start_offsets[r]
        seq_len   = seq_lens[r]
        t_slice   = slice(start, start + seq_len)          # token rows for this request

        for h in range(H_q):
            kv_h = h // num_kv_groups

            # ── Load q/k/v in NATIVE dtype (no upfront fp32 cast) ────────────
            # Matches hybrid_attention.py: feature map runs in x.dtype, with
            # explicit `.float()` only at matmul/exp boundaries.
            q_h = q[t_slice, h,    :]   # [S, D] q.dtype
            k_h = k[t_slice, kv_h, :]   # [S, D] q.dtype
            v_h = v[t_slice, kv_h, :]   # [S, D] q.dtype

            # ── Load W [D, F] for this kv-head (shared across GQA group) ─────
            # Cast to q.dtype is done inside _hybrid_feature_map, matching
            # `self.hedgehog_weights.to(dtype=x.dtype)` in hybrid_attention.py.
            W_h = W[kv_h]               # [D, F]

            # ── Read committed state for (slot, kv_h); state is fp32 ─────────
            S_pos = read_state.S_state[slot, kv_h, :F_dim,  :].float()  # [F, D]
            S_neg = read_state.S_state[slot, kv_h,  F_dim:, :].float()  # [F, D]
            Z_pos = read_state.Z_state[slot, kv_h, :F_dim   ].float()   # [F]
            Z_neg = read_state.Z_state[slot, kv_h,  F_dim:  ].float()   # [F]

            # ── Hedgehog feature maps with x.dtype round-trip ────────────────
            # Matches hybrid_attention.py._feature_map exactly: einsum in
            # x.dtype, softmax in fp32, cast back to x.dtype.
            phi_q = _hybrid_feature_map(q_h, W_h)   # [S, 2F] q.dtype
            phi_k = _hybrid_feature_map(k_h, W_h)   # [S, 2F] q.dtype
            phi_q_pos, phi_q_neg = phi_q[:, :F_dim], phi_q[:, F_dim:]
            phi_k_pos, phi_k_neg = phi_k[:, :F_dim], phi_k[:, F_dim:]

            # ── Linear branch readout (cast phi to fp32 at matmul boundary) ──
            # Matches: lin_num = phi_q.float() @ S_state, S_state already fp32.
            phi_q_pos_f = phi_q_pos.float()
            phi_q_neg_f = phi_q_neg.float()
            lin_num = phi_q_pos_f @ S_pos + phi_q_neg_f @ S_neg              # [S, D]
            lin_den = (phi_q_pos_f * Z_pos).sum(-1) + (phi_q_neg_f * Z_neg).sum(-1)  # [S]
            lin_den = lin_den.clamp_min(eps)

            # ── Softmax branch: q.float() @ k.float().T, exp in fp32 ─────────
            # Matches hybrid_attention.py: scores = q_n.float() @ k_n.float().T * scale.
            # No padding mask: extend assumes contiguous valid tokens.
            scores  = (q_h.float() @ k_h.float().t()) * scale                # [S, S] fp32
            row_max = scores.amax(dim=-1, keepdim=True)
            a_sm    = torch.exp(scores - row_max)                             # [S, S] fp32
            sm_num  = a_sm @ v_h.float()                                      # [S, D] fp32
            sm_den  = a_sm.sum(dim=-1).clamp_min(eps)                         # [S]

            # ── Shared-normalization blend ───────────────────────────────────
            # w = sigmoid(alpha_h.float()) — per-head scalar in fp32.
            w = torch.sigmoid(alpha[h].float())                               # scalar
            den      = (w * sm_den + lin_den).clamp_min(eps)                  # [S]
            out_tile = (w * sm_num + lin_num) / den[:, None]                  # [S, D] fp32

            # Final cast to out_dtype matches `(num / den).to(out_dtype)`.
            out[t_slice, h, :] = out_tile.to(out_dtype)

            # ── State update: phi_k.float() @ v.float(), accum in fp32 ───────
            # Matches hybrid_attention.py forward:
            #   S_state += phi_k_n.transpose(-2, -1).float() @ v[..., n].float()
            #   Z_state += phi_k_n.float().sum(dim=-2)
            phi_k_pos_f = phi_k_pos.float()
            phi_k_neg_f = phi_k_neg.float()
            v_h_f       = v_h.float()
            S_pos_new = S_pos + phi_k_pos_f.t() @ v_h_f   # [F, D]
            S_neg_new = S_neg + phi_k_neg_f.t() @ v_h_f   # [F, D]
            Z_pos_new = Z_pos + phi_k_pos_f.sum(0)         # [F]
            Z_neg_new = Z_neg + phi_k_neg_f.sum(0)         # [F]

            # State is per kv-head; all q-heads in the group write the same values.
            write_state.S_state[slot, kv_h, :F_dim,  :] = S_pos_new
            write_state.S_state[slot, kv_h,  F_dim:, :] = S_neg_new
            write_state.Z_state[slot, kv_h, :F_dim   ]  = Z_pos_new
            write_state.Z_state[slot, kv_h,  F_dim:  ]  = Z_neg_new

    return out


def ref_hybrid_extend_per_q(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: List[int],
    req_indices: torch.Tensor,
    read_state: HybridStateView,
    write_state: HybridStateView,
    W: torch.Tensor,
    alpha: torch.Tensor,
    scale: float = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Pure PyTorch reference for per-q-head hybrid extend (HedgehogHybridAttnPerQ).

    Same token math as `ref_hybrid_extend`, but:
      - W : [H_q, D, F]   (one projection per query head)
      - State : [slots, H_q, 2F, D] / [slots, H_q, 2F]  (per q-head)
      - k, v remain [T, H_kv, D]; query head h uses kv row kv_h = h // (H_q // H_kv).

    Matches the per-q Triton path in the served kernel (no repeat_kv): the same
    head indexing, so a mismatch here is a real kernel bug, not a layout skew.
    """
    T, H_q, D = q.shape
    _, H_kv, _ = k.shape
    if W.shape[0] != H_q:
        raise ValueError(f"W must have shape [H_q, D, F]; got W.shape[0]={W.shape[0]} vs H_q={H_q}")
    num_kv_groups = H_q // H_kv
    F_dim = W.shape[-1]
    R = len(seq_lens)
    scale = scale or D ** -0.5
    out_dtype = q.dtype
    out = torch.zeros_like(q)

    start_offsets = [0] * R
    for r in range(1, R):
        start_offsets[r] = start_offsets[r - 1] + seq_lens[r - 1]

    for r in range(R):
        slot = int(req_indices[r].item())
        start = start_offsets[r]
        seq_len = seq_lens[r]
        t_slice = slice(start, start + seq_len)

        for h in range(H_q):
            kv_h = h // num_kv_groups
            q_h = q[t_slice, h, :]
            k_h = k[t_slice, kv_h, :]
            v_h = v[t_slice, kv_h, :]
            W_h = W[h]

            S_pos = read_state.S_state[slot, h, :F_dim, :].float()
            S_neg = read_state.S_state[slot, h, F_dim:, :].float()
            Z_pos = read_state.Z_state[slot, h, :F_dim].float()
            Z_neg = read_state.Z_state[slot, h, F_dim:].float()

            phi_q = _hybrid_feature_map(q_h, W_h)
            phi_k = _hybrid_feature_map(k_h, W_h)
            phi_q_pos, phi_q_neg = phi_q[:, :F_dim], phi_q[:, F_dim:]
            phi_k_pos, phi_k_neg = phi_k[:, :F_dim], phi_k[:, F_dim:]

            phi_q_pos_f = phi_q_pos.float()
            phi_q_neg_f = phi_q_neg.float()
            lin_num = phi_q_pos_f @ S_pos + phi_q_neg_f @ S_neg
            lin_den = (phi_q_pos_f * Z_pos).sum(-1) + (phi_q_neg_f * Z_neg).sum(-1)
            lin_den = lin_den.clamp_min(eps)

            scores = (q_h.float() @ k_h.float().t()) * scale
            row_max = scores.amax(dim=-1, keepdim=True)
            a_sm = torch.exp(scores - row_max)
            sm_num = a_sm @ v_h.float()
            sm_den = a_sm.sum(dim=-1).clamp_min(eps)

            w = torch.sigmoid(alpha[h].float())
            den = (w * sm_den + lin_den).clamp_min(eps)
            out_tile = (w * sm_num + lin_num) / den[:, None]
            out[t_slice, h, :] = out_tile.to(out_dtype)

            phi_k_pos_f = phi_k_pos.float()
            phi_k_neg_f = phi_k_neg.float()
            v_h_f = v_h.float()
            S_pos_new = S_pos + phi_k_pos_f.t() @ v_h_f
            S_neg_new = S_neg + phi_k_neg_f.t() @ v_h_f
            Z_pos_new = Z_pos + phi_k_pos_f.sum(0)
            Z_neg_new = Z_neg + phi_k_neg_f.sum(0)

            write_state.S_state[slot, h, :F_dim, :] = S_pos_new
            write_state.S_state[slot, h, F_dim:, :] = S_neg_new
            write_state.Z_state[slot, h, :F_dim] = Z_pos_new
            write_state.Z_state[slot, h, F_dim:] = Z_neg_new

    return out
