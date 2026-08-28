"""
Triton hybrid attention kernel for SGLang serving.

Drop-in replacement for the pure-PyTorch BlockSoftmaxLinearHybrid reference.
Math is equivalent (within bf16 quantization noise) to:
  src/llada_fast/modeling/hybrid_attention.py::BlockSoftmaxLinearHybrid._block_out

The Triton kernels are taken from
  scripts/kernels_comparison/hedgehog_hybrid_attn.py
and adapted here to:
  - Per-q-head state layout `[slots, H, 2F, D]`, matching SGLang's existing
    layer_cache.temporal `[slots, 2, H, 2F, D]`.
  - GQA: K/V stay `[T, H_kv, D]`; Triton uses `kv_h = q_h // (H_q // H_kv)` for K/V
    loads and `q_h` for per-q-head `W` and recurrent state (no Python repeat_kv).
  - Multi-block extend via a Python loop chaining state in-place.
  - SGLang `forward_qkv(q, k, v, layer_cache, metadata)` entry point and
    `make_state_view_from_layer_cache(layer_cache)` adapter.

Compute pattern per (request, q-head, block):
  phi_q = [softmax(q @ W), softmax(-(q @ W))]
  phi_k = [softmax(k @ W), softmax(-(k @ W))]
  lin_num = phi_q @ S_state         # S_state holds blocks 0..n-1 (committed)
  lin_den = phi_q @ Z_state         # Z_state holds blocks 0..n-1 (committed)
  scores  = (q.float() @ k.float().T) * scale
  a_sm    = softmax(scores) along key axis (fp32, masked)
  sm_num  = a_sm @ v
  sm_den  = sum(a_sm, dim=-1)
  out     = (w * sm_num + lin_num) / (w * sm_den + lin_den), w = sigmoid(alpha)

State advances strictly causally: output for block n is computed BEFORE
phi_k(block n) is added to S_state / Z_state.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

import triton
import triton.language as tl

from sglang.srt.models.llada_profile import timed_region


# =============================================================================
# State container
# =============================================================================


@dataclass
class HybridStateView:
    """
    S_state : [slots, H, 2F, D]  fp32   (per-q-head)
        S_pos = S[:, :, :F, :]   ← Σ_t phi_k_pos(t)^T v(t)
        S_neg = S[:, :, F:, :]   ← Σ_t phi_k_neg(t)^T v(t)

    Z_state : [slots, H, 2F]     fp32
        Z_pos = Z[:, :, :F]      ← Σ_t phi_k_pos(t)
        Z_neg = Z[:, :, F:]      ← Σ_t phi_k_neg(t)
    """
    S_state: torch.Tensor
    Z_state: torch.Tensor


# =============================================================================
# Block planning (multi-block)
# =============================================================================


@dataclass
class RequestBlockPlan:
    req_indices: torch.Tensor       # [R] pool slot indices
    seq_lens: torch.Tensor          # [R]
    start_offsets: torch.Tensor     # [R]
    num_blocks: torch.Tensor        # [R] = ceil(seq_lens / block_size)
    max_blocks: int
    block_size: int


def build_request_block_plan(
    seq_lens: List[int],
    req_indices: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> RequestBlockPlan:
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
    req_indices_t = req_indices.to(device=device, dtype=torch.long)

    if seq_lens_t.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return RequestBlockPlan(
            req_indices=empty,
            seq_lens=empty,
            start_offsets=empty,
            num_blocks=empty,
            max_blocks=0,
            block_size=block_size,
        )

    start_offsets = torch.zeros_like(seq_lens_t)
    if seq_lens_t.numel() > 1:
        start_offsets[1:] = torch.cumsum(seq_lens_t[:-1], dim=0)
    num_blocks = (seq_lens_t + block_size - 1) // block_size

    return RequestBlockPlan(
        req_indices=req_indices_t,
        seq_lens=seq_lens_t,
        start_offsets=start_offsets,
        num_blocks=num_blocks,
        max_blocks=int(num_blocks.max().item()),
        block_size=block_size,
    )


# =============================================================================
# State update kernel
# Grid = (R_active, H)
# One program per (request, q-head). K/V rows use kv_h = q_h // num_kv_groups;
# W and recurrent state use q_h (per-q-head checkpoint layout).
# =============================================================================


@triton.jit
def _hybrid_update_state_kernel_rw(
    k_ptr, v_ptr,
    req_indices_ptr,
    start_offsets_ptr, seq_lens_ptr,
    S_read_ptr, Z_read_ptr,
    S_write_ptr, Z_write_ptr,
    W_ptr,
    k_scale, v_scale,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_Sr_slot, stride_Sr_h, stride_Sr_f, stride_Sr_d,
    stride_Zr_slot, stride_Zr_h, stride_Zr_f,
    stride_Sw_slot, stride_Sw_h, stride_Sw_f, stride_Sw_d,
    stride_Zw_slot, stride_Zw_h, stride_Zw_f,
    stride_W_h, stride_W_d, stride_W_f,
    num_kv_groups,
    UPDATE_USES_KV_HEADS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h if UPDATE_USES_KV_HEADS else pid_h // num_kv_groups
    state_h = pid_h

    slot = tl.load(req_indices_ptr + pid_r)
    start_off = tl.load(start_offsets_ptr + pid_r)
    seq_len = tl.load(seq_lens_ptr + pid_r)

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    offs_f = tl.arange(0, F)
    smask = offs_s < seq_len

    # Load k, v in NATIVE dtype to mirror hybrid_attention.py:
    #   einsum runs in x.dtype with W cast to x.dtype.
    k_native = tl.load(
        k_ptr + (start_off + offs_s[:, None]) * stride_k_t
        + kv_h * stride_k_h
        + offs_d[None, :] * stride_k_d,
        mask=smask[:, None],
        other=0.0,
    )
    native_dtype = k_native.dtype

    v_native = tl.load(
        v_ptr + (start_off + offs_s[:, None]) * stride_v_t
        + kv_h * stride_v_h
        + offs_d[None, :] * stride_v_d,
        mask=smask[:, None],
        other=0.0,
    )
    v_f32 = v_native.to(tl.float32) * v_scale

    # W cast to native dtype: matches `self.hedgehog_weights.to(dtype=x.dtype)`.
    W_native = tl.load(
        W_ptr
        + state_h * stride_W_h
        + offs_d[:, None] * stride_W_d
        + offs_f[None, :] * stride_W_f,
    ).to(native_dtype)  # [D, F] native dtype

    Sr_base = S_read_ptr + slot * stride_Sr_slot + state_h * stride_Sr_h
    Zr_base = Z_read_ptr + slot * stride_Zr_slot + state_h * stride_Zr_h

    S_pos = tl.load(
        Sr_base
        + offs_f[:, None] * stride_Sr_f
        + offs_d[None, :] * stride_Sr_d
    ).to(tl.float32)

    S_neg = tl.load(
        Sr_base
        + (offs_f + F)[:, None] * stride_Sr_f
        + offs_d[None, :] * stride_Sr_d
    ).to(tl.float32)

    Z_pos = tl.load(Zr_base + offs_f * stride_Zr_f).to(tl.float32)
    Z_neg = tl.load(Zr_base + (offs_f + F) * stride_Zr_f).to(tl.float32)

    # Apply k_scale in native dtype before the einsum.
    k_scaled = (k_native.to(tl.float32) * k_scale).to(native_dtype)

    # u_k = k @ W in tensor cores (fp32 accumulator). Skipping the bf16 round-trip
    # (diff vs PyTorch is below bf16 quantization noise; saves cycles).
    u_k = tl.dot(k_scaled, W_native, allow_tf32=True)

    # Softmax in fp32 via hardware-fast exp2(x * LOG2E).
    LOG2E = 1.4426950408889634
    m = tl.max(u_k, axis=1)
    e = tl.exp2((u_k - m[:, None]) * LOG2E)
    phi_k_pos = e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)
    phi_k_pos = tl.where(smask[:, None], phi_k_pos, 0.0)

    m = tl.max(-u_k, axis=1)
    e = tl.exp2((-u_k - m[:, None]) * LOG2E)
    phi_k_neg = e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)
    phi_k_neg = tl.where(smask[:, None], phi_k_neg, 0.0)

    Sw_base = S_write_ptr + slot * stride_Sw_slot + state_h * stride_Sw_h
    Zw_base = Z_write_ptr + slot * stride_Zw_slot + state_h * stride_Zw_h

    tl.store(
        Sw_base
        + offs_f[:, None] * stride_Sw_f
        + offs_d[None, :] * stride_Sw_d,
        S_pos + tl.dot(tl.trans(phi_k_pos), v_f32, allow_tf32=True),
    )

    tl.store(
        Sw_base
        + (offs_f + F)[:, None] * stride_Sw_f
        + offs_d[None, :] * stride_Sw_d,
        S_neg + tl.dot(tl.trans(phi_k_neg), v_f32, allow_tf32=True),
    )

    tl.store(
        Zw_base + offs_f * stride_Zw_f,
        Z_pos + tl.sum(phi_k_pos, axis=0),
    )

    tl.store(
        Zw_base + (offs_f + F) * stride_Zw_f,
        Z_neg + tl.sum(phi_k_neg, axis=0),
    )


# =============================================================================
# Output kernel
# Grid = (R_active, H)
#
# Single-pass linear path: q@W computed once, softmax over F resolved in
# registers, then phi_pos/phi_neg fuse with one load of S_state and Z_state.
# =============================================================================


@triton.jit
def _hybrid_output_kernel_streaming(
    q_ptr, k_ptr, v_ptr, out_ptr,
    req_indices_ptr,
    start_offsets_ptr, seq_lens_ptr,
    S_ptr, Z_ptr,
    W_ptr, alpha_gate_ptr,
    k_scale, v_scale,
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_out_t, stride_out_h, stride_out_d,
    stride_S_slot, stride_S_h, stride_S_f, stride_S_d,
    stride_Z_slot, stride_Z_h, stride_Z_f,
    stride_W_h, stride_W_d, stride_W_f,
    num_kv_groups,
    SHARE_KV_GROUPS: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_r = tl.program_id(0)
    q_h = tl.program_id(1)
    kv_h = q_h // num_kv_groups
    state_h = kv_h if SHARE_KV_GROUPS else q_h

    slot = tl.load(req_indices_ptr + pid_r)
    start_off = tl.load(start_offsets_ptr + pid_r)
    seq_len = tl.load(seq_lens_ptr + pid_r)

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    smask = offs_s < seq_len

    # Load q/k/v in NATIVE dtype.
    q_native = tl.load(
        q_ptr
        + (start_off + offs_s[:, None]) * stride_q_t
        + q_h * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=smask[:, None],
        other=0.0,
    )
    native_dtype = q_native.dtype

    k_native = tl.load(
        k_ptr
        + (start_off + offs_s[:, None]) * stride_k_t
        + kv_h * stride_k_h
        + offs_d[None, :] * stride_k_d,
        mask=smask[:, None],
        other=0.0,
    )
    v_native = tl.load(
        v_ptr
        + (start_off + offs_s[:, None]) * stride_v_t
        + kv_h * stride_v_h
        + offs_d[None, :] * stride_v_d,
        mask=smask[:, None],
        other=0.0,
    )
    v_f32 = v_native.to(tl.float32) * v_scale

    # k_scale applied in native dtype to match the einsum input precision.
    k_scaled = (k_native.to(tl.float32) * k_scale).to(native_dtype)

    LOG2E = 1.4426950408889634

    # ---- Softmax branch: q.float() @ k.float().T * scale (pure fp32) ----
    q_f32 = q_native.to(tl.float32)
    k_f32 = k_scaled.to(tl.float32)
    scores = tl.dot(q_f32, tl.trans(k_f32), allow_tf32=True) * scale

    # Mask invalid keys with finite -finfo.max (matches torch.finfo(scores.dtype).min)
    NEG_FP32_MAX = -3.4028234663852886e38
    scores = tl.where(smask[None, :], scores, NEG_FP32_MAX)
    row_max = tl.max(scores, axis=1)
    a_sm = tl.exp2((scores - row_max[:, None]) * LOG2E)
    a_sm = tl.where(smask[:, None] & smask[None, :], a_sm, 0.0)

    sm_den = tl.maximum(tl.sum(a_sm, axis=1), eps)       # [S]
    sm_num = tl.dot(a_sm, v_f32, allow_tf32=True)        # [S, D]

    S_base = S_ptr + slot * stride_S_slot + state_h * stride_S_h
    Z_base = Z_ptr + slot * stride_Z_slot + state_h * stride_Z_h
    W_base = W_ptr + state_h * stride_W_h

    # ---- Single-pass linear branch: F=64 fits comfortably in SRAM. This avoids
    # reloading W and recomputing q@W across multiple BLOCK_F passes.
    offs_f = tl.arange(0, F)
    W = tl.load(
        W_base
        + offs_d[:, None] * stride_W_d
        + offs_f[None, :] * stride_W_f,
    ).to(native_dtype)  # [D, F]

    u = tl.dot(q_native, W, allow_tf32=True)  # [S, F] fp32 accum
    u = tl.where(smask[:, None], u, -float("inf"))

    max_pos = tl.where(smask, tl.max(u, axis=1), 0.0)
    max_neg = tl.where(smask, tl.max(-u, axis=1), 0.0)

    e_pos = tl.exp2((u - max_pos[:, None]) * LOG2E)
    e_neg = tl.exp2((-u - max_neg[:, None]) * LOG2E)
    e_pos = tl.where(smask[:, None], e_pos, 0.0)
    e_neg = tl.where(smask[:, None], e_neg, 0.0)

    sum_pos = tl.maximum(tl.sum(e_pos, axis=1), eps)
    sum_neg = tl.maximum(tl.sum(e_neg, axis=1), eps)

    phi_pos = e_pos / sum_pos[:, None]
    phi_neg = e_neg / sum_neg[:, None]

    S_pos = tl.load(
        S_base + offs_f[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
    ).to(tl.float32)
    S_neg = tl.load(
        S_base
        + (offs_f + F)[:, None] * stride_S_f
        + offs_d[None, :] * stride_S_d,
    ).to(tl.float32)
    Z_pos = tl.load(Z_base + offs_f * stride_Z_f).to(tl.float32)
    Z_neg = tl.load(Z_base + (offs_f + F) * stride_Z_f).to(tl.float32)

    lin_num = tl.dot(phi_pos, S_pos, allow_tf32=True) + tl.dot(
        phi_neg, S_neg, allow_tf32=True
    )
    lin_den = tl.sum(phi_pos * Z_pos[None, :], axis=1) + tl.sum(
        phi_neg * Z_neg[None, :], axis=1
    )
    lin_den = tl.maximum(lin_den, eps)

    w = tl.load(alpha_gate_ptr + q_h).to(tl.float32)

    den = tl.maximum(w * sm_den[:, None] + lin_den[:, None], eps)
    out = tl.where(smask[:, None], (w * sm_num + lin_num) / den, 0.0)

    tl.store(
        out_ptr
        + (start_off + offs_s[:, None]) * stride_out_t
        + q_h * stride_out_h
        + offs_d[None, :] * stride_out_d,
        out.to(native_dtype),
        mask=smask[:, None],
    )




@triton.jit
def _hybrid_output_update_kernel_fused_perq(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    req_indices_ptr,
    start_offsets_ptr,
    seq_lens_ptr,
    S_read_ptr,
    Z_read_ptr,
    S_write_ptr,
    Z_write_ptr,
    W_ptr,
    alpha_gate_ptr,
    k_scale,
    v_scale,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_Sr_slot,
    stride_Sr_h,
    stride_Sr_f,
    stride_Sr_d,
    stride_Zr_slot,
    stride_Zr_h,
    stride_Zr_f,
    stride_Sw_slot,
    stride_Sw_h,
    stride_Sw_f,
    stride_Sw_d,
    stride_Zw_slot,
    stride_Zw_h,
    stride_Zw_f,
    stride_W_h,
    stride_W_d,
    stride_W_f,
    num_kv_groups,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
):
    pid_r = tl.program_id(0)
    q_h = tl.program_id(1)
    kv_h = q_h // num_kv_groups

    slot = tl.load(req_indices_ptr + pid_r)
    start_off = tl.load(start_offsets_ptr + pid_r)
    seq_len = tl.load(seq_lens_ptr + pid_r)

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    offs_f = tl.arange(0, F)
    smask = offs_s < seq_len

    q_native = tl.load(
        q_ptr
        + (start_off + offs_s[:, None]) * stride_q_t
        + q_h * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=smask[:, None],
        other=0.0,
    )
    native_dtype = q_native.dtype

    k_native = tl.load(
        k_ptr
        + (start_off + offs_s[:, None]) * stride_k_t
        + kv_h * stride_k_h
        + offs_d[None, :] * stride_k_d,
        mask=smask[:, None],
        other=0.0,
    )
    v_native = tl.load(
        v_ptr
        + (start_off + offs_s[:, None]) * stride_v_t
        + kv_h * stride_v_h
        + offs_d[None, :] * stride_v_d,
        mask=smask[:, None],
        other=0.0,
    )
    # Scale in fp32 for numerical correctness, then downcast to native dtype so
    # the four matmuls below (scores, sm_num, lin_num pos/neg, u_k, S update)
    # route through BF16 WGMMA on Hopper instead of TF32 WGMMA. Accumulators
    # stay fp32 inside tl.dot. Benchmarked at +5–13% over the TF32 path with
    # bf16-stored state — see scripts/kernels_comparison/benchmark_perq.py.
    v_f32 = v_native.to(tl.float32) * v_scale
    v_op = v_f32.to(native_dtype)
    k_op = (k_native.to(tl.float32) * k_scale).to(native_dtype)

    W = tl.load(
        W_ptr
        + q_h * stride_W_h
        + offs_d[:, None] * stride_W_d
        + offs_f[None, :] * stride_W_f,
    ).to(native_dtype)

    LOG2E = 1.4426950408889634

    scores = tl.dot(q_native, tl.trans(k_op)) * scale

    NEG_FP32_MAX = -3.4028234663852886e38
    scores = tl.where(smask[None, :], scores, NEG_FP32_MAX)
    row_max = tl.max(scores, axis=1)
    a_sm = tl.exp2((scores - row_max[:, None]) * LOG2E)
    a_sm = tl.where(smask[:, None] & smask[None, :], a_sm, 0.0)

    sm_den = tl.maximum(tl.sum(a_sm, axis=1), eps)
    sm_num = tl.dot(a_sm.to(native_dtype), v_op)

    Sr_base = S_read_ptr + slot * stride_Sr_slot + q_h * stride_Sr_h
    Zr_base = Z_read_ptr + slot * stride_Zr_slot + q_h * stride_Zr_h

    S_pos = tl.load(
        Sr_base + offs_f[:, None] * stride_Sr_f + offs_d[None, :] * stride_Sr_d,
    ).to(tl.float32)
    S_neg = tl.load(
        Sr_base
        + (offs_f + F)[:, None] * stride_Sr_f
        + offs_d[None, :] * stride_Sr_d,
    ).to(tl.float32)
    Z_pos = tl.load(Zr_base + offs_f * stride_Zr_f).to(tl.float32)
    Z_neg = tl.load(Zr_base + (offs_f + F) * stride_Zr_f).to(tl.float32)

    u_q = tl.dot(q_native, W)
    u_q = tl.where(smask[:, None], u_q, -float("inf"))

    max_q_pos = tl.where(smask, tl.max(u_q, axis=1), 0.0)
    max_q_neg = tl.where(smask, tl.max(-u_q, axis=1), 0.0)
    e_q_pos = tl.exp2((u_q - max_q_pos[:, None]) * LOG2E)
    e_q_neg = tl.exp2((-u_q - max_q_neg[:, None]) * LOG2E)
    e_q_pos = tl.where(smask[:, None], e_q_pos, 0.0)
    e_q_neg = tl.where(smask[:, None], e_q_neg, 0.0)

    phi_q_pos = e_q_pos / tl.maximum(tl.sum(e_q_pos, axis=1)[:, None], eps)
    phi_q_neg = e_q_neg / tl.maximum(tl.sum(e_q_neg, axis=1)[:, None], eps)

    lin_num = tl.dot(
        phi_q_pos.to(native_dtype), S_pos.to(native_dtype)
    ) + tl.dot(
        phi_q_neg.to(native_dtype), S_neg.to(native_dtype)
    )
    lin_den = tl.sum(phi_q_pos * Z_pos[None, :], axis=1) + tl.sum(
        phi_q_neg * Z_neg[None, :], axis=1
    )
    lin_den = tl.maximum(lin_den, eps)

    w = tl.load(alpha_gate_ptr + q_h).to(tl.float32)

    den = tl.maximum(w * sm_den[:, None] + lin_den[:, None], eps)
    out = tl.where(smask[:, None], (w * sm_num + lin_num) / den, 0.0)

    tl.store(
        out_ptr
        + (start_off + offs_s[:, None]) * stride_out_t
        + q_h * stride_out_h
        + offs_d[None, :] * stride_out_d,
        out.to(native_dtype),
        mask=smask[:, None],
    )

    u_k = tl.dot(k_op, W)

    max_k_pos = tl.max(u_k, axis=1)
    e_k_pos = tl.exp2((u_k - max_k_pos[:, None]) * LOG2E)
    phi_k_pos = e_k_pos / tl.maximum(tl.sum(e_k_pos, axis=1)[:, None], eps)
    phi_k_pos = tl.where(smask[:, None], phi_k_pos, 0.0)

    max_k_neg = tl.max(-u_k, axis=1)
    e_k_neg = tl.exp2((-u_k - max_k_neg[:, None]) * LOG2E)
    phi_k_neg = e_k_neg / tl.maximum(tl.sum(e_k_neg, axis=1)[:, None], eps)
    phi_k_neg = tl.where(smask[:, None], phi_k_neg, 0.0)

    Sw_base = S_write_ptr + slot * stride_Sw_slot + q_h * stride_Sw_h
    Zw_base = Z_write_ptr + slot * stride_Zw_slot + q_h * stride_Zw_h

    tl.store(
        Sw_base + offs_f[:, None] * stride_Sw_f + offs_d[None, :] * stride_Sw_d,
        S_pos + tl.dot(tl.trans(phi_k_pos).to(native_dtype), v_op),
    )
    tl.store(
        Sw_base
        + (offs_f + F)[:, None] * stride_Sw_f
        + offs_d[None, :] * stride_Sw_d,
        S_neg + tl.dot(tl.trans(phi_k_neg).to(native_dtype), v_op),
    )
    tl.store(Zw_base + offs_f * stride_Zw_f, Z_pos + tl.sum(phi_k_pos, axis=0))
    tl.store(Zw_base + (offs_f + F) * stride_Zw_f, Z_neg + tl.sum(phi_k_neg, axis=0))


# =============================================================================
# Main module
# =============================================================================


def _pick_block_f(feature_dim: int) -> int:
    for cand in (16, 8, 4, 2, 1):
        if feature_dim % cand == 0 and cand <= feature_dim:
            return cand
    return 1


class BlockSoftmaxLinearHybrid(nn.Module):
    """
    Drop-in for the prior pure-PyTorch reference. Same external API; output
    is equivalent within bf16 quantization noise.

    Per-q-head W shape `[H, D, F]` and alpha `[1, H, 1, 1]`, matching
    src/llada_fast/modeling/hybrid_attention.py checkpoints. State is per-q-head
    `[slots, H, 2F, D]` to match SGLang's existing layer_cache.temporal layout.
    """

    def __init__(
        self,
        config,
        num_heads: int,
        num_kv_heads: int,
        block_d: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        head_dim = int(
            getattr(config, "head_dim", None)
            or (config.hidden_size // config.num_attention_heads)
        )
        feature_dim = int(getattr(config, "feature_dim"))
        block_size = int(getattr(config, "block_size", 32))

        if block_size != 32:
            raise ValueError(
                f"This Triton kernel is specialized for block_size=32, got {block_size}"
            )

        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        # When True, MambaPool Z/S leading dim is num_kv_heads (see LLaDAFastConfigAdapter).
        self.share_hedgehog_kv_groups = bool(
            getattr(config, "share_hedgehog_kv_groups", False)
        )
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.scaling = head_dim ** -0.5
        self.BLOCK_F = _pick_block_f(self.feature_dim)

        # W/state are per-kv-head for KV-shared checkpoints and per-q-head
        # otherwise. Alpha remains per-q-head because it gates the final blend.
        weight_heads = self.num_kv_heads if self.share_hedgehog_kv_groups else self.num_heads
        weight = torch.eye(head_dim, feature_dim).unsqueeze(0).expand(
            weight_heads, -1, -1
        )
        self.hedgehog_weights = nn.Parameter(weight.clone())  # [Hkv or H, D, F]

        # Per-q-head alpha: shape [1, H, 1, 1] for reference compatibility.
        # Lazy-cache sigmoid(alpha) as [H] fp32 on first _run_kernels
        # (load_weights uses .data.copy_ without firing hooks).
        self.alpha = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self._alpha_gate_cache: Optional[torch.Tensor] = None

        # Triton passes raw strides; avoid a per-forward .contiguous() on weights.
        hw = self.hedgehog_weights.data
        if not hw.is_contiguous():
            self.hedgehog_weights.data = hw.contiguous()

    # -- forward over flat token stream (multi-block) -------------------------

    def _run_kernels(
        self,
        q: torch.Tensor,                    # [T, H, D]
        k: torch.Tensor,                    # [T, Hkv, D] or [T, H, D] after expand
        v: torch.Tensor,                    # [T, Hkv, D] or [T, H, D]
        plan: RequestBlockPlan,
        S_read: torch.Tensor,               # [slots, Hkv or H, 2F, D]
        Z_read: torch.Tensor,               # [slots, Hkv or H, 2F]
        S_write: torch.Tensor,
        Z_write: torch.Tensor,
        out: torch.Tensor,                  # [T, H, D] — caller provides buffer
    ) -> torch.Tensor:
        """
        Run Triton hybrid kernels; write attention into `out` and recurrent state
        into ``S_write`` / ``Z_write``.

        Callers (e.g. ``LLaDAFastAttnBackend.forward_qkv``) must pass **contiguous**
        ``q``, ``k``, ``v`` shaped ``[T, H, D]`` and ``[T, Hkv, D]`` — GQA uses
        ``num_kv_groups`` inside Triton; **no** Python ``repeat_kv`` on ``k``/``v``.

        State tensors are the **full MambaPool** views ``[slots, Hkv or H, …]``.
        Triton indexes rows via ``req_indices_ptr`` (Mamba slot ids), so we
        **do not** gather state into a dense fp32 buffer.
        """
        H = self.num_heads
        Hkv = self.num_kv_heads
        D = self.head_dim
        S = self.block_size
        F = self.feature_dim

        if k.shape[1] != v.shape[1]:
            raise ValueError(
                f"k and v head dims must match; got k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}"
            )
        if k.shape[1] == Hkv:
            num_kv_groups = H // Hkv
        elif k.shape[1] == H:
            num_kv_groups = 1
        else:
            raise ValueError(
                f"k must be [T, {Hkv}, D] (GQA) or expanded [T, {H}, D]; got {tuple(k.shape)}"
            )

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        if plan.max_blocks == 0:
            return out

        if self._alpha_gate_cache is None:
            self._alpha_gate_cache = (
                torch.sigmoid(self.alpha.detach().reshape(-1).float()).contiguous()
            )
        alpha_gate = self._alpha_gate_cache
        W = self.hedgehog_weights

        S_r, Z_r = S_read, Z_read
        S_w, Z_w = S_write, Z_write

        # ── Fast path: every request fits in one block (typical dLLM extend).
        if plan.max_blocks == 1:
            R_act = plan.req_indices.numel()
            if R_act == 0:
                return out

            # === ORIGINAL FAST PATH (bf16-matmul fused kernel) — RE-ENABLED ON H200 ===
            # The fused kernel downcasts operands to bf16 for the four tl.dot calls
            # (scores, sm_num, lin_num pos/neg, S update) to route through Hopper
            # BF16 WGMMA (+5–13% vs TF32). Falls through to the streaming + update
            # path below when share_hedgehog_kv_groups=True (which the fused kernel
            # doesn't handle).
            # num_warps is not a free tuning knob. It changes only the reduction order,
            # but JointThreshold compares confidence against a fixed threshold, so a
            # sub-ulp logit difference flips an unmask decision and generation diverges.
            # Measured, HumanEval 164 no-edit / b256 throughput:  8 -> 113, 3985.7 tok/s
            # and 4 -> 120, 3532.3 tok/s.  8 is set for throughput; use 4 for accuracy.
            if not self.share_hedgehog_kv_groups:  # fused bf16-WGMMA fast path (Hopper)
                with timed_region(
                    "hybrid.fused_output_update_kernel",
                    path="hybrid",
                    tokens=int(q.shape[0]),
                    requests=int(R_act),
                    heads=int(H),
                    kv_heads=int(Hkv),
                    share_kv=0,
                    max_blocks=int(plan.max_blocks),
                ):
                    _hybrid_output_update_kernel_fused_perq[(R_act, H)](
                        q, k, v, out,
                        plan.req_indices, plan.start_offsets, plan.seq_lens,
                        S_r, Z_r, S_w, Z_w,
                        W, alpha_gate,
                        1.0, 1.0,
                        q.stride(0), q.stride(1), q.stride(2),
                        k.stride(0), k.stride(1), k.stride(2),
                        v.stride(0), v.stride(1), v.stride(2),
                        out.stride(0), out.stride(1), out.stride(2),
                        S_r.stride(0), S_r.stride(1), S_r.stride(2), S_r.stride(3),
                        Z_r.stride(0), Z_r.stride(1), Z_r.stride(2),
                        S_w.stride(0), S_w.stride(1), S_w.stride(2), S_w.stride(3),
                        Z_w.stride(0), Z_w.stride(1), Z_w.stride(2),
                        W.stride(0), W.stride(1), W.stride(2),
                        num_kv_groups,
                        eps=self.eps,
                        scale=self.scaling,
                        BLOCK_S=S, D=D, F=F,
                        num_warps=8, num_stages=1,
                    )
                return out
            # share_hedgehog_kv_groups=True falls through to streaming variant.
            # =======================================================================

            # === NEW FAST PATH (TF32 streaming + state update) =====================
            # Mirrors the slow-path math (output kernel sees pre-update state,
            # then state-update kernel writes the new state) but runs it once
            # without the multi-block Python loop / .nonzero() / .clamp() ops.
            # Matmul inputs are fp32 with allow_tf32=True on every tl.dot,
            # matching HF's torch.matmul behavior (PyTorch 2.0+ defaults
            # torch.backends.cuda.matmul.allow_tf32 to False, i.e. strict fp32).
            with timed_region(
                "hybrid.output_kernel_tf32",
                path="hybrid",
                tokens=int(q.shape[0]),
                requests=int(R_act),
                heads=int(H),
                kv_heads=int(Hkv),
                share_kv=int(self.share_hedgehog_kv_groups),
                max_blocks=int(plan.max_blocks),
            ):
                _hybrid_output_kernel_streaming[(R_act, H)](
                    q, k, v, out,
                    plan.req_indices,
                    plan.start_offsets,
                    plan.seq_lens,
                    S_r, Z_r,
                    W, alpha_gate,
                    1.0, 1.0,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    S_r.stride(0), S_r.stride(1), S_r.stride(2), S_r.stride(3),
                    Z_r.stride(0), Z_r.stride(1), Z_r.stride(2),
                    W.stride(0), W.stride(1), W.stride(2),
                    num_kv_groups,
                    SHARE_KV_GROUPS=self.share_hedgehog_kv_groups,
                    eps=self.eps,
                    scale=self.scaling,
                    BLOCK_S=S,
                    D=D,
                    F=F,
                    BLOCK_F=self.BLOCK_F,
                    num_warps=4,
                    num_stages=1,
                )

            update_heads = Hkv if self.share_hedgehog_kv_groups else H
            with timed_region(
                "hybrid.update_kernel_tf32",
                path="hybrid",
                tokens=int(k.shape[0]),
                requests=int(R_act),
                heads=int(update_heads),
                kv_heads=int(Hkv),
                share_kv=int(self.share_hedgehog_kv_groups),
                max_blocks=int(plan.max_blocks),
            ):
                _hybrid_update_state_kernel_rw[(R_act, update_heads)](
                    k, v,
                    plan.req_indices,
                    plan.start_offsets,
                    plan.seq_lens,
                    S_r, Z_r,
                    S_w, Z_w,
                    W,
                    1.0, 1.0,
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    S_r.stride(0), S_r.stride(1), S_r.stride(2), S_r.stride(3),
                    Z_r.stride(0), Z_r.stride(1), Z_r.stride(2),
                    S_w.stride(0), S_w.stride(1), S_w.stride(2), S_w.stride(3),
                    Z_w.stride(0), Z_w.stride(1), Z_w.stride(2),
                    W.stride(0), W.stride(1), W.stride(2),
                    num_kv_groups,
                    UPDATE_USES_KV_HEADS=self.share_hedgehog_kv_groups,
                    eps=self.eps,
                    BLOCK_S=S,
                    D=D,
                    F=F,
                    num_warps=4,
                    num_stages=1,
                )
            return out
            # =======================================================================

        # ── Slow path: multi-block extend (seq_len > block_size for some req).
        for block_idx in range(plan.max_blocks):
            active_mask = plan.num_blocks > block_idx
            if not active_mask.any():
                break

            active = torch.nonzero(active_mask, as_tuple=False).flatten().to(torch.long)
            pool_idx = plan.req_indices[active].contiguous()
            R_act = active.numel()

            blk_starts = (plan.start_offsets[active] + block_idx * S).contiguous()
            blk_vlens = (
                (plan.seq_lens[active] - block_idx * S).clamp(0, S).to(torch.long).contiguous()
            )

            if block_idx == 0:
                S_rb, Z_rb = S_r, Z_r
            else:
                S_rb, Z_rb = S_w, Z_w

            _hybrid_output_kernel_streaming[(R_act, H)](
                q, k, v, out,
                pool_idx,
                blk_starts, blk_vlens,
                S_rb, Z_rb,
                W, alpha_gate,
                1.0, 1.0,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                S_rb.stride(0), S_rb.stride(1), S_rb.stride(2), S_rb.stride(3),
                Z_rb.stride(0), Z_rb.stride(1), Z_rb.stride(2),
                W.stride(0), W.stride(1), W.stride(2),
                num_kv_groups,
                SHARE_KV_GROUPS=self.share_hedgehog_kv_groups,
                eps=self.eps,
                scale=self.scaling,
                BLOCK_S=S,
                D=D,
                F=F,
                BLOCK_F=self.BLOCK_F,
                num_warps=4,
                num_stages=1,
            )

            update_heads = Hkv if self.share_hedgehog_kv_groups else H
            _hybrid_update_state_kernel_rw[(R_act, update_heads)](
                k, v,
                pool_idx,
                blk_starts, blk_vlens,
                S_rb, Z_rb,
                S_w, Z_w,
                W,
                1.0, 1.0,
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                S_rb.stride(0), S_rb.stride(1), S_rb.stride(2), S_rb.stride(3),
                Z_rb.stride(0), Z_rb.stride(1), Z_rb.stride(2),
                S_w.stride(0), S_w.stride(1), S_w.stride(2), S_w.stride(3),
                Z_w.stride(0), Z_w.stride(1), Z_w.stride(2),
                W.stride(0), W.stride(1), W.stride(2),
                num_kv_groups,
                UPDATE_USES_KV_HEADS=self.share_hedgehog_kv_groups,
                eps=self.eps,
                BLOCK_S=S,
                D=D,
                F=F,
                num_warps=4,
                num_stages=1,
            )

        return out

    def forward(
        self,
        q: torch.Tensor,                    # [T, H, D]
        k: torch.Tensor,                    # [T, H or Hkv, D]
        v: torch.Tensor,                    # [T, H or Hkv, D]
        seq_lens: List[int],
        req_indices: torch.Tensor,          # [R]
        read_state: HybridStateView,
        write_state: Optional[HybridStateView] = None,
        plan: Optional[RequestBlockPlan] = None,
    ) -> torch.Tensor:
        if write_state is None:
            write_state = read_state

        device = q.device
        S = self.block_size
        if plan is None:
            plan = build_request_block_plan(
                seq_lens=seq_lens,
                req_indices=req_indices,
                block_size=S,
                device=device,
            )

        out = torch.zeros_like(q)
        return self._run_kernels(
            q,
            k,
            v,
            plan,
            read_state.S_state,
            read_state.Z_state,
            write_state.S_state,
            write_state.Z_state,
            out,
        )

    # -- SGLang entry point ---------------------------------------------------

    def forward_qkv(self, q, k, v, layer_cache, metadata) -> torch.Tensor:
        T = q.shape[0]
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
        committed, working = make_state_view_from_layer_cache(layer_cache)
        plan = getattr(metadata, "plan", None)
        if plan is None:
            plan = build_request_block_plan(
                seq_lens=metadata.seq_lens,
                req_indices=metadata.state_slot_indices,
                block_size=self.block_size,
                device=q.device,
            )
        out = torch.zeros_like(q.reshape(T, H, D))
        self._run_kernels(
            q.reshape(T, H, D),
            k.reshape(T, Hkv, D),
            v.reshape(T, Hkv, D),
            plan,
            committed.S_state,
            committed.Z_state,
            working.S_state,
            working.Z_state,
            out,
        )
        return out.reshape(T, H * D)

    @torch.no_grad()
    def reset_states(self, state: HybridStateView, slots=None):
        if slots is None:
            state.S_state.zero_()
            state.Z_state.zero_()
        else:
            state.S_state[slots] = 0
            state.Z_state[slots] = 0


# =============================================================================
# SGLang MambaPool state adapter
# =============================================================================


def make_state_view_from_layer_cache(layer_cache):
    """
    Layout (unchanged from prior reference):
        layer_cache.temporal: [slots, 2, H, 2F, D]
            temporal[:, 0] = committed S
            temporal[:, 1] = working   S
        layer_cache.conv[0]:  [slots, H*2F, 1] or [slots, H*2F]   committed Z
        layer_cache.conv[1]:  [slots, H*2F, 1] or [slots, H*2F]   working   Z
    """
    S_full = layer_cache.temporal
    if S_full.ndim != 5 or S_full.shape[1] != 2:
        raise ValueError(
            f"Expected temporal shape [slots, 2, H, 2F, D], got {tuple(S_full.shape)}"
        )

    S_committed = S_full[:, 0]
    S_working = S_full[:, 1]

    if len(layer_cache.conv) < 2:
        raise ValueError("Expected conv[0]=committed Z and conv[1]=working Z")

    z0 = layer_cache.conv[0]
    z1 = layer_cache.conv[1]
    if z0.ndim == 3 and z0.shape[-1] == 1:
        z0 = z0[..., 0]
        z1 = z1[..., 0]

    H = S_committed.shape[1]
    twoF = S_committed.shape[2]
    Z_committed = z0.view(-1, H, twoF)
    Z_working = z1.view(-1, H, twoF)

    committed = HybridStateView(S_state=S_committed, Z_state=Z_committed)
    working = HybridStateView(S_state=S_working, Z_state=Z_working)
    return committed, working



# """
# Triton-optimized hybrid attention kernel for SGLang serving.

# Goal:
# - keep the same external interface as the reference pure-PyTorch version
# - move the hot path to Triton
# - remove Python per-request gather/scatter loops
# - fuse:
#     * block gather
#     * linear branch numerator/denominator
#     * intra-block softmax branch
#     * shared-normalization blend
#     * recurrent state update

# Notes:
# - recurrence across blocks is still sequential by design
# - the heavy inner work per active-request x head is fused
# - supports fp16/bf16 inputs, accumulates in fp32
# - exact "feature map = [softmax(u), softmax(-u)]" is preserved
# - block_size should realistically be 16/32/64 for best Triton codegen
# """


# # import math
# # from dataclasses import dataclass
# # from typing import Optional

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import triton
# # import triton.language as tl


# # # =============================================================================
# # # Block planning
# # # =============================================================================


# # @dataclass
# # class RequestBlockPlan:
# #     req_indices: torch.Tensor       # [R] pool slot indices
# #     seq_lens: torch.Tensor          # [R]
# #     start_offsets: torch.Tensor     # [R]
# #     num_blocks: torch.Tensor        # [R]
# #     max_blocks: int
# #     block_size: int


# # def build_request_block_plan(
# #     seq_lens: list[int],
# #     req_indices: torch.Tensor,
# #     block_size: int,
# #     device: torch.device,
# # ) -> RequestBlockPlan:
# #     seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
# #     req_indices_t = req_indices.to(device=device, dtype=torch.long)

# #     if seq_lens_t.numel() == 0:
# #         empty = torch.empty((0,), dtype=torch.long, device=device)
# #         return RequestBlockPlan(
# #             req_indices=empty,
# #             seq_lens=empty,
# #             start_offsets=empty,
# #             num_blocks=empty,
# #             max_blocks=0,
# #             block_size=block_size,
# #         )

# #     start_offsets = torch.zeros_like(seq_lens_t)
# #     if seq_lens_t.numel() > 1:
# #         start_offsets[1:] = torch.cumsum(seq_lens_t[:-1], dim=0)

# #     num_blocks = (seq_lens_t + block_size - 1) // block_size
# #     return RequestBlockPlan(
# #         req_indices=req_indices_t,
# #         seq_lens=seq_lens_t,
# #         start_offsets=start_offsets,
# #         num_blocks=num_blocks,
# #         max_blocks=int(num_blocks.max().item()),
# #         block_size=block_size,
# #     )


# # # =============================================================================
# # # State
# # # =============================================================================


# # @dataclass
# # class HybridStateView:
# #     S_state: torch.Tensor  # [slots, H, 2F, D] fp32
# #     Z_state: torch.Tensor  # [slots, H, 2F]    fp32


# # # =============================================================================
# # # GQA helper
# # # =============================================================================


# # def _repeat_kv(x: torch.Tensor, num_q_heads: int, num_kv_heads: int, head_dim: int):
# #     if num_q_heads == num_kv_heads:
# #         return x
# #     groups = num_q_heads // num_kv_heads
# #     t = x.shape[0]
# #     x = x.reshape(t, num_kv_heads, head_dim)
# #     x = x[:, :, None, :].expand(t, num_kv_heads, groups, head_dim)
# #     return x.reshape(t, num_q_heads, head_dim)


# # # =============================================================================
# # # Main fused kernel:
# # # - reads old S/Z
# # # - computes block output
# # # - updates S only
# # # - does NOT update Z
# # # =============================================================================


# # @triton.jit
# # def _hybrid_block_kernel(
# #     q_ptr, k_ptr, v_ptr, out_ptr,
# #     active_req_ptr,
# #     start_ptr,
# #     valid_len_ptr,
# #     req_slot_ptr,
# #     S_state_ptr,
# #     Z_state_ptr,
# #     W_ptr,
# #     alpha_ptr,
# #     T,
# #     H,
# #     D,
# #     Fdim,
# #     stride_q_t, stride_q_h, stride_q_d,
# #     stride_k_t, stride_k_h, stride_k_d,
# #     stride_v_t, stride_v_h, stride_v_d,
# #     stride_out_t, stride_out_h, stride_out_d,
# #     stride_S_slot, stride_S_h, stride_S_f, stride_S_d,
# #     stride_Z_slot, stride_Z_h, stride_Z_f,
# #     stride_W_h, stride_W_d, stride_W_f,
# #     eps,
# #     scale,
# #     BLOCK_S: tl.constexpr,
# #     BLOCK_D: tl.constexpr,
# #     BLOCK_F: tl.constexpr,
# # ):
# #     pid0 = tl.program_id(0)  # active request
# #     pid1 = tl.program_id(1)  # head
# #     pid2 = tl.program_id(2)  # D tile

# #     offs_s = tl.arange(0, BLOCK_S)
# #     offs_d = pid2 * BLOCK_D + tl.arange(0, BLOCK_D)
# #     dmask = offs_d < D

# #     req_idx = tl.load(active_req_ptr + pid0)
# #     token_start = tl.load(start_ptr + req_idx)
# #     valid_len = tl.load(valid_len_ptr + req_idx)
# #     slot = tl.load(req_slot_ptr + req_idx)
# #     h = pid1

# #     smask = offs_s < valid_len
# #     neg_inf = -1.0e9

# #     # ── Capture native dtype early (before any loop-scoped loads) ──
# #     out_dtype = tl.load(v_ptr + token_start * stride_v_t + h * stride_v_h).dtype

# #     # ── Load v_tile [BLOCK_S, BLOCK_D] for output and S update ──
# #     v_ptrs = (
# #         v_ptr
# #         + (token_start + offs_s[:, None]) * stride_v_t
# #         + h * stride_v_h
# #         + offs_d[None, :] * stride_v_d
# #     )
# #     v_tile = tl.load(v_ptrs, mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

# #     # ── Blend weight ──
# #     one = tl.full((), 1.0, tl.float32)
# #     alpha_val = tl.load(alpha_ptr + h).to(tl.float32)
# #     w = one / (one + tl.exp(-alpha_val))

# #     # ── Intra-block softmax: q @ k^T in bf16 (matching reference) ──
# #     scores = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)
# #     for dk in range(0, D, BLOCK_D):
# #         offs_dk = dk + tl.arange(0, BLOCK_D)
# #         dkmask = offs_dk < D
# #         q2 = tl.load(
# #             q_ptr + (token_start + offs_s[:, None]) * stride_q_t + h * stride_q_h + offs_dk[None, :] * stride_q_d,
# #             mask=smask[:, None] & dkmask[None, :], other=0.0)
# #         k2 = tl.load(
# #             k_ptr + (token_start + offs_s[:, None]) * stride_k_t + h * stride_k_h + offs_dk[None, :] * stride_k_d,
# #             mask=smask[:, None] & dkmask[None, :], other=0.0)
# #         scores += tl.dot(q2, tl.trans(k2))

# #     scores = (scores * scale).to(out_dtype)
# #     scores = tl.where(smask[None, :], scores, neg_inf)
# #     row_max = tl.max(scores, axis=1)
# #     a_sm = tl.exp(scores - row_max[:, None]).to(tl.float32)
# #     a_sm = tl.where(smask[:, None] & smask[None, :], a_sm, 0.0)
# #     sm_den = tl.maximum(tl.sum(a_sm, axis=1), eps)
# #     sm_num = tl.dot(a_sm, v_tile)

# #     # ── Preload entire Z_old into registers (2*Fdim values) ──
# #     # This avoids the D-tile race: all D-tiles read Z from registers,
# #     # pid2==0 writes Z_new to global memory after the output store.
# #     offs_f_all = tl.arange(0, BLOCK_F)  # assuming Fdim <= BLOCK_F (single tile)
# #     fmask_all = offs_f_all < Fdim

# #     Z_base = Z_state_ptr + slot * stride_Z_slot + h * stride_Z_h
# #     Z_pos_reg = tl.load(Z_base + offs_f_all * stride_Z_f, mask=fmask_all, other=0.0).to(tl.float32)
# #     Z_neg_reg = tl.load(Z_base + (offs_f_all + Fdim) * stride_Z_f, mask=fmask_all, other=0.0).to(tl.float32)

# #     # ── Single-pass feature map + linear branch + S update ──
# #     # For Fdim <= BLOCK_F, the F loop runs once — no need for online softmax.
# #     # u = q @ W and u_k = k @ W computed once, softmax directly.
# #     lin_num = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
# #     lin_den = tl.zeros((BLOCK_S,), dtype=tl.float32)
# #     Z_pos_delta = tl.zeros((BLOCK_F,), dtype=tl.float32)
# #     Z_neg_delta = tl.zeros((BLOCK_F,), dtype=tl.float32)

# #     for ff in range(0, Fdim, BLOCK_F):
# #         offs_f = ff + tl.arange(0, BLOCK_F)
# #         fmask = offs_f < Fdim

# #         # Compute u_q = q @ W, u_k = k @ W (bf16 dots, fp32 accum)
# #         u_q = tl.zeros((BLOCK_S, BLOCK_F), dtype=tl.float32)
# #         u_k = tl.zeros((BLOCK_S, BLOCK_F), dtype=tl.float32)
# #         for dk in range(0, D, BLOCK_D):
# #             offs_dk = dk + tl.arange(0, BLOCK_D)
# #             dkmask = offs_dk < D
# #             q2 = tl.load(
# #                 q_ptr + (token_start + offs_s[:, None]) * stride_q_t + h * stride_q_h + offs_dk[None, :] * stride_q_d,
# #                 mask=smask[:, None] & dkmask[None, :], other=0.0)
# #             k2 = tl.load(
# #                 k_ptr + (token_start + offs_s[:, None]) * stride_k_t + h * stride_k_h + offs_dk[None, :] * stride_k_d,
# #                 mask=smask[:, None] & dkmask[None, :], other=0.0)
# #             w2 = tl.load(
# #                 W_ptr + h * stride_W_h + offs_dk[:, None] * stride_W_d + offs_f[None, :] * stride_W_f,
# #                 mask=dkmask[:, None] & fmask[None, :], other=0.0)
# #             u_q += tl.dot(q2, w2)
# #             u_k += tl.dot(k2, w2)

# #         # Direct softmax (single F-tile: no online reduction needed)
# #         u_q = tl.where(fmask[None, :], u_q, -1.0e30)
# #         u_k = tl.where(fmask[None, :], u_k, -1.0e30)

# #         # phi_q = [softmax(u_q), softmax(-u_q)] with bf16 round-trip
# #         m = tl.max(u_q, axis=1)
# #         e = tl.exp(u_q - m[:, None])
# #         phi_q_pos = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(out_dtype).to(tl.float32)
# #         m = tl.max(-u_q, axis=1)
# #         e = tl.exp(-u_q - m[:, None])
# #         phi_q_neg = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(out_dtype).to(tl.float32)

# #         m = tl.max(u_k, axis=1)
# #         e = tl.exp(u_k - m[:, None])
# #         phi_k_pos = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(out_dtype).to(tl.float32)
# #         m = tl.max(-u_k, axis=1)
# #         e = tl.exp(-u_k - m[:, None])
# #         phi_k_neg = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(out_dtype).to(tl.float32)

# #         # Zero invalid rows
# #         phi_q_pos = tl.where(smask[:, None] & fmask[None, :], phi_q_pos, 0.0)
# #         phi_q_neg = tl.where(smask[:, None] & fmask[None, :], phi_q_neg, 0.0)
# #         phi_k_pos = tl.where(smask[:, None] & fmask[None, :], phi_k_pos, 0.0)
# #         phi_k_neg = tl.where(smask[:, None] & fmask[None, :], phi_k_neg, 0.0)

# #         # ── Linear branch: lin_num += phi_q @ S_old, lin_den += phi_q . Z_old ──
# #         S_base = S_state_ptr + slot * stride_S_slot + h * stride_S_h

# #         S_pos = tl.load(S_base + offs_f[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
# #                          mask=fmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
# #         S_neg = tl.load(S_base + (offs_f + Fdim)[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
# #                          mask=fmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

# #         # Z read from REGISTERS (preloaded), not global — no D-tile race
# #         Z_pos_tile = tl.where(fmask, Z_pos_reg, 0.0) if ff == 0 else tl.load(Z_base + offs_f * stride_Z_f, mask=fmask, other=0.0).to(tl.float32)
# #         Z_neg_tile = tl.where(fmask, Z_neg_reg, 0.0) if ff == 0 else tl.load(Z_base + (offs_f + Fdim) * stride_Z_f, mask=fmask, other=0.0).to(tl.float32)

# #         lin_num += tl.dot(phi_q_pos, S_pos) + tl.dot(phi_q_neg, S_neg)
# #         lin_den += tl.sum(phi_q_pos * Z_pos_tile[None, :], axis=1) + tl.sum(phi_q_neg * Z_neg_tile[None, :], axis=1)

# #         # ── S update: S_new = S_old + phi_k^T @ v_tile ──
# #         tl.store(S_base + offs_f[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
# #                  S_pos + tl.dot(tl.trans(phi_k_pos), v_tile), mask=fmask[:, None] & dmask[None, :])
# #         tl.store(S_base + (offs_f + Fdim)[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
# #                  S_neg + tl.dot(tl.trans(phi_k_neg), v_tile), mask=fmask[:, None] & dmask[None, :])

# #         # ── Accumulate Z delta (same phi_k used for S) ──
# #         if ff == 0:
# #             Z_pos_delta = tl.sum(phi_k_pos, axis=0)
# #             Z_neg_delta = tl.sum(phi_k_neg, axis=0)
# #         else:
# #             Z_pos_delta += tl.sum(phi_k_pos, axis=0)
# #             Z_neg_delta += tl.sum(phi_k_neg, axis=0)

# #     lin_den = tl.maximum(lin_den, eps)

# #     # ── Blend and store output ──
# #     den = w * sm_den[:, None] + lin_den[:, None]
# #     den = tl.maximum(den, eps)
# #     out_tile = (w * sm_num + lin_num) / den
# #     out_tile = tl.where(smask[:, None] & dmask[None, :], out_tile, 0.0)

# #     out_ptrs = (
# #         out_ptr
# #         + (token_start + offs_s[:, None]) * stride_out_t
# #         + h * stride_out_h
# #         + offs_d[None, :] * stride_out_d
# #     )
# #     tl.store(out_ptrs, out_tile.to(out_dtype), mask=smask[:, None] & dmask[None, :])

# #     # ── Z update from pid2==0 only (no race: Z reads were from registers) ──
# #     if pid2 == 0:
# #         tl.store(Z_base + offs_f_all * stride_Z_f,
# #                  Z_pos_reg + Z_pos_delta, mask=fmask_all)
# #         tl.store(Z_base + (offs_f_all + Fdim) * stride_Z_f,
# #                  Z_neg_reg + Z_neg_delta, mask=fmask_all)


# # def _launch_hybrid_block_kernel(
# #     q: torch.Tensor,
# #     k: torch.Tensor,
# #     v: torch.Tensor,
# #     out: torch.Tensor,
# #     active_req: torch.Tensor,
# #     starts: torch.Tensor,
# #     valid_lens: torch.Tensor,
# #     req_slots: torch.Tensor,
# #     state: HybridStateView,
# #     hedgehog_weights: torch.Tensor,
# #     alpha: torch.Tensor,
# #     block_size: int,
# #     head_dim: int,
# #     feature_dim: int,
# #     eps: float,
# # ):
# #     H = q.shape[1]
# #     D = head_dim
# #     Fdim = feature_dim

# #     BLOCK_S = block_size
# #     BLOCK_D = 64 if D <= 64 else 128
# #     BLOCK_F = max(64, Fdim)  # must be >= Fdim for Z preload

# #     grid = (active_req.numel(), H, triton.cdiv(D, BLOCK_D))
# #     _hybrid_block_kernel[grid](
# #         q, k, v, out,
# #         active_req,
# #         starts,
# #         valid_lens,
# #         req_slots,
# #         state.S_state,
# #         state.Z_state,
# #         hedgehog_weights,
# #         alpha,
# #         q.shape[0],
# #         H,
# #         D,
# #         Fdim,
# #         q.stride(0), q.stride(1), q.stride(2),
# #         k.stride(0), k.stride(1), k.stride(2),
# #         v.stride(0), v.stride(1), v.stride(2),
# #         out.stride(0), out.stride(1), out.stride(2),
# #         state.S_state.stride(0), state.S_state.stride(1), state.S_state.stride(2), state.S_state.stride(3),
# #         state.Z_state.stride(0), state.Z_state.stride(1), state.Z_state.stride(2),
# #         hedgehog_weights.stride(0), hedgehog_weights.stride(1), hedgehog_weights.stride(2),
# #         eps,
# #         1.0 / math.sqrt(D),
# #         BLOCK_S=BLOCK_S,
# #         BLOCK_D=BLOCK_D,
# #         BLOCK_F=BLOCK_F,
# #         num_warps=8 if BLOCK_D == 128 else 4,
# #         num_stages=2,
# #     )


# # def _launch_z_update_kernel(
# #     phi_k_buf: torch.Tensor,
# #     active_req: torch.Tensor,
# #     valid_lens: torch.Tensor,
# #     req_slots: torch.Tensor,
# #     state: HybridStateView,
# #     num_heads: int,
# #     block_size: int,
# #     feature_dim: int,
# # ):
# #     BLOCK_S = block_size
# #     BLOCK_F = 64 if feature_dim <= 64 else 128

# #     grid = (active_req.numel(), num_heads)
# #     _z_update_kernel[grid](
# #         phi_k_buf,
# #         active_req,
# #         valid_lens,
# #         req_slots,
# #         state.Z_state,
# #         feature_dim,
# #         phi_k_buf.stride(0), phi_k_buf.stride(1), phi_k_buf.stride(2), phi_k_buf.stride(3),
# #         state.Z_state.stride(0), state.Z_state.stride(1), state.Z_state.stride(2),
# #         BLOCK_S=BLOCK_S,
# #         BLOCK_F=BLOCK_F,
# #         num_warps=4,
# #         num_stages=2,
# #     )


# # # =============================================================================
# # # Main module
# # # =============================================================================


# # class BlockSoftmaxLinearHybrid(nn.Module):
# #     def __init__(
# #         self,
# #         config,
# #         num_heads: int,
# #         num_kv_heads: int,
# #         eps: float = 1e-6,
# #     ):
# #         super().__init__()
# #         head_dim = int(
# #             getattr(config, "head_dim", None)
# #             or (config.hidden_size // config.num_attention_heads)
# #         )
# #         feature_dim = int(getattr(config, "feature_dim"))
# #         block_size = int(getattr(config, "block_size", 32))

# #         if head_dim not in (64, 80, 96, 128):
# #             raise ValueError(f"Unsupported head_dim={head_dim}; use 64/80/96/128")

# #         self.num_heads = int(num_heads)
# #         self.num_kv_heads = int(num_kv_heads)
# #         self.head_dim = int(head_dim)
# #         self.feature_dim = int(feature_dim)
# #         self.block_size = int(block_size)
# #         self.eps = float(eps)

# #         weight = torch.eye(head_dim, feature_dim).unsqueeze(0).expand(
# #             num_heads, -1, -1
# #         )
# #         self.hedgehog_weights = nn.Parameter(weight.clone().contiguous())  # [H, D, F]
# #         self.alpha = nn.Parameter(torch.zeros(num_heads, device=weight.device))  # [H]

# #     def _feature_map_reference(self, x: torch.Tensor) -> torch.Tensor:
# #         u = torch.einsum("bhsd,hdf->bhsf", x, self.hedgehog_weights.to(dtype=x.dtype))
# #         u_f32 = u.float()
# #         return torch.cat(
# #             [F.softmax(u_f32, dim=-1), F.softmax(-u_f32, dim=-1)], dim=-1
# #         ).to(dtype=x.dtype)

# #     def forward(
# #         self,
# #         q: torch.Tensor,            # [T, H, D]
# #         k: torch.Tensor,            # [T, H or Hkv, D]
# #         v: torch.Tensor,            # [T, H or Hkv, D]
# #         seq_lens: list[int],
# #         req_indices: torch.Tensor,  # [R]
# #         read_state: HybridStateView,
# #         write_state: Optional[HybridStateView] = None,
# #     ) -> torch.Tensor:
# #         if write_state is None:
# #             write_state = read_state

# #         H = self.num_heads
# #         D = self.head_dim
# #         S = self.block_size
# #         out_dtype = q.dtype

# #         if k.shape[1] != H:
# #             k = _repeat_kv(k, H, self.num_kv_heads, D)
# #         if v.shape[1] != H:
# #             v = _repeat_kv(v, H, self.num_kv_heads, D)

# #         q = q.contiguous()
# #         k = k.contiguous()
# #         v = v.contiguous()

# #         plan = build_request_block_plan(
# #             seq_lens=seq_lens,
# #             req_indices=req_indices,
# #             block_size=S,
# #             device=q.device,
# #         )

# #         out = torch.zeros_like(q)
# #         if plan.max_blocks == 0:
# #             return out

# #         # Local double-buffered state — matches reference semantics exactly:
# #         # clone committed state into local fp32 tensors, update locals
# #         # block-by-block, write final result to working buffer once at the end.
# #         # This prevents denoising passes from observing partially advanced state.
# #         R = plan.req_indices.numel()
# #         S_cur = read_state.S_state[plan.req_indices].clone().contiguous()
# #         Z_cur = read_state.Z_state[plan.req_indices].clone().contiguous()
# #         S_tmp = torch.empty_like(S_cur)
# #         Z_tmp = torch.empty_like(Z_cur)

# #         # No phi_k buffer needed — Z update is merged into main kernel
# #         # using register-preloaded Z (no D-tile race).

# #         for block_idx in range(plan.max_blocks):
# #             active_mask = plan.num_blocks > block_idx
# #             if not active_mask.any():
# #                 break

# #             active = torch.nonzero(active_mask, as_tuple=False).flatten()

# #             # Full R-sized arrays — kernel indexes via active_req indirection
# #             all_starts = plan.start_offsets + block_idx * S
# #             all_vlens = (plan.seq_lens - block_idx * S).clamp(0, S)
# #             # local_slots maps active index → local buffer index (0..R-1)
# #             local_slots = torch.arange(R, device=q.device, dtype=torch.long)

# #             # Copy current state for active slots to tmp (kernel reads/writes tmp in-place)
# #             S_tmp[active] = S_cur[active]
# #             Z_tmp[active] = Z_cur[active]

# #             tmp_state = HybridStateView(S_tmp, Z_tmp)

# #             _launch_hybrid_block_kernel(
# #                 q=q, k=k, v=v, out=out,
# #                 active_req=active,
# #                 starts=all_starts,
# #                 valid_lens=all_vlens,
# #                 req_slots=local_slots,
# #                 state=tmp_state,
# #                 hedgehog_weights=self.hedgehog_weights,
# #                 alpha=self.alpha,
# #                 block_size=S, head_dim=D,
# #                 feature_dim=self.feature_dim, eps=self.eps,
# #             )

# #             # Chain state forward — copy updated active slots back
# #             S_cur[active] = S_tmp[active]
# #             Z_cur[active] = Z_tmp[active]

# #         # Write final chained state to working buffer once
# #         write_state.S_state[plan.req_indices] = S_cur
# #         write_state.Z_state[plan.req_indices] = Z_cur

# #         return out

# #     def forward_qkv(self, q, k, v, layer_cache, metadata):
# #         T = q.shape[0]
# #         H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
# #         committed, working = make_state_view_from_layer_cache(layer_cache)

# #         q = q.contiguous().reshape(T, H, D)
# #         k = k.contiguous().reshape(T, Hkv, D)
# #         v = v.contiguous().reshape(T, Hkv, D)

# #         out = self.forward(
# #             q,
# #             k,
# #             v,
# #             seq_lens=metadata.seq_lens,
# #             req_indices=metadata.state_slot_indices,
# #             read_state=committed,
# #             write_state=working,
# #         )
# #         return out.reshape(T, H * D)

# #     @torch.no_grad()
# #     def reset_states(self, state: HybridStateView, slots=None):
# #         if slots is None:
# #             state.S_state.zero_()
# #             state.Z_state.zero_()
# #         else:
# #             state.S_state[slots] = 0
# #             state.Z_state[slots] = 0


# # def make_state_view_from_layer_cache(layer_cache):
# #     S_full = layer_cache.temporal

# #     S_committed = S_full[:, 0]
# #     S_working = S_full[:, 1]

# #     z0 = layer_cache.conv[0]
# #     z1 = layer_cache.conv[1]
# #     if z0.ndim == 3 and z0.shape[-1] == 1:
# #         z0 = z0[..., 0]
# #         z1 = z1[..., 0]

# #     H = S_committed.shape[1]
# #     twoF = S_committed.shape[2]
# #     Z_committed = z0.reshape(-1, H, twoF)
# #     Z_working = z1.reshape(-1, H, twoF)

# #     committed = HybridStateView(S_state=S_committed, Z_state=Z_committed)
# #     working = HybridStateView(S_state=S_working, Z_state=Z_working)
# #     return committed, working


# # =============================================================================
# # ACTIVE IMPLEMENTATION — Persistent fused kernel (ThunderKittens-inspired)
# #
# # Key design (from HazyResearch/ThunderKittens hedgehog.cu):
# #   1. One program per (request, head) — loops over ALL blocks internally.
# #   2. No D-tiling — full D per program. No D-tile race on Z.
# #   3. Z state in registers across blocks. S streamed from local buffer.
# #   4. Single phi_k for both S and Z updates — no recomputation drift.
# #   5. Single-pass feature map (F <= BLOCK_F).
# # =============================================================================

# import math
# from dataclasses import dataclass
# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F_torch
# import triton
# import triton.language as tl


# @dataclass
# class RequestBlockPlan:
#     req_indices: torch.Tensor
#     seq_lens: torch.Tensor
#     start_offsets: torch.Tensor
#     num_blocks: torch.Tensor
#     max_blocks: int
#     block_size: int


# def build_request_block_plan(seq_lens, req_indices, block_size, device):
#     seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
#     req_indices_t = req_indices.to(device=device, dtype=torch.long)
#     if seq_lens_t.numel() == 0:
#         empty = torch.empty((0,), dtype=torch.long, device=device)
#         return RequestBlockPlan(empty, empty, empty, empty, 0, block_size)
#     start_offsets = torch.zeros_like(seq_lens_t)
#     if seq_lens_t.numel() > 1:
#         start_offsets[1:] = torch.cumsum(seq_lens_t[:-1], dim=0)
#     num_blocks = (seq_lens_t + block_size - 1) // block_size
#     return RequestBlockPlan(
#         req_indices_t, seq_lens_t, start_offsets, num_blocks,
#         int(num_blocks.max().item()), block_size,
#     )


# @dataclass
# class HybridStateView:
#     S_state: torch.Tensor
#     Z_state: torch.Tensor


# def _repeat_kv(x, num_q_heads, num_kv_heads, head_dim):
#     if num_q_heads == num_kv_heads:
#         return x
#     groups = num_q_heads // num_kv_heads
#     t = x.shape[0]
#     return (
#         x.view(t, num_kv_heads, head_dim)[:, :, None, :]
#         .expand(t, num_kv_heads, groups, head_dim)
#         .reshape(t, num_q_heads, head_dim)
#     )


# # Grid: (R, H) — one program per (request, head), loops over blocks internally
# @triton.jit
# def _hybrid_persistent_kernel(
#     q_ptr, k_ptr, v_ptr, out_ptr,
#     start_offsets_ptr, seq_lens_ptr, num_blocks_ptr,
#     S_ptr, Z_ptr,
#     W_ptr, alpha_ptr,
#     stride_q_t, stride_q_h, stride_q_d,
#     stride_k_t, stride_k_h, stride_k_d,
#     stride_v_t, stride_v_h, stride_v_d,
#     stride_out_t, stride_out_h, stride_out_d,
#     stride_S_r, stride_S_h, stride_S_f, stride_S_d,
#     stride_Z_r, stride_Z_h, stride_Z_f,
#     stride_W_h, stride_W_d, stride_W_f,
#     eps: tl.constexpr,
#     scale: tl.constexpr,
#     BLOCK_S: tl.constexpr,
#     D: tl.constexpr,
#     F: tl.constexpr,
#     MAX_BLOCKS: tl.constexpr,
# ):
#     pid_r = tl.program_id(0)
#     pid_h = tl.program_id(1)
#     h = pid_h

#     seq_len = tl.load(seq_lens_ptr + pid_r)
#     start_off = tl.load(start_offsets_ptr + pid_r)
#     n_blocks = tl.load(num_blocks_ptr + pid_r)

#     alpha_val = tl.load(alpha_ptr + h).to(tl.float32)
#     w = 1.0 / (1.0 + tl.exp(-alpha_val))

#     offs_s = tl.arange(0, BLOCK_S)
#     offs_d = tl.arange(0, D)
#     offs_f = tl.arange(0, F)

#     # Capture native dtype
#     native_dtype = tl.load(q_ptr).dtype

#     # Load W [D, F] for this head — constant across blocks
#     W = tl.load(W_ptr + h * stride_W_h + offs_d[:, None] * stride_W_d + offs_f[None, :] * stride_W_f)

#     # Z in registers — persists across blocks
#     Z_base = Z_ptr + pid_r * stride_Z_r + h * stride_Z_h
#     Z_pos = tl.load(Z_base + offs_f * stride_Z_f).to(tl.float32)
#     Z_neg = tl.load(Z_base + (offs_f + F) * stride_Z_f).to(tl.float32)

#     S_base = S_ptr + pid_r * stride_S_r + h * stride_S_h

#     for block_idx in range(MAX_BLOCKS):
#         # No break in Triton — guard entire body instead
#         active = block_idx < n_blocks
#         token_start = start_off + block_idx * BLOCK_S
#         valid_len = tl.where(active, tl.minimum(seq_len - block_idx * BLOCK_S, BLOCK_S), 0)
#         smask = offs_s < valid_len

#         # Load q, k, v [BLOCK_S, D] — full D, no tiling
#         q_base = (token_start + offs_s[:, None]) * stride_q_t + h * stride_q_h
#         q = tl.load(q_ptr + q_base + offs_d[None, :] * stride_q_d, mask=smask[:, None], other=0.0)
#         k_base = (token_start + offs_s[:, None]) * stride_k_t + h * stride_k_h
#         k = tl.load(k_ptr + k_base + offs_d[None, :] * stride_k_d, mask=smask[:, None], other=0.0)
#         v_base = (token_start + offs_s[:, None]) * stride_v_t + h * stride_v_h
#         v = tl.load(v_ptr + v_base + offs_d[None, :] * stride_v_d, mask=smask[:, None], other=0.0)
#         v_f32 = v.to(tl.float32)

#         # Softmax branch: q @ k^T in native dtype
#         scores = tl.dot(q, tl.trans(k))
#         scores = (scores * scale).to(native_dtype)
#         scores = tl.where(smask[None, :], scores, -1.0e9)
#         row_max = tl.max(scores, axis=1)
#         a_sm = tl.exp(scores - row_max[:, None]).to(tl.float32)
#         a_sm = tl.where(smask[:, None] & smask[None, :], a_sm, 0.0)
#         sm_den = tl.maximum(tl.sum(a_sm, axis=1), eps)
#         sm_num = tl.dot(a_sm, v_f32)

#         # Feature maps with bf16 round-trip
#         u_q = tl.dot(q, W)
#         u_k = tl.dot(k, W)

#         m = tl.max(u_q, axis=1)
#         e = tl.exp(u_q - m[:, None])
#         phi_q_pos = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(native_dtype).to(tl.float32)
#         m = tl.max(-u_q, axis=1)
#         e = tl.exp(-u_q - m[:, None])
#         phi_q_neg = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(native_dtype).to(tl.float32)

#         m = tl.max(u_k, axis=1)
#         e = tl.exp(u_k - m[:, None])
#         phi_k_pos = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(native_dtype).to(tl.float32)
#         m = tl.max(-u_k, axis=1)
#         e = tl.exp(-u_k - m[:, None])
#         phi_k_neg = (e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)).to(native_dtype).to(tl.float32)

#         phi_q_pos = tl.where(smask[:, None], phi_q_pos, 0.0)
#         phi_q_neg = tl.where(smask[:, None], phi_q_neg, 0.0)
#         phi_k_pos = tl.where(smask[:, None], phi_k_pos, 0.0)
#         phi_k_neg = tl.where(smask[:, None], phi_k_neg, 0.0)

#         # Linear branch: S from local buffer, Z from registers
#         S_pos = tl.load(S_base + offs_f[:, None] * stride_S_f + offs_d[None, :] * stride_S_d).to(tl.float32)
#         S_neg = tl.load(S_base + (offs_f + F)[:, None] * stride_S_f + offs_d[None, :] * stride_S_d).to(tl.float32)

#         lin_num = tl.dot(phi_q_pos, S_pos) + tl.dot(phi_q_neg, S_neg)
#         lin_den = tl.sum(phi_q_pos * Z_pos[None, :], axis=1) + tl.sum(phi_q_neg * Z_neg[None, :], axis=1)
#         lin_den = tl.maximum(lin_den, eps)

#         # Blend and store
#         den = tl.maximum(w * sm_den[:, None] + lin_den[:, None], eps)
#         out_tile = (w * sm_num + lin_num) / den
#         out_tile = tl.where(smask[:, None], out_tile, 0.0)

#         out_base = (token_start + offs_s[:, None]) * stride_out_t + h * stride_out_h
#         tl.store(out_ptr + out_base + offs_d[None, :] * stride_out_d,
#                  out_tile.to(native_dtype), mask=smask[:, None])

#         # State update AFTER output (strictly causal)
#         # When block is inactive (smask all False), phi_k is all zeros,
#         # so delta_S = 0 and delta_Z = 0 — state unchanged. Safe to write.
#         tl.store(S_base + offs_f[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
#                  S_pos + tl.dot(tl.trans(phi_k_pos), v_f32))
#         tl.store(S_base + (offs_f + F)[:, None] * stride_S_f + offs_d[None, :] * stride_S_d,
#                  S_neg + tl.dot(tl.trans(phi_k_neg), v_f32))
#         Z_pos += tl.sum(phi_k_pos, axis=0)
#         Z_neg += tl.sum(phi_k_neg, axis=0)

#     # Write Z back once
#     tl.store(Z_base + offs_f * stride_Z_f, Z_pos)
#     tl.store(Z_base + (offs_f + F) * stride_Z_f, Z_neg)


# class BlockSoftmaxLinearHybrid(nn.Module):
#     def __init__(self, config, num_heads: int, num_kv_heads: int, eps: float = 1e-6):
#         super().__init__()
#         head_dim = int(getattr(config, "head_dim", None)
#                        or config.hidden_size // config.num_attention_heads)
#         feature_dim = int(getattr(config, "feature_dim"))
#         block_size = int(getattr(config, "block_size", 32))
#         self.num_heads = num_heads
#         self.num_kv_heads = num_kv_heads
#         self.head_dim = head_dim
#         self.feature_dim = feature_dim
#         self.block_size = block_size
#         self.eps = eps
#         w = torch.eye(head_dim, feature_dim).unsqueeze(0).expand(num_heads, -1, -1)
#         self.hedgehog_weights = nn.Parameter(w.clone().contiguous())
#         self.alpha = nn.Parameter(torch.zeros(num_heads))

#     @torch.no_grad()
#     def reset_states(self, state: HybridStateView, slots=None):
#         if slots is None:
#             state.S_state.zero_()
#             state.Z_state.zero_()
#         else:
#             state.S_state[slots] = 0
#             state.Z_state[slots] = 0

#     def forward(self, q, k, v, seq_lens, req_indices, read_state, write_state=None):
#         if write_state is None:
#             write_state = read_state
#         H, D, S = self.num_heads, self.head_dim, self.block_size

#         if k.shape[1] != H:
#             k = _repeat_kv(k, H, self.num_kv_heads, D)
#         if v.shape[1] != H:
#             v = _repeat_kv(v, H, self.num_kv_heads, D)
#         q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

#         plan = build_request_block_plan(seq_lens, req_indices, S, q.device)
#         out = torch.zeros_like(q)
#         if plan.max_blocks == 0:
#             return out

#         R = plan.req_indices.numel()
#         S_local = read_state.S_state[plan.req_indices].clone().contiguous()
#         Z_local = read_state.Z_state[plan.req_indices].clone().contiguous()

#         _hybrid_persistent_kernel[(R, H)](
#             q, k, v, out,
#             plan.start_offsets, plan.seq_lens, plan.num_blocks,
#             S_local, Z_local,
#             self.hedgehog_weights.to(q.dtype).contiguous(), self.alpha,
#             q.stride(0), q.stride(1), q.stride(2),
#             k.stride(0), k.stride(1), k.stride(2),
#             v.stride(0), v.stride(1), v.stride(2),
#             out.stride(0), out.stride(1), out.stride(2),
#             S_local.stride(0), S_local.stride(1), S_local.stride(2), S_local.stride(3),
#             Z_local.stride(0), Z_local.stride(1), Z_local.stride(2),
#             self.hedgehog_weights.stride(0), self.hedgehog_weights.stride(1), self.hedgehog_weights.stride(2),
#             eps=self.eps, scale=D ** -0.5,
#             BLOCK_S=S, D=D, F=self.feature_dim, MAX_BLOCKS=plan.max_blocks,
#             num_warps=4, num_stages=1,
#         )

#         write_state.S_state[plan.req_indices] = S_local
#         write_state.Z_state[plan.req_indices] = Z_local
#         return out

#     def forward_qkv(self, q, k, v, layer_cache, metadata):
#         T = q.shape[0]
#         H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
#         committed, working = make_state_view_from_layer_cache(layer_cache)
#         out = self.forward(
#             q.reshape(T, H, D), k.reshape(T, Hkv, D), v.reshape(T, Hkv, D),
#             seq_lens=metadata.seq_lens, req_indices=metadata.state_slot_indices,
#             read_state=committed, write_state=working,
#         )
#         return out.reshape(T, H * D)


# def make_state_view_from_layer_cache(layer_cache):
#     S_full = layer_cache.temporal
#     if S_full.ndim != 5 or S_full.shape[1] != 2:
#         raise ValueError(f"Expected temporal [slots,2,H,2F,D], got {tuple(S_full.shape)}")
#     S_committed, S_working = S_full[:, 0], S_full[:, 1]
#     if len(layer_cache.conv) < 2:
#         raise ValueError("Need conv[0]=committed Z, conv[1]=working Z")
#     z0, z1 = layer_cache.conv[0], layer_cache.conv[1]
#     if z0.ndim == 3 and z0.shape[-1] == 1:
#         z0, z1 = z0[..., 0], z1[..., 0]
#     H, twoF = S_committed.shape[1], S_committed.shape[2]
#     return (
#         HybridStateView(S_committed, z0.view(-1, H, twoF)),
#         HybridStateView(S_working, z1.view(-1, H, twoF)),
#     )


# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0

"""
Pure full-linear Hedgehog attention kernel for SGLang serving.

This is intended to match the current PyTorch full-linear reference semantics:

For each request/head/block:

    phi_q = [softmax(q @ W), softmax(-(q @ W))]
    phi_k = [softmax(k @ W), softmax(-(k @ W))]

    S_cur = phi_k^T @ v
    Z_cur = sum(phi_k)

    S_query = S_old + S_cur
    Z_query = Z_old + Z_cur

    out = (phi_q @ S_query) / (phi_q @ Z_query)

    S_old <- S_query
    Z_old <- Z_query

Important:
- No intra-block softmax branch.
- No alpha / blend gate.
- Current block is included in the query-time state exactly once.
- Persistent local state is advanced exactly once after computing the block output.
- Reads from committed state, writes final chained state to working state.
"""

# from __future__ import annotations

# import torch
# import torch.nn as nn
# import triton
# import triton.language as tl

# from dataclasses import dataclass
# from typing import Optional


# # =============================================================================
# # Block planning
# # =============================================================================


# @dataclass
# class RequestBlockPlan:
#     req_indices: torch.Tensor       # [R] pool slot indices
#     seq_lens: torch.Tensor          # [R]
#     start_offsets: torch.Tensor     # [R]
#     num_blocks: torch.Tensor        # [R]
#     max_blocks: int
#     block_size: int


# def build_request_block_plan(
#     seq_lens,
#     req_indices,
#     block_size: int,
#     device: torch.device,
# ) -> RequestBlockPlan:
#     seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
#     req_indices_t = req_indices.to(device=device, dtype=torch.long)

#     if seq_lens_t.numel() == 0:
#         empty = torch.empty((0,), dtype=torch.long, device=device)
#         return RequestBlockPlan(
#             req_indices=empty,
#             seq_lens=empty,
#             start_offsets=empty,
#             num_blocks=empty,
#             max_blocks=0,
#             block_size=block_size,
#         )

#     start_offsets = torch.zeros_like(seq_lens_t)
#     if seq_lens_t.numel() > 1:
#         start_offsets[1:] = torch.cumsum(seq_lens_t[:-1], dim=0)

#     num_blocks = (seq_lens_t + block_size - 1) // block_size

#     return RequestBlockPlan(
#         req_indices=req_indices_t,
#         seq_lens=seq_lens_t,
#         start_offsets=start_offsets,
#         num_blocks=num_blocks,
#         max_blocks=int(num_blocks.max().item()),
#         block_size=block_size,
#     )


# # =============================================================================
# # State view
# # =============================================================================


# @dataclass
# class HybridStateView:
#     # S_state: [slots, H, 2F, D], usually fp32
#     S_state: torch.Tensor

#     # Z_state: [slots, H, 2F], usually fp32
#     Z_state: torch.Tensor


# # =============================================================================
# # GQA helper
# # =============================================================================


# def _repeat_kv(
#     x: torch.Tensor,
#     num_q_heads: int,
#     num_kv_heads: int,
#     head_dim: int,
# ) -> torch.Tensor:
#     if num_q_heads == num_kv_heads:
#         return x

#     assert num_q_heads % num_kv_heads == 0, (
#         f"num_q_heads={num_q_heads} must be divisible by "
#         f"num_kv_heads={num_kv_heads}"
#     )

#     groups = num_q_heads // num_kv_heads
#     t = x.shape[0]

#     return (
#         x.view(t, num_kv_heads, head_dim)[:, :, None, :]
#         .expand(t, num_kv_heads, groups, head_dim)
#         .reshape(t, num_q_heads, head_dim)
#     )


# # =============================================================================
# # Triton persistent full-linear kernel
# # =============================================================================


# @triton.jit
# def _linear_persistent_kernel(
#     q_ptr,
#     k_ptr,
#     v_ptr,
#     out_ptr,
#     start_offsets_ptr,
#     seq_lens_ptr,
#     num_blocks_ptr,
#     S_ptr,
#     Z_ptr,
#     W_ptr,
#     stride_q_t: tl.constexpr,
#     stride_q_h: tl.constexpr,
#     stride_q_d: tl.constexpr,
#     stride_k_t: tl.constexpr,
#     stride_k_h: tl.constexpr,
#     stride_k_d: tl.constexpr,
#     stride_v_t: tl.constexpr,
#     stride_v_h: tl.constexpr,
#     stride_v_d: tl.constexpr,
#     stride_out_t: tl.constexpr,
#     stride_out_h: tl.constexpr,
#     stride_out_d: tl.constexpr,
#     stride_S_r: tl.constexpr,
#     stride_S_h: tl.constexpr,
#     stride_S_f: tl.constexpr,
#     stride_S_d: tl.constexpr,
#     stride_Z_r: tl.constexpr,
#     stride_Z_h: tl.constexpr,
#     stride_Z_f: tl.constexpr,
#     stride_W_h: tl.constexpr,
#     stride_W_d: tl.constexpr,
#     stride_W_f: tl.constexpr,
#     eps: tl.constexpr,
#     BLOCK_S: tl.constexpr,
#     D: tl.constexpr,
#     F: tl.constexpr,
#     MAX_BLOCKS: tl.constexpr,
# ):
#     """
#     Grid: (R, H)

#     One Triton program handles one (request, head) pair and loops over all
#     blocks sequentially, preserving the recurrent state order exactly.
#     """

#     pid_r = tl.program_id(0)
#     pid_h = tl.program_id(1)

#     seq_len = tl.load(seq_lens_ptr + pid_r)
#     start_off = tl.load(start_offsets_ptr + pid_r)
#     n_blocks = tl.load(num_blocks_ptr + pid_r)

#     offs_s = tl.arange(0, BLOCK_S)
#     offs_d = tl.arange(0, D)
#     offs_f = tl.arange(0, F)

#     # Infer native activation dtype from q.
#     native_dtype = tl.load(q_ptr).dtype

#     # W for this head: [D, F].
#     W = tl.load(
#         W_ptr
#         + pid_h * stride_W_h
#         + offs_d[:, None] * stride_W_d
#         + offs_f[None, :] * stride_W_f
#     )

#     S_base = S_ptr + pid_r * stride_S_r + pid_h * stride_S_h
#     Z_base = Z_ptr + pid_r * stride_Z_r + pid_h * stride_Z_h

#     # Keep Z in registers across blocks.
#     Z_pos = tl.load(Z_base + offs_f * stride_Z_f).to(tl.float32)
#     Z_neg = tl.load(Z_base + (offs_f + F) * stride_Z_f).to(tl.float32)

#     for block_idx in range(MAX_BLOCKS):
#         active = block_idx < n_blocks

#         token_start = start_off + block_idx * BLOCK_S
#         valid_len = tl.where(
#             active,
#             tl.minimum(seq_len - block_idx * BLOCK_S, BLOCK_S),
#             0,
#         )
#         smask = offs_s < valid_len

#         # ---------------------------------------------------------------------
#         # Load q/k/v block: [BLOCK_S, D]
#         # ---------------------------------------------------------------------

#         q_base = (token_start + offs_s[:, None]) * stride_q_t + pid_h * stride_q_h
#         k_base = (token_start + offs_s[:, None]) * stride_k_t + pid_h * stride_k_h
#         v_base = (token_start + offs_s[:, None]) * stride_v_t + pid_h * stride_v_h

#         q = tl.load(
#             q_ptr + q_base + offs_d[None, :] * stride_q_d,
#             mask=smask[:, None],
#             other=0.0,
#         )

#         k = tl.load(
#             k_ptr + k_base + offs_d[None, :] * stride_k_d,
#             mask=smask[:, None],
#             other=0.0,
#         )

#         v = tl.load(
#             v_ptr + v_base + offs_d[None, :] * stride_v_d,
#             mask=smask[:, None],
#             other=0.0,
#         )
#         v_f32 = v.to(tl.float32)

#         # ---------------------------------------------------------------------
#         # Hedgehog feature map:
#         #
#         # PyTorch reference:
#         #   u = einsum(...).to(x.dtype)
#         #   u_f32 = u.float()
#         #   phi = cat([softmax(u_f32), softmax(-u_f32)]).to(x.dtype)
#         #
#         # Here:
#         #   u_q/u_k are dot products with W already in activation dtype.
#         #   softmax is done in fp32.
#         #   phi is cast to native dtype, then back to fp32 for matmuls,
#         #   matching the PyTorch "return phi.to(dtype=x.dtype)" behavior.
#         # ---------------------------------------------------------------------

#         u_q = tl.dot(q, W)
#         u_k = tl.dot(k, W)

#         # phi_q_pos = softmax(u_q)
#         m = tl.max(u_q, axis=1)
#         e = tl.exp(u_q - m[:, None])
#         phi_q_pos = e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)
#         phi_q_pos = phi_q_pos.to(native_dtype).to(tl.float32)

#         # phi_q_neg = softmax(-u_q)
#         m = tl.max(-u_q, axis=1)
#         e = tl.exp(-u_q - m[:, None])
#         phi_q_neg = e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)
#         phi_q_neg = phi_q_neg.to(native_dtype).to(tl.float32)

#         # phi_k_pos = softmax(u_k)
#         m = tl.max(u_k, axis=1)
#         e = tl.exp(u_k - m[:, None])
#         phi_k_pos = e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)
#         phi_k_pos = phi_k_pos.to(native_dtype).to(tl.float32)

#         # phi_k_neg = softmax(-u_k)
#         m = tl.max(-u_k, axis=1)
#         e = tl.exp(-u_k - m[:, None])
#         phi_k_neg = e / tl.maximum(tl.sum(e, axis=1)[:, None], eps)
#         phi_k_neg = phi_k_neg.to(native_dtype).to(tl.float32)

#         # Zero invalid/padded rows.
#         phi_q_pos = tl.where(smask[:, None], phi_q_pos, 0.0)
#         phi_q_neg = tl.where(smask[:, None], phi_q_neg, 0.0)
#         phi_k_pos = tl.where(smask[:, None], phi_k_pos, 0.0)
#         phi_k_neg = tl.where(smask[:, None], phi_k_neg, 0.0)

#         # ---------------------------------------------------------------------
#         # Load old recurrent state.
#         # ---------------------------------------------------------------------

#         S_pos_old = tl.load(
#             S_base
#             + offs_f[:, None] * stride_S_f
#             + offs_d[None, :] * stride_S_d
#         ).to(tl.float32)

#         S_neg_old = tl.load(
#             S_base
#             + (offs_f + F)[:, None] * stride_S_f
#             + offs_d[None, :] * stride_S_d
#         ).to(tl.float32)

#         # ---------------------------------------------------------------------
#         # Current block state.
#         # ---------------------------------------------------------------------

#         S_pos_cur = tl.dot(tl.trans(phi_k_pos), v_f32)
#         S_neg_cur = tl.dot(tl.trans(phi_k_neg), v_f32)

#         Z_pos_cur = tl.sum(phi_k_pos, axis=0)
#         Z_neg_cur = tl.sum(phi_k_neg, axis=0)

#         # Query-time state includes previous state + current block.
#         # This is the key full-linear behavior.
#         S_pos_query = S_pos_old + S_pos_cur
#         S_neg_query = S_neg_old + S_neg_cur

#         Z_pos_query = Z_pos + Z_pos_cur
#         Z_neg_query = Z_neg + Z_neg_cur

#         # ---------------------------------------------------------------------
#         # Pure linear output.
#         # ---------------------------------------------------------------------

#         lin_num = (
#             tl.dot(phi_q_pos, S_pos_query)
#             + tl.dot(phi_q_neg, S_neg_query)
#         )

#         lin_den = (
#             tl.sum(phi_q_pos * Z_pos_query[None, :], axis=1)
#             + tl.sum(phi_q_neg * Z_neg_query[None, :], axis=1)
#         )
#         lin_den = tl.maximum(lin_den, eps)

#         out_tile = lin_num / lin_den[:, None]
#         out_tile = tl.where(smask[:, None], out_tile, 0.0)

#         # ---------------------------------------------------------------------
#         # Store output.
#         # ---------------------------------------------------------------------

#         out_base = (token_start + offs_s[:, None]) * stride_out_t + pid_h * stride_out_h

#         tl.store(
#             out_ptr + out_base + offs_d[None, :] * stride_out_d,
#             out_tile.to(native_dtype),
#             mask=smask[:, None],
#         )

#         # ---------------------------------------------------------------------
#         # Advance local recurrent state exactly once.
#         # ---------------------------------------------------------------------

#         tl.store(
#             S_base
#             + offs_f[:, None] * stride_S_f
#             + offs_d[None, :] * stride_S_d,
#             S_pos_query,
#         )

#         tl.store(
#             S_base
#             + (offs_f + F)[:, None] * stride_S_f
#             + offs_d[None, :] * stride_S_d,
#             S_neg_query,
#         )

#         Z_pos = Z_pos_query
#         Z_neg = Z_neg_query

#     # Write Z registers back once after all blocks.
#     tl.store(Z_base + offs_f * stride_Z_f, Z_pos)
#     tl.store(Z_base + (offs_f + F) * stride_Z_f, Z_neg)


# # =============================================================================
# # Main module
# # =============================================================================


# class BlockSoftmaxLinearHybrid(nn.Module):
#     """
#     Name kept for drop-in compatibility, but this module now implements
#     pure full-linear Hedgehog attention, not softmax+linear hybrid attention.
#     """

#     def __init__(
#         self,
#         config,
#         num_heads: int,
#         num_kv_heads: int,
#         eps: float = 1e-6,
#     ):
#         super().__init__()

#         head_dim = int(
#             getattr(config, "head_dim", None)
#             or config.hidden_size // config.num_attention_heads
#         )
#         feature_dim = int(getattr(config, "feature_dim"))
#         block_size = int(getattr(config, "block_size", 32))

#         if feature_dim <= 0:
#             raise ValueError(f"feature_dim must be positive, got {feature_dim}")

#         if head_dim <= 0:
#             raise ValueError(f"head_dim must be positive, got {head_dim}")

#         # This kernel assumes one F tile exactly: offs_f = arange(0, F).
#         # Very large F may compile poorly. For current Hedgehog F=64, this is fine.
#         self.num_heads = int(num_heads)
#         self.num_kv_heads = int(num_kv_heads)
#         self.head_dim = int(head_dim)
#         self.feature_dim = int(feature_dim)
#         self.block_size = int(block_size)
#         self.eps = float(eps)

#         weight = torch.eye(head_dim, feature_dim).unsqueeze(0).expand(
#             self.num_heads, -1, -1
#         )

#         self.hedgehog_weights = nn.Parameter(weight.clone().contiguous())

#         # Optional compatibility parameter:
#         # Keep this only if older checkpoints contain "alpha".
#         # It is intentionally unused by the full-linear kernel.
#         self.alpha = nn.Parameter(torch.zeros(self.num_heads), requires_grad=False)

#     @torch.no_grad()
#     def reset_states(
#         self,
#         state: HybridStateView,
#         slots: Optional[torch.Tensor] = None,
#     ) -> None:
#         if slots is None:
#             state.S_state.zero_()
#             state.Z_state.zero_()
#         else:
#             state.S_state[slots] = 0
#             state.Z_state[slots] = 0

#     def forward(
#         self,
#         q: torch.Tensor,                  # [T, H, D]
#         k: torch.Tensor,                  # [T, H or Hkv, D]
#         v: torch.Tensor,                  # [T, H or Hkv, D]
#         seq_lens: list[int],
#         req_indices: torch.Tensor,        # [R]
#         read_state: HybridStateView,
#         write_state: Optional[HybridStateView] = None,
#     ) -> torch.Tensor:
#         if write_state is None:
#             write_state = read_state

#         H = self.num_heads
#         Hkv = self.num_kv_heads
#         D = self.head_dim
#         S = self.block_size
#         F = self.feature_dim

#         if q.ndim != 3:
#             raise ValueError(f"q must have shape [T,H,D], got {tuple(q.shape)}")
#         if k.ndim != 3:
#             raise ValueError(f"k must have shape [T,Hkv,D], got {tuple(k.shape)}")
#         if v.ndim != 3:
#             raise ValueError(f"v must have shape [T,Hkv,D], got {tuple(v.shape)}")

#         if q.shape[1] != H or q.shape[2] != D:
#             raise ValueError(
#                 f"q shape mismatch: expected [T,{H},{D}], got {tuple(q.shape)}"
#             )

#         if k.shape[2] != D or v.shape[2] != D:
#             raise ValueError(
#                 f"k/v head_dim mismatch: expected D={D}, "
#                 f"got k={tuple(k.shape)}, v={tuple(v.shape)}"
#             )

#         # GQA expansion.
#         if k.shape[1] != H:
#             k = _repeat_kv(k, H, Hkv, D)
#         if v.shape[1] != H:
#             v = _repeat_kv(v, H, Hkv, D)

#         q = q.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()

#         plan = build_request_block_plan(
#             seq_lens=seq_lens,
#             req_indices=req_indices,
#             block_size=S,
#             device=q.device,
#         )

#         out = torch.zeros_like(q)

#         if plan.max_blocks == 0:
#             return out

#         R = plan.req_indices.numel()

#         # Read committed state into local contiguous working tensors.
#         # The Triton kernel mutates these local tensors block-by-block.
#         S_local = read_state.S_state[plan.req_indices].clone().contiguous()
#         Z_local = read_state.Z_state[plan.req_indices].clone().contiguous()

#         expected_S_shape_tail = (H, 2 * F, D)
#         expected_Z_shape_tail = (H, 2 * F)

#         if tuple(S_local.shape[1:]) != expected_S_shape_tail:
#             raise ValueError(
#                 f"S_state shape mismatch: expected [R,{H},{2 * F},{D}], "
#                 f"got {tuple(S_local.shape)}"
#             )

#         if tuple(Z_local.shape[1:]) != expected_Z_shape_tail:
#             raise ValueError(
#                 f"Z_state shape mismatch: expected [R,{H},{2 * F}], "
#                 f"got {tuple(Z_local.shape)}"
#             )

#         W = self.hedgehog_weights.to(q.dtype).contiguous()

#         _linear_persistent_kernel[(R, H)](
#             q,
#             k,
#             v,
#             out,
#             plan.start_offsets,
#             plan.seq_lens,
#             plan.num_blocks,
#             S_local,
#             Z_local,
#             W,
#             q.stride(0),
#             q.stride(1),
#             q.stride(2),
#             k.stride(0),
#             k.stride(1),
#             k.stride(2),
#             v.stride(0),
#             v.stride(1),
#             v.stride(2),
#             out.stride(0),
#             out.stride(1),
#             out.stride(2),
#             S_local.stride(0),
#             S_local.stride(1),
#             S_local.stride(2),
#             S_local.stride(3),
#             Z_local.stride(0),
#             Z_local.stride(1),
#             Z_local.stride(2),
#             W.stride(0),
#             W.stride(1),
#             W.stride(2),
#             eps=self.eps,
#             BLOCK_S=S,
#             D=D,
#             F=F,
#             MAX_BLOCKS=plan.max_blocks,
#             num_warps=4,
#             num_stages=1,
#         )

#         # Write final chained state to working buffer once.
#         # Scheduler/backend can promote working -> committed between blocks.
#         write_state.S_state[plan.req_indices] = S_local.to(write_state.S_state.dtype)
#         write_state.Z_state[plan.req_indices] = Z_local.to(write_state.Z_state.dtype)

#         return out

#     def forward_qkv(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         layer_cache,
#         metadata,
#     ) -> torch.Tensor:
#         T = q.shape[0]
#         H = self.num_heads
#         Hkv = self.num_kv_heads
#         D = self.head_dim

#         committed, working = make_state_view_from_layer_cache(layer_cache)

#         q = q.contiguous().reshape(T, H, D)
#         k = k.contiguous().reshape(T, Hkv, D)
#         v = v.contiguous().reshape(T, Hkv, D)

#         out = self.forward(
#             q=q,
#             k=k,
#             v=v,
#             seq_lens=metadata.seq_lens,
#             req_indices=metadata.state_slot_indices,
#             read_state=committed,
#             write_state=working,
#         )

#         return out.reshape(T, H * D)


# # =============================================================================
# # SGLang MambaPool state adapter
# # =============================================================================


# def make_state_view_from_layer_cache(layer_cache):
#     """
#     Expected layout:

#         layer_cache.temporal: [slots, 2, H, 2F, D]
#             temporal[:, 0] = committed S
#             temporal[:, 1] = working S

#         layer_cache.conv[0]: [slots, H*2F, 1] or [slots, H*2F]
#             committed Z

#         layer_cache.conv[1]: [slots, H*2F, 1] or [slots, H*2F]
#             working Z
#     """

#     S_full = layer_cache.temporal

#     if S_full.ndim != 5 or S_full.shape[1] != 2:
#         raise ValueError(
#             f"Expected layer_cache.temporal shape [slots,2,H,2F,D], "
#             f"got {tuple(S_full.shape)}"
#         )

#     S_committed = S_full[:, 0]
#     S_working = S_full[:, 1]

#     if len(layer_cache.conv) < 2:
#         raise ValueError("Expected conv[0]=committed Z and conv[1]=working Z")

#     z0 = layer_cache.conv[0]
#     z1 = layer_cache.conv[1]

#     if z0.ndim == 3 and z0.shape[-1] == 1:
#         z0 = z0[..., 0]
#     if z1.ndim == 3 and z1.shape[-1] == 1:
#         z1 = z1[..., 0]

#     H = S_committed.shape[1]
#     twoF = S_committed.shape[2]

#     Z_committed = z0.reshape(-1, H, twoF)
#     Z_working = z1.reshape(-1, H, twoF)

#     committed = HybridStateView(
#         S_state=S_committed,
#         Z_state=Z_committed,
#     )

#     working = HybridStateView(
#         S_state=S_working,
#         Z_state=Z_working,
#     )

#     return committed, working