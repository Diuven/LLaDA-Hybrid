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
"""
SGLang attention backend for LLaDA Fast hybrid (softmax + linear) layers.

Recurrent state (S_state, Z_state) is stored in SGLang's MambaPool via
HybridReqToTokenPool.  State slot indices are obtained from
get_mamba_indices() and the per-layer cache is accessed through
mamba2_layer_cache(layer_id).  No separate hedgehog pool is used — the
MambaPool layout is reused with the following mapping:

  layer_cache.temporal : [slots+1, 2, H/tp, 2F, D]  ← S_state (dim 1: committed=0, working=1)
  layer_cache.conv[0]  : [slots+1, H/tp*2F, 1]     ← Z_committed
  layer_cache.conv[1]  : [slots+1, H/tp*2F, 1]     ← Z_working
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.llada_fast_hybrid_kernel import RequestBlockPlan
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_LLADA_FAST_BLOCK_SIZE = 32


def _qkv_as_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    T: int,
    H: int,
    Hkv: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [T,H,D] / [T,Hkv,D] without an extra view when already shaped."""
    if q.dim() == 3 and q.shape[1] == H and q.shape[2] == D:
        q_r = q
    else:
        q_r = q.view(T, H, D)
    if k.dim() == 3 and k.shape[1] == Hkv and k.shape[2] == D:
        k_r = k
    else:
        k_r = k.view(T, Hkv, D)
    if v.dim() == 3 and v.shape[1] == Hkv and v.shape[2] == D:
        v_r = v
    else:
        v_r = v.view(T, Hkv, D)
    return q_r, k_r, v_r


@dataclass
class LLaDAFastMetadata:
    """Lightweight metadata computed once per forward pass."""

    # req_pool_indices → MambaPool slot indices (via HybridReqToTokenPool.get_mamba_indices)
    state_slot_indices: torch.Tensor  # [batch_size] int64

    # Pre-built block plan reused across all hybrid layers in this step.
    plan: RequestBlockPlan

    # If False, TritonHybridBlock.run() skips the pool write-back so that
    # each denoising pass re-computes from the same committed checkpoint.
    commit: bool = True


class LLaDAFastAttnBackend(AttentionBackend):
    """
    Linear-attention backend slot for LLaDA Fast inside HybridLinearAttnBackend.

    Lifecycle (per forward pass)
    ----------------------------
    HybridLinearAttnBackend.init_forward_metadata()
        → self.init_forward_metadata(forward_batch)   [this class]

    LLaDAFastHybridAttention.forward()
        → forward_batch.attn_backend.linear_attn_backend.forward_qkv(
              mixer, q, k, v, layer_id)               [this class]
    """

    def __init__(self, model_runner):
        super().__init__()
        self.device = model_runner.device
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.metadata: Optional[LLaDAFastMetadata] = None

        max_bs = self.req_to_token_pool.size
        self.start_offsets_buf = torch.zeros(
            max_bs + 1, dtype=torch.long, device=self.device
        )
        self.num_blocks_buf = torch.zeros(
            max_bs, dtype=torch.long, device=self.device
        )

        self._max_bs = max_bs
        self._out_buf: Optional[torch.Tensor] = None
        self._out_buf_flat: Optional[torch.Tensor] = None

        self._layer_state_cache: dict = {}
        # Indexed by global layer_id for O(1) lookup (hybrid layers only).
        self._layer_states_by_id: list | None = None

        self._H: int = 0
        self._Hkv: int = 0
        self._z_heads: int = 0  # Z pool rows: H when per-q-head state, Hkv when share_hedgehog_kv_groups
        self._D: int = 0
        self._HD: int = 0
        self._twoF: int = 0

        # CUDA graph persistent metadata buffers (allocated in
        # init_cuda_graph_state, refreshed in-place by init_forward_metadata_*).
        self._cg_state_slots: Optional[list] = None
        self._cg_start_offsets: Optional[torch.Tensor] = None
        self._cg_num_blocks: Optional[torch.Tensor] = None
        self._cg_seq_lens: Optional[torch.Tensor] = None
        self._cg_pad_slot: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Called by HybridLinearAttnBackend before the layer loop
    # ------------------------------------------------------------------ #

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        state_slot_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )

        # IMPORTANT: use extend_seq_lens (tokens actually present in q/k/v),
        # NOT seq_lens (prefix + current block total).  During DLLM_EXTEND,
        # seq_lens = prefix + block but q/k/v only hold the current block
        # (extend_seq_lens = block_size).  For EXTEND with no prefix,
        # extend_seq_lens == seq_lens, so this is always safe.
        seq_lens = forward_batch.extend_seq_lens
        bs = seq_lens.shape[0]

        start_offsets = forward_batch.extend_start_loc
        if start_offsets is None:
            if bs > 1:
                torch.cumsum(seq_lens[:-1], dim=0, out=self.start_offsets_buf[1:bs])
            start_offsets = self.start_offsets_buf[:bs]

        block_size = _LLADA_FAST_BLOCK_SIZE
        num_blocks = self.num_blocks_buf[:bs]
        seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
            max_blocks = (
                max((int(x) + block_size - 1) // block_size for x in seq_lens_cpu)
                if bs > 0
                else 0
            )
            if max_blocks <= 1:
                num_blocks.fill_(1)
            else:
                torch.div(
                    seq_lens + (block_size - 1),
                    block_size,
                    rounding_mode="floor",
                    out=num_blocks,
                )
        else:
            torch.div(
                seq_lens + (block_size - 1),
                block_size,
                rounding_mode="floor",
                out=num_blocks,
            )
            max_blocks = int(num_blocks.max().item()) if bs > 0 else 0

        plan = RequestBlockPlan(
            req_indices=state_slot_indices,
            seq_lens=seq_lens,
            start_offsets=start_offsets,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            block_size=block_size,
        )

        self.metadata = LLaDAFastMetadata(
            state_slot_indices=state_slot_indices,
            plan=plan,
            commit=False,  # never promote during forward; scheduler promotes between blocks
        )

        self._ensure_layer_state_cache()

    def _ensure_layer_state_cache(self) -> None:
        """Build stable views over the per-layer MambaPool state, once.

        Used by both the eager path and CUDA graph capture/replay.  CUDA graph
        capture runs before any eager ``init_forward_metadata``, so the capture
        path must call this itself or ``forward_qkv`` sees an empty cache.
        """
        if self._layer_state_cache:
            return
        for layer_id in self.req_to_token_pool.mamba_map:
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            S_full = layer_cache.temporal
            z0, z1 = layer_cache.conv[0], layer_cache.conv[1]
            if z0.ndim == 3 and z0.shape[-1] == 1:
                z0 = z0[..., 0]
                z1 = z1[..., 0]
            # These are stable views over the state pool; the tensor
            # contents change per request, but the storage/layout does not.
            state_heads = S_full.shape[2]
            twoF = S_full.shape[3]
            self._layer_state_cache[layer_id] = (
                S_full[:, 0],
                S_full[:, 1],
                z0.view(-1, state_heads, twoF),
                z1.view(-1, state_heads, twoF),
            )
        max_lid = max(self._layer_state_cache.keys())
        self._layer_states_by_id = [None] * (max_lid + 1)
        for lid, tup in self._layer_state_cache.items():
            self._layer_states_by_id[lid] = tup

    # ------------------------------------------------------------------ #
    # CUDA graph support
    #
    # The captured graph records tensor *addresses*; replay only rewrites
    # their *contents* in place.  For dLLM block decode every request has
    # extend_seq_lens == block_size, so num_blocks is always 1 and the
    # block geometry (start_offsets / num_blocks / seq_lens) is constant —
    # only the per-request state-pool slot indices change between replays.
    # ------------------------------------------------------------------ #

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Pre-allocate persistent metadata buffers for graph capture/replay."""
        bsz = _LLADA_FAST_BLOCK_SIZE
        dev = self.device
        # Per-bs slot buffers — refreshed in place each replay.  int32 to
        # match HybridReqToTokenPool.get_mamba_indices() so the captured
        # Triton kernel variant is reused (a dtype change would recompile).
        self._cg_state_slots = [
            torch.zeros(i + 1, dtype=torch.int32, device=dev)
            for i in range(max_bs)
        ]
        # Constant block geometry — written once, shared read-only by every
        # captured batch size.  extend_seq_lens == block_size ⇒ num_blocks 1.
        self._cg_start_offsets = torch.arange(
            0, (max_bs + 1) * bsz, bsz, dtype=torch.long, device=dev
        )
        self._cg_num_blocks = torch.ones(max_bs, dtype=torch.long, device=dev)
        self._cg_seq_lens = torch.full(
            (max_bs,), bsz, dtype=torch.long, device=dev
        )

    def _pad_state_slot(self) -> int:
        """MambaPool scratch row index.

        ``layer_cache.temporal`` is shaped ``[slots + 1, ...]``; the trailing
        row is a scratch slot.  Padded (dummy) graph requests are pointed here
        so their kernel writes cannot corrupt a real request's state.
        """
        if self._cg_pad_slot is None:
            assert self._layer_states_by_id is not None, (
                "init_forward_metadata must run (warmup) before graph replay"
            )
            states = next(s for s in self._layer_states_by_id if s is not None)
            # states = (S_committed, S_working, Z_committed, Z_working);
            # S_committed == temporal[:, 0] → shape [slots + 1, ...].
            self._cg_pad_slot = int(states[0].shape[0]) - 1
        return self._cg_pad_slot

    def _build_cuda_graph_metadata(
        self, bs: int, req_pool_indices: torch.Tensor, num_pad: int = 0
    ) -> None:
        # Capture runs before any eager forward — make sure the per-layer
        # state-pool views exist so forward_qkv can index them.
        self._ensure_layer_state_cache()
        slots = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        slot_buf = self._cg_state_slots[bs - 1]
        slot_buf.copy_(slots)  # in-place — keeps the captured address stable
        if num_pad:
            # Dummy padding requests: divert their state writes to the pool
            # scratch row so they cannot race a real request's slot.
            slot_buf[bs - num_pad:].fill_(self._pad_state_slot())
        plan = RequestBlockPlan(
            req_indices=slot_buf,
            seq_lens=self._cg_seq_lens[:bs],
            start_offsets=self._cg_start_offsets[:bs],
            num_blocks=self._cg_num_blocks[:bs],
            max_blocks=1,  # dLLM block decode: one block per request
            block_size=_LLADA_FAST_BLOCK_SIZE,
        )
        self.metadata = LLaDAFastMetadata(
            state_slot_indices=slot_buf, plan=plan, commit=False
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
    ) -> None:
        self._build_cuda_graph_metadata(bs, req_pool_indices)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        num_pad = 0
        if seq_lens_cpu is not None:
            fill = self.get_cuda_graph_seq_len_fill_value()
            num_pad = int((seq_lens_cpu[:bs] == fill).sum())
        self._build_cuda_graph_metadata(bs, req_pool_indices, num_pad=num_pad)

    def get_cuda_graph_seq_len_fill_value(self):
        # Must agree with the full (triton) sub-backend, which returns 1.
        return 1

    # ------------------------------------------------------------------ #
    # Called by LLaDAFastHybridAttention.forward() for each hybrid layer
    # ------------------------------------------------------------------ #

    def forward_qkv(
        self,
        mixer: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        assert self.metadata is not None, "init_forward_metadata must be called first"

        T = q.shape[0]

        if self._H == 0:
            self._H = mixer.num_heads
            self._Hkv = mixer.num_kv_heads
            self._D = mixer.head_dim
            self._HD = self._H * self._D
            self._twoF = 2 * mixer.feature_dim
            # Pool layout matches LLaDAFastConfigAdapter.mamba2_cache_params
            self._z_heads = (
                self._Hkv
                if getattr(mixer, "share_hedgehog_kv_groups", False)
                else self._H
            )

        states = self._layer_states_by_id[layer_id]
        S_committed, S_working, Z_committed, Z_working = states

        q_r, k_r, v_r = _qkv_as_thd(q, k, v, T, self._H, self._Hkv, self._D)

        if self._out_buf is None or self._out_buf.dtype != q.dtype:
            max_T = self._max_bs * _LLADA_FAST_BLOCK_SIZE
            self._out_buf = torch.empty(
                max_T,
                self._H,
                self._D,
                device=self.device,
                dtype=q.dtype,
            )
            self._out_buf_flat = self._out_buf.view(max_T, self._HD)
        out_view = self._out_buf[:T]

        mixer._run_kernels(
            q_r,
            k_r,
            v_r,
            self.metadata.plan,
            S_committed,
            Z_committed,
            S_working,
            Z_working,
            out_view,
        )
        return self._out_buf_flat[:T]

    def forward(self, mixer, hidden_states, layer_id):
        raise NotImplementedError(
            "Use forward_qkv() — QKV must be split before the hybrid kernel."
        )

    def promote_hybrid_state(self, req_pool_indices: torch.Tensor) -> None:
        slot_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        for layer_id in self.req_to_token_pool.mamba_map:
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            layer_cache.temporal[slot_indices, 0] = layer_cache.temporal[slot_indices, 1]
            layer_cache.conv[0][slot_indices] = layer_cache.conv[1][slot_indices]

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError
