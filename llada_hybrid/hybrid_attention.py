"""
BlockSoftmaxLinearHybrid: complementary combination of two attention mechanisms.

Design
──────
For each block n (tokens at positions [n·S, (n+1)·S)):

  softmax stream:  bidirectional attention within block n only (exact, S×S).
  linear stream:   kernel linear attention over all previous blocks 0..n-1
                   via a recurrent O(D·F) state (approximate, cross-block).

  Shared-normalization mix (LoLCATs-style):

    out_n = (w · sm_num_n  +  lin_num_n) / (w · sm_den_n  +  lin_den_n)

  where
    sm_num_n = Σ_j exp(q·kⱼ/√d) · vⱼ   for j in block n  (unnorm. weighted sum)
    sm_den_n = Σ_j exp(q·kⱼ/√d)          for j in block n  (partition function)
    lin_num_n = φ(q) @ S_state             (accumulated from blocks 0..n-1)
    lin_den_n = φ(q) @ Z_state             (linear normalizer)
    w         = sigmoid(alpha)             (learnable scalar gate)

Why shared normalization instead of additive output blending
─────────────────────────────────────────────────────────────
Additive blending  out = w·softmax_out + (1−w)·linear_out  couples the
linear branch's gradient magnitude to (1−w).  With alpha=4.0 this is 0.018,
giving hedgehog_weights only 1/55th of the gradient they deserve.

In the shared-normalization form the linear branch always contributes with
factor 1 (independent of alpha).  Gradients to hedgehog_weights flow
unattenuated through lin_num and lin_den regardless of w.  Gradient to alpha
is balanced because both terms compete in the same denominator.

Numerically, when lin_den ≈ 0 (early training / first block) the output
degrades gracefully to pure intra-block softmax:
  (w·sm_num) / (w·sm_den) = sm_num / sm_den = standard softmax output.

State update order: linear output is computed BEFORE the state is updated
with block n, so the linear stream is strictly causal across blocks.

Stage-1 distillation
─────────────────────
  · alpha: init = 0.0 → sigmoid = 0.5 (balanced start, LoLCATs-style).
  · Only linear_attention.{hedgehog_weights, alpha} are trainable in stage-1
    hybrid mode; Q/K/V stay frozen so the softmax path is exact throughout.
  · First block: lin_den = eps, output ≈ intra-block softmax. Stable signal
    from step 1.

API — same as OrderInvariantKernelLinearAttention:
  forward(q, k, v, attention_mask=None, key_padding_mask=None) → (B, H, L, D)
  attention_mask is NOT used (softmax is intra-block; linear is self-contained).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_attention import OrderInvariantKernelLinearAttention


class BlockSoftmaxLinearHybrid(nn.Module):

    def __init__(self, config, block_size: int | None = None):
        super().__init__()
        self.num_heads   = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim    = config.head_dim
        self.block_size  = int(block_size or getattr(config, "block_size", 32))
        self.feature_dim = int(config.feature_dim)
        self.eps         = 1e-6
        self.scaling     = self.head_dim ** -0.5

        # If `share_hedgehog_kv_groups` is True, hedgehog_weights are stored
        # per-kv-head (Hkv, D, F) — 4× smaller for GQA 4:1. The forward
        # broadcasts each W to its GQA group of q-heads via a reshape+einsum.
        self.share_kv_groups = bool(getattr(config, "share_hedgehog_kv_groups", False))
        if self.share_kv_groups:
            assert self.num_heads % self.num_kv_heads == 0, (
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads "
                f"({self.num_kv_heads}) when share_hedgehog_kv_groups=True"
            )
            self.num_kv_groups = self.num_heads // self.num_kv_heads
            W_heads = self.num_kv_heads
        else:
            self.num_kv_groups = 1
            W_heads = self.num_heads

        # Init W: (W_heads, D, F).  torch.eye(D, F) leaves rows F..D-1 all zeros when D > F,
        # so half of head_dim never affects φ at step 0.  Fill the tail block with small
        # noise so every dimension participates from the start (F unchanged).
        D, F = self.head_dim, self.feature_dim
        if D > F:
            w = torch.zeros(W_heads, D, F, dtype=torch.float32)
            w[:, :F, :] = torch.eye(F, dtype=torch.float32).unsqueeze(0).expand(W_heads, -1, -1)
            std = min(0.02, D ** -0.5)
            w[:, F:, :] = std * torch.randn(W_heads, D - F, F, dtype=torch.float32)
            weight = w
        else:
            weight = torch.eye(D, F, dtype=torch.float32).unsqueeze(0).expand(
                W_heads, -1, -1
            ).contiguous()
        self.hedgehog_weights = nn.Parameter(weight.clone())   # (W_heads, D, F)

        # Gate: per-q-head scalar regardless of W sharing — alpha is cheap and
        # the per-q-head gate matters for blending.
        # Shape (1, H, 1, 1) broadcasts over (B, H, S, D) and (B, H, S, 1).
        self.alpha = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

    # ── Hedgehog feature map ─────────────────────────────────────────────────

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, N, S, D)  →  (B, H, N, S, 2F)"""
        W = self.hedgehog_weights.to(dtype=x.dtype)
        if self.share_kv_groups:
            # W shape (Hkv, D, F). Reshape x's H dim into (Hkv, num_kv_groups)
            # so the einsum broadcasts each W[hkv] across its GQA group.
            B, _H, N, S, D_ = x.shape
            x = x.view(B, self.num_kv_heads, self.num_kv_groups, N, S, D_)
            u = torch.einsum("bgknsd,gdf->bgknsf", x, W).reshape(
                B, self.num_heads, N, S, -1
            )
        else:
            # W shape (H, D, F) — original per-q-head path.
            u = torch.einsum("bhnsd,hdf->bhnsf", x, W)
        u_f32 = u.float()
        return torch.cat(
            [F.softmax(u_f32, dim=-1), F.softmax(-u_f32, dim=-1)], dim=-1
        ).to(dtype=x.dtype)

    # ── Shared-normalization block output ────────────────────────────────────

    # def _block_out(
    #     self,
    #     q_n:       torch.Tensor,         # (B, H, S, D)
    #     k_n:       torch.Tensor,         # (B, H, S, D)
    #     v_n:       torch.Tensor,         # (B, H, S, D)
    #     phi_q_n:   torch.Tensor,         # (B, H, S, 2F)
    #     S_state:   torch.Tensor,         # (B, H, 2F, D) fp32, previous blocks only
    #     Z_state:   torch.Tensor,         # (B, H, 2F)    fp32, previous blocks only
    #     w:         torch.Tensor,         # unused in pure-linear mode
    #     kv_mask_n: torch.Tensor | None,  # (B, 1, S), 1=real key
    #     out_dtype: torch.dtype,
    # ) -> torch.Tensor:
    #     """
    #     Pure Hedgehog block output.
    #
    #     Linear stream now sees:
    #     previous blocks 0..n-1 through S_state/Z_state
    #     current block n through a temporary S_cur/Z_cur
    #
    #     No intra-block softmax branch is used.
    #     """
    #
    #     # ── Current-block Hedgehog state ─────────────────────────────────────
    #     # k_n: (B,H,S,D) -> (B,H,1,S,D) -> feature map -> (B,H,S,2F)
    #     phi_k_cur = self._feature_map(k_n.unsqueeze(2)).squeeze(2)
    #
    #     v_cur = v_n
    #     if kv_mask_n is not None:
    #         # kv_mask_n: (B,1,S) -> (B,1,S,1), broadcasts over heads/features
    #         m = kv_mask_n.to(device=phi_k_cur.device, dtype=phi_k_cur.dtype).unsqueeze(-1)
    #         phi_k_cur = phi_k_cur * m
    #         v_cur = v_cur * m.to(v_cur.dtype)
    #
    #     S_cur = torch.matmul(
    #         phi_k_cur.transpose(-2, -1).float(),
    #         v_cur.float(),
    #     )  # (B,H,2F,D)
    #
    #     Z_cur = phi_k_cur.float().sum(dim=-2)  # (B,H,2F)
    #
    #     # Query-time state includes previous blocks + current block.
    #     S_query = S_state + S_cur
    #     Z_query = Z_state + Z_cur
    #
    #     # ── Pure linear numerator / denominator ──────────────────────────────
    #     lin_num = torch.matmul(phi_q_n.float(), S_query.float())  # (B,H,S,D)
    #     lin_den = torch.matmul(
    #         phi_q_n.float(),
    #         Z_query.float().unsqueeze(-1),
    #     ).clamp_min(self.eps)  # (B,H,S,1)
    #
    #     return (lin_num / lin_den).to(out_dtype)

    def _block_out(
        self,
        q_n:       torch.Tensor,         # (B, H, S, D)
        k_n:       torch.Tensor,         # (B, H, S, D)
        v_n:       torch.Tensor,         # (B, H, S, D)
        phi_q_n:   torch.Tensor,         # (B, H, S, 2F)
        S_state:   torch.Tensor,         # (B, H, 2F, D) fp32, previous blocks only
        Z_state:   torch.Tensor,         # (B, H, 2F)    fp32, previous blocks only
        w:         torch.Tensor,         # (1, H, 1, 1) gate = sigmoid(alpha)
        kv_mask_n: torch.Tensor | None,  # (B, 1, S), 1=real key
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Hybrid block output: shared-normalization mix of
          - intra-block softmax (radix attention) for the current block, and
          - Hedgehog linear attention over previous blocks 0..n-1.

        out = (w * sm_num  +  lin_num) / (w * sm_den  +  lin_den)

        where
          sm_num  = Σ_{j in block n} exp(q·kⱼ/√d) · vⱼ   (unnorm. softmax weighted sum)
          sm_den  = Σ_{j in block n} exp(q·kⱼ/√d)          (partition function)
          lin_num = φ(q) @ S_state                           (linear, cross-block 0..n-1)
          lin_den = φ(q) @ Z_state                           (linear normalizer)
          w       = sigmoid(alpha)                           (learnable gate)

        Shared normalization keeps hedgehog_weights gradients unattenuated by w.
        When lin_den ≈ 0 (first block / early training) output degrades gracefully
        to pure intra-block softmax.
        """

        # ── Linear numerator / denominator (cross-block, previous only) ──────
        lin_num = torch.matmul(phi_q_n.float(), S_state)               # (B, H, S, D)
        lin_den = torch.matmul(phi_q_n.float(), Z_state.unsqueeze(-1)) # (B, H, S, 1)
        lin_den = lin_den.clamp_min(self.eps)

        # ── Intra-block softmax (radix attention) for current block ──────────
        # FP32 QK matmul + numerically stable exp (online softmax / LoLCATs-style).
        scores = torch.matmul(q_n.float(), k_n.float().transpose(-1, -2)) * self.scaling
        # (B, H, S, S) fp32
        if kv_mask_n is not None:
            # kv_mask_n: (B, 1, S) → (B, 1, 1, S) — mask key dimension
            scores = scores.masked_fill(
                kv_mask_n.unsqueeze(-2) == 0,
                torch.finfo(scores.dtype).min,  # finite → no NaN in exp
            )
        scores_max = scores.amax(dim=-1, keepdim=True)          # online max for stability
        a_sm    = torch.exp(scores - scores_max)                 # (B, H, S, S) fp32
        sm_num  = torch.matmul(a_sm, v_n.float())               # (B, H, S, D)
        sm_den  = a_sm.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # (B, H, S, 1)

        # ── Shared-normalization mix ──────────────────────────────────────────
        num = w * sm_num + lin_num
        den = (w * sm_den + lin_den).clamp_min(self.eps)
        return (num / den).to(out_dtype)


    # ------------------------------------------------------------------
    # OLD HYBRID SOFTMAX + LINEAR CODE — disabled
    # ------------------------------------------------------------------
    # lin_num = torch.matmul(phi_q_n.float(), S_state)
    # lin_den = torch.matmul(phi_q_n.float(), Z_state.unsqueeze(-1))
    # lin_den = lin_den.clamp_min(self.eps)
    #
    # scores = torch.matmul(q_n.float(), k_n.float().transpose(-1, -2)) * self.scaling
    # if kv_mask_n is not None:
    #     scores = scores.masked_fill(
    #         kv_mask_n.unsqueeze(-2) == 0,
    #         torch.finfo(scores.dtype).min,
    #     )
    # scores_max = scores.amax(dim=-1, keepdim=True)
    # a_sm = torch.exp(scores - scores_max)
    # sm_num = torch.matmul(a_sm, v_n.float())
    # sm_den = a_sm.sum(dim=-1, keepdim=True).clamp_min(self.eps)
    #
    # num = w * sm_num + lin_num
    # den = (w * sm_den + lin_den).clamp_min(self.eps)
    # return (num / den).to(out_dtype)

    # def _block_out(
    #     self,
    #     q_n:       torch.Tensor,         # (B, H, S, D)  raw query
    #     k_n:       torch.Tensor,         # (B, H, S, D)  raw key  (unmasked)
    #     v_n:       torch.Tensor,         # (B, H, S, D)  value    (padding already zeroed)
    #     phi_q_n:   torch.Tensor,         # (B, H, S, 2F)
    #     S_state:   torch.Tensor,         # (B, H, 2F, D) fp32
    #     Z_state:   torch.Tensor,         # (B, H, 2F)    fp32
    #     w:         torch.Tensor,         # scalar sigmoid(alpha)
    #     kv_mask_n: torch.Tensor | None,  # (B, 1, S) float, 1=real key
    #     out_dtype: torch.dtype,
    # ) -> torch.Tensor:                   # (B, H, S, D)
    #     """
    #     Compute one block's output with shared-normalization:
    #       (w·sm_num + lin_num) / (w·sm_den + lin_den)
    #     linear_factor = 1 (independent of alpha).
    #     """
    #     # ── Linear numerator / denominator ──────────────────────────────────
    #     lin_num = torch.matmul(phi_q_n.float(), S_state)             # (B, H, S, D)
    #     lin_den = torch.matmul(phi_q_n.float(), Z_state.unsqueeze(-1))  # (B, H, S, 1)
    #     lin_den = lin_den.clamp_min(self.eps)

    #     # ── Softmax numerator / denominator (unnormalised exp) ────────────
    #     # FP32 QK + FP32 exp (LoLCATs-consistent): cast q/k before matmul so
    #     # `scores - scores_max` and `torch.exp(...)` run in fp32, not bf16.
    #     scores = torch.matmul(q_n.float(), k_n.float().transpose(-1, -2)) * self.scaling  # (B, H, S, S) fp32
    #     if kv_mask_n is not None:
    #         # kv_mask_n: (B, 1, S) → unsqueeze query dim → (B, 1, 1, S)
    #         # Use finfo.min (finite) rather than -inf so all-masked rows give
    #         # max - max = 0 and a finite (zero-contribution) exp instead of NaN.
    #         scores = scores.masked_fill(
    #             kv_mask_n.unsqueeze(-2) == 0,
    #             torch.finfo(scores.dtype).min,
    #         )
    #     scores_max = scores.amax(dim=-1, keepdim=True)
    #     a_sm    = torch.exp(scores - scores_max)                       # (B, H, S, S) fp32
    #     sm_num  = torch.matmul(a_sm, v_n.float())                     # (B, H, S, D)
    #     sm_den  = a_sm.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # (B, H, S, 1)

    #     # ── Combined output ───────────────────────────────────────────────
    #     num = w * sm_num + lin_num
    #     den = (w * sm_den + lin_den).clamp_min(self.eps)
    #     return (num / den).to(out_dtype)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        query_states:  torch.Tensor,   # (B, H, L, D)
        key_states:    torch.Tensor,
        value_states:  torch.Tensor,
        attention_mask=None,           # ignored
        key_padding_mask=None,         # (B, L) float/bool, 1=real token
        initial_S_state: torch.Tensor | None = None,
        initial_Z_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple:
        B, H, L, D = query_states.shape
        out_dtype  = query_states.dtype
        S          = self.block_size

        num_blocks = (L + S - 1) // S
        padded_L   = num_blocks * S
        pad_len    = padded_L - L

        q, k, v = query_states, key_states, value_states
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        q = q.view(B, H, num_blocks, S, D)
        k = k.view(B, H, num_blocks, S, D)
        v = v.view(B, H, num_blocks, S, D)

        # key_valid: (B, 1, N, S, 1) — covers both kpm and block-padding.
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(device=q.device, dtype=torch.float32)[:, :padded_L]
            if kpm.shape[-1] < padded_L:
                kpm = F.pad(kpm, (0, padded_L - kpm.shape[-1]))
            key_valid = kpm.view(B, 1, num_blocks, S, 1)
        elif pad_len > 0:
            valid_1d = torch.ones(padded_L, device=q.device)
            valid_1d[L:] = 0.0
            key_valid = valid_1d.view(1, 1, num_blocks, S, 1)
        else:
            key_valid = None

        phi_q = self._feature_map(q)   # (B, H, N, S, 2F)
        phi_k = self._feature_map(k)   # (B, H, N, S, 2F)
        D_feat = 2 * self.feature_dim

        # Zero out padding keys in the linear stream.
        if key_valid is not None:
            kv = key_valid.to(device=phi_k.device, dtype=phi_q.dtype)
            phi_k = phi_k * kv
            v     = v     * kv.to(v.dtype)

        if initial_S_state is not None:
            S_state = initial_S_state
        else:
            S_state = torch.zeros(B, H, D_feat, D, dtype=torch.float32, device=q.device)
        if initial_Z_state is not None:
            Z_state = initial_Z_state
        else:
            Z_state = torch.zeros(B, H, D_feat,    dtype=torch.float32, device=q.device)
        w       = torch.sigmoid(self.alpha.float())

        out_blocks = []
        for n in range(num_blocks):
            kv_mask_n = key_valid[:, :, n, :, 0] if key_valid is not None else None

            out_blocks.append(self._block_out(
                q[:, :, n], k[:, :, n], v[:, :, n],
                phi_q[:, :, n],
                S_state, Z_state,
                w, kv_mask_n, out_dtype,
            ))

            # Update recurrent state AFTER querying (strictly causal).
            phi_k_n = phi_k[:, :, n]
            S_state = S_state + torch.matmul(phi_k_n.transpose(-2, -1).float(), v[:, :, n].float())
            Z_state = Z_state + phi_k_n.float().sum(dim=-2)

        out = torch.stack(out_blocks, dim=2).view(B, H, padded_L, D)
        if pad_len > 0:
            out = out[:, :, :L, :]
        if return_final_state:
            return out, S_state, Z_state
        return out

    # ── BD3LM staircase forward ───────────────────────────────────────────────

    def forward_bd3lm(
        self,
        query_states:    torch.Tensor,   # (B, H, 2L, D)
        key_states:      torch.Tensor,   # (B, H, 2L, D)
        value_states:    torch.Tensor,   # (B, H, 2L, D)
        half_len:        int,            # L
        key_padding_mask=None,           # (B, 2L) float/bool, 1=real token
    ) -> torch.Tensor:
        """
        BD3LM staircase with shared-normalization hybrid attention.

        Per block n, execution order is:

          Phase 1 — noisy query (pre-update state):
            Linear stream sees clean blocks 0..n-1 (S_state before update).
            Intra-block softmax attends to noisy block n tokens.
            Combined: noisy block n can use clean context + its own noisy tokens.

          Phase 2 — clean query (pre-update state):
            Linear stream sees clean blocks 0..n-1 (same S_state as Phase 1).
            Intra-block softmax attends to clean block n tokens.
            Combined: clean block n sees clean 0..n-1 via linear + clean n via softmax.

          Phase 3 — state update:
            Persistent S_state and Z_state += clean block n keys/values.
            After this update the state contains clean blocks 0..n.

        Phases 1 and 2 both use the PRE-update state intentionally — the intra-block
        softmax already handles within-block self-attention for both noisy and clean
        queries, so the linear stream only needs to supply cross-block context (0..n-1).
        This differs from pure linear (forward_bd3lm in OrderInvariantKernelLinearAttention)
        where Phase 3 happens between noisy and clean queries because there is no softmax.

        Shared normalization: linear branch contributes with factor 1 (independent of
        the gate alpha), so hedgehog_weights gradients are not attenuated by the gate.

        Returns (B, H, 2L, D) — [noisy outputs | clean outputs].
        """
        B, H, total_L, D = query_states.shape
        assert total_L == 2 * half_len

        out_dtype = query_states.dtype
        S         = self.block_size

        q_noisy = query_states[:, :, :half_len, :]
        q_clean = query_states[:, :, half_len:,  :]
        k_noisy = key_states  [:, :, :half_len, :]
        k_clean = key_states  [:, :, half_len:,  :]
        v_noisy = value_states[:, :, :half_len, :]
        v_clean = value_states[:, :, half_len:,  :]

        num_blocks = (half_len + S - 1) // S
        padded_L   = num_blocks * S
        pad_len    = padded_L - half_len

        def _blk(x):
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            return x.view(B, H, num_blocks, S, D)

        q_noisy, q_clean = _blk(q_noisy), _blk(q_clean)
        k_noisy, k_clean = _blk(k_noisy), _blk(k_clean)
        v_noisy, v_clean = _blk(v_noisy), _blk(v_clean)

        phi_q_noisy = self._feature_map(q_noisy)
        phi_q_clean = self._feature_map(q_clean)
        phi_k_clean = self._feature_map(k_clean)
        D_feat = 2 * self.feature_dim

        # Build per-half key-validity masks: (B, 1, N, S, 1).
        # kv_noisy used for: softmax score masking of noisy block, v_noisy zeroing.
        # kv_clean used for: linear state masking, softmax score masking of clean block.
        def _build_kv(kpm_half):
            kpm = kpm_half.float()
            if kpm.shape[-1] < padded_L:
                kpm = F.pad(kpm, (0, padded_L - kpm.shape[-1]), value=0.0)
            return kpm.view(B, 1, num_blocks, S, 1).to(device=phi_k_clean.device,
                                                        dtype=phi_k_clean.dtype)

        if key_padding_mask is not None:
            kv_noisy = _build_kv(key_padding_mask[:, :half_len])
            kv_clean = _build_kv(key_padding_mask[:, half_len:])
        elif pad_len > 0:
            valid_1d = torch.ones(padded_L, device=phi_k_clean.device,
                                  dtype=phi_k_clean.dtype)
            valid_1d[half_len:] = 0.0
            pad_mask = valid_1d.view(1, 1, num_blocks, S, 1)
            kv_noisy = kv_clean = pad_mask
        else:
            kv_noisy = kv_clean = None

        # Apply key masks to values and linear keys.
        if kv_noisy is not None:
            v_noisy     = v_noisy     * kv_noisy.to(v_noisy.dtype)
        if kv_clean is not None:
            phi_k_clean = phi_k_clean * kv_clean
            v_clean     = v_clean     * kv_clean.to(v_clean.dtype)

        S_state = torch.zeros(B, H, D_feat, D, dtype=torch.float32, device=q_noisy.device)
        Z_state = torch.zeros(B, H, D_feat,    dtype=torch.float32, device=q_noisy.device)
        w       = torch.sigmoid(self.alpha.float())

        out_noisy_blocks = []
        out_clean_blocks = []

        for n in range(num_blocks):
            kvm_noisy_n = kv_noisy[:, :, n, :, 0] if kv_noisy is not None else None
            kvm_clean_n = kv_clean[:, :, n, :, 0] if kv_clean is not None else None

            # Phase 1: noisy query — linear sees clean 0..n-1 (pre-update).
            out_noisy_blocks.append(self._block_out(
                q_noisy[:, :, n], k_noisy[:, :, n], v_noisy[:, :, n],
                phi_q_noisy[:, :, n],
                S_state, Z_state,
                w, kvm_noisy_n, out_dtype,
            ))

            # Phase 2: clean query — linear sees clean 0..n-1 (same pre-update state).
            # Intra-block softmax covers clean block n; linear supplies cross-block context.
            out_clean_blocks.append(self._block_out(
                q_clean[:, :, n], k_clean[:, :, n], v_clean[:, :, n],
                phi_q_clean[:, :, n],
                S_state, Z_state,
                w, kvm_clean_n, out_dtype,
            ))

            # Phase 3: state update — add clean block n; state now contains clean 0..n.
            phi_kn = phi_k_clean[:, :, n]
            S_state = S_state + torch.matmul(phi_kn.transpose(-2, -1).float(), v_clean[:, :, n].float())
            Z_state = Z_state + phi_kn.float().sum(dim=-2)

        out_n = torch.stack(out_noisy_blocks, dim=2).reshape(B, H, padded_L, D)
        out_c = torch.stack(out_clean_blocks, dim=2).reshape(B, H, padded_L, D)
        if pad_len > 0:
            out_n = out_n[:, :, :half_len, :]
            out_c = out_c[:, :, :half_len, :]

        return torch.cat([out_n, out_c], dim=2)


