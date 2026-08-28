"""Correctness test for the fused hybrid attention kernel.

The kernel under test is imported from the installed SGLang in sglang/, so this
checks exactly what serving runs -- not a copy that can drift from it. It is
compared against ``reference.py``, a pure-PyTorch implementation of the same
math, over a sweep of batch sizes, prefix lengths and ragged-length cases.

Usage:
    python kernel/test_kernel.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import itertools

import torch

from state import HybridStateView, build_request_block_plan
from production import HedgehogHybridAttnProd
from reference import ref_hybrid_extend_per_q

DEVICE = "cuda"
DTYPE = torch.bfloat16
H_Q = 16
H_KV = 4
D = 128
F = 64
BLOCK_SIZE = 32

OUT_ATOL = 5e-2
STATE_ATOL = 5e-2
Z_ATOL = 1e-2

BATCH_SIZES = [1, 4, 8, 16]
PREFIX_LENS = [0, 64, 256, 1024, 4096]
SEQ_LENS = [8, 16, 32]
ALPHAS = [-2.0, 0.0, 2.0]


@dataclass(frozen=True)
class StateVariant:
    name: str
    s_dtype: torch.dtype
    z_dtype: torch.dtype


STATE_VARIANTS = [
    StateVariant("Sfp32_Zfp32", torch.float32, torch.float32),
    StateVariant("Sbf16_Zfp32", torch.bfloat16, torch.float32),
]

# Two production paths to validate:
#   forward       — streaming output kernel + separate state-update kernel
#                   (current dispatch path for plan.max_blocks > 1 OR share_kv groups)
#   forward_fused — single fused output+state-update kernel with bf16 WGMMA matmuls
#                   (re-enabled in BlockSoftmaxLinearHybrid.forward for plan.max_blocks==1
#                   and !share_hedgehog_kv_groups on H200 / sm_90)
KERNEL_METHODS = ["forward", "forward_fused"]


def make_inputs(
    B,
    seq_len,
    prefix_len,
    alpha_val,
    device=DEVICE,
    dtype=DTYPE,
    s_dtype=torch.float32,
    z_dtype=torch.float32,
):
    T = B * seq_len
    q = torch.randn(T, H_Q, D, device=device, dtype=dtype)
    k = torch.randn(T, H_KV, D, device=device, dtype=dtype)
    v = torch.randn(T, H_KV, D, device=device, dtype=dtype)

    scale = (prefix_len ** 0.5) if prefix_len > 0 else 0.1
    committed = HybridStateView(
        S_state=torch.randn(B, H_Q, 2 * F, D, device=device, dtype=s_dtype) * scale * 0.01,
        Z_state=torch.randn(B, H_Q, 2 * F, device=device, dtype=z_dtype).abs() * scale * 0.1,
    )
    committed_frozen = HybridStateView(
        S_state=committed.S_state.clone(),
        Z_state=committed.Z_state.clone(),
    )

    def fresh_working():
        return HybridStateView(
            S_state=committed.S_state.clone(),
            Z_state=committed.Z_state.clone(),
        )

    seq_lens = [seq_len] * B
    req_indices = torch.arange(B, dtype=torch.long, device=device)

    return q, k, v, committed, committed_frozen, fresh_working, seq_lens, req_indices


def check(name, triton_val, ref_val, atol, results):
    diff = (triton_val.float() - ref_val.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff <= atol
    results.append((name, ok, max_diff, mean_diff))
    return ok


def run():
    print("\nCorrectness test (PRODUCTION kernel): Triton HedgehogHybridAttnProd vs PyTorch reference")
    print(f"Config: H_Q={H_Q} H_KV={H_KV} D={D} F={F} device={DEVICE} dtype={DTYPE}")
    print("State variants: " + ", ".join(v.name for v in STATE_VARIANTS))
    print(f"Tolerances: out={OUT_ATOL}  S_state={STATE_ATOL}  Z_state={Z_ATOL}\n")

    kernel = HedgehogHybridAttnProd(H_Q, H_KV, D, F, BLOCK_SIZE).to(DEVICE)
    first_fail_printed = [False]

    total, passed, failed = 0, 0, 0
    per_method = {m: {"total": 0, "passed": 0, "failed": 0} for m in KERNEL_METHODS}
    print(f"{'case':>56}  {'method':>20}  {'variant':>12}  {'out':>10}  {'S_state':>10}  {'Z_state':>10}  {'freeze':>8}  {'result':>6}")
    print("-" * 150)

    for B, prefix_len, seq_len, alpha_val, state_variant, method_name in itertools.product(
        BATCH_SIZES, PREFIX_LENS, SEQ_LENS, ALPHAS, STATE_VARIANTS, KERNEL_METHODS
    ):
        with torch.no_grad():
            kernel.alpha.fill_(alpha_val)

        q, k, v, committed, committed_frozen, fresh_working, seq_lens, req_indices = make_inputs(
            B,
            seq_len,
            prefix_len,
            alpha_val,
            s_dtype=state_variant.s_dtype,
            z_dtype=state_variant.z_dtype,
        )

        working_triton = fresh_working()
        plan = build_request_block_plan(seq_lens, req_indices, BLOCK_SIZE, DEVICE)
        kernel_fn = getattr(kernel, method_name)
        out_triton = kernel_fn(q, k, v, plan, committed, working_triton)

        working_ref = fresh_working()
        out_ref = ref_hybrid_extend_per_q(
            q,
            k,
            v,
            seq_lens,
            req_indices,
            committed,
            working_ref,
            W=kernel.hedgehog_weights.detach(),
            alpha=kernel.alpha.detach(),
        )

        committed_unchanged = torch.allclose(committed.S_state, committed_frozen.S_state, atol=0) and torch.allclose(
            committed.Z_state, committed_frozen.Z_state, atol=0
        )

        results = []
        out_ok = check("out", out_triton, out_ref, OUT_ATOL, results)
        s_ok = check("S_state", working_triton.S_state, working_ref.S_state, STATE_ATOL, results)
        z_ok = check("Z_state", working_triton.Z_state, working_ref.Z_state, Z_ATOL, results)

        all_ok = out_ok and s_ok and z_ok and committed_unchanged
        total += 1
        per_method[method_name]["total"] += 1
        if all_ok:
            passed += 1
            per_method[method_name]["passed"] += 1
        else:
            failed += 1
            per_method[method_name]["failed"] += 1

        label = f"B={B:2d} prefix={prefix_len:5d} seq={seq_len:2d} alpha={alpha_val:5.1f}"
        freeze_sym = "✓" if committed_unchanged else "✗"
        result_sym = "PASS" if all_ok else "FAIL"
        r = {x[0]: x for x in results}

        if not all_ok or total <= 8:
            print(
                f"  {label:>52}  "
                f"  {method_name:>18}  "
                f"  {state_variant.name:>10}  "
                f"  {r['out'][2]:>8.4f}  "
                f"  {r['S_state'][2]:>8.4f}  "
                f"  {r['Z_state'][2]:>8.4f}  "
                f"  {freeze_sym:>6}  "
                f"  {result_sym}"
            )

        if not all_ok and not first_fail_printed[0]:
            first_fail_printed[0] = True
            print(f"\n  [FIRST FAIL DIAGNOSIS — {label} {method_name} {state_variant.name}]")
            out_diff = (out_triton.float() - out_ref.float()).abs()
            for h in range(H_Q):
                od = out_diff[:, h, :].max().item()
                sd = (working_triton.S_state[:, h] - working_ref.S_state[:, h]).abs().max().item()
                zd = (working_triton.Z_state[:, h] - working_ref.Z_state[:, h]).abs().max().item()
                flag = "  <-- FAIL" if (od > OUT_ATOL or sd > STATE_ATOL or zd > Z_ATOL) else ""
                print(f"    h={h:2d}  out={od:.5f}  S={sd:.5f}  Z={zd:.5f}{flag}")
            print()

    print("-" * 150)
    print(f"\nSummary: {passed}/{total} passed, {failed} failed")
    for m in KERNEL_METHODS:
        s = per_method[m]
        print(f"  {m:>20s}: {s['passed']}/{s['total']} passed, {s['failed']} failed")
    print()

    if failed == 0:
        print("✓ All tests passed — production kernel matches PyTorch reference.")
    else:
        print("✗ Some tests FAILED — see rows above.")


def deep_dive(state_variant=STATE_VARIANTS[-1]):
    print(f"\n── Deep dive: B=4 prefix=1024 seq=32 alpha=0.0 (prod kernel, {state_variant.name}) ──")
    alpha_val = 0.0
    kernel = HedgehogHybridAttnProd(H_Q, H_KV, D, F, BLOCK_SIZE).to(DEVICE)
    with torch.no_grad():
        kernel.alpha.fill_(alpha_val)

    q, k, v, committed, _committed_frozen, fresh_working, seq_lens, req_indices = make_inputs(
        4,
        32,
        1024,
        alpha_val,
        s_dtype=state_variant.s_dtype,
        z_dtype=state_variant.z_dtype,
    )

    working_triton = fresh_working()
    plan = build_request_block_plan(seq_lens, req_indices, BLOCK_SIZE, DEVICE)
    out_triton = kernel.forward(q, k, v, plan, committed, working_triton)

    working_ref = fresh_working()
    out_ref = ref_hybrid_extend_per_q(
        q,
        k,
        v,
        seq_lens,
        req_indices,
        committed,
        working_ref,
        W=kernel.hedgehog_weights.detach(),
        alpha=kernel.alpha.detach(),
    )

    diff = (out_triton.float() - out_ref.float()).abs()
    print(f"  output max |err| = {diff.max():.6f}")
    print(f"  output mean|err| = {diff.mean():.6f}")
    print("  output per-head max |err|:")
    for h in range(H_Q):
        d = diff[:, h, :].max().item()
        print(f"    head {h:2d}: {d:.6f}")

    s_diff = (working_triton.S_state - working_ref.S_state).abs()
    z_diff = (working_triton.Z_state - working_ref.Z_state).abs()
    print(f"  S_state max |err| = {s_diff.max():.6f}")
    print(f"  Z_state max |err| = {z_diff.max():.6f}")


if __name__ == "__main__":
    run()
    deep_dive()
