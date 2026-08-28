"""Continuous-batching throughput benchmark.

A fixed pool of alpaca-cleaned prompts is submitted in one ``engine.generate``
call per measured pass. The engine admits up to ``--batch-size`` sequences
concurrently and refills at block boundaries, so ``--batch-size`` is the
*concurrency cap*, not a per-call batch size. Throughput is
``completion_tokens / wall_clock`` over the whole pass.

Why the pool is fixed and the prompts are identical across passes: what is being
measured is the engine's scheduling under contention, not prompt diversity.
Warmup uses a disjoint prompt so no state from it can bias a measured pass (the
radix cache is off at the engine level for the same reason).

Two arms, same command, different ``--model-path``:

    python eval/throughput.py --model-path inclusionAI/LLaDA2.1-mini \
        --batch-size 256 --gen-length 2048 --out results/teacher_b256.json

    python eval/throughput.py --model-path models/llada-hybrid-6l \
        --batch-size 256 --gen-length 2048 --out results/hybrid_b256.json

Run-to-run spread is real and not small: the kernels are not batch-invariant, so
reduction order follows batch composition, which under continuous batching
follows wall-clock KV release. Report a median over several passes and, when
comparing arms, alternate them rather than running one arm to completion first.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import List

import torch

from engine import add_engine_args, load_engine

# Disjoint from the measured pool so warmup cannot seed any cache the
# measurement then benefits from.
WARMUP_PROMPT = (
    "A train leaves Chicago at 9am travelling at 80 mph. "
    "Another leaves New York at 10am at 100 mph. When do they meet?"
)


def load_prompt_pool(n: int, seed: int) -> List[str]:
    """``n`` alpaca-cleaned instructions, deterministically sampled.

    The user turn concatenates instruction and input the way Alpaca's reference
    template does.
    """
    from datasets import load_dataset

    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    prompts = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        extra = (row.get("input") or "").strip()
        prompts.append(f"{instruction}\n\n{extra}" if extra else instruction)
    if len(prompts) < n:
        print(f"[pool] WARNING: asked for {n} prompts, dataset yielded {len(prompts)}")
    return prompts


def run_pass(engine, prompts: List[str], gen_length: int):
    """One full submission. Returns (texts, completion_tokens, wall_seconds)."""
    sampling_params = {
        "max_new_tokens": gen_length,
        "temperature": 0,
        # Fixed token budget per request: otherwise throughput would partly
        # measure how early each arm chooses to stop.
        "ignore_eos": True,
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = engine.generate(prompt=prompts, sampling_params=sampling_params)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if isinstance(results, dict):
        results = [results]
    texts = [r.get("text", "") for r in results]
    tokens = sum(r.get("meta_info", {}).get("completion_tokens", 0) for r in results)
    return texts, tokens, t1 - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(ap)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="concurrency cap (engine max_running_requests)")
    ap.add_argument("--gen-length", type=int, default=2048)
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=3, help="measured passes")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--dump-completions", default=None,
                    help="directory to write each pass's completions to")
    args = ap.parse_args()

    pool = load_prompt_pool(args.pool_size, args.seed)

    engine, tokenizer = load_engine(
        args.model_path,
        algo_config=args.algo_config,
        tp_size=args.tp_size,
        max_running_requests=args.batch_size,
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=args.max_total_tokens,
    )

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    measured_prompts = [chat(p) for p in pool]
    warmup_prompts = [chat(WARMUP_PROMPT)] * args.batch_size

    print(
        f"[throughput] pool={len(pool)} concurrency={args.batch_size} "
        f"gen_length={args.gen_length} warmup={args.warmup} passes={args.passes}"
    )

    tps_all, wall_all, token_all = [], [], []
    for i in range(args.warmup + args.passes):
        is_warmup = i < args.warmup
        tag = "warmup" if is_warmup else f"pass {i - args.warmup + 1}/{args.passes}"
        prompts = warmup_prompts if is_warmup else measured_prompts
        print(f"  [{tag}] submitting {len(prompts)} prompts ...")

        texts, tokens, wall = run_pass(engine, prompts, args.gen_length)
        tps = tokens / wall if wall > 0 else 0.0
        print(f"  [{tag}] wall={wall:.1f}s tokens={tokens} TPS={tps:.1f}")

        if is_warmup:
            continue
        tps_all.append(tps)
        wall_all.append(wall)
        token_all.append(tokens)

        if args.dump_completions:
            os.makedirs(args.dump_completions, exist_ok=True)
            path = os.path.join(
                args.dump_completions,
                f"b{args.batch_size}_gen{args.gen_length}_p{i - args.warmup + 1}.json",
            )
            with open(path, "w") as f:
                json.dump({"n": len(texts), "texts": texts}, f)
            print(f"  [{tag}] completions -> {path}")

    summary = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "gen_length": args.gen_length,
        "pool_size": len(pool),
        "seed": args.seed,
        "passes": args.passes,
        "tps": round(statistics.median(tps_all), 1),
        "tps_mean": round(statistics.fmean(tps_all), 1),
        "tps_all": [round(t, 1) for t in tps_all],
        "tps_sd": round(statistics.stdev(tps_all), 1) if len(tps_all) > 1 else None,
        "wall_seconds": round(statistics.median(wall_all), 2),
        "wall_all": [round(w, 2) for w in wall_all],
        "tokens_total": token_all[0] if token_all else 0,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[throughput] wrote {args.out}")


if __name__ == "__main__":
    main()
