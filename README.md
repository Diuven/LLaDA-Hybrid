# LLaDA-Hybrid

Code and weights for **[Retrofitting Linear Attention into Diffusion Language Models](https://arxiv.org/abs/2608.06628)** (Kim, Roh, Kim).

LLaDA-Hybrid replaces the softmax attention in 6 of the 20 layers of
[LLaDA 2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) with
**block-hybrid attention**: exact bidirectional softmax attention *within* the
active 32-token denoising block, and a fixed-size Hedgehog linear-attention state
over all previously committed blocks. Those six layers stop writing a KV cache
entirely, so their cost of reaching prior context is independent of how much
context there is.

This repository ships the retrofitted weights, the PyTorch model definition, and
the SGLang source -- fused Triton kernel included -- that the serving
measurements were taken on.

---

## Throughput

One H200, SGLang continuous batching, JointThreshold decoding (threshold 0.7,
editing off, block size 32), 1024-prompt Alpaca-cleaned pool, generation length
2048, `mem_fraction_static=0.75`, seed 42.

Table 2 of the paper:

| concurrent requests | teacher (tok/s) | LLaDA-Hybrid (tok/s) | speedup |
|---|---|---|---|
| 16 | 1845.7 | 2761.7 | 1.50× |
| 32 | 2047.5 | 3101.9 | 1.51× |
| 64 | 2457.9 | 4026.4 | 1.64× |
| 128 | 2310.2 | 3994.4 | 1.73× |
| 256 | 2350.3 | 3767.5 | 1.60× |

---

## Install

```bash
git clone https://github.com/Diuven/LLaDA-Hybrid.git
cd LLaDA-Hybrid
pip install -e .
```

Serving needs the SGLang in `sglang/`, which is checked into this repository as
source rather than as a patch: it is `sgl-project/sglang@2b47bd3a` (2026-03-23,
between v0.5.9 and v0.5.10) with the hybrid backend, kernel and model class
already in it, and it is the exact tree the numbers above were measured on.

```bash
pip install -e sglang/python      # CUDA build, takes a while
```

`sglang/UPSTREAM.md` gives the base commit, the list of files that differ from
upstream, and the one-line `diff -ru` that reproduces the full change set.
Measured with torch 2.9.1+cu128, transformers 4.57.1, Python 3.12; see
`requirements.txt`.

---

## Quickstart

```bash
# Join the released adapter to the base weights (symlinks, nothing is copied)
python scripts/prepare_model.py --out models/llada-hybrid-6l

# Throughput, both arms through the same code path
python eval/throughput.py --model-path models/llada-hybrid-6l \
    --batch-size 256 --gen-length 2048 --passes 2 --out results/hybrid.json
python eval/throughput.py --model-path inclusionAI/LLaDA2.1-mini \
    --batch-size 256 --gen-length 2048 --passes 2 --out results/teacher.json

# The fused kernel against a pure-PyTorch reference (720 cases)
python kernel/test_kernel.py
```

---

## The checkpoint

`checkpoints/llada-hybrid-6l/adapter.safetensors` is **13.4 MB**, because it holds
everything that differs from the public base checkpoint and nothing else:

| keys | count | what |
|---|---|---|
| `hedgehog.*` | 12 | feature map `W` `[16,128,64]` and gate `alpha` `[16]` for layers 0, 4, 8, 12, 16, 18 |
| `lora.*` | 80 | LoRA `A`/`B`, rank 16, on `query_key_value` and `dense` in all 20 layers |

Every other weight is bit-identical to `inclusionAI/LLaDA2.1-mini`, so the base is
pulled from the Hub rather than re-uploaded. `scripts/prepare_model.py` writes a
`config.json` and symlinks the base shards into a directory SGLang serves directly.

**The LoRA is merged into the base weights at load** (`W += (alpha/r)·B@A`, in
fp32). The served network is therefore architecturally identical to the distilled
model — same kernels, same FLOPs, same KV footprint, and no adapter cost at
inference time. The factors ship unmerged and unscaled so the merge arithmetic
stays exactly what the reported numbers were measured with.

---

## Method

Within a layer, the active 32-token block attends to itself with exact softmax
attention, while every committed block is summarised by a Hedgehog feature map
`phi(x) = [softmax(xW), softmax(-xW)]` giving a fixed-size `(S, Z)`
numerator/normaliser pair per query head, updated once per block. The two branches
are merged under a **shared** denominator with a learned per-head gate
`w = sigma(alpha)`:

```
out(q) = (w · sm_num(q) + lin_num(q)) / (w · sm_den(q) + lin_den(q))
```

Block diffusion makes this natural: the decoder only ever needs the active block
at full resolution. The only new parameters per linearized layer are `W` and
`alpha`.

The retrofit is two-stage — layer-local attention transfer against the frozen
softmax teacher, then LoRA adaptation under the masked-diffusion objective. See
the paper for the recipe. **Training code is not part of this release**; this
repository is inference and serving only.

---

## Layout

```
checkpoints/llada-hybrid-6l/   adapter.safetensors (13.4 MB) + adapter_config.json
llada_hybrid/                  PyTorch model definition
sglang/                        SGLang @2b47bd3a with the hybrid backend, as source
eval/                          throughput measurement
kernel/                        fused kernel vs pure-PyTorch reference
scripts/                       build_adapter.py, prepare_model.py
results/                       throughput JSON from this repository
```

## What this changes in SGLang

19 files differ from the base commit; `sglang/UPSTREAM.md` lists them all and
shows how to diff the vendored tree against a clean upstream checkout. The
substantive parts:

- `srt/layers/attention/llada_fast_hybrid_kernel.py` — the fused Triton kernel. One
  launch computes the within-block softmax branch, the linear readout from the
  recurrent state, the gated merge and the causal state update, without writing
  intermediates to HBM.
- `srt/models/llada_fast.py` — the model class; swaps in hybrid attention at
  `__init__` and applies the adapter in `load_weights`.
- `srt/layers/attention/llada_fast_backend.py` — maps request pool slots to
  recurrent-state slots, following SGLang's Mamba hybrid pattern, so the state
  reuses the existing slot allocator and continuous-batching scheduler.
- `srt/configs/llada_fast.py` — tells `HybridLinearKVPool` that only the 14 softmax
  layers need KV slots. This is where the memory saving comes from.
- `srt/layers/logits_processor.py` — keep dLLM extend logits in bf16; the fp32 cast
  doubled peak VRAM for a tensor used only for argmax and softmax.

## Reproducibility

Throughput is not exactly reproducible run to run. The matmul kernels are not
batch-invariant, so reduction order follows batch composition; under continuous
batching composition follows wall-clock KV release; and the decoder's
`confidence > 0.7` test then turns a last-bit logit perturbation into a different
unmask decision. Completions vary between identical runs as well as timings.
Report a median over several passes, alternate the arms rather than running one to
completion first, and do not compare runs from different sessions.

## Acknowledgments

Built on [LLaDA 2.1](https://github.com/inclusionAI/LLaDA2.X) by InclusionAI and
[SGLang](https://github.com/sgl-project/sglang). The retrofit recipe follows
[LoLCATs](https://github.com/HazyResearch/lolcats) and the feature map follows
[Hedgehog](https://arxiv.org/abs/2402.04347).

All research and code in this repository was carried out at MIT, prior to and
independently of the authors' subsequent employment.

MIT licensed. The base weights and SGLang carry their own licenses.
