# Provenance

This directory is [sgl-project/sglang](https://github.com/sgl-project/sglang) at

    2b47bd3a348bf5ca52a3a4910b2e22c851798576   # 2026-03-23, between v0.5.9 and v0.5.10

with the LLaDA-Hybrid changes applied, checked in as source. It is the exact
tree the throughput numbers in the README were measured on. Install it with:

    pip install -e sglang/python

To see everything this work changes in the serving engine, diff it against a
clean upstream checkout:

    git clone https://github.com/sgl-project/sglang.git /tmp/sglang-upstream
    git -C /tmp/sglang-upstream checkout --detach 2b47bd3a348bf5ca52a3a4910b2e22c851798576
    diff -ru -x .git /tmp/sglang-upstream .

19 files differ. New files:

| file | what |
|---|---|
| `python/sglang/srt/layers/attention/llada_fast_hybrid_kernel.py` | the fused Triton kernel — one launch computes the within-block softmax branch, the linear readout from the recurrent state, the gated merge and the causal state update, without writing intermediates to HBM |
| `python/sglang/srt/layers/attention/llada_fast_backend.py` | attention backend; maps request pool slots to recurrent-state slots following SGLang's Mamba hybrid pattern, so the state reuses the existing slot allocator and continuous-batching scheduler |
| `python/sglang/srt/models/llada_fast.py` | the model class; swaps in hybrid attention at `__init__` and applies the adapter in `load_weights` |
| `python/sglang/srt/configs/llada_fast.py` | tells `HybridLinearKVPool` that only the 14 softmax layers need KV slots — this is where the memory saving comes from |
| `python/sglang/srt/models/llada_profile.py` | per-stage timing hooks, off by default |

Modified files: `srt/configs/{__init__,model_config}.py`,
`srt/dllm/config.py`, `srt/dllm/algorithm/{joint_threshold,low_confidence}.py`,
`srt/layers/attention/{attention_registry,triton_backend}.py`,
`srt/layers/logits_processor.py`, `srt/layers/moe/topk.py`,
`srt/managers/scheduler.py`,
`srt/model_executor/{forward_batch_info,model_runner,model_runner_kv_cache_mixin}.py`,
`srt/models/llada2.py`. The logits-processor change keeps dLLM extend logits in
bf16; the fp32 cast doubled peak VRAM for a tensor used only for argmax and
softmax.

SGLang is Apache-2.0; its `LICENSE` is preserved in this directory unchanged.
