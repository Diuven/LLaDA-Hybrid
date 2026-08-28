"""SGLang engine construction, shared by every benchmark in this directory.

Both arms load through the same function: the architecture is read from the
model directory's ``config.json``, so the teacher is ``inclusionAI/LLaDA2.1-mini``
and the hybrid is a directory produced by ``scripts/prepare_model.py``. Nothing
downstream branches on which one is running.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_ALGO_CONFIG = str(Path(__file__).resolve().parent / "configs" / "joint_threshold.yaml")


def load_engine(
    model_path: str,
    *,
    algo_config: Optional[str] = None,
    tp_size: int = 1,
    max_running_requests: Optional[int] = None,
    mem_fraction_static: Optional[float] = None,
    max_total_tokens: Optional[int] = None,
) -> Tuple[object, object]:
    """Start an SGLang Engine running the JointThreshold dLLM decoder.

    ``max_running_requests`` is the concurrency cap: the scheduler admits up to
    that many sequences at once and refills at block boundaries. It is what the
    throughput benchmark sweeps.

    The non-default engine settings are all forced by the dLLM path:

    ``attention_backend="triton"``
        flashinfer / FA3 mis-handle the dLLM batch shape.
    ``disable_radix_cache=True``
        prefix reuse across benchmark prompts would make throughput a function
        of prompt overlap rather than of the model.
    ``disable_cuda_graph``
        the denoising loop changes the batch width between passes (converged
        requests are dropped from the forward), so captured graphs are replayed
        at the wrong width. Set ``SGLANG_DISABLE_CUDA_GRAPH=0`` to override.
    """
    from sglang import Engine
    from transformers import AutoTokenizer

    engine_kw = dict(
        model_path=model_path,
        tp_size=tp_size,
        trust_remote_code=True,
        dllm_algorithm="JointThreshold",
        dllm_algorithm_config=algo_config or DEFAULT_ALGO_CONFIG,
        attention_backend="triton",
        disable_radix_cache=True,
        disable_cuda_graph=os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "1") != "0",
        # Loading 16B of MoE weights off a shared filesystem can exceed SGLang's
        # 300s default, which SIGKILLs the scheduler mid-load.
        watchdog_timeout=float(os.environ.get("SGLANG_WATCHDOG_TIMEOUT", "1800")),
    )
    for key, value in (
        ("max_running_requests", max_running_requests),
        ("mem_fraction_static", mem_fraction_static),
        ("max_total_tokens", max_total_tokens),
    ):
        if value is not None:
            engine_kw[key] = value

    print(f"[engine] model_path={model_path}")
    print(f"[engine] algo_config={engine_kw['dllm_algorithm_config']}")
    engine = Engine(**engine_kw)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("[engine] ready")
    return engine, tokenizer


def add_engine_args(parser) -> None:
    """Attach the engine flags every benchmark here accepts."""
    parser.add_argument(
        "--model-path",
        required=True,
        help="teacher: inclusionAI/LLaDA2.1-mini; "
        "hybrid: a directory built by scripts/prepare_model.py",
    )
    parser.add_argument("--algo-config", default=None, help="dLLM decoder YAML")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-total-tokens", type=int, default=None)
