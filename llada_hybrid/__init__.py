"""LLaDA-Hybrid: LLaDA 2.1-mini with six attention layers replaced by
block-softmax + Hedgehog linear attention.

This package is the PyTorch reference implementation. It defines the model the
released adapter was distilled into and is the definition the SGLang kernel is
checked against; serving throughput comes from the SGLang path (see the repo
README), not from here.
"""

from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModel, LLaDA2MoeModelLM
from .attention import LLaDA2MoeAttention
from .linear_attention import OrderInvariantKernelLinearAttention
from .hybrid_attention import BlockSoftmaxLinearHybrid
from .decoder import LLaDA2MoeDecoderLayer
from .cache import BlockDiffusionCache

__all__ = [
    "LLaDA2MoeConfig",
    "LLaDA2MoeModel",
    "LLaDA2MoeModelLM",
    "LLaDA2MoeAttention",
    "LLaDA2MoeDecoderLayer",
    "OrderInvariantKernelLinearAttention",
    "BlockSoftmaxLinearHybrid",
    "BlockDiffusionCache",
]
