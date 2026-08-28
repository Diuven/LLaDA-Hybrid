# llada_fast/modeling/configuration_llada2_moe.py
from transformers.configuration_utils import PretrainedConfig


class LLaDA2MoeConfig(PretrainedConfig):
    model_type = "llada2_moe"

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,  # NOTE: 0 means "use num_attention_heads" (compat)
        hidden_act="silu",
        use_qkv_bias=False,
        use_qk_norm=True,
        use_bias=True,
        rms_norm_eps=1e-05,
        norm_head=False,
        tie_word_embeddings=False,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=156892,
        eos_token_id=156892,
        # Some LLaDA tokenizers expose mask_token_id; keep optional here.
        mask_token_id=156895,
        # MoE configs
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        use_linear_attention=False,
        use_block_softmax_hybrid=False,
        block_size=32,
        feature_dim=64,
        linear_attention_layers=None,
        **kwargs,
    ):
        self.num_hidden_layers = int(num_hidden_layers)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = intermediate_size
        self.num_attention_heads = int(num_attention_heads)

        # FIX: if num_key_value_heads is 0/None, treat as "same as num_attention_heads"
        if num_key_value_heads is None or int(num_key_value_heads) <= 0:
            self.num_key_value_heads = int(num_attention_heads)
        else:
            self.num_key_value_heads = int(num_key_value_heads)

        self.hidden_act = hidden_act
        self.use_qkv_bias = bool(use_qkv_bias)
        self.use_qk_norm = bool(use_qk_norm)
        self.use_bias = bool(use_bias)
        self.norm_head = bool(norm_head)
        self.rms_norm_eps = float(rms_norm_eps)
        self.embedding_dropout = float(embedding_dropout)
        self.attention_dropout = float(attention_dropout)
        self.output_dropout = float(output_dropout)
        self.initializer_range = float(initializer_range)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.use_cache = bool(use_cache)
        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window)
        self.max_window_layers = int(max_window_layers)
        self.rope_scaling = rope_scaling

        self.eos_token_id = int(eos_token_id)
        self.mask_token_id = int(mask_token_id)

        self.head_dim = int(head_dim) if head_dim is not None else (self.hidden_size // self.num_attention_heads)

        # MoE configs
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = int(first_k_dense_replace)
        self.output_router_logits = bool(output_router_logits)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.partial_rotary_factor = float(partial_rotary_factor)

        self.use_linear_attention = bool(use_linear_attention)
        self.use_block_softmax_hybrid = bool(use_block_softmax_hybrid)
        self.block_size = int(block_size)
        self.feature_dim = int(feature_dim)
        self.linear_attention_layers = linear_attention_layers

        super().__init__(
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
            tie_word_embeddings=bool(tie_word_embeddings),
            **kwargs,
        )