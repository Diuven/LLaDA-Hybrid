# llada_fast/modeling/modeling_llada2_moe.py
#
# Facade module: imports submodules and re-exports all public classes.
# HuggingFace auto_map points here, so all class names must remain importable.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_llada2_moe import LLaDA2MoeConfig

# ── Re-exports from submodules ────────────────────────────────────────────────
from .norm import LLaDA2MoeRMSNorm                              # noqa: F401
from .rotary import LLaDA2MoeRotaryEmbedding, apply_rotary_pos_emb  # noqa: F401
from .moe import LLaDA2MoeMLP, LLaDA2MoeGate, LLaDA2MoeSparseMoeBlock  # noqa: F401
from .attention import (                                          # noqa: F401
    LLaDA2MoeAttention,
    repeat_kv,
    eager_attention_forward,
)
from .decoder import LLaDA2MoeDecoderLayer                       # noqa: F401

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LLaDA2MoeConfig"

LLADA2MOE_START_DOCSTRING = r"""LLaDA2 MoE model."""
LLADA2MOE_INPUTS_DOCSTRING = r"""
    Inputs:
      - input_ids: (B, L)
      - attention_mask: MUST be (B, 1, L, L) block mask for this implementation.
"""


# ── PreTrainedModel base ─────────────────────────────────────────────────────


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ── Core transformer model ───────────────────────────────────────────────────


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LLaDA2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flex_attention = config._attn_implementation == "flex_attention"
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        half_len: Optional[int] = None,
        prefix_cache=None,
        cache_mode: Optional[str] = None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting use_cache=False.")
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        block_attention_mask = attention_mask
        if block_attention_mask is not None:
            block_attention_mask = block_attention_mask.detach()

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)

            if attention_mask.size() == (batch_size, 1, seq_length, seq_length):
                attention_mask = torch.where(
                    attention_mask.bool(),
                    torch.zeros_like(attention_mask),
                    torch.full_like(attention_mask, float("-inf")),
                )
            else:
                raise ValueError(
                    f"LLaDA2 only supports 4D block attention masks of shape {(batch_size,1,seq_length,seq_length)}; got {attention_mask.size()=}."
                )

            if key_padding_mask is not None:
                kpm_bool = key_padding_mask if key_padding_mask.dtype == torch.bool else key_padding_mask.bool()
                attention_mask = attention_mask.masked_fill(
                    ~kpm_bool[:, None, None, :], float("-inf")
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                    key_padding_mask,
                    block_attention_mask,
                    half_len,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    key_padding_mask=key_padding_mask,
                    block_attention_mask=block_attention_mask,
                    half_len=half_len,
                    prefix_cache=prefix_cache,
                    cache_mode=cache_mode,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits] if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


# ── Language model head ──────────────────────────────────────────────────────


class LLaDA2MoeModelLM(LLaDA2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        prefix_cache=None,
        cache_mode: Optional[str] = None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""Inference-only forward. This release ships no training path, so no
        loss is computed and ``loss`` in the returned object is always ``None``.

        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            key_padding_mask=key_padding_mask,
            prefix_cache=prefix_cache,
            cache_mode=cache_mode,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        aux_loss = None

        if not return_dict:
            return (logits,) + outputs[1:]

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {"past_key_values": past_key_values, "use_cache": kwargs.get("use_cache", False), "attention_mask": attention_mask}
        )
        if "position_ids" in kwargs:
            model_inputs["position_ids"] = kwargs["position_ids"]
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = True,
        minimal_topk: int = 1,
        threshold: float = 0.7,
        editing_threshold: float = 0.5,
        max_post_steps: int = 16,
        eos_id: Optional[int] = None,
        mask_id: Optional[int] = None,
        num_to_transfer: int = 1,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool = False,
    ):
        if use_kv_cache:
            return self._generate_with_kv_cache(
                inputs=inputs, temperature=temperature,
                block_length=block_length, steps=steps,
                gen_length=gen_length, top_p=top_p, top_k=top_k,
                eos_early_stop=eos_early_stop, minimal_topk=minimal_topk,
                threshold=threshold, editing_threshold=editing_threshold,
                max_post_steps=max_post_steps, eos_id=eos_id, mask_id=mask_id,
                num_to_transfer=num_to_transfer,
                repetition_penalty=repetition_penalty,
            )

        steps = min(int(steps), int(gen_length) // int(minimal_topk))
        input_ids = inputs.to(self.device)
        batch_size = input_ids.shape[0]

        if eos_id is None:
            eos_id = getattr(self.config, "eos_token_id", None)
        if mask_id is None:
            mask_id = getattr(self.config, "mask_token_id", 156895)

        if eos_id is not None:
            eos_id = int(eos_id)
        if mask_id is not None:
            mask_id = int(mask_id)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(self.dtype)

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        prefill_blocks = prompt_length // block_length

        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end].expand(
                batch_size, -1, -1, -1
            )
            cur_position_ids = position_ids[:, :current_window_end]

            block_start_pos = num_block * block_length

            post_steps = 0
            step_count = 0
            while True:
                if step_count >= steps:
                    break
                step_count += 1
                old_block_tokens = cur_x[:, -block_length:].clone()
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if not torch.any(active_block_mask):
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                prompt_mask_in_block = torch.zeros(block_length, dtype=torch.bool, device=self.device)
                if block_start_pos < prompt_length:
                    prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                    prompt_mask_in_block[:prompt_end_in_block] = True

                outputs = self.forward(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                logits = outputs.logits

                active_logits = logits[:, -block_length:, :]

                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        past_tokens = x[b, :current_window_end]
                        unique_tokens = past_tokens[past_tokens != mask_id].unique()
                        if len(unique_tokens) > 0:
                            u_toks = unique_tokens.unsqueeze(0).expand(block_length, -1)
                            scores = torch.gather(active_logits[b], 1, u_toks)
                            scores = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
                            active_logits[b].scatter_(1, u_toks, scores)

                x0, x0_p = self._sample_with_temperature_topk_topp(active_logits, temperature=temperature, top_k=top_k, top_p=top_p)

                mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if active_block_mask.sum() > 0:
                    mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    for b in range(batch_size):
                        ab = active_block_mask[b]
                        if ab.sum() == 0:
                            continue
                        conf = mask_confidence[b]
                        high_conf = (conf > threshold) & ab
                        if int(high_conf.sum()) >= num_to_transfer:
                            mask_transfer_index[b] = high_conf
                        else:
                            num_available = int(ab.sum().item())
                            k = min(int(num_to_transfer), num_available)
                            _, idx = torch.topk(conf, k=k)
                            mask_transfer_index[b, idx] = True

                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                non_mask_positions = ~active_block_mask
                non_prompt_positions = ~prompt_mask_in_block
                editable_positions = non_mask_positions & non_prompt_positions.unsqueeze(0).expand(batch_size, -1)
                editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
                token_changed = x0 != old_block_tokens

                for b in range(batch_size):
                    high_conf_edit = (editing_confidence[b] > editing_threshold) & editable_positions[b] & token_changed[b]
                    editing_transfer_index[b] = high_conf_edit

                final_transfer_index = mask_transfer_index | editing_transfer_index
                if final_transfer_index.any():
                    cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]

                if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                    break

            x[:, :current_window_end] = cur_x
            print(f"  [block {num_block - prefill_blocks}/{num_blocks - prefill_blocks}] denoising_steps={step_count}")

            if eos_early_stop and eos_id is not None:
                generated_part = x[:, prompt_length:current_window_end]
                if (generated_part == mask_id).sum() == 0:
                    if (generated_part == eos_id).any():
                        break

        generated_answer = x[:, : prompt_length + gen_length]

        if not eos_early_stop:
            return generated_answer[:, prompt_length:]

        if eos_id is None:
            return generated_answer[:, prompt_length:]

        trimmed = generated_answer[:, prompt_length:].clone()
        for b in range(batch_size):
            gen = trimmed[b]
            eos_pos = (gen == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                cut = int(eos_pos[0].item()) + 1
                if cut < gen_length:
                    trimmed[b, cut:] = eos_id
        return trimmed

    def _init_prefix_cache(self, batch_size: int):
        """Create a BlockDiffusionCache and populate layer types."""
        from .cache import BlockDiffusionCache

        num_layers = self.config.num_hidden_layers
        cache = BlockDiffusionCache(num_layers, self.device)

        for li, layer in enumerate(self.model.layers):
            attn = layer.attention
            if getattr(attn, "is_linear_active", False) and hasattr(attn, "linear_attention"):
                cache.layer_type[li] = "linear"
                lin = attn.linear_attention
                feature_dim_2x = 2 * int(lin.feature_dim)
                head_dim = int(attn.head_dim)
                num_heads = int(attn.num_heads)
                cache.init_recurrent_state(li, batch_size, num_heads, feature_dim_2x, head_dim)
            else:
                cache.layer_type[li] = "softmax"

        return cache

    def _commit_block_to_cache(
        self,
        block_tokens: torch.Tensor,
        block_position_ids: torch.Tensor,
        prefix_cache,
    ):
        """Run a clean forward pass on finalized block tokens and update cache."""
        self.forward(
            input_ids=block_tokens,
            position_ids=block_position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            prefix_cache=prefix_cache,
            cache_mode="commit",
        )

    @torch.no_grad()
    def _generate_with_kv_cache(
        self,
        inputs,
        temperature,
        block_length,
        steps,
        gen_length,
        top_p,
        top_k,
        eos_early_stop,
        minimal_topk,
        threshold,
        editing_threshold,
        max_post_steps,
        eos_id,
        mask_id,
        num_to_transfer,
        repetition_penalty,
    ):
        steps = min(int(steps), int(gen_length) // int(minimal_topk))
        input_ids = inputs.to(self.device)
        batch_size = input_ids.shape[0]

        if eos_id is None:
            eos_id = getattr(self.config, "eos_token_id", None)
        if mask_id is None:
            mask_id = getattr(self.config, "mask_token_id", 156895)
        if eos_id is not None:
            eos_id = int(eos_id)
        if mask_id is not None:
            mask_id = int(mask_id)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        # Initialize prefix cache
        prefix_cache = self._init_prefix_cache(batch_size)

        # Prefill: commit full prompt blocks into cache
        prefill_blocks = prompt_length // block_length
        for b in range(prefill_blocks):
            s = b * block_length
            e = s + block_length
            self._commit_block_to_cache(
                block_tokens=x[:, s:e],
                block_position_ids=position_ids[:, s:e],
                prefix_cache=prefix_cache,
            )

        # Generate block by block
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]

            block_start_pos = num_block * block_length
            post_steps = 0
            step_count = 0

            while True:
                if step_count >= steps:
                    break
                step_count += 1
                old_block_tokens = cur_x[:, -block_length:].clone()
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if not torch.any(active_block_mask):
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                prompt_mask_in_block = torch.zeros(block_length, dtype=torch.bool, device=self.device)
                if block_start_pos < prompt_length:
                    prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                    prompt_mask_in_block[:prompt_end_in_block] = True

                # Forward pass on current block only, using cached prefix
                block_tokens = cur_x[:, -block_length:]
                block_pos = position_ids[:, block_start_pos:current_window_end]

                outputs = self.forward(
                    input_ids=block_tokens,
                    position_ids=block_pos,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    prefix_cache=prefix_cache,
                    cache_mode="read",
                )
                active_logits = outputs.logits

                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        past_tokens = cur_x[b]
                        unique_tokens = past_tokens[past_tokens != mask_id].unique()
                        if len(unique_tokens) > 0:
                            u_toks = unique_tokens.unsqueeze(0).expand(block_length, -1)
                            scores = torch.gather(active_logits[b], 1, u_toks)
                            scores = torch.where(
                                scores > 0,
                                scores / repetition_penalty,
                                scores * repetition_penalty,
                            )
                            active_logits[b].scatter_(1, u_toks, scores)

                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p,
                )

                mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if active_block_mask.sum() > 0:
                    mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    for b in range(batch_size):
                        ab = active_block_mask[b]
                        if ab.sum() == 0:
                            continue
                        conf = mask_confidence[b]
                        high_conf = (conf > threshold) & ab
                        if int(high_conf.sum()) >= num_to_transfer:
                            mask_transfer_index[b] = high_conf
                        else:
                            num_available = int(ab.sum().item())
                            k = min(int(num_to_transfer), num_available)
                            _, idx = torch.topk(conf, k=k)
                            mask_transfer_index[b, idx] = True

                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                non_mask_positions = ~active_block_mask
                non_prompt_positions = ~prompt_mask_in_block
                editable_positions = non_mask_positions & non_prompt_positions.unsqueeze(0).expand(batch_size, -1)
                editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
                token_changed = x0 != old_block_tokens

                for b in range(batch_size):
                    high_conf_edit = (
                        (editing_confidence[b] > editing_threshold)
                        & editable_positions[b]
                        & token_changed[b]
                    )
                    editing_transfer_index[b] = high_conf_edit

                final_transfer_index = mask_transfer_index | editing_transfer_index
                if final_transfer_index.any():
                    cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]

                if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                    break

            # Write back and commit finalized block into cache
            x[:, :current_window_end] = cur_x
            print(f"  [block {num_block - prefill_blocks}/{num_blocks - prefill_blocks}] denoising_steps={step_count}")
            self._commit_block_to_cache(
                block_tokens=x[:, block_start_pos:current_window_end],
                block_position_ids=position_ids[:, block_start_pos:current_window_end],
                prefix_cache=prefix_cache,
            )

            if eos_early_stop and eos_id is not None:
                generated_part = x[:, prompt_length:current_window_end]
                if (generated_part == mask_id).sum() == 0:
                    if (generated_part == eos_id).any():
                        break

        generated_answer = x[:, :prompt_length + gen_length]

        if not eos_early_stop:
            return generated_answer[:, prompt_length:]

        if eos_id is None:
            return generated_answer[:, prompt_length:]

        trimmed = generated_answer[:, prompt_length:].clone()
        for b in range(batch_size):
            gen = trimmed[b]
            eos_pos = (gen == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                cut = int(eos_pos[0].item()) + 1
                if cut < gen_length:
                    trimmed[b, cut:] = eos_id
        return trimmed

    @staticmethod
    def _top_k_logits(logits, k):
        if k is None or k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_logits(logits, p):
        if p is None or p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)

        if temperature == 0.0:
            token = torch.argmax(logits, dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
            token_prob = torch.gather(probs, -1, token)
            return token.view(*orig_shape), token_prob.view(*orig_shape)

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)
