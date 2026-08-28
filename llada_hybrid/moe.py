"""Mixture of Experts layers for LLaDA2 MoE."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from .configuration_llada2_moe import LLaDA2MoeConfig


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(self, scores: torch.Tensor):
        num_tokens, _ = scores.size()
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)
        return probs, top_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = torch.sigmoid(logits.float()).type_as(logits)
        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)
        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
        topk_weight = (
            scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        )
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight, logits


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [LLaDA2MoeMLP(config=config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]
        )
        self.gate = LLaDA2MoeGate(config)
        if config.num_shared_experts is not None and config.num_shared_experts > 0:
            self.shared_experts = LLaDA2MoeMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states_rep = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states_rep)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(hidden_states_rep[mask])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)

        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens_tensor in enumerate(tokens_per_expert):
            num_tokens = int(num_tokens_tensor.item())
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert_out = self.experts[i](sorted_tokens[start_idx:end_idx])
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
