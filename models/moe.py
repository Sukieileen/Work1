import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.normality import NormalPrototypeBank


class LatentMoEClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=2,
        num_experts=4,
        top_k=2,
        bottleneck_dim=None,
        temperature=1.5,
        gate_dropout=0.1,
        balance_loss_weight=1e-2,
        diversity_loss_weight=1e-3,
        z_loss_weight=0.0,
        use_normality_anchor=True,
        prototype_scale=1.0,
        prototype_margin_global=1.0,
        prototype_margin_expert=1.0,
        router_use_distance=True,
    ):
        super(LatentMoEClassifier, self).__init__()
        if num_experts < 1:
            raise ValueError('num_experts must be positive.')
        if top_k < 1:
            raise ValueError('top_k must be positive.')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else max(input_dim // 4, 1)
        self.temperature = max(float(temperature), 1e-6)
        self.balance_loss_weight = balance_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.z_loss_weight = z_loss_weight
        self.use_normality_anchor = bool(use_normality_anchor)
        self.prototype_scale = float(prototype_scale)
        self.router_use_distance = bool(router_use_distance and self.use_normality_anchor)
        self.eps = 1e-9

        self.input_norm = nn.LayerNorm(input_dim)
        self.router_feature_dim = input_dim + (2 if self.router_use_distance else 0)
        self.router_feature_norm = nn.LayerNorm(self.router_feature_dim)
        self.router_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity()
        self.router = nn.Linear(self.router_feature_dim, num_experts)
        self.down_projs = nn.ModuleList([
            nn.Linear(input_dim, self.bottleneck_dim) for _ in range(num_experts)
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(self.bottleneck_dim, input_dim) for _ in range(num_experts)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        self.prototype_bank = (
            NormalPrototypeBank(
                input_dim=input_dim,
                num_experts=num_experts,
                margin_global=prototype_margin_global,
                margin_expert=prototype_margin_expert,
                eps=self.eps,
            )
            if self.use_normality_anchor else None
        )

        self._last_auxiliary_loss = None
        self._last_metrics = {}
        self._last_cache = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        for module in list(self.down_projs) + list(self.up_projs) + list(self.heads):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        if self.prototype_bank is not None:
            self.prototype_bank.reset_parameters()

    def _build_router_features(self, inputs, normalized_inputs, global_distance):
        if not self.router_use_distance:
            return normalized_inputs
        feature_norm = torch.norm(inputs, p=2, dim=-1, keepdim=True) / math.sqrt(float(self.input_dim))
        return torch.cat([normalized_inputs, global_distance.unsqueeze(-1), feature_norm], dim=-1)

    def _select_from_cache(self, key, batch_slice=None):
        cached_value = self._last_cache.get(key)
        if cached_value is None:
            return None
        if batch_slice is None:
            return cached_value
        return cached_value[batch_slice]

    def forward(self, inputs):
        normalized_inputs = self.input_norm(inputs)
        if self.prototype_bank is not None:
            global_distance = self.prototype_bank.global_distance(inputs)
        else:
            global_distance = inputs.new_zeros(inputs.size(0))

        router_features = self._build_router_features(inputs, normalized_inputs, global_distance)
        router_inputs = self.router_dropout(self.router_feature_norm(router_features))
        router_logits = self.router(router_inputs)
        routing_probs = F.softmax(router_logits / self.temperature, dim=-1)

        if self.top_k == self.num_experts:
            routing_mask = torch.ones_like(routing_probs)
        else:
            topk_indices = torch.topk(routing_probs, k=self.top_k, dim=-1).indices
            routing_mask = torch.zeros_like(routing_probs)
            routing_mask.scatter_(1, topk_indices, 1.0)

        sparse_probs = routing_probs * routing_mask
        sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + self.eps)

        expert_logits = []
        expert_deltas = []
        expert_representations = []
        for down_proj, up_proj, head in zip(self.down_projs, self.up_projs, self.heads):
            hidden = F.gelu(down_proj(normalized_inputs))
            delta = up_proj(hidden)
            expert_representation = inputs + delta
            expert_logits.append(head(expert_representation))
            expert_deltas.append(delta)
            expert_representations.append(expert_representation)

        expert_logits = torch.stack(expert_logits, dim=1)
        expert_deltas = torch.stack(expert_deltas, dim=1)
        expert_representations = torch.stack(expert_representations, dim=1)
        if self.prototype_bank is not None:
            expert_distances = self.prototype_bank.expert_distance(expert_representations)
            proto_logits = self.prototype_scale * torch.stack([-expert_distances, expert_distances], dim=-1)
            expert_logits = expert_logits + proto_logits
        else:
            expert_distances = expert_logits.new_zeros(expert_logits.size(0), self.num_experts)
        final_logits = torch.sum(sparse_probs.unsqueeze(-1) * expert_logits, dim=1)

        balance_loss = self._compute_balance_loss(routing_probs, routing_mask)
        diversity_loss = self._compute_diversity_loss(expert_deltas, sparse_probs)
        z_loss = router_logits.logsumexp(dim=-1).pow(2).mean()
        auxiliary_loss = (
            self.balance_loss_weight * balance_loss +
            self.diversity_loss_weight * diversity_loss +
            self.z_loss_weight * z_loss
        )

        self._last_auxiliary_loss = auxiliary_loss
        self._last_metrics = {
            'moe_balance_loss': balance_loss.detach(),
            'moe_diversity_loss': diversity_loss.detach(),
            'moe_z_loss': z_loss.detach(),
            'moe_aux_loss': auxiliary_loss.detach(),
            'moe_router_entropy': (
                -(routing_probs * torch.log(routing_probs + self.eps)).sum(dim=-1).mean()
            ).detach(),
            'moe_importance_std': routing_probs.mean(dim=0).std(unbiased=False).detach(),
            'moe_load_std': (
                routing_mask.sum(dim=0) / (routing_mask.sum() + self.eps)
            ).std(unbiased=False).detach(),
        }
        self._last_cache = {
            'base_repr': inputs,
            'normalized_inputs': normalized_inputs,
            'expert_repr': expert_representations,
            'routing_probs': sparse_probs,
            'routing_mask': routing_mask,
            'global_distance': global_distance,
            'expert_distance': expert_distances,
        }
        return final_logits

    def _compute_balance_loss(self, routing_probs, routing_mask):
        importance = routing_probs.mean(dim=0)
        load = routing_mask.sum(dim=0) / (routing_mask.sum() + self.eps)
        importance_loss = self.num_experts * importance.pow(2).sum() - 1.0
        load_loss = self.num_experts * load.pow(2).sum() - 1.0
        return 0.5 * (importance_loss + load_loss)

    def _compute_diversity_loss(self, expert_deltas, sparse_probs):
        if self.num_experts < 2:
            return expert_deltas.new_zeros(())

        normalized_deltas = F.normalize(expert_deltas, p=2, dim=-1, eps=self.eps)
        diversity_loss = expert_deltas.new_zeros(())
        pair_count = 0
        for first_expert in range(self.num_experts):
            for second_expert in range(first_expert + 1, self.num_experts):
                similarity = (
                    normalized_deltas[:, first_expert, :] * normalized_deltas[:, second_expert, :]
                ).sum(dim=-1)
                co_activation = sparse_probs[:, first_expert] * sparse_probs[:, second_expert]
                diversity_loss = diversity_loss + (co_activation * similarity.pow(2)).mean()
                pair_count += 1

        return diversity_loss / max(pair_count, 1)

    def get_auxiliary_loss(self):
        if self._last_auxiliary_loss is None:
            return self.router.weight.new_zeros(())
        return self._last_auxiliary_loss

    def get_metrics(self):
        return self._last_metrics

    def get_prototype_loss(self, targets, anomaly_id, batch_slice=None, normal_only=False):
        if self.prototype_bank is None:
            return self.router.weight.new_zeros(())

        base_repr = self._select_from_cache('base_repr', batch_slice=batch_slice)
        expert_repr = self._select_from_cache('expert_repr', batch_slice=batch_slice)
        routing_probs = self._select_from_cache('routing_probs', batch_slice=batch_slice)
        if base_repr is None or expert_repr is None or routing_probs is None:
            return self.router.weight.new_zeros(())

        return self.prototype_bank.compute_loss(
            base_repr,
            expert_repr,
            routing_probs,
            targets,
            anomaly_id,
            normal_only=normal_only,
        )

    def get_prototype_separation_loss(self):
        if self.prototype_bank is None:
            return self.router.weight.new_zeros(())
        return self.prototype_bank.separation_loss()

    def get_prototype_metrics(self):
        if self.prototype_bank is None:
            return {}
        return self.prototype_bank.get_metrics()
