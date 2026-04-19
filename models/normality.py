import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalPrototypeBank(nn.Module):
    def __init__(self, input_dim, num_experts, margin_global=1.0, margin_expert=1.0, eps=1e-9):
        super(NormalPrototypeBank, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.margin_global = float(margin_global)
        self.margin_expert = float(margin_expert)
        self.eps = float(eps)

        self.global_prototype = nn.Parameter(torch.empty(input_dim))
        self.expert_prototypes = nn.Parameter(torch.empty(num_experts, input_dim))
        self._last_metrics = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.global_prototype, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_prototypes, mean=0.0, std=0.02)

    def _squared_distance(self, inputs, prototypes):
        diff = inputs - prototypes
        # Normalize by feature dimension to keep router and proto-logit scales stable.
        return diff.pow(2).mean(dim=-1)

    def global_distance(self, hiddens):
        return self._squared_distance(hiddens, self.global_prototype.unsqueeze(0))

    def expert_distance(self, expert_hiddens):
        return self._squared_distance(expert_hiddens, self.expert_prototypes.unsqueeze(0))

    def separation_loss(self):
        if self.num_experts < 2:
            return self.expert_prototypes.new_zeros(())

        normalized = F.normalize(self.expert_prototypes, p=2, dim=-1, eps=self.eps)
        similarity = torch.matmul(normalized, normalized.transpose(0, 1))
        pair_indices = torch.triu_indices(self.num_experts, self.num_experts, offset=1)
        pairwise_similarity = similarity[pair_indices[0], pair_indices[1]]
        if pairwise_similarity.numel() == 0:
            return similarity.new_zeros(())
        return pairwise_similarity.pow(2).mean()

    def _masked_mean(self, values, mask):
        if mask is None or not torch.any(mask):
            return values.new_zeros(())
        return values[mask].mean()

    def compute_loss(self, base_repr, expert_repr, routing_probs, targets, anomaly_id, normal_only=False):
        if targets.dim() > 1:
            label_ids = targets.argmax(dim=-1)
        else:
            label_ids = targets
        label_ids = label_ids.long()

        normal_mask = label_ids != anomaly_id
        anomaly_mask = label_ids == anomaly_id
        if normal_only:
            anomaly_mask = torch.zeros_like(normal_mask)

        global_distance_sq = self.global_distance(base_repr)
        expert_distance_sq = self.expert_distance(expert_repr)
        global_distance = torch.sqrt(global_distance_sq + self.eps)
        expert_distance = torch.sqrt(expert_distance_sq + self.eps)
        weighted_expert_distance_sq = (routing_probs * expert_distance_sq).sum(dim=-1)
        weighted_expert_distance = (routing_probs * expert_distance).sum(dim=-1)

        pull_loss = base_repr.new_zeros(())
        if torch.any(normal_mask):
            pull_loss = (
                global_distance_sq[normal_mask].mean() +
                weighted_expert_distance_sq[normal_mask].mean()
            )

        push_loss = base_repr.new_zeros(())
        margin_violation = base_repr.new_zeros(())
        if torch.any(anomaly_mask):
            global_push = F.relu(self.margin_global - global_distance[anomaly_mask]).pow(2).mean()
            expert_push = (
                routing_probs[anomaly_mask] *
                F.relu(self.margin_expert - expert_distance[anomaly_mask]).pow(2)
            ).sum(dim=-1).mean()
            push_loss = global_push + expert_push

            global_violation = (global_distance[anomaly_mask] < self.margin_global).float().mean()
            expert_violation = (
                routing_probs[anomaly_mask] *
                (expert_distance[anomaly_mask] < self.margin_expert).float()
            ).sum(dim=-1).mean()
            margin_violation = 0.5 * (global_violation + expert_violation)

        sep_loss = self.separation_loss()
        self._last_metrics = {
            'proto_pull_loss': pull_loss.detach(),
            'proto_push_loss': push_loss.detach(),
            'proto_sep_loss': sep_loss.detach(),
            'proto_global_normal_dist': self._masked_mean(global_distance.detach(), normal_mask),
            'proto_global_anomaly_dist': self._masked_mean(global_distance.detach(), label_ids == anomaly_id),
            'proto_expert_normal_dist': self._masked_mean(weighted_expert_distance.detach(), normal_mask),
            'proto_expert_anomaly_dist': self._masked_mean(
                weighted_expert_distance.detach(),
                label_ids == anomaly_id,
            ),
            'proto_margin_violation': margin_violation.detach(),
        }
        return pull_loss + push_loss

    def get_metrics(self):
        return self._last_metrics
