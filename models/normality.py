import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalPrototypeBank(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts,
        margin_global=1.0,
        margin_expert=1.0,
        use_global_prototype=False,
        diversity_margin=0.5,
        eps=1e-9,
    ):
        super(NormalPrototypeBank, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.margin_global = float(margin_global)
        self.margin_expert = float(margin_expert)
        self.use_global_prototype = bool(use_global_prototype)
        self.diversity_margin = float(diversity_margin)
        self.eps = float(eps)

        if self.use_global_prototype:
            self.global_prototype = nn.Parameter(torch.empty(input_dim))
        else:
            self.register_parameter('global_prototype', None)
        self.expert_prototypes = nn.Parameter(torch.empty(num_experts, input_dim))
        self._last_metrics = {}
        self.reset_parameters()

    def reset_parameters(self):
        if self.global_prototype is not None:
            nn.init.normal_(self.global_prototype, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_prototypes, mean=0.0, std=0.02)

    def _squared_distance(self, inputs, prototypes):
        diff = inputs - prototypes
        # Normalize by feature dimension to keep router and proto-logit scales stable.
        return diff.pow(2).mean(dim=-1)

    def global_distance(self, hiddens):
        if self.global_prototype is not None:
            return self._squared_distance(hiddens, self.global_prototype.unsqueeze(0))
        return self.expert_distance_from_base(hiddens).min(dim=-1).values

    def expert_distance_from_base(self, hiddens):
        return self._squared_distance(hiddens.unsqueeze(1), self.expert_prototypes.unsqueeze(0))

    def expert_distance_from_expert_repr(self, expert_hiddens):
        return self._squared_distance(expert_hiddens, self.expert_prototypes.unsqueeze(0))

    def expert_distance(self, expert_hiddens):
        return self.expert_distance_from_expert_repr(expert_hiddens)

    def separation_loss(self):
        if self.num_experts < 2 or self.diversity_margin <= 0:
            return self.expert_prototypes.new_zeros(())

        pair_indices = torch.triu_indices(self.num_experts, self.num_experts, offset=1)
        pairwise_distance = torch.cdist(self.expert_prototypes, self.expert_prototypes, p=2)
        pairwise_distance = pairwise_distance[pair_indices[0], pair_indices[1]]
        if pairwise_distance.numel() == 0:
            return self.expert_prototypes.new_zeros(())
        return F.relu(self.diversity_margin - pairwise_distance).pow(2).mean()

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

        base_expert_distance_sq = self.expert_distance_from_base(base_repr)
        expert_distance_sq = self.expert_distance_from_expert_repr(expert_repr)
        base_expert_distance = torch.sqrt(base_expert_distance_sq + self.eps)
        expert_distance = torch.sqrt(expert_distance_sq + self.eps)

        weighted_base_distance_sq = (routing_probs * base_expert_distance_sq).sum(dim=-1)
        weighted_expert_distance_sq = (routing_probs * expert_distance_sq).sum(dim=-1)
        weighted_base_distance = (routing_probs * base_expert_distance).sum(dim=-1)
        weighted_expert_distance = (routing_probs * expert_distance).sum(dim=-1)
        min_base_distance_sq, _ = base_expert_distance_sq.min(dim=-1)
        min_base_distance = torch.sqrt(min_base_distance_sq + self.eps)

        global_distance_sq = None
        global_distance = None
        if self.global_prototype is not None:
            global_distance_sq = self.global_distance(base_repr)
            global_distance = torch.sqrt(global_distance_sq + self.eps)

        pull_terms = []
        if torch.any(normal_mask):
            pull_terms.append(weighted_base_distance_sq[normal_mask].mean())
            pull_terms.append(weighted_expert_distance_sq[normal_mask].mean())
            if global_distance_sq is not None:
                pull_terms.append(global_distance_sq[normal_mask].mean())
        pull_loss = torch.stack(pull_terms).mean() if pull_terms else base_repr.new_zeros(())

        push_terms = []
        margin_violations = []
        if torch.any(anomaly_mask):
            base_push = (
                routing_probs[anomaly_mask] *
                F.relu(self.margin_expert - base_expert_distance[anomaly_mask]).pow(2)
            ).sum(dim=-1).mean()
            expert_push = (
                routing_probs[anomaly_mask] *
                F.relu(self.margin_expert - expert_distance[anomaly_mask]).pow(2)
            ).sum(dim=-1).mean()
            push_terms.extend([base_push, expert_push])

            base_violation = (
                routing_probs[anomaly_mask] *
                (base_expert_distance[anomaly_mask] < self.margin_expert).float()
            ).sum(dim=-1).mean()
            expert_violation = (
                routing_probs[anomaly_mask] *
                (expert_distance[anomaly_mask] < self.margin_expert).float()
            ).sum(dim=-1).mean()
            margin_violations.extend([base_violation, expert_violation])

            if global_distance is not None:
                push_terms.append(F.relu(self.margin_global - global_distance[anomaly_mask]).pow(2).mean())
                margin_violations.append((global_distance[anomaly_mask] < self.margin_global).float().mean())

        push_loss = torch.stack(push_terms).mean() if push_terms else base_repr.new_zeros(())
        margin_violation = (
            torch.stack(margin_violations).mean() if margin_violations else base_repr.new_zeros(())
        )

        sep_loss = self.separation_loss()
        metrics = {
            'proto_pull_loss': pull_loss.detach(),
            'proto_push_loss': push_loss.detach(),
            'proto_sep_loss': sep_loss.detach(),
            'proto_base_weighted_normal_dist': self._masked_mean(weighted_base_distance.detach(), normal_mask),
            'proto_base_weighted_anomaly_dist': self._masked_mean(
                weighted_base_distance.detach(),
                label_ids == anomaly_id,
            ),
            'proto_base_min_normal_dist': self._masked_mean(min_base_distance.detach(), normal_mask),
            'proto_base_min_anomaly_dist': self._masked_mean(min_base_distance.detach(), label_ids == anomaly_id),
            'proto_expert_normal_dist': self._masked_mean(weighted_expert_distance.detach(), normal_mask),
            'proto_expert_anomaly_dist': self._masked_mean(
                weighted_expert_distance.detach(),
                label_ids == anomaly_id,
            ),
            'proto_margin_violation': margin_violation.detach(),
            'proto_use_global': base_repr.new_tensor(1.0 if self.global_prototype is not None else 0.0),
        }
        if global_distance is not None:
            metrics.update({
                'proto_global_normal_dist': self._masked_mean(global_distance.detach(), normal_mask),
                'proto_global_anomaly_dist': self._masked_mean(global_distance.detach(), label_ids == anomaly_id),
            })
        self._last_metrics = metrics
        return pull_loss + push_loss

    def get_metrics(self):
        return self._last_metrics
