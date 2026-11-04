from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def intervention_indices(sites, count):
    batch_size = len(sites)
    if batch_size == 0 or count < 1:
        raise ValueError("An intervention requires a nonempty batch and a positive count")
    different = sites[:, None] != sites[None, :]
    other = ~torch.eye(batch_size, device=sites.device, dtype=torch.bool)
    allowed = torch.where(different.any(dim=1, keepdim=True), different, other)
    allowed = torch.where(allowed.any(dim=1, keepdim=True), allowed, torch.ones_like(allowed))
    return torch.multinomial(allowed.float(), count, replacement=True).T


class IHNDLoss(nn.Module):
    def __init__(self, interventions=4, alpha=1.0, beta=0.05, lambda_var=1.0):
        super().__init__()
        if interventions < 1 or min(alpha, beta, lambda_var) < 0:
            raise ValueError("Invalid objective coefficients")
        self.interventions = int(interventions)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_var = float(lambda_var)

    def forward(self, output, batch):
        donors = intervention_indices(batch.sites, self.interventions)
        gates = torch.sigmoid(output["shortcut_logits"].detach()[donors])
        intervention_logits = output["logits"][None] * gates
        losses = F.cross_entropy(intervention_logits.reshape(-1, 2), batch.labels.repeat(self.interventions), reduction="none").reshape(self.interventions, batch.size)
        risks = losses.mean(dim=1)
        risk_variance = risks.var(unbiased=False)
        invariant = risks.mean() + self.lambda_var*risk_variance
        shortcut = F.cross_entropy(output["shortcut_logits"], batch.labels)
        probability = output["rationale_probability"]
        sparsity = output["logits"].new_zeros(batch.size)
        sparsity.index_add_(0, batch.edge_graph, probability)
        sizes = torch.bincount(batch.edge_graph, minlength=batch.size).to(sparsity.dtype).clamp_min(1)
        sparsity = (sparsity/sizes).mean()
        total = invariant + self.alpha*shortcut + self.beta*sparsity
        return {"loss": total, "invariant_risk": risks.mean(), "risk_variance": risk_variance, "shortcut_loss": shortcut, "sparsity": sparsity, "donors": donors}
