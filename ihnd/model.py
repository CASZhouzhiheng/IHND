from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def grouped_mean(values, groups, count):
    result = values.new_zeros((count, values.shape[-1]))
    result.index_add_(0, groups, values)
    sizes = torch.bincount(groups, minlength=count).to(values.dtype).clamp_min(1)
    return result / sizes[:, None]


def edge_mean(z, nodes, edges, count, chunk_size):
    result = z.new_zeros((count, z.shape[-1]))
    for start in range(0, len(nodes), chunk_size):
        end = start + chunk_size
        result.index_add_(0, edges[start:end], z[nodes[start:end]])
    cardinality = torch.bincount(edges, minlength=count).to(z.dtype).clamp_min(1)
    return result / cardinality[:, None]


class HypergraphDiffusion(nn.Module):
    def __init__(self, hidden_dim, tau=1.0, incidence_chunk_size=32768):
        super().__init__()
        self.tau = float(tau)
        self.chunk_size = int(incidence_chunk_size)
        self.attention = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1), nn.LeakyReLU(0.2))

    def forward(self, z, nodes, edges, weights):
        if len(nodes) == 0:
            return z
        active = weights[edges] > 0
        nodes, edges = nodes[active], edges[active]
        if len(nodes) == 0:
            return z
        edge_count = len(weights)
        degree = z.new_zeros(len(z))
        degree.index_add_(0, nodes, weights[edges])
        inverse = torch.where(degree > 0, degree.clamp_min(1e-12).rsqrt(), torch.zeros_like(degree))
        normalized = z * inverse[:, None]
        mean_z = edge_mean(z, nodes, edges, edge_count, self.chunk_size)
        mean_normalized = edge_mean(normalized, nodes, edges, edge_count, self.chunk_size)
        scores = []
        for start in range(0, len(nodes), self.chunk_size):
            n, e = nodes[start:start+self.chunk_size], edges[start:start+self.chunk_size]
            scores.append(self.attention(torch.cat((z[n], mean_z[e]), dim=-1)).squeeze(-1))
        scores = torch.cat(scores)
        maximum = scores.new_full((len(z),), -torch.inf)
        maximum.scatter_reduce_(0, nodes, scores.detach(), reduce="amax", include_self=True)
        numerator = torch.exp(scores - maximum[nodes])
        denominator = scores.new_zeros(len(z))
        denominator.index_add_(0, nodes, numerator)
        coefficient = numerator / denominator[nodes].clamp_min(1e-12)
        flux_mean = z.new_zeros((edge_count, z.shape[-1]))
        for start in range(0, len(nodes), self.chunk_size):
            stop = start + self.chunk_size
            n, e = nodes[start:stop], edges[start:stop]
            flux = coefficient[start:stop, None] * (normalized[n]-mean_normalized[e])
            flux_mean.index_add_(0, e, flux)
        cardinality = torch.bincount(edges, minlength=edge_count).to(z.dtype).clamp_min(1)
        flux_mean = flux_mean / cardinality[:, None]
        divergence = torch.zeros_like(z)
        for start in range(0, len(nodes), self.chunk_size):
            stop = start + self.chunk_size
            n, e = nodes[start:stop], edges[start:stop]
            flux = coefficient[start:stop, None] * (normalized[n]-mean_normalized[e])
            divergence.index_add_(0, n, flux-flux_mean[e])
        return z - self.tau * inverse[:, None] * divergence


class HNDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, tau, incidence_chunk_size, gradient_checkpointing):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([HypergraphDiffusion(hidden_dim, tau, incidence_chunk_size) for _ in range(num_layers)])
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, x, nodes, edges, weights):
        z = self.projection(x)
        for layer in self.layers:
            if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:
                z = checkpoint(layer, z, nodes, edges, weights, use_reentrant=False)
            else:
                z = layer(z, nodes, edges, weights)
        return z


class IHND(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, tau=1.0, temperature=1.0, num_classes=2, incidence_chunk_size=32768, gradient_checkpointing=True):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_layers < 1 or temperature <= 0 or tau < 0 or num_classes != 2 or incidence_chunk_size < 1:
            raise ValueError("Invalid IHND configuration")
        self.temperature = float(temperature)
        self.chunk_size = int(incidence_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder = HNDEncoder(input_dim, hidden_dim, num_layers, tau, incidence_chunk_size, gradient_checkpointing)
        self.rationale = nn.Sequential(nn.Linear(hidden_dim+1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.invariant_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        self.shortcut_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def rationale_logits(self, z, nodes, edges, weights):
        if len(weights) == 0:
            return z.new_empty(0)
        representations = edge_mean(z, nodes, edges, len(weights), self.chunk_size)
        output = []
        for start in range(0, len(weights), self.chunk_size):
            stop = start + self.chunk_size
            output.append(self.rationale(torch.cat((representations[start:stop], weights[start:stop, None]), dim=-1)).squeeze(-1))
        return torch.cat(output)

    def forward(self, batch):
        z = self.encoder(batch.x, batch.node_index, batch.edge_index, batch.weights)
        if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:
            logits = checkpoint(self.rationale_logits, z, batch.node_index, batch.edge_index, batch.weights, use_reentrant=False)
        else:
            logits = self.rationale_logits(z, batch.node_index, batch.edge_index, batch.weights)
        probability = torch.sigmoid(logits)
        if self.training:
            uniform = torch.rand_like(logits).clamp(1e-6, 1-1e-6)
            mask = torch.sigmoid((logits + torch.log(uniform)-torch.log1p(-uniform))/self.temperature)
        else:
            mask = probability
        invariant_weights = batch.weights * mask
        invariant_nodes = self.encoder(batch.x, batch.node_index, batch.edge_index, invariant_weights)
        invariant = grouped_mean(invariant_nodes, batch.node_graph, batch.size)
        output = {"logits": self.invariant_head(invariant), "invariant": invariant, "rationale_probability": probability, "mask": mask, "invariant_weights": invariant_weights}
        if self.training:
            with torch.no_grad():
                shortcut_nodes = self.encoder(batch.x, batch.node_index, batch.edge_index, batch.weights*(1-mask))
                shortcut = grouped_mean(shortcut_nodes, batch.node_graph, batch.size)
            output["shortcut_logits"] = self.shortcut_head(shortcut.detach())
            output["shortcut"] = shortcut.detach()
        return output
