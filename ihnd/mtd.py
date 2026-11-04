from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch

from .data import load_timeseries, read_manifest, write_csv


def candidate_edges():
    return np.asarray([(a, b, -1) for a, b in itertools.combinations(range(90), 2)] + list(itertools.combinations(range(90), 3)), dtype=np.int32)


def normalized_derivatives(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 90 or values.shape[0] < 4 or not np.isfinite(values).all():
        raise ValueError("Expected finite time-by-90 ROI signals with at least four time points")
    derivative = np.diff(values, axis=0)
    standard_deviation = derivative.std(axis=0, ddof=0)
    if np.any(standard_deviation <= 0) or not np.isfinite(standard_deviation).all():
        raise ValueError("All 90 ROI derivative sequences must have finite positive variance")
    return np.ascontiguousarray((derivative/standard_deviation).T, dtype=np.float32)


def torch_strength(q, edges):
    a = q[:, edges[:, 0].long()]
    b = q[:, edges[:, 1].long()]
    c = q[:, edges[:, 2].clamp_min(0).long()]
    triples = (edges[:, 2] >= 0)[None, :, None]
    product = a*b*torch.where(triples, c, torch.ones_like(c))
    same_sign = ((a >= 0) & (b >= 0) & (~triples | (c >= 0))) | ((a <= 0) & (b <= 0) & (~triples | (c <= 0)))
    centered = product-product.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1)
    numerator = (centered.abs()*same_sign).mean(dim=-1)
    return torch.where(variance > 0, numerator/variance.clamp_min(torch.finfo(q.dtype).tiny).sqrt(), torch.zeros_like(numerator))


class MTDBuilder:
    def __init__(self, device="cpu", permutations=10000, p_threshold=0.05, permutation_batch_size=16, edge_chunk_size=2048, backend="auto"):
        self.device = torch.device(device)
        self.permutations = int(permutations)
        self.p_threshold = float(p_threshold)
        self.permutation_batch_size = int(permutation_batch_size)
        self.edge_chunk_size = int(edge_chunk_size)
        if self.permutations < 1 or not 0 < p_threshold < 1 or min(self.permutation_batch_size, self.edge_chunk_size) < 1:
            raise ValueError("Invalid MTD construction settings")
        if backend not in ("auto", "native", "torch"):
            raise ValueError("Backend must be auto, native, or torch")
        self.kernel = None
        if backend == "native" and self.device.type != "cuda":
            raise ValueError("The native backend requires CUDA")
        if self.device.type == "cuda" and backend in ("auto", "native"):
            try:
                from .native import NativeStrength
                self.kernel = NativeStrength(self.device)
            except (OSError, RuntimeError, AttributeError) as error:
                if backend == "native":
                    raise
                warnings.warn(f"Using the torch MTD backend: {error}", RuntimeWarning)
        self.backend = "native" if self.kernel is not None else "torch"

    @torch.no_grad()
    def build(self, values, seed=42):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        q = normalized_derivatives(values)
        edges = candidate_edges()
        edge_gpu = torch.as_tensor(edges, device=self.device)
        observed = torch.empty(len(edges), device=self.device)
        counts = torch.zeros(len(edges), device=self.device, dtype=torch.int32)
        q_gpu = torch.as_tensor(q[None], device=self.device)
        if self.kernel is not None:
            self.kernel(q_gpu, edge_gpu, observed, observed, counts)
        else:
            for start in range(0, len(edges), self.edge_chunk_size):
                stop = start+self.edge_chunk_size
                observed[start:stop] = torch_strength(q_gpu, edge_gpu[start:stop])[0]
        generator = np.random.default_rng(seed)
        for start in range(0, self.permutations, self.permutation_batch_size):
            count = min(self.permutation_batch_size, self.permutations-start)
            shuffled = generator.permuted(np.broadcast_to(q, (count, *q.shape)), axis=2).copy()
            shuffled_gpu = torch.as_tensor(shuffled, device=self.device)
            if self.kernel is not None:
                self.kernel(shuffled_gpu, edge_gpu, observed, observed, counts, mode=1)
            else:
                for first in range(0, len(edges), self.edge_chunk_size):
                    last = first+self.edge_chunk_size
                    null = torch_strength(shuffled_gpu, edge_gpu[first:last])
                    counts[first:last] += (null >= observed[None, first:last]).sum(dim=0).to(torch.int32)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        strength = observed.cpu().numpy()
        exceedances = counts.cpu().numpy()
        pvalues = (exceedances.astype(np.float64)+1)/(self.permutations+1)
        keep = pvalues < self.p_threshold
        elapsed = time.perf_counter()-started
        return {"node_features": q, "retained_nodes_zero_based": edges[keep], "weights": strength[keep], "retained_pvalues": pvalues[keep], "all_candidates_zero_based": edges, "all_strengths": strength, "all_null_exceedances": exceedances, "all_pvalues": pvalues}, elapsed


def prepare_dataset(manifest, output, config, device, backend=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest, require_graphs=False)
    settings = dict(config["construction"])
    variable = settings.pop("mat_variable", "Data")
    if backend is not None:
        settings["backend"] = backend
    builder = MTDBuilder(device=device, **settings)
    prepared, report = [], []
    for number, row in enumerate(rows, 1):
        identifier = row["subject_id"]
        key = hashlib.sha256(identifier.encode()).hexdigest()[:20]
        seed = int.from_bytes(hashlib.sha256(f'{config["training"]["seed"]}:{identifier}'.encode()).digest()[:4], "little")
        filename = output/f"{key}.npz"
        metadata_path = output/f"{key}.json"
        provenance = {"subject_id": identifier, "timeseries_sha256": hashlib.sha256(Path(row["timeseries_path"]).read_bytes()).hexdigest(), "settings": settings, "mat_variable": variable, "seed": seed, "format_version": 1}
        if filename.exists():
            if not metadata_path.is_file():
                raise RuntimeError(f"Graph metadata is missing: {filename}")
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous["provenance"] != provenance:
                raise RuntimeError(f"Existing graph configuration differs: {filename}; select a new output directory")
            details = previous
        else:
            values = load_timeseries(row["timeseries_path"], variable)
            graph, elapsed = builder.build(values, seed=seed)
            temporary = filename.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **graph)
            temporary.replace(filename)
            edges = graph["retained_nodes_zero_based"]
            details = {"provenance": provenance, "backend": builder.backend, "device": str(device), "construction_seconds": elapsed, "timepoints": int(values.shape[0]), "pairs": int(np.sum(edges[:, 2] < 0)), "triples": int(np.sum(edges[:, 2] >= 0))}
            metadata_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
        prepared.append({"subject_id": identifier, "site": row["site"], "label": row["label"], "graph_path": filename.name})
        report.append({"subject_id": identifier, **{k: v for k, v in details.items() if k != "provenance"}})
        print(f"Prepared {number}/{len(rows)}: {identifier}, hyperedges={details['pairs']+details['triples']}", flush=True)
    write_csv(output/"manifest.csv", prepared, ["subject_id", "site", "label", "graph_path"])
    (output/"construction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
