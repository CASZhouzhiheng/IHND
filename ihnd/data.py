from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def read_manifest(path, require_graphs=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject_id", "site", "label", "graph_path" if require_graphs else "timeseries_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row["subject_id"] or not row["site"] or row["subject_id"] in seen:
            raise ValueError("Subject IDs must be unique and nonempty; site is required")
        seen.add(row["subject_id"])
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1):
            raise ValueError("Labels must be 0 for NC and 1 for MDD")
        key = "graph_path" if require_graphs else "timeseries_path"
        target = Path(row[key])
        row[key] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
        if not Path(row[key]).is_file():
            raise FileNotFoundError(row[key])
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_timeseries(path, variable="Data"):
    path = Path(path)
    if path.suffix.lower() == ".mat":
        arrays = loadmat(path, variable_names=[variable])
        if variable not in arrays:
            raise KeyError(f"Variable {variable} is absent from {path}")
        values = arrays[variable]
    elif path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            values = arrays["timeseries"]
    else:
        raise ValueError("Time series must be MAT, NPY, or NPZ files")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 90 or values.shape[0] < 4:
        raise ValueError(f"Expected a time-by-90 matrix with at least four time points: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite time series: {path}")
    return values


def read_graph(path):
    with np.load(path, allow_pickle=False) as arrays:
        x = np.array(arrays["node_features"], dtype=np.float32, copy=True)
        edges = np.array(arrays["retained_nodes_zero_based"], dtype=np.int64, copy=True)
        weights = np.array(arrays["weights"], dtype=np.float32, copy=True)
    if x.ndim != 2 or x.shape[0] != 90 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError(f"Invalid node features: {path}")
    if edges.shape != (len(weights), 3):
        raise ValueError(f"Invalid hyperedge array: {path}")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"Hyperedge weights must be finite and positive: {path}")
    if len(edges):
        valid = edges >= 0
        if np.any(edges < -1) or np.any(edges >= 90) or np.any(~valid[:, :2]):
            raise ValueError(f"Invalid ROI membership: {path}")
        if np.any(edges[:, 0] >= edges[:, 1]) or np.any(valid[:, 2] & (edges[:, 1] >= edges[:, 2])):
            raise ValueError(f"Hyperedge nodes must be distinct and ordered: {path}")
    return x, edges, weights


@dataclass
class HypergraphBatch:
    x: torch.Tensor
    node_index: torch.Tensor
    edge_index: torch.Tensor
    weights: torch.Tensor
    node_graph: torch.Tensor
    edge_graph: torch.Tensor
    labels: torch.Tensor
    sites: torch.Tensor
    subject_ids: list[str]
    site_names: list[str]

    @property
    def size(self):
        return len(self.subject_ids)

    def to(self, device):
        tensors = {name: getattr(self, name).to(device) for name in ("x", "node_index", "edge_index", "weights", "node_graph", "edge_graph", "labels", "sites")}
        return HypergraphBatch(**tensors, subject_ids=self.subject_ids, site_names=self.site_names)


class SubjectDataset(Dataset):
    def __init__(self, rows, input_timepoints):
        self.rows = list(rows)
        self.input_timepoints = int(input_timepoints)
        if self.input_timepoints < 1:
            raise ValueError("input_timepoints must be positive")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, edges, weights = read_graph(row["graph_path"])
        length = self.input_timepoints
        if x.shape[1] >= length:
            x = np.ascontiguousarray(x[:, :length])
        else:
            x = np.pad(x, ((0, 0), (0, length-x.shape[1])))
        return row, torch.from_numpy(x), torch.from_numpy(edges), torch.from_numpy(weights)


def collate_subjects(items):
    features, nodes, edge_ids, weights, node_graph, edge_graph = [], [], [], [], [], []
    ids, site_names, labels = [], [], []
    offset = 0
    for graph_id, (row, x, edges, weight) in enumerate(items):
        count = len(edges)
        present = edges >= 0
        indices = torch.arange(count, dtype=torch.long)[:, None].expand_as(edges)
        nodes.append(edges[present] + 90*graph_id)
        edge_ids.append(indices[present] + offset)
        features.append(x)
        weights.append(weight)
        node_graph.append(torch.full((90,), graph_id, dtype=torch.long))
        edge_graph.append(torch.full((count,), graph_id, dtype=torch.long))
        ids.append(row["subject_id"])
        site_names.append(row["site"])
        labels.append(int(row["label"]))
        offset += count
    sites = {value: i for i, value in enumerate(sorted(set(site_names)))}
    return HypergraphBatch(torch.cat(features), torch.cat(nodes), torch.cat(edge_ids), torch.cat(weights), torch.cat(node_graph), torch.cat(edge_graph), torch.tensor(labels, dtype=torch.long), torch.tensor([sites[v] for v in site_names], dtype=torch.long), ids, site_names)


def infer_timepoints(rows, requested=None):
    lengths = []
    for row in rows:
        with np.load(row["graph_path"], allow_pickle=False) as graph:
            lengths.append(int(graph["node_features"].shape[1]))
    chosen = int(requested) if requested is not None else min(lengths)
    if chosen < 1:
        raise ValueError("Input feature length must be positive")
    return chosen
