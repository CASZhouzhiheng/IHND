from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SubjectDataset, collate_subjects, infer_timepoints, read_manifest, write_csv
from .model import IHND
from .objective import IHNDLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def classification_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"N": len(y), "ACC": float(accuracy_score(y, prediction)), "F1": float(f1_score(y, prediction, pos_label=1, zero_division=0)), "SEN": float(tp/(tp+fn)) if tp+fn else None, "SPE": float(tn/(tn+fp)) if tn+fp else None, "AUC": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None}


def create_splits(rows, settings):
    labels = np.asarray([row["label"] for row in rows])
    sites = np.asarray([row["site"] for row in rows])
    indices = np.arange(len(rows))
    protocol = settings["protocol"]
    seed = int(settings["seed"])
    if protocol == "loso":
        if len(np.unique(sites)) < 2:
            raise ValueError("LOSO evaluation requires at least two sites")
        folds = [(str(site), indices[sites != site], indices[sites == site]) for site in sorted(np.unique(sites))]
    elif protocol == "fivefold":
        count = int(settings.get("folds", 5))
        if count != 5:
            raise ValueError("The fivefold protocol requires folds=5")
        if np.bincount(labels, minlength=2).min() < count:
            raise ValueError("Each diagnosis requires at least five participants for five-fold evaluation")
        splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
        folds = [(f"fold_{i+1:02d}", source, test) for i, (source, test) in enumerate(splitter.split(indices, labels))]
    else:
        raise ValueError("Protocol must be loso or fivefold")
    result = []
    for index, (name, source, test) in enumerate(folds):
        train, validation = train_test_split(source, test_size=float(settings["validation_fraction"]), stratify=labels[source], random_state=seed+index)
        if any(set(a) & set(b) for a, b in ((train, validation), (train, test), (validation, test))):
            raise RuntimeError("Partitions overlap")
        if len(train)+len(validation)+len(test) != len(rows):
            raise RuntimeError("Incomplete partition")
        if protocol == "loso" and set(sites[train]) & set(sites[test]):
            raise RuntimeError("Target-site leakage")
        result.append({"name": name, "train": train.tolist(), "validation": validation.tolist(), "test": test.tolist(), "seed": seed+index})
    return result


def data_loader(rows, input_dim, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(SubjectDataset(rows, input_dim), batch_size=int(batch_size), shuffle=shuffle, collate_fn=collate_subjects, num_workers=0, generator=generator)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    records = []
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        total_loss += float(F.cross_entropy(output["logits"], batch.labels, reduction="sum"))
        probability = output["logits"].softmax(dim=-1)[:, 1].cpu().tolist()
        labels = batch.labels.cpu().tolist()
        for i, value in enumerate(probability):
            records.append({"subject_id": batch.subject_ids[i], "site": batch.site_names[i], "label": labels[i], "prediction": int(value >= 0.5), "probability_mdd": value})
    if not records:
        raise ValueError("Evaluation partition is empty")
    metrics = classification_metrics([r["label"] for r in records], [r["probability_mdd"] for r in records])
    metrics["loss"] = total_loss/len(records)
    return metrics, records


def alignment_report(rows, input_dim):
    shorter, longer = 0, 0
    for row in rows:
        with np.load(row["graph_path"], allow_pickle=False) as graph:
            length = graph["node_features"].shape[1]
        shorter += length < input_dim
        longer += length > input_dim
    return {"participants": len(rows), "right_zero_padded": int(shorter), "right_truncated": int(longer), "input_timepoints": int(input_dim)}


def save_checkpoint(path, model, input_dim, config, epoch, metrics):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"model_state": state, "input_dim": int(input_dim), "model_config": dict(config["model"]), "training_config": dict(config["training"]), "epoch": int(epoch), "validation_metrics": metrics, "label_map": {"NC": 0, "MDD": 1}}, path)


def train_fold(rows, split, config, output, device):
    settings = config["training"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"best.pt").exists():
        raise FileExistsError(f"Existing checkpoint: {output/'best.pt'}; select a new run directory")
    set_seed(split["seed"])
    partitions = {name: [rows[i] for i in split[name]] for name in ("train", "validation", "test")}
    input_dim = infer_timepoints(partitions["train"], settings.get("input_timepoints"))
    alignment = {name: alignment_report(values, input_dim) for name, values in partitions.items()}
    (output/"feature_alignment.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    split_rows = [{"subject_id": row["subject_id"], "site": row["site"], "label": row["label"], "partition": name} for name, subset in partitions.items() for row in subset]
    write_csv(output/"split.csv", split_rows, ["subject_id", "site", "label", "partition"])
    loaders = {name: data_loader(values, input_dim, settings["batch_size"], name == "train", split["seed"]) for name, values in partitions.items()}
    model = IHND(input_dim, **config["model"]).to(device)
    criterion = IHNDLoss(**{key: settings[key] for key in ("interventions", "alpha", "beta", "lambda_var")})
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    best_accuracy, best_loss = -1.0, float("inf")
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, int(settings["epochs"])+1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "invariant_risk", "risk_variance", "shortcut_loss", "sparsity")}
        count = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            objective = criterion(model(batch), batch)
            if not torch.isfinite(objective["loss"]):
                raise FloatingPointError(f"Nonfinite training loss in {split['name']}, epoch {epoch}")
            objective["loss"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite gradient in {split['name']}, epoch {epoch}")
            optimizer.step()
            count += batch.size
            for name in totals:
                totals[name] += float(objective[name].detach())*batch.size
        validation, _ = evaluate(model, loaders["validation"], device)
        history.append({"epoch": epoch, **{key: value/count for key, value in totals.items()}, "validation_ACC": validation["ACC"], "validation_loss": validation["loss"]})
        if validation["ACC"] > best_accuracy or (validation["ACC"] == best_accuracy and validation["loss"] < best_loss):
            best_accuracy, best_loss = validation["ACC"], validation["loss"]
            save_checkpoint(output/"best.pt", model, input_dim, config, epoch, validation)
        write_csv(output/"history.csv", history, list(history[0]))
        print(f"{split['name']} epoch {epoch}/{settings['epochs']}: loss={history[-1]['loss']:.5f}, validation_ACC={100*validation['ACC']:.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    checkpoint_data = torch.load(output/"best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_data["model_state"])
    metrics, predictions = evaluate(model, loaders["test"], device)
    metrics.update({"fold": split["name"], "best_epoch": checkpoint_data["epoch"], "training_seconds": elapsed})
    for row in predictions:
        row["fold"] = split["name"]
    write_csv(output/"test_predictions.csv", predictions, ["fold", "subject_id", "site", "label", "prediction", "probability_mdd"])
    (output/"metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, predictions


def train(manifest, output, config, device, fold=None):
    rows = read_manifest(manifest)
    settings = config["training"]
    if int(settings["epochs"]) < 1 or int(settings["batch_size"]) < 1 or not 0 < float(settings["validation_fraction"]) < 1:
        raise ValueError("Invalid training settings")
    torch.set_num_threads(int(settings.get("cpu_threads", 8)))
    all_splits = create_splits(rows, settings)
    splits = all_splits if fold is None else [s for s in all_splits if s["name"] == fold]
    if not splits:
        raise ValueError(f"Unknown fold {fold}; available: {[s['name'] for s in all_splits]}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"run_config.json").exists():
        raise FileExistsError(f"Run directory already contains a configuration: {output}")
    (output/"run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (output/"environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    results, all_predictions = [], []
    for split in splits:
        name = split["name"]
        if Path(name).name != name or name in (".", "..") or any(v in name for v in ("/", "\\", ":")):
            raise ValueError("Site and fold identifiers must be valid directory names")
        result, predictions = train_fold(rows, split, config, output/name, device)
        results.append(result)
        all_predictions.extend(predictions)
    write_csv(output/"fold_metrics.csv", results, ["fold", "N", "ACC", "F1", "SEN", "SPE", "AUC", "best_epoch", "training_seconds", "loss"])
    write_csv(output/"predictions.csv", all_predictions, ["fold", "subject_id", "site", "label", "prediction", "probability_mdd"])
    summary = {"protocol": settings["protocol"], "folds_completed": len(results), "folds_available": len(all_splits), "units": "percent", "aggregation": "equal-weight arithmetic mean and sample standard deviation across evaluated folds", "metrics": {}}
    for name in ("ACC", "F1", "SEN", "SPE", "AUC"):
        values = np.asarray([100*r[name] for r in results if r[name] is not None], dtype=float)
        summary["metrics"][name] = {"mean": round(float(values.mean()), 5) if len(values) else None, "std": round(float(values.std(ddof=1)), 5) if len(values) > 1 else None, "valid_folds": len(values), "undefined_folds": [r["fold"] for r in results if r[name] is None]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def predict(checkpoint_path, manifest, output, device, batch_size=32):
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    rows = read_manifest(manifest)
    model = IHND(checkpoint_data["input_dim"], **checkpoint_data["model_config"]).to(device)
    model.load_state_dict(checkpoint_data["model_state"])
    loader = data_loader(rows, checkpoint_data["input_dim"], batch_size)
    metrics, predictions = evaluate(model, loader, device)
    write_csv(output, predictions, ["subject_id", "site", "label", "prediction", "probability_mdd"])
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
