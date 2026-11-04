from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


DEFAULTS = {
    "construction": {"permutations": 10000, "p_threshold": 0.05, "permutation_batch_size": 16, "edge_chunk_size": 2048, "backend": "auto", "mat_variable": "Data"},
    "model": {"hidden_dim": 64, "num_layers": 2, "tau": 1.0, "temperature": 1.0, "num_classes": 2, "incidence_chunk_size": 32768, "gradient_checkpointing": True},
    "training": {"protocol": "loso", "folds": 5, "validation_fraction": 0.125, "epochs": 150, "batch_size": 32, "learning_rate": 0.01, "weight_decay": 0.001, "interventions": 4, "alpha": 1.0, "beta": 0.05, "lambda_var": 1.0, "seed": 42, "input_timepoints": None, "cpu_threads": 8},
}


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        for section, values in supplied.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f"Unknown configuration section: {section}")
            if set(values)-set(config[section]):
                raise ValueError(f"Unknown {section} settings: {sorted(set(values)-set(config[section]))}")
            config[section].update(values)
    return config


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Select a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def main():
    parser = argparse.ArgumentParser(prog="ihnd")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config")
    prepare.add_argument("--device", default="auto")
    prepare.add_argument("--backend", choices=["auto", "native", "torch"])
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--config")
    training.add_argument("--device", default="auto")
    training.add_argument("--protocol", choices=["loso", "fivefold"])
    training.add_argument("--fold")
    inference = commands.add_parser("predict")
    inference.add_argument("--manifest", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", default="auto")
    inference.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = choose_device(args.device)
    if args.command == "prepare":
        from .mtd import prepare_dataset
        prepare_dataset(args.manifest, args.output, load_config(args.config), device, args.backend)
    elif args.command == "train":
        from .training import train
        config = load_config(args.config)
        if args.protocol:
            config["training"]["protocol"] = args.protocol
        train(args.manifest, args.output, config, device, args.fold)
    else:
        from .training import predict
        predict(args.checkpoint, args.manifest, args.output, device, args.batch_size)


if __name__ == "__main__":
    main()
