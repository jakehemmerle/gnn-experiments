#!/usr/bin/env python3
"""Config-driven experiment launcher.

Supports node classification (Epic 1) and will support link prediction (Epic 2).

Usage:
    uv run python -m src.run --config configs/my_experiment.yaml
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .config import get_device, generate_run_id, ensure_dirs
from .utils import set_seed
from .models import get_model
from .datasets import get_dataset, get_task_for_dataset
from .trainers import NodeClassificationTrainer


def run_node_classification(config, device, results_dir, run_id):
    """Run node classification experiment."""
    dataset_name = config.dataset.name
    model_name = config.model.name

    print(f"\nLoading {dataset_name} dataset...")
    data, train_mask, test_mask = get_dataset(
        dataset_name,
        root=str(Path(__file__).parent.parent / "data" / dataset_name),
        train_ratio=config.data.train_ratio
    )
    data = data.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    print(f"Features: {data.num_node_features}")
    print(f"Train: {train_mask.sum():,}, Test: {test_mask.sum():,}")

    # Build model kwargs from config
    model_kwargs = {
        "in_channels": data.num_node_features,
        "out_channels": config.model.get("out_channels", 2),
        "hidden_channels": config.model.hidden_channels,
        "dropout": config.model.dropout,
    }

    # Add model-specific kwargs
    if model_name == "GAT":
        model_kwargs["heads"] = config.model.get("heads", 8)
    elif model_name == "GraphSAGE":
        model_kwargs["aggr"] = config.model.get("aggr", "mean")

    model = get_model(model_name, task="node_classification", **model_kwargs).to(device)

    # Train
    trainer = NodeClassificationTrainer(
        model=model,
        data=data,
        train_mask=train_mask,
        test_mask=test_mask,
        config=config,
        device=device,
        results_dir=results_dir
    )

    print(f"\nTraining {model_name} for {config.training.epochs} epochs...")
    results = trainer.train(log_interval=20)
    trainer.save(run_id, model_name, dataset_name)

    return results


def run_link_prediction(config, device, results_dir, run_id):
    """Run link prediction experiment.

    Epic 2 will implement this function with:
    - FB15k-237 dataset loading
    - KGE model training (TransE, DistMult, RotatE)
    - MRR/Hits@k evaluation
    """
    raise NotImplementedError(
        "Link prediction not yet implemented. "
        "Epic 2 will add support for FB15k-237 with TransE/DistMult/RotatE."
    )


def main():
    parser = argparse.ArgumentParser(description="Run GNN experiment from config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Setup
    set_seed(config.seed)
    device = get_device()

    # Determine task from dataset or explicit config
    task = config.get("task", None)
    if task is None:
        task = get_task_for_dataset(config.dataset.name)

    # Generate run ID and results directory
    run_id = generate_run_id(config.dataset.name, config.model.name)
    results_dir = Path(__file__).parent.parent / "results" / run_id
    ensure_dirs(results_dir)

    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Task: {task}")
    print(f"Config: {config_path}")

    # Dispatch to appropriate trainer
    if task == "node_classification":
        results = run_node_classification(config, device, results_dir, run_id)
    elif task == "link_prediction":
        results = run_link_prediction(config, device, results_dir, run_id)
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"\nResults saved to {results_dir}")
    print(f"Final Test Accuracy: {results['test_acc']:.4f}")


if __name__ == "__main__":
    main()
