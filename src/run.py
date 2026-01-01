#!/usr/bin/env python3
"""Config-driven experiment launcher.

Supports node classification (Epic 1) and will support link prediction (Epic 2).

Usage:
    # Single GPU
    uv run python -m src.run --config configs/my_experiment.yaml

    # Multi-GPU (e.g., 2 GPUs)
    uv run torchrun --nproc_per_node=2 -m src.run --config configs/my_experiment.yaml
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .config import get_device, generate_run_id, ensure_dirs
from .distributed import setup_distributed, cleanup_distributed, is_main_process
from .utils import set_seed
from .models import get_model
from .datasets import get_dataset, get_task_for_dataset
from .trainers import NodeClassificationTrainer, LinkPredictionTrainer


def run_node_classification(config, device, results_dir, run_id, rank, world_size):
    """Run node classification experiment."""
    dataset_name = config.dataset.name
    model_name = config.model.name

    if is_main_process():
        print(f"\nLoading {dataset_name} dataset...")
    data, train_mask, test_mask = get_dataset(
        dataset_name,
        root=str(Path(__file__).parent.parent / "data" / dataset_name),
        train_ratio=config.data.train_ratio
    )

    if is_main_process():
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
        results_dir=results_dir,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process():
        print(f"\nTraining {model_name} for {config.training.epochs} epochs...")
    results = trainer.train(log_interval=20)
    if is_main_process():
        trainer.save(run_id, model_name, dataset_name)

    return results


def run_link_prediction(config, device, results_dir, run_id, rank, world_size):
    """Run link prediction experiment with KGE models."""
    dataset_name = config.dataset.name
    model_name = config.model.name

    if is_main_process():
        print(f"\nLoading {dataset_name} dataset...")
    train_data, val_data, test_data, num_entities, num_relations = get_dataset(
        dataset_name,
        root=str(Path(__file__).parent.parent / "data" / dataset_name)
    )

    if is_main_process():
        print(f"Entities: {num_entities:,}, Relations: {num_relations}")
        print(f"Train: {train_data.num_edges:,}, Val: {val_data.num_edges:,}, Test: {test_data.num_edges:,}")

    # Build model kwargs from config
    model_kwargs = {
        "num_nodes": num_entities,
        "num_relations": num_relations,
        "hidden_channels": config.model.hidden_channels,
    }

    # Add model-specific kwargs
    if hasattr(config.model, "margin"):
        model_kwargs["margin"] = config.model.margin

    model = get_model(model_name, task="link_prediction", **model_kwargs).to(device)

    if is_main_process():
        print(f"\nModel: {model_name}")
        print(f"Embedding dim: {config.model.hidden_channels}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = LinkPredictionTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        device=device,
        results_dir=results_dir,
        rank=rank,
        world_size=world_size,
    )

    log_interval = config.training.get("log_every", 50)
    if is_main_process():
        print(f"\nTraining {model_name} for {config.training.epochs} epochs...")
    results = trainer.train(log_interval=log_interval)
    if is_main_process():
        trainer.save(run_id, model_name, dataset_name)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run GNN experiment from config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Initialize distributed training (if running under torchrun)
    rank, world_size = setup_distributed()

    try:
        # Load config
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = OmegaConf.load(config_path)

        # Setup
        set_seed(config.seed)
        device = get_device(rank)

        # Determine task from dataset or explicit config
        task = config.get("task", None)
        if task is None:
            task = get_task_for_dataset(config.dataset.name)

        # Generate run ID and results directory
        run_id = generate_run_id(config.dataset.name, config.model.name)
        results_dir = Path(__file__).parent.parent / "results" / run_id
        if is_main_process():
            ensure_dirs(results_dir)

        if is_main_process():
            print(f"Device: {device}")
            if world_size > 1:
                print(f"Distributed: {world_size} GPUs")
            print(f"Run ID: {run_id}")
            print(f"Task: {task}")
            print(f"Config: {config_path}")

        # Dispatch to appropriate trainer
        if task == "node_classification":
            results = run_node_classification(
                config, device, results_dir, run_id, rank, world_size
            )
        elif task == "link_prediction":
            results = run_link_prediction(
                config, device, results_dir, run_id, rank, world_size
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        if is_main_process():
            print(f"\nResults saved to {results_dir}")
            if task == "node_classification":
                print(f"Final Test Accuracy: {results['test_acc']:.4f}")
            elif task == "link_prediction":
                print(f"Test MRR: {results['test_mrr']:.4f}")
                print(f"Test Hits@{results['k']}: {results['test_hits_at_k']:.4f}")
            print(f"Training Time: {results['training_time_seconds']:.2f}s")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
