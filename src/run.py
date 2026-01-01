#!/usr/bin/env python3
"""Config-driven experiment launcher.

Unified entry point for all GNN experiments with proper validation,
logging, and experiment tracking.

Usage:
    uv run python -m src.run --config configs/elliptic_gcn.yaml
    uv run python -m src.run --config configs/fb15k_transe.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .config import (
    get_device,
    generate_run_id,
    load_config,
    validate_config,
    merge_with_defaults,
)
from .paths import get_results_dir, ensure_dirs
from .utils import set_seed
from .models import get_model
from .datasets import get_dataset, get_task_for_dataset
from .trainers import NodeClassificationTrainer, LinkPredictionTrainer
from .logging_config import setup_logging, get_logger


def run_node_classification(config, device, results_dir, run_id, logger):
    """Run node classification experiment."""
    dataset_name = config.dataset.name
    model_name = config.model.name

    logger.info(f"Loading {dataset_name} dataset...")
    dataset = get_dataset(
        dataset_name,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio
    )

    logger.info(f"Nodes: {dataset.num_nodes:,}, Edges: {dataset.num_edges:,}")
    logger.info(f"Features: {dataset.num_features}")
    logger.info(f"Train: {dataset.train_mask.sum():,}, Val: {dataset.val_mask.sum():,}, Test: {dataset.test_mask.sum():,}")

    # Build model kwargs from config
    model_kwargs = {
        "in_channels": dataset.num_features,
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
        dataset=dataset,
        config=config,
        device=device,
        results_dir=results_dir
    )

    log_interval = config.training.get("log_interval", 20)
    logger.info(f"Training {model_name} for {config.training.epochs} epochs...")
    results = trainer.train(log_interval=log_interval)
    trainer.save(run_id, model_name, dataset_name)

    return results


def run_link_prediction(config, device, results_dir, run_id, logger):
    """Run link prediction experiment with KGE models."""
    dataset_name = config.dataset.name
    model_name = config.model.name

    logger.info(f"Loading {dataset_name} dataset...")
    train_data, val_data, test_data, num_entities, num_relations = get_dataset(
        dataset_name
    )

    logger.info(f"Entities: {num_entities:,}, Relations: {num_relations}")
    logger.info(f"Train: {train_data.num_edges:,}, Val: {val_data.num_edges:,}, Test: {test_data.num_edges:,}")

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

    logger.info(f"Model: {model_name}")
    logger.info(f"Embedding dim: {config.model.hidden_channels}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = LinkPredictionTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        device=device,
        results_dir=results_dir
    )

    log_interval = config.training.get("log_every", 50)
    logger.info(f"Training {model_name} for {config.training.epochs} epochs...")
    results = trainer.train(log_interval=log_interval)
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

    # Load and validate config
    config = load_config(args.config)

    # Validate config
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")

    # Merge with defaults
    config = merge_with_defaults(config)

    # Setup
    set_seed(config.seed)
    device = get_device()

    # Determine task from dataset or explicit config
    task = config.get("task", None)
    if task is None:
        task = get_task_for_dataset(config.dataset.name)

    # Generate run ID and results directory
    run_id = generate_run_id(config.dataset.name, config.model.name)
    results_dir = get_results_dir(run_id)
    ensure_dirs(results_dir)

    # Setup logging
    logger = setup_logging(log_file=results_dir / "experiment.log")
    logger.info(f"Device: {device}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Task: {task}")
    logger.info(f"Config: {args.config}")

    # Dispatch to appropriate trainer
    if task == "node_classification":
        results = run_node_classification(config, device, results_dir, run_id, logger)
    elif task == "link_prediction":
        results = run_link_prediction(config, device, results_dir, run_id, logger)
    else:
        raise ValueError(f"Unknown task: {task}")

    logger.info(f"Results saved to {results_dir}")
    if task == "node_classification":
        logger.info(f"Final Test Accuracy: {results['test_acc']:.4f}")
        logger.info(f"Best Val Accuracy: {results['best_val_acc']:.4f}")
    elif task == "link_prediction":
        logger.info(f"Test MRR: {results['test_mrr']:.4f}")
        k = results['k']
        logger.info(f"Test Hits@{k}: {results[f'test_hits@{k}']:.4f}")
    logger.info(f"Training Time: {results['training_time_seconds']:.2f}s")


if __name__ == "__main__":
    main()
