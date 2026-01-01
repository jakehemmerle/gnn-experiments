#!/usr/bin/env python3
"""Elliptic Bitcoin fraud detection with GraphSAGE."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.config import get_device, generate_run_id, ensure_dirs
from src.utils import set_seed
from src.datasets.elliptic import load_elliptic
from src.models.graphsage import GraphSAGE
from src.trainers.node_classifier import NodeClassificationTrainer


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "elliptic_graphsage.yaml"
    config = OmegaConf.load(config_path)

    # Setup
    set_seed(config.seed)
    device = get_device()
    run_id = generate_run_id("elliptic", "graphsage")
    results_dir = Path(__file__).parent.parent / "results" / run_id
    ensure_dirs(results_dir)

    print(f"Device: {device}")
    print(f"Run ID: {run_id}")

    # Load data
    print("\nLoading Elliptic Bitcoin dataset...")
    data, train_mask, test_mask = load_elliptic(
        root=str(Path(__file__).parent.parent / "data" / "elliptic"),
        train_ratio=config.data.train_ratio
    )
    data = data.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    print(f"Features: {data.num_node_features}")
    print(f"Train: {train_mask.sum():,}, Test: {test_mask.sum():,}")

    # Create model
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=config.model.hidden_channels,
        out_channels=2,
        dropout=config.model.dropout,
        aggr=config.model.aggr
    ).to(device)

    # Train using centralized trainer
    trainer = NodeClassificationTrainer(
        model=model,
        data=data,
        train_mask=train_mask,
        test_mask=test_mask,
        config=config,
        device=device,
        results_dir=results_dir
    )

    print(f"\nTraining GraphSAGE for {config.training.epochs} epochs...")
    results = trainer.train(log_interval=20)
    trainer.save(run_id, "GraphSAGE", "elliptic")

    print(f"\nResults saved to {results_dir}")
    print(f"Final Test Accuracy: {results['test_acc']:.4f}")


if __name__ == "__main__":
    main()
