#!/usr/bin/env python3
"""Elliptic Bitcoin fraud detection with GraphSAGE."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.config import get_device, generate_run_id, ensure_dirs
from src.utils import set_seed, save_checkpoint, save_results
from src.datasets.elliptic import load_elliptic
from src.models.graphsage import GraphSAGE


def train(model, data, train_mask, optimizer):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, train_mask, test_mask):
    """Evaluate model on train and test sets."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_correct = (pred[train_mask] == data.y[train_mask]).sum()
    train_acc = train_correct / train_mask.sum()

    test_correct = (pred[test_mask] == data.y[test_mask]).sum()
    test_acc = test_correct / test_mask.sum()

    return train_acc.item(), test_acc.item()


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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )

    # Training loop
    print(f"\nTraining GraphSAGE for {config.training.epochs} epochs...")
    best_test_acc = 0.0

    for epoch in range(1, config.training.epochs + 1):
        loss = train(model, data, train_mask, optimizer)

        if epoch % 20 == 0 or epoch == 1:
            train_acc, test_acc = evaluate(model, data, train_mask, test_mask)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc

    # Final evaluation
    train_acc, test_acc = evaluate(model, data, train_mask, test_mask)

    # Save results
    results = {
        "run_id": run_id,
        "model": "GraphSAGE",
        "dataset": "elliptic",
        "device": str(device),
        "hyperparameters": OmegaConf.to_container(config),
        "results": {
            "final_loss": loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc
        }
    }

    save_results(results_dir / "metrics.json", results)
    save_checkpoint(
        results_dir / "checkpoint.pt",
        model, optimizer, config.training.epochs,
        {"train_acc": train_acc, "test_acc": test_acc}
    )

    # Save config used for this run
    OmegaConf.save(config, results_dir / "config.yaml")

    print(f"\nResults saved to {results_dir}")
    print(f"Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
