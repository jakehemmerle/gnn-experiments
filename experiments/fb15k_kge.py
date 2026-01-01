#!/usr/bin/env python3
"""FB15k-237 knowledge graph link prediction with configurable KGE models.

Usage:
    uv run python experiments/fb15k_kge.py                     # Uses TransE by default
    uv run python experiments/fb15k_kge.py --model DistMult    # Use DistMult
    uv run python experiments/fb15k_kge.py --model RotatE      # Use RotatE
    uv run python experiments/fb15k_kge.py --config configs/fb15k_rotate.yaml
"""

import sys
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.config import get_device, generate_run_id, ensure_dirs
from src.utils import set_seed, save_checkpoint, save_results
from src.datasets.fb15k import load_fb15k237
from src.models.kge import get_kge_model


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_examples = 0

    for head, rel, tail in loader:
        head = head.to(device)
        rel = rel.to(device)
        tail = tail.to(device)

        optimizer.zero_grad()
        loss = model.loss(head, rel, tail)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * head.size(0)
        total_examples += head.size(0)

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, data, batch_size, k, device):
    """Evaluate model using filtered ranking protocol."""
    model.eval()

    mean_rank, mrr, hits_at_k = model.test(
        head_index=data.edge_index[0].to(device),
        rel_type=data.edge_type.to(device),
        tail_index=data.edge_index[1].to(device),
        batch_size=batch_size,
        k=k,
        log=False
    )

    return mean_rank, mrr, hits_at_k


def get_default_config(model_name: str) -> OmegaConf:
    """Get default configuration for a model."""
    configs = {
        'transe': {
            'model': {'name': 'TransE', 'hidden_channels': 50, 'margin': 1.0},
            'training': {'lr': 0.01, 'epochs': 500, 'batch_size': 2048, 'log_every': 50},
            'evaluation': {'batch_size': 20000, 'k': [1, 3, 10]},
            'seed': 42
        },
        'distmult': {
            'model': {'name': 'DistMult', 'hidden_channels': 50},
            'training': {'lr': 0.0001, 'epochs': 500, 'batch_size': 2048, 'log_every': 50},
            'evaluation': {'batch_size': 20000, 'k': [1, 3, 10]},
            'seed': 42
        },
        'rotate': {
            'model': {'name': 'RotatE', 'hidden_channels': 128, 'margin': 9.0},
            'training': {'lr': 0.001, 'epochs': 500, 'batch_size': 2048, 'log_every': 50},
            'evaluation': {'batch_size': 20000, 'k': [1, 3, 10]},
            'seed': 42
        }
    }
    return OmegaConf.create(configs.get(model_name.lower(), configs['transe']))


def main():
    parser = argparse.ArgumentParser(description='Train KGE models on FB15k-237')
    parser.add_argument('--model', type=str, default='TransE',
                       choices=['TransE', 'DistMult', 'RotatE'],
                       help='KGE model to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (overrides --model)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    args = parser.parse_args()

    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = get_default_config(args.model)

    # Command-line overrides
    if args.epochs:
        config.training.epochs = args.epochs

    model_name = config.model.name

    # Setup
    set_seed(config.seed)
    device = get_device()
    run_id = generate_run_id("fb15k237", model_name.lower())
    results_dir = Path(__file__).parent.parent / "results" / run_id
    ensure_dirs(results_dir)

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")

    # Load data
    print("\nLoading FB15k-237 dataset...")
    train_data, val_data, test_data, num_entities, num_relations = load_fb15k237(
        root=str(Path(__file__).parent.parent / "data" / "fb15k237")
    )

    print(f"Entities: {num_entities:,}, Relations: {num_relations}")
    print(f"Train: {train_data.num_edges:,}, Val: {val_data.num_edges:,}, Test: {test_data.num_edges:,}")

    # Create model
    model_kwargs = {
        'model_name': model_name,
        'num_nodes': num_entities,
        'num_relations': num_relations,
        'hidden_channels': config.model.hidden_channels
    }
    if hasattr(config.model, 'margin'):
        model_kwargs['margin'] = config.model.margin

    model = get_kge_model(**model_kwargs).to(device)

    print(f"\nEmbedding dim: {config.model.hidden_channels}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loader
    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=train_data.edge_type,
        tail_index=train_data.edge_index[1],
        batch_size=config.training.batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # Training loop
    print(f"\nTraining {model_name} for {config.training.epochs} epochs...")
    best_val_mrr = 0.0
    k = max(config.evaluation.k)

    for epoch in range(1, config.training.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)

        if epoch % config.training.log_every == 0 or epoch == 1:
            val_rank, val_mrr, val_hits = evaluate(
                model, val_data,
                config.evaluation.batch_size,
                k, device
            )

            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val MRR: {val_mrr:.4f}, Hits@{k}: {val_hits:.4f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                save_checkpoint(
                    results_dir / "best_checkpoint.pt",
                    model, optimizer, epoch,
                    {"val_mrr": val_mrr, "val_hits": val_hits}
                )

    # Final evaluation
    print("\nEvaluating on test set...")
    test_rank, test_mrr, test_hits = evaluate(
        model, test_data,
        config.evaluation.batch_size,
        k, device
    )

    print(f"\nTest Results:")
    print(f"  Mean Rank: {test_rank:.1f}")
    print(f"  MRR: {test_mrr:.4f}")
    print(f"  Hits@{k}: {test_hits:.4f}")

    # Save results
    results = {
        "run_id": run_id,
        "model": model_name,
        "dataset": "FB15k-237",
        "device": str(device),
        "hyperparameters": OmegaConf.to_container(config),
        "results": {
            "final_loss": loss,
            "test_mean_rank": float(test_rank),
            "test_mrr": float(test_mrr),
            "test_hits_at_k": float(test_hits),
            "k": k,
            "best_val_mrr": best_val_mrr
        }
    }

    save_results(results_dir / "metrics.json", results)
    OmegaConf.save(config, results_dir / "config.yaml")

    save_checkpoint(
        results_dir / "checkpoint.pt",
        model, optimizer, config.training.epochs,
        {"test_mrr": test_mrr}
    )

    print(f"\nResults saved to {results_dir}")
    print(f"Test MRR: {test_mrr:.4f}")


if __name__ == "__main__":
    main()
