#!/usr/bin/env python3
"""FB15k-237 knowledge graph link prediction with TransE."""

import sys
from pathlib import Path

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
    """
    Evaluate model using filtered ranking protocol.

    Returns mean_rank, MRR, and Hits@k metrics.
    """
    model.eval()

    # Use PyG's built-in test method for filtered ranking
    # Returns: (mean_rank, mrr, hits_at_k)
    mean_rank, mrr, hits_at_k = model.test(
        head_index=data.edge_index[0].to(device),
        rel_type=data.edge_type.to(device),
        tail_index=data.edge_index[1].to(device),
        batch_size=batch_size,
        k=k,
        log=False  # Disable progress bar for cleaner output
    )

    return mean_rank, mrr, hits_at_k


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "fb15k_transe.yaml"
    config = OmegaConf.load(config_path)

    # Setup
    set_seed(config.seed)
    device = get_device()
    run_id = generate_run_id("fb15k237", "transe")
    results_dir = Path(__file__).parent.parent / "results" / run_id
    ensure_dirs(results_dir)

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
    model = get_kge_model(
        model_name=config.model.name,
        num_nodes=num_entities,
        num_relations=num_relations,
        hidden_channels=config.model.hidden_channels
    ).to(device)

    print(f"\nModel: {config.model.name}")
    print(f"Embedding dim: {config.model.hidden_channels}")

    # Create data loader using model's built-in loader
    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=train_data.edge_type,
        tail_index=train_data.edge_index[1],
        batch_size=config.training.batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # Training loop
    print(f"\nTraining TransE for {config.training.epochs} epochs...")
    best_val_mrr = 0.0
    k = max(config.evaluation.k)  # Use max k for Hits@k

    for epoch in range(1, config.training.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)

        if epoch % config.training.log_every == 0 or epoch == 1:
            # Evaluate on validation set
            val_rank, val_mrr, val_hits = evaluate(
                model, val_data,
                config.evaluation.batch_size,
                k, device
            )

            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val MRR: {val_mrr:.4f}, Hits@{k}: {val_hits:.4f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                # Save best model
                save_checkpoint(
                    results_dir / "best_checkpoint.pt",
                    model, optimizer, epoch,
                    {"val_mrr": val_mrr, "val_hits": val_hits}
                )

    # Final evaluation on test set
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
        "model": config.model.name,
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

    # Save final checkpoint
    save_checkpoint(
        results_dir / "checkpoint.pt",
        model, optimizer, config.training.epochs,
        {"test_mrr": test_mrr}
    )

    print(f"\nResults saved to {results_dir}")
    print(f"Test MRR: {test_mrr:.4f}")


if __name__ == "__main__":
    main()
