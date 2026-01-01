# GNN Experiments

Graph Neural Network experiments using PyTorch Geometric.

## Current Experiments

### Elliptic Bitcoin Fraud Detection
Node classification on the Elliptic Bitcoin dataset to detect illicit transactions.

- **Dataset**: 203,769 transactions, 234,355 edges
- **Model**: 2-layer GCN
- **Accuracy**: ~95.7% on test set

## Setup

```bash
# Install dependencies
uv sync

# Run the Elliptic Bitcoin experiment
uv run python experiments/elliptic_gcn.py
```

## Project Structure

```
├── src/
│   ├── config.py           # Device selection, config loading
│   ├── utils.py            # Seeds, checkpointing
│   ├── datasets/           # Dataset loaders
│   └── models/             # GNN architectures
├── configs/                # YAML hyperparameter configs
├── experiments/            # Runnable experiment scripts
├── results/                # Training outputs (gitignored)
└── data/                   # Datasets (gitignored, auto-downloaded)
```

## Requirements

- Python 3.11+
- PyTorch 2.5+
- PyTorch Geometric 2.7+
