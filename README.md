# GNN Experiments

Graph Neural Network experiments using PyTorch Geometric.

## Quick Start

```bash
# Install dependencies
uv sync

# Run an experiment via config file
uv run python -m src.run --config configs/elliptic_gcn.yaml
```

## Running Experiments

### Option 1: Config-Driven (Recommended)

Run any experiment using a YAML config file:

```bash
# GCN on Elliptic Bitcoin
uv run python -m src.run --config configs/elliptic_gcn.yaml

# GAT on Elliptic Bitcoin
uv run python -m src.run --config configs/elliptic_gat.yaml

# GraphSAGE on Elliptic Bitcoin
uv run python -m src.run --config configs/elliptic_graphsage.yaml
```

Create custom experiments by copying `configs/template.yaml`:

```yaml
dataset:
  name: elliptic

model:
  name: GAT
  hidden_channels: 64
  heads: 8
  dropout: 0.6

training:
  lr: 0.005
  weight_decay: 5e-4
  epochs: 200

data:
  train_ratio: 0.8

seed: 42
```

### Option 2: Standalone Scripts

Run experiments directly:

```bash
uv run python experiments/elliptic_gcn.py
uv run python experiments/elliptic_gat.py
uv run python experiments/elliptic_graphsage.py
```

## Available Models

| Model | Description | Key Params |
|-------|-------------|------------|
| GCN | Graph Convolutional Network | `hidden_channels`, `dropout` |
| GAT | Graph Attention Network | `hidden_channels`, `heads`, `dropout` |
| GraphSAGE | Scalable inductive learning | `hidden_channels`, `dropout`, `aggr` |

## Results

Results are saved to `results/{dataset}_{model}_{timestamp}/`:
- `metrics.json` - Final metrics and hyperparameters
- `config.yaml` - Config used for this run
- `checkpoint.pt` - Model weights

## Current Experiments

### Elliptic Bitcoin Fraud Detection
Node classification on the Elliptic Bitcoin dataset to detect illicit transactions.

- **Dataset**: 203,769 transactions, 234,355 edges, 165 features
- **Task**: Binary classification (licit vs illicit)
- **Models**: GCN (~95.7%), GAT, GraphSAGE

## Project Structure

```text
├── src/
│   ├── config.py           # Device selection, run ID generation
│   ├── utils.py            # Seeds, checkpointing
│   ├── run.py              # Config-driven experiment launcher
│   ├── datasets/           # Dataset loaders
│   ├── models/             # GNN architectures (GCN, GAT, GraphSAGE)
│   └── trainers/           # Training loops
├── configs/                # YAML experiment configs
├── experiments/            # Standalone experiment scripts
├── results/                # Training outputs (gitignored)
├── data/                   # Datasets (gitignored, auto-downloaded)
└── docs/                   # Documentation
```

## Requirements

- Python 3.11+
- PyTorch 2.5+
- PyTorch Geometric 2.7+
- CUDA 12.x (optional, falls back to CPU)
