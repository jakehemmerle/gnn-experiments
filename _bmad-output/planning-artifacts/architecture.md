---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - technical-gnn-blockchain-knowledge-graphs-research-2025-12-31.md
workflowType: 'architecture'
project_name: 'gnn-experimentation-system'
user_name: 'Jake'
date: '2025-12-31'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
- Execute GNN training pipelines on two real-world datasets:
  - Elliptic Bitcoin: Node classification (fraud detection, 203K nodes)
  - FB15k-237: Link prediction (knowledge graph completion, 14K entities)
- Support multiple GNN architectures with minimal code changes (GCN, GAT, TransE, DistMult, RGCN, RotatE)
- Handle dataset loading, train/test splitting, and evaluation metrics
- Provide reproducible training runs

**Non-Functional Requirements:**
- CUDA-first execution with automatic CPU fallback
- Clean environment isolation via uv
- Fast experimentation cycles
- Extensible architecture for adding new models/datasets

**Scale & Complexity:**
- Primary domain: ML Experimentation
- Complexity level: Low-Medium
- Estimated architectural components: 4-6 modules

### Technical Constraints & Dependencies

| Constraint | Value |
|------------|-------|
| Python | 3.11 |
| Package Manager | uv |
| Core Library | PyTorch Geometric 2.7.0 |
| PyTorch | 2.5+ (torch.compile support) |
| CUDA | 12.9 (primary) |
| Fallback | CPU (automatic) |
| Platform | Linux/Windows with CUDA, macOS CPU-only |

**Device Selection Pattern:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Cross-Cutting Concerns Identified

- **Experiment reproducibility**: Random seeds, hyperparameter logging
- **Model persistence**: Save/load trained models
- **Results tracking**: Metrics, loss curves, evaluation scores
- **Code organization**: Separation of data, models, training, evaluation
- **Configuration management**: Hyperparameters, paths, device settings

## Starter Template Evaluation

### Primary Technology Domain

ML/Data Science Experimentation with PyTorch Geometric

### Starter Options Considered

| Option | Description | Verdict |
|--------|-------------|---------|
| Cookiecutter Data Science v2 | Standard DS template | Too heavy for experimentation |
| victoresque/pytorch-template | Config + checkpoints + base classes | Over-engineered for 2 datasets |
| **Custom Minimal Structure** | Tailored for GNN experiments | ✅ Selected |

### Selected Starter: Custom Minimal Structure

**Rationale:** Focused experimentation system needs speed and simplicity over framework overhead. Follows PyG examples patterns while adding just enough organization.

**Initialization Command:**

```bash
mkdir gnn-experiments && cd gnn-experiments
uv init
uv add torch --index-url https://download.pytorch.org/whl/cu129
uv add torch-geometric
```

### Project Structure

```
gnn-experiments/
├── pyproject.toml          # uv project config
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters, device selection
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── elliptic.py     # Elliptic Bitcoin loader
│   │   └── fb15k.py        # FB15k-237 loader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py          # GCN for node classification
│   │   ├── gat.py          # GAT variant
│   │   └── kge.py          # TransE, DistMult, RotatE
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── node_classifier.py
│   │   └── link_predictor.py
│   └── utils.py            # Seeds, metrics, saving
├── experiments/
│   ├── elliptic_gcn.py     # Run: uv run experiments/elliptic_gcn.py
│   └── fb15k_transe.py
├── checkpoints/            # Saved models
├── results/                # Metrics, logs
└── data/                   # Auto-downloaded by PyG
```

### Architectural Decisions from Structure

| Decision | Choice |
|----------|--------|
| Package Management | uv with pyproject.toml |
| Source Layout | src/ layout (importable module) |
| Config Approach | Python module (config.py), not YAML/JSON |
| Experiment Scripts | Standalone scripts in experiments/ |
| Model Organization | One file per architecture |
| Data Storage | PyG auto-download to data/ |

## Core Architectural Decisions

### Decision Summary

| Category | Decision | Choice |
|----------|----------|--------|
| Data Loading | Strategy | PyG built-in datasets directly |
| Data Caching | Approach | PyG default disk cache |
| Model Definition | Pattern | Class per model file |
| Hyperparameters | Handling | Constructor args with defaults |
| Training Loop | Pattern | Explicit loops (no Lightning) |
| Validation | Strategy | Train/Val/Test (dataset-dependent) |
| Checkpointing | Approach | Full checkpoint (model + optimizer + metrics) |
| Configuration | Management | OmegaConf (YAML-based) |
| Reproducibility | Seeding | Manual seeds in utils.py |
| Logging | Format | Print + JSON |
| Results Storage | Structure | Combined logs + checkpoints per run |
| Model Saving | Format | state_dict only |

---

### Data Architecture

**Dataset Loading:** Use PyG built-in loaders directly
- `EllipticBitcoinDataset(root='./data/elliptic')`
- `FB15k_237(root='./data/fb15k-237', split='train')`

**Caching:** PyG default - downloads once to `data/`, reuses on subsequent runs.

**Rationale:** No abstraction needed for 2 well-designed datasets. PyG handles download, caching, and preprocessing.

---

### Model Architecture

**Pattern:** One class per model file
```
src/models/
├── gcn.py      # class GCN(nn.Module)
├── gat.py      # class GAT(nn.Module)
└── kge.py      # TransE, DistMult, RotatE
```

**Hyperparameters:** Constructor arguments with sensible defaults
```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.5):
```

**Rationale:** Explicit, readable, easy to modify. No factory indirection.

---

### Training Pipeline

**Loop Pattern:** Explicit training loops
```python
for epoch in range(1, config.training.epochs + 1):
    loss = train()
    if epoch % 20 == 0:
        train_acc, test_acc = test()
```

**Validation Strategy:**
- Elliptic Bitcoin: Train/Test split on labeled nodes
- FB15k-237: Pre-defined Train/Val/Test splits

**Checkpointing:** Full checkpoint per run
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'metrics': {'train_acc': train_acc, 'test_acc': test_acc}
}, f'results/{run_id}/checkpoint.pt')
```

**Rationale:** Explicit loops aid learning. Full checkpoints enable resume and comparison.

---

### Experiment Management

**Configuration:** OmegaConf with YAML files
```yaml
# configs/elliptic_gcn.yaml
model:
  name: GCN
  hidden_channels: 64
  num_layers: 2
  dropout: 0.5

training:
  lr: 0.01
  weight_decay: 5e-4
  epochs: 200

seed: 42
```

**Loading:**
```python
from omegaconf import OmegaConf
config = OmegaConf.load("configs/elliptic_gcn.yaml")
```

**Reproducibility:** Manual seed function
```python
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
```

---

### Results & Persistence

**Folder Structure:** Combined logs + checkpoints per run
```
results/
└── elliptic_gcn_2025-12-31_14-32-07/
    ├── metrics.json
    └── checkpoint.pt
```

**Naming Convention:** `{dataset}_{model}_{YYYY-MM-DD}_{HH-MM-SS}`

**JSON Log Format:**
```json
{
    "run_id": "elliptic_gcn_2025-12-31_14-32-07",
    "model": "GCN",
    "dataset": "elliptic",
    "device": "cuda",

    "hyperparameters": {
        "hidden_channels": 64,
        "num_layers": 2,
        "dropout": 0.5,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "epochs": 200
    },

    "results": {
        "training_time_seconds": 127.4,
        "final_loss": 0.3142,
        "train_acc": 0.9234,
        "test_acc": 0.8876
    },

    "seed": 42
}
```

**Model Saving:** state_dict only (smaller, standard practice)

**Results Comparison:** Manual inspection initially; add comparison script if needed later

