# Epic 2 Implementation Guide

This guide documents the extensibility points and patterns for implementing Epic 2 (Knowledge Graph Link Prediction) in parallel with Epic 1.

## Epic 2 Overview

**Goal:** Train knowledge graph embedding models on FB15k-237 to predict missing facts/relations.

**Stories:**
- 2.1: Add FB15k-237 Dataset Loader
- 2.2: Add KGE Models (TransE, DistMult, RotatE)
- 2.3: Create FB15k TransE Experiment
- 2.4: Extend Config Launcher for Link Prediction

## Extensibility Points

### 1. Dataset Registry (`src/datasets/__init__.py`)

Add the FB15k-237 loader to the registry:

```python
from .fb15k import load_fb15k237

DATASETS: dict[str, tuple[Callable, str]] = {
    "elliptic": (load_elliptic, "node_classification"),
    "fb15k237": (load_fb15k237, "link_prediction"),  # Add this
}
```

### 2. Dataset Loader (`src/datasets/fb15k.py`)

Create the FB15k-237 loader following the elliptic.py pattern:

```python
"""FB15k-237 dataset loader for knowledge graph link prediction."""

from torch_geometric.datasets import FB15k_237

def load_fb15k237(root: str = './data/fb15k237'):
    """
    Load FB15k-237 dataset with train/val/test splits.

    Returns:
        train_data: Training triples
        val_data: Validation triples
        test_data: Test triples
        num_entities: Number of unique entities
        num_relations: Number of unique relations
    """
    train_data = FB15k_237(root=root, split='train')[0]
    val_data = FB15k_237(root=root, split='val')[0]
    test_data = FB15k_237(root=root, split='test')[0]

    num_entities = train_data.num_nodes
    num_relations = train_data.edge_type.max().item() + 1

    return train_data, val_data, test_data, num_entities, num_relations
```

### 3. Model Registry (`src/models/__init__.py`)

Add KGE models to the link prediction registry:

```python
from .kge import get_kge_model

LINK_PREDICTION_MODELS: dict[str, Callable] = {
    "TransE": lambda **kwargs: get_kge_model("TransE", **kwargs),
    "DistMult": lambda **kwargs: get_kge_model("DistMult", **kwargs),
    "RotatE": lambda **kwargs: get_kge_model("RotatE", **kwargs),
}
```

### 4. KGE Models (`src/models/kge.py`)

Create unified interface for PyG's built-in KGE models:

```python
"""Knowledge Graph Embedding models."""

from torch_geometric.nn.kge import TransE, DistMult, RotatE

def get_kge_model(name: str, num_entities: int, num_relations: int, hidden_channels: int = 50):
    """Factory function for KGE models."""
    models = {
        "TransE": TransE,
        "DistMult": DistMult,
        "RotatE": RotatE,
    }
    if name not in models:
        raise ValueError(f"Unknown KGE model: {name}")

    return models[name](
        num_nodes=num_entities,
        num_relations=num_relations,
        hidden_channels=hidden_channels
    )
```

### 5. Link Prediction Trainer (`src/trainers/link_predictor.py`)

Create trainer with MRR/Hits@k evaluation:

```python
"""Link prediction training loop."""

import torch
from torch_geometric.nn.kge import KGEModel

class LinkPredictionTrainer:
    """Trainer for knowledge graph link prediction."""

    def __init__(self, model: KGEModel, train_data, val_data, config, device, results_dir):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        # ... setup optimizer

    def train_epoch(self) -> float:
        """Train for one epoch using negative sampling."""
        self.model.train()
        # Use model.loss() with positive and negative samples
        pass

    @torch.no_grad()
    def evaluate(self, data) -> dict:
        """Evaluate with MRR and Hits@k metrics."""
        self.model.eval()
        # Use model.test() for ranking metrics
        return {"mrr": mrr, "hits@1": h1, "hits@10": h10}
```

### 6. Config Launcher (`src/run.py`)

Implement the `run_link_prediction()` function:

```python
def run_link_prediction(config, device, results_dir, run_id):
    """Run link prediction experiment."""
    dataset_name = config.dataset.name
    model_name = config.model.name

    # Load FB15k-237
    train_data, val_data, test_data, num_entities, num_relations = get_dataset(
        dataset_name,
        root=str(Path(__file__).parent.parent / "data" / dataset_name)
    )

    # Create KGE model
    model = get_model(
        model_name,
        task="link_prediction",
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_channels=config.model.hidden_channels
    ).to(device)

    # Train with LinkPredictionTrainer
    trainer = LinkPredictionTrainer(...)
    results = trainer.train()

    return results
```

## Config Template for Link Prediction

```yaml
# configs/fb15k_transe.yaml
dataset:
  name: fb15k237

task: link_prediction

model:
  name: TransE
  hidden_channels: 50
  margin: 1.0  # TransE-specific

training:
  lr: 0.01
  epochs: 500
  negative_samples: 10

seed: 42
```

## Evaluation Metrics

Link prediction uses ranking-based metrics:
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct entities
- **Hits@1**: Percentage of correct entities ranked first
- **Hits@10**: Percentage of correct entities in top 10

Target baselines (FB15k-237):
- TransE: MRR ~0.29, Hits@10 ~0.47
- DistMult: MRR ~0.24, Hits@10 ~0.42
- RotatE: MRR ~0.34, Hits@10 ~0.53

## File Structure After Epic 2

```text
src/
├── datasets/
│   ├── __init__.py      # Updated registry
│   ├── elliptic.py
│   └── fb15k.py         # NEW
├── models/
│   ├── __init__.py      # Updated registry
│   ├── gcn.py
│   ├── gat.py
│   ├── graphsage.py
│   └── kge.py           # NEW
├── trainers/
│   ├── __init__.py
│   ├── node_classifier.py
│   └── link_predictor.py  # NEW
└── run.py               # Updated with link prediction support

experiments/
├── elliptic_gcn.py
├── elliptic_gat.py
├── elliptic_graphsage.py
└── fb15k_transe.py      # NEW

configs/
├── elliptic_gcn.yaml
├── elliptic_gat.yaml
├── elliptic_graphsage.yaml
├── fb15k_transe.yaml    # NEW
├── fb15k_distmult.yaml  # NEW
└── template.yaml
```
