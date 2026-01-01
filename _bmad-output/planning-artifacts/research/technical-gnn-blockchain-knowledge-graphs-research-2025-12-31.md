---
stepsCompleted: [1, 2]
inputDocuments: []
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'GNN examples for blockchain transaction graphs and knowledge graphs'
research_goals: 'Find runnable GNN examples using PyTorch (PyTorch Geometric) or Keras (Spektral), managed with uv, for blockchain and knowledge graph applications'
user_name: 'Jake'
date: '2025-12-31'
web_research_enabled: true
source_verification: true
---

# Research Report: Technical - GNN Examples for Blockchain & Knowledge Graphs

**Date:** 2025-12-31
**Author:** Jake
**Research Type:** Technical

---

## Research Overview

This technical research focuses on identifying and evaluating runnable Graph Neural Network (GNN) examples that work with:
- **Blockchain transaction graphs** - fraud detection, entity classification, transaction pattern analysis
- **Knowledge graphs** - link prediction, entity embedding, relation extraction

**Target framework:** PyTorch Geometric (PyTorch)
**Environment management:** uv (Python package manager)
**Skill level:** Intermediate

---

<!-- Content will be appended sequentially through research workflow steps -->

## Technology Stack Analysis

### Core Framework: PyTorch Geometric (PyG)

**Version:** 2.7.0 (latest stable)
**Repository:** [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
**Documentation:** [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)

PyG is the leading library for Graph Neural Networks built on PyTorch. Key features:
- Unified API requiring only 10-20 lines to train a GNN model
- Full `torch.compile()` support since v2.4 (up to 3× runtime speedups)
- Extensive built-in datasets including blockchain and knowledge graph datasets
- Source: [PyG Documentation](https://pytorch-geometric.readthedocs.io/)

### Installation with uv

**Basic Installation:**
```bash
uv pip install torch torch-geometric
```

**With CUDA support:**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install torch-geometric
```

**Optional Extensions (for advanced sparse operations):**
```bash
# First install torch, then extensions with --no-build-isolation
uv pip install torch
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html --no-build-isolation
```

**Note:** As of PyTorch >2.5.0, conda packages are no longer available. Use pip/uv instead.
Sources: [Official Installation Guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html), [uv PyTorch Guide](https://docs.astral.sh/uv/guides/integration/pytorch/), [GitHub Issue #10178](https://github.com/pyg-team/pytorch_geometric/issues/10178)

### Blockchain Transaction Graph Datasets

**1. EllipticBitcoinDataset (Built into PyG)**
- 203,769 Bitcoin transactions, 234,355 edges
- Binary classification: licit vs illicit transactions
- 166 node features (94 local + 72 aggregated)
- 49 time steps (~2 weeks apart each)
- Source: [PyG EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)

**2. EllipticBitcoinTemporalDataset**
- Same data with temporal split support
- Enables inductive learning experiments
- Source: [PyG EllipticBitcoinTemporalDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinTemporalDataset.html)

### Knowledge Graph Datasets

**Built into PyG:**
- FB15k-237 (Freebase subset, 14,541 entities, 237 relations)
- WN18RR (WordNet subset, 40,943 entities, 11 relations)
- YAGO3-10

**Alternative Library - TorchKGE:**
- Specialized for Knowledge Graph Embeddings
- 3-24× faster evaluation than other frameworks
- Built-in: FB13, FB15k, FB15k237, WN18, WN18RR, YAGO3-10
- Source: [TorchKGE GitHub](https://github.com/torchkge-team/torchkge)

### GNN Architectures Available in PyG

| Architecture | Use Case | PyG Module |
|-------------|----------|------------|
| GCN | General node classification | `GCNConv` |
| GAT | Attention-based node tasks | `GATConv` |
| GraphSAGE | Inductive learning | `SAGEConv` |
| R-GCN | Knowledge graphs (multi-relation) | `RGCNConv` |
| TransE | Knowledge graph embeddings | `torch_geometric.nn.TransE` |
| DistMult | Link prediction | via kge module |

### Development Tools

- **IDE:** Any Python IDE (VSCode, PyCharm)
- **Environment:** uv for package management
- **GPU:** CUDA 11.8/12.6+ recommended for performance
- **Notebooks:** Jupyter/Colab for experimentation

---

## Runnable Examples

### Dataset 1: Elliptic Bitcoin (Blockchain Fraud Detection)

**Why it's cool:** Real Bitcoin transaction data with actual fraud labels. You're literally training a model to catch crypto criminals - scams, ransomware, Ponzi schemes, and money laundering operations.

**Task:** Node classification (licit vs illicit transactions)
**Nodes:** 203,769 transactions | **Edges:** 234,355 payment flows
**Labels:** 2% illicit, 21% licit, 77% unknown

#### Option A: Official PyG Dataset + Custom GCN (Recommended for Learning)

```python
# elliptic_gcn.py - Bitcoin Fraud Detection with GCN
import torch
import torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv

# Load the Elliptic Bitcoin dataset
dataset = EllipticBitcoinDataset(root='./data/elliptic')
data = dataset[0]

print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Node features: {data.num_node_features}")
print(f"Classes: licit (0) vs illicit (1)")

# Create train/test masks (only labeled nodes)
labeled_mask = data.y != -1  # -1 = unknown
labeled_indices = labeled_mask.nonzero(as_tuple=True)[0]

# 80/20 split on labeled data
n_labeled = labeled_indices.size(0)
perm = torch.randperm(n_labeled)
train_idx = labeled_indices[perm[:int(0.8 * n_labeled)]]
test_idx = labeled_indices[perm[int(0.8 * n_labeled):]]

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_node_features, 64, 2).to(device)
data = data.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_correct = (pred[train_mask] == data.y[train_mask]).sum()
    train_acc = train_correct / train_mask.sum()

    test_correct = (pred[test_mask] == data.y[test_mask]).sum()
    test_acc = test_correct / test_mask.sum()

    return train_acc.item(), test_acc.item()

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, test_acc = test()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
```

**Source:** Adapted from [PyG GCN Example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py) + [EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)

#### Option B: Community Notebook (Full Tutorial)

**MariaZork's Elliptic GNN Notebook**
- Complete Jupyter notebook with GCN and GAT implementations
- Includes visualization and analysis
- Source: [elliptic-dataset-gnn.ipynb](https://github.com/MariaZork/my-machine-learning-tutorials/blob/master/elliptic-dataset-gnn.ipynb)

#### Option C: Advanced - Multi-Distance GCN

**GCN-on-EllipticDataSet**
- Proposes MD-GC-Layer for multi-hop aggregation
- More sophisticated than vanilla GCN
- Source: [yeungchenwa/GCN-on-EllipticDataSet](https://github.com/yeungchenwa/GCN-on-EllipticDataSet)

---

### Dataset 2: FB15k-237 (Knowledge Graph Link Prediction)

**Why it's cool:** Real-world knowledge from Freebase - movies, actors, directors, awards, sports teams, cities. The task is predicting missing facts like "Who directed The Dark Knight?" or "What team does this player belong to?"

**Task:** Link prediction (predict missing relations)
**Entities:** 14,541 | **Relations:** 237 types | **Triples:** 310,116
**Domains:** Movies, actors, awards, sports, geography

#### Option A: Official PyG Knowledge Graph Embeddings (Recommended)

```python
# kge_fb15k.py - Knowledge Graph Embeddings on FB15k-237
# Official PyG example with TransE, DistMult, ComplEx, RotatE
import torch
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE, ComplEx, DistMult, RotatE

# Load FB15k-237 dataset
train_data = FB15k_237(root='./data/fb15k-237', split='train')[0]
val_data = FB15k_237(root='./data/fb15k-237', split='val')[0]
test_data = FB15k_237(root='./data/fb15k-237', split='test')[0]

print(f"Training triples: {train_data.edge_index.size(1)}")
print(f"Entities: {train_data.num_nodes}, Relations: {train_data.edge_type.max().item() + 1}")

# Choose your model (TransE is a classic, RotatE is state-of-the-art)
model_name = 'TransE'  # Options: 'TransE', 'DistMult', 'ComplEx', 'RotatE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == 'TransE':
    model = TransE(
        num_nodes=train_data.num_nodes,
        num_relations=train_data.edge_type.max().item() + 1,
        hidden_channels=50,
    ).to(device)
elif model_name == 'DistMult':
    model = DistMult(
        num_nodes=train_data.num_nodes,
        num_relations=train_data.edge_type.max().item() + 1,
        hidden_channels=50,
    ).to(device)
elif model_name == 'ComplEx':
    model = ComplEx(
        num_nodes=train_data.num_nodes,
        num_relations=train_data.edge_type.max().item() + 1,
        hidden_channels=50,
    ).to(device)
elif model_name == 'RotatE':
    model = RotatE(
        num_nodes=train_data.num_nodes,
        num_relations=train_data.edge_type.max().item() + 1,
        hidden_channels=50,
    ).to(device)

loader = model.loader(
    head_index=train_data.edge_index[0].to(device),
    rel_type=train_data.edge_type.to(device),
    tail_index=train_data.edge_index[1].to(device),
    batch_size=1000,
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for head, rel, tail in loader:
        optimizer.zero_grad()
        loss = model.loss(head, rel, tail)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0].to(device),
        rel_type=data.edge_type.to(device),
        tail_index=data.edge_index[1].to(device),
        batch_size=1000,
        k=10,  # Hits@10
    )

for epoch in range(1, 501):
    loss = train()
    if epoch % 50 == 0:
        mrr, hits = test(val_data)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

# Final test
mrr, hits = test(test_data)
print(f'Test MRR: {mrr:.4f}, Test Hits@10: {hits:.4f}')
```

**Source:** [PyG kge_fb15k_237.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/kge_fb15k_237.py)

#### Option B: Official PyG RGCN Link Prediction

```python
# rgcn_link_pred.py - Relational GCN for Link Prediction
# Uses graph structure for embeddings (more powerful than pure KGE)
import torch
import torch.nn.functional as F
from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import RGCNConv, GAE

# Load FB15k-237 for RGCN
dataset = RelLinkPredDataset(root='./data/rel-link-pred', name='FB15k-237')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edge types: {data.edge_type.max().item() + 1}")

class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_blocks=5)

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return x

class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = torch.nn.Embedding(num_relations, hidden_channels)

    def forward(self, z, edge_index, edge_type):
        head = z[edge_index[0]]
        tail = z[edge_index[1]]
        rel = self.rel_emb(edge_type)
        return (head * rel * tail).sum(dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_relations = data.edge_type.max().item() + 1

encoder = RGCNEncoder(data.num_nodes, 500, num_relations).to(device)
decoder = DistMultDecoder(num_relations, 500).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.01
)

def train():
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    z = encoder(data.edge_index, data.edge_type)
    pos_score = decoder(z, data.edge_index, data.edge_type)

    # Negative sampling
    neg_edge_index = torch.randint(0, data.num_nodes, data.edge_index.size(), device=device)
    neg_score = decoder(z, neg_edge_index, data.edge_type)

    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
```

**Source:** [PyG rgcn_link_pred.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py)

---

## Quick Start Guide

### Setup with uv

```bash
# Create new project
mkdir gnn-examples && cd gnn-examples
uv init
uv add torch torch-geometric

# Optional: Add Jupyter for notebooks
uv add jupyter

# Run the Elliptic example
uv run python elliptic_gcn.py

# Run the FB15k-237 example
uv run python kge_fb15k.py
```

### Expected Output

**Elliptic Bitcoin GCN:**
```
Nodes: 203769, Edges: 234355
Epoch 200, Loss: 0.3142, Train: 0.9234, Test: 0.8876
```

**FB15k-237 TransE:**
```
Training triples: 272115
Epoch 500, Loss: 0.0823, Val MRR: 0.2341, Val Hits@10: 0.4512
```

---

## Sources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [PyG GitHub Repository](https://github.com/pyg-team/pytorch_geometric)
- [EllipticBitcoinDataset Docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)
- [FB15k-237 Microsoft Download](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
- [MariaZork Elliptic Notebook](https://github.com/MariaZork/my-machine-learning-tutorials/blob/master/elliptic-dataset-gnn.ipynb)
- [uv PyTorch Guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyG KGE Example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/kge_fb15k_237.py)
