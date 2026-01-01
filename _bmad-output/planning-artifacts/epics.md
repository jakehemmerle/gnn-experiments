---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - architecture.md
  - research/technical-gnn-blockchain-knowledge-graphs-research-2025-12-31.md
---

# GNN Experimentation System - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for gnn-experimentation-system, decomposing the requirements from the Architecture and Research documents into implementable stories.

## Requirements Inventory

### Functional Requirements

**Node Classification (Elliptic Bitcoin - Fraud Detection):**
FR1: Execute GNN training pipelines on Elliptic Bitcoin dataset ✅ DONE
FR2: Support GCN architecture for node classification ✅ DONE
FR3: Support GAT architecture for node classification
FR4: Support GraphSAGE architecture (scalable, inductive learning)

**Link Prediction (FB15k-237 - Knowledge Graphs):**
FR5: Execute GNN training pipelines on FB15k-237 dataset
FR6: Support TransE for knowledge graph embeddings
FR7: Support DistMult for knowledge graph link prediction
FR8: Support RotatE for state-of-the-art KG embeddings

**Infrastructure (Done):**
FR9: Handle dataset loading with automatic train/test splitting ✅ DONE
FR10: Provide reproducible training runs with configurable random seeds ✅ DONE
FR11: Save and load trained model checkpoints ✅ DONE
FR12: Track and persist experiment results (metrics, hyperparameters, loss) ✅ DONE
FR13: Support YAML-based configuration management with OmegaConf ✅ DONE

**Hyperparameter Optimization:**
FR14: Support grid search for hyperparameter tuning
FR15: Support automatic optimization via Optuna (Bayesian optimization)

**Backlog (Future):**
BL1: GPS Graph Transformer - hybrid MPNN + Transformer architecture
BL2: TGN (Temporal Graph Networks) - for dynamic/temporal graphs
BL3: Pareto-based genetic optimization (NSGA-II/III) for multi-objective HPO

### NonFunctional Requirements

NFR1: CUDA-first execution with automatic CPU fallback ✅ DONE
NFR2: Clean environment isolation via uv ✅ DONE
NFR3: Fast experimentation cycles (minimal boilerplate)
NFR4: Extensible architecture for adding new models/datasets
NFR5: Python 3.11+ compatibility ✅ DONE
NFR6: PyTorch Geometric 2.7.0+ compatibility ✅ DONE
NFR7: PyTorch 2.5+ with torch.compile support for performance

### Additional Requirements

From Architecture:
- Custom minimal project structure (no heavy frameworks like PyTorch Lightning)
- One class per model file pattern
- Explicit training loops for learning clarity
- Combined logs + checkpoints per run in results/ folder
- Run ID naming: {dataset}_{model}_{YYYY-MM-DD}_{HH-MM-SS}

From Research:
- Use PyG built-in datasets (EllipticBitcoinDataset, FB15k_237)
- Evaluation metrics: Accuracy for node classification, MRR/Hits@10 for link prediction

### FR Coverage Map

| FR | Epic | Description |
|----|------|-------------|
| FR1 | ✅ Done | Elliptic Bitcoin training |
| FR2 | ✅ Done | GCN architecture |
| FR3 | Epic 1 | GAT architecture |
| FR4 | Epic 1 | GraphSAGE architecture |
| FR5 | Epic 2 | FB15k-237 dataset |
| FR6 | Epic 2 | TransE |
| FR7 | Epic 2 | DistMult |
| FR8 | Epic 2 | RotatE |
| FR9-13 | ✅ Done | Infrastructure |
| FR14 | Epic 3 (Story 3.1) | Grid search |
| FR15 | Epic 3 (Story 3.2) | Optuna |

## Epic List

### Epic 1: Enhanced Fraud Detection Models
**Goal:** Train and compare multiple GNN architectures on Elliptic Bitcoin to find the best fraud detection model.

**User Outcome:** Researchers can experiment with GAT and GraphSAGE alongside GCN, comparing accuracy and training characteristics for fraud detection.

**FRs covered:** FR3 (GAT), FR4 (GraphSAGE)

**Parallelizable:** ✅ Can run in parallel with Epic 2

---

### Epic 2: Knowledge Graph Link Prediction
**Goal:** Train knowledge graph embedding models on FB15k-237 to predict missing facts/relations.

**User Outcome:** Researchers can train TransE, DistMult, and RotatE models, comparing MRR/Hits@10 metrics for knowledge graph completion tasks.

**FRs covered:** FR5 (FB15k-237 dataset), FR6 (TransE), FR7 (DistMult), FR8 (RotatE)

**Parallelizable:** ✅ Can run in parallel with Epic 1

---

### Epic 3: Hyperparameter Optimization
**Goal:** Automatically discover optimal hyperparameters for any model configuration.

**User Outcome:** Researchers can run grid search or Optuna-based optimization to find best learning rate, hidden dimensions, dropout, etc. without manual tuning.

**FRs covered:** FR14 (Grid search), FR15 (Optuna)

**Parallelizable:** ⚡ Can start in parallel, applies to models from Epic 1 & 2

---

### Backlog Epic: Advanced Architectures (Future)
**Goal:** Expand to cutting-edge architectures for specialized use cases.

**Items:**
- BL1: GPS Graph Transformer (hybrid MPNN + Transformer)
- BL2: TGN for temporal/dynamic graphs
- BL3: Pareto genetic optimization (NSGA-II/III)

---

## Parallelization Map

```
         ┌─────────────────┐     ┌─────────────────┐
         │    EPIC 1       │     │    EPIC 2       │
         │ Fraud Detection │     │ Knowledge Graph │
         │  (GAT, SAGE)    │     │ (TransE, etc.)  │
         └────────┬────────┘     └────────┬────────┘
                  │    Can run in         │
                  │◄──── PARALLEL ───────►│
                  └───────────┬───────────┘
                              ▼
                    ┌─────────────────┐
                    │    EPIC 3       │
                    │   HPO (Grid +   │
                    │     Optuna)     │
                    └─────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    BACKLOG      │
                    │ (GPS, TGN, etc) │
                    └─────────────────┘
```

---

## Epic 1: Enhanced Fraud Detection Models - Stories

### Story 1.1: Add GAT Model

**As a** ML researcher,
**I want** a GAT (Graph Attention Network) model implementation,
**So that** I can leverage attention mechanisms for fraud detection on Elliptic Bitcoin.

**Acceptance Criteria:**

**Given** the existing project structure with GCN model
**When** I create `src/models/gat.py`
**Then** it contains a 2-layer GAT class with configurable heads, hidden channels, and dropout
**And** the forward method accepts (x, edge_index) matching GCN interface
**And** default hyperparameters are heads=8 (layer 1), heads=1 (output)

---

### Story 1.2: Create Elliptic GAT Experiment

**As a** ML researcher,
**I want** a runnable GAT experiment on Elliptic Bitcoin,
**So that** I can compare GAT accuracy against the GCN baseline (~95.7%).

**Acceptance Criteria:**

**Given** the GAT model from Story 1.1
**When** I run `uv run python experiments/elliptic_gat.py`
**Then** it trains GAT on Elliptic Bitcoin for 200 epochs
**And** prints train/test accuracy every 20 epochs
**And** saves results to `results/elliptic_gat_{timestamp}/` including:
  - `metrics.json` (results + hyperparameters)
  - `config.yaml` (full config used for this run)
  - `checkpoint.pt` (model weights)

---

### Story 1.3: Add GraphSAGE Model

**As a** ML researcher,
**I want** a GraphSAGE model implementation,
**So that** I can use inductive learning for scalable fraud detection.

**Acceptance Criteria:**

**Given** the existing model structure
**When** I create `src/models/graphsage.py`
**Then** it contains a 2-layer GraphSAGE class using SAGEConv
**And** supports configurable hidden channels, dropout, and aggregation type
**And** the forward method matches the GCN/GAT interface

---

### Story 1.4: Create Elliptic GraphSAGE Experiment

**As a** ML researcher,
**I want** a runnable GraphSAGE experiment on Elliptic Bitcoin,
**So that** I can compare its performance for fraud detection.

**Acceptance Criteria:**

**Given** the GraphSAGE model from Story 1.3
**When** I run `uv run python experiments/elliptic_graphsage.py`
**Then** it trains GraphSAGE on Elliptic Bitcoin for 200 epochs
**And** prints train/test accuracy every 20 epochs
**And** saves results to `results/elliptic_graphsage_{timestamp}/` including:
  - `metrics.json`, `config.yaml`, `checkpoint.pt`

---

### Story 1.5: Add Config-Driven Experiment Launcher

**As a** ML researcher,
**I want** to launch any experiment from a config file,
**So that** I can easily run variations without editing code.

**Acceptance Criteria:**

**Given** a YAML config file (e.g., `my_experiment.yaml`)
**When** I run `uv run python -m src.run --config my_experiment.yaml`
**Then** it loads the config, selects the correct dataset/model, and runs training
**And** the config specifies: dataset, model, hyperparameters, epochs, seed
**And** a template config `configs/template.yaml` exists with documented options
**And** the full config is saved to the results folder

---

## Epic 2: Knowledge Graph Link Prediction - Stories

### Story 2.1: Add FB15k-237 Dataset Loader

**As a** ML researcher,
**I want** a FB15k-237 dataset loader,
**So that** I can train knowledge graph models on real-world Freebase data.

**Acceptance Criteria:**

**Given** the existing datasets structure
**When** I create `src/datasets/fb15k.py`
**Then** it loads FB15k-237 using PyG's `FB15k_237` class
**And** returns train/val/test splits as separate Data objects
**And** exposes num_entities and num_relations for model initialization
**And** follows the same pattern as elliptic.py

---

### Story 2.2: Add KGE Models (TransE, DistMult, RotatE)

**As a** ML researcher,
**I want** knowledge graph embedding model wrappers,
**So that** I can train and compare TransE, DistMult, and RotatE on FB15k-237.

**Acceptance Criteria:**

**Given** PyG's built-in KGE models
**When** I create `src/models/kge.py`
**Then** it provides a unified interface for TransE, DistMult, and RotatE
**And** each model is selectable via a `model_name` parameter
**And** all share common hyperparameters: hidden_channels, num_entities, num_relations
**And** includes a `get_kge_model(name, ...)` factory function

---

### Story 2.3: Create FB15k TransE Experiment

**As a** ML researcher,
**I want** a runnable TransE experiment on FB15k-237,
**So that** I can establish a baseline for knowledge graph completion.

**Acceptance Criteria:**

**Given** the FB15k loader and KGE models
**When** I run `uv run python experiments/fb15k_transe.py`
**Then** it trains TransE on FB15k-237 for 500 epochs
**And** prints loss and validation MRR/Hits@10 every 50 epochs
**And** saves results to `results/fb15k_transe_{timestamp}/` including:
  - `metrics.json`, `config.yaml`, `checkpoint.pt`
**And** final test MRR/Hits@10 is reported

---

### Story 2.4: Extend Config Launcher for Link Prediction

**As a** ML researcher,
**I want** the config launcher to support link prediction tasks,
**So that** I can run DistMult and RotatE experiments via config files.

**Acceptance Criteria:**

**Given** the config launcher from Story 1.5
**When** I specify `task: link_prediction` and `model: DistMult` in config
**Then** it runs the appropriate training loop with MRR/Hits@k evaluation
**And** supports all three KGE models (TransE, DistMult, RotatE)
**And** template config includes link prediction examples

---

## Epic 3: Hyperparameter Optimization - Stories

### Story 3.1: Add Grid Search HPO

**As a** ML researcher,
**I want** to run grid search over hyperparameter combinations,
**So that** I can systematically explore the hyperparameter space without manual iteration.

**Acceptance Criteria:**

**Given** a config file with `hpo.method: grid` and `hpo.search_space` defined
**When** I run `uv run python -m src.hpo --config configs/hpo_grid.yaml`
**Then** it runs all combinations of specified hyperparameters
**And** saves results for each run to `results/hpo_{timestamp}/`
**And** generates a summary CSV with all runs ranked by performance
**And** identifies the best configuration automatically

---

### Story 3.2: Add Optuna Bayesian HPO

**As a** ML researcher,
**I want** Optuna-based Bayesian optimization for hyperparameters,
**So that** I can efficiently find optimal settings without exhaustive search.

**Acceptance Criteria:**

**Given** a config file with `hpo.method: optuna` and search bounds
**When** I run `uv run python -m src.hpo --config configs/hpo_optuna.yaml`
**Then** it runs Optuna with TPE sampler for the specified number of trials
**And** supports pruning of unpromising trials (MedianPruner)
**And** saves Optuna study to SQLite for resumability
**And** outputs best hyperparameters and final metrics

---

### Story 3.3: Add HPO Results Report

**As a** ML researcher,
**I want** a simple comparison view of HPO results,
**So that** I can quickly identify winning configurations and understand parameter sensitivity.

**Acceptance Criteria:**

**Given** completed HPO runs in `results/hpo_{timestamp}/`
**When** I run `uv run python -m src.hpo.report --run-dir results/hpo_xxx`
**Then** it generates a markdown report with:
  - Top 5 configurations ranked by metric
  - Parameter importance (for Optuna runs)
  - Best config YAML ready for production use

---

## Backlog Epic: Advanced Architectures - Stories

### BL1: GPS Graph Transformer

**As a** ML researcher,
**I want** a GPS (General, Powerful, Scalable) Graph Transformer implementation,
**So that** I can experiment with hybrid MPNN + Transformer architectures for improved expressiveness.

**Notes:** Combines message-passing with global attention. Useful for tasks requiring long-range dependencies.

---

### BL2: Temporal Graph Networks (TGN)

**As a** ML researcher,
**I want** TGN support for dynamic/temporal graphs,
**So that** I can model evolving relationships over time (e.g., transaction sequences).

**Notes:** Memory-based architecture tracking node states across time. Requires temporal dataset support.

---

### BL3: Pareto Genetic Optimization (NSGA-II/III)

**As a** ML researcher,
**I want** multi-objective HPO using genetic algorithms,
**So that** I can optimize for multiple competing objectives (e.g., accuracy vs training time).

**Notes:** NSGA-II/III for Pareto-optimal solution sets. Useful when trade-offs matter.

---

### BL4: PROTEIN Cost-Aware Bayesian HPO

**As a** ML researcher,
**I want** PufferAI's PROTEIN algorithm for cost-aware hyperparameter optimization,
**So that** I can find configurations that balance performance and training cost on the Pareto frontier.

**Background:**
- Based on ImbueAI's CARBS (Cost AwaRe Bayesian Search)
- Models Pareto frontier of cost (wall-clock time/steps) vs score
- Fixes CARBS edge cases (degenerate random sampling, noise sensitivity)
- Fixes data normalization for Gaussian Processes
- Simplified implementation (~500 LOC vs 2,500)
- Part of PufferLib 3.0 (open source at puffer.ai)

**Reference:** https://x.com/jsuarez5341/status/1938287195305005500

**Stories (to be detailed when prioritized):**
- Integrate PufferLib sweep module as dependency
- Add cost tracking (time/steps) to training loop
- Implement Pareto-aware trial selection
- Add cost vs performance visualization

