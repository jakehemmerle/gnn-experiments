"""Pytest fixtures for GNN experiments tests."""

import pytest
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from omegaconf import OmegaConf


@pytest.fixture
def device():
    """Get the test device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_node_classification_config():
    """Sample configuration for node classification."""
    return OmegaConf.create({
        "dataset": {"name": "elliptic"},
        "model": {
            "name": "GCN",
            "hidden_channels": 32,
            "dropout": 0.5,
            "out_channels": 2
        },
        "training": {
            "lr": 0.01,
            "weight_decay": 5e-4,
            "epochs": 5,
            "log_interval": 1
        },
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15
        },
        "seed": 42
    })


@pytest.fixture
def sample_link_prediction_config():
    """Sample configuration for link prediction."""
    return OmegaConf.create({
        "dataset": {"name": "fb15k237"},
        "model": {
            "name": "TransE",
            "hidden_channels": 32,
            "margin": 1.0
        },
        "training": {
            "lr": 0.01,
            "epochs": 5,
            "batch_size": 512,
            "log_every": 1
        },
        "evaluation": {
            "batch_size": 1000,
            "k": 10
        },
        "seed": 42
    })


@pytest.fixture
def mock_graph_data():
    """Create mock graph data for testing."""
    from torch_geometric.data import Data

    num_nodes = 100
    num_edges = 500
    num_features = 32

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))

    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def mock_node_classification_data(mock_graph_data):
    """Create mock NodeClassificationData for testing."""
    from src.datasets.elliptic import NodeClassificationData

    num_nodes = mock_graph_data.num_nodes

    # Create random masks
    indices = torch.randperm(num_nodes)
    n_train = int(0.7 * num_nodes)
    n_val = int(0.15 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_val]] = True
    test_mask[indices[n_train + n_val:]] = True

    return NodeClassificationData(
        data=mock_graph_data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=2
    )
