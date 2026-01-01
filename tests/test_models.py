"""Tests for model architectures."""

import pytest
import torch

from src.models import get_model, GCN, GAT, GraphSAGE
from src.models.kge import get_kge_model


class TestGCN:
    """Tests for GCN model."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = GCN(in_channels=32, hidden_channels=64, out_channels=2)
        x = torch.randn(100, 32)
        edge_index = torch.randint(0, 100, (2, 500))

        out = model(x, edge_index)
        assert out.shape == (100, 2)

    def test_training_mode(self):
        """Model should support training and eval modes."""
        model = GCN(in_channels=32, hidden_channels=64, out_channels=2)
        model.train()
        assert model.training

        model.eval()
        assert not model.training

    def test_parameters_are_learnable(self):
        """Model parameters should require gradients."""
        model = GCN(in_channels=32, hidden_channels=64, out_channels=2)
        for param in model.parameters():
            assert param.requires_grad


class TestGAT:
    """Tests for GAT model."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = GAT(in_channels=32, hidden_channels=8, out_channels=2, heads=4)
        x = torch.randn(100, 32)
        edge_index = torch.randint(0, 100, (2, 500))

        out = model(x, edge_index)
        assert out.shape == (100, 2)

    def test_multi_head_attention(self):
        """GAT should use multi-head attention."""
        model = GAT(in_channels=32, hidden_channels=8, out_channels=2, heads=8)
        # First layer should concatenate heads
        assert model.conv1.heads == 8


class TestGraphSAGE:
    """Tests for GraphSAGE model."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = GraphSAGE(in_channels=32, hidden_channels=64, out_channels=2)
        x = torch.randn(100, 32)
        edge_index = torch.randint(0, 100, (2, 500))

        out = model(x, edge_index)
        assert out.shape == (100, 2)

    def test_different_aggregators(self):
        """Should support different aggregation methods."""
        for aggr in ["mean", "max"]:
            model = GraphSAGE(in_channels=32, hidden_channels=64, out_channels=2, aggr=aggr)
            assert model.conv1.aggr == aggr


class TestGetModel:
    """Tests for model factory function."""

    def test_get_gcn(self):
        """Should create GCN model."""
        model = get_model(
            "GCN",
            task="node_classification",
            in_channels=32,
            hidden_channels=64,
            out_channels=2
        )
        assert isinstance(model, GCN)

    def test_get_gat(self):
        """Should create GAT model."""
        model = get_model(
            "GAT",
            task="node_classification",
            in_channels=32,
            hidden_channels=8,
            out_channels=2,
            heads=4
        )
        assert isinstance(model, GAT)

    def test_get_graphsage(self):
        """Should create GraphSAGE model."""
        model = get_model(
            "GraphSAGE",
            task="node_classification",
            in_channels=32,
            hidden_channels=64,
            out_channels=2
        )
        assert isinstance(model, GraphSAGE)

    def test_unknown_model_raises(self):
        """Should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown node classification model"):
            get_model("UnknownModel", task="node_classification", in_channels=32)

    def test_unknown_task_raises(self):
        """Should raise ValueError for unknown task."""
        with pytest.raises(ValueError, match="Unknown task type"):
            get_model("GCN", task="unknown_task", in_channels=32)


class TestKGEModels:
    """Tests for knowledge graph embedding models."""

    def test_get_transe(self):
        """Should create TransE model."""
        model = get_kge_model(
            "TransE",
            num_nodes=1000,
            num_relations=50,
            hidden_channels=32
        )
        assert model is not None

    def test_get_distmult(self):
        """Should create DistMult model."""
        model = get_kge_model(
            "DistMult",
            num_nodes=1000,
            num_relations=50,
            hidden_channels=32
        )
        assert model is not None

    def test_get_rotate(self):
        """Should create RotatE model."""
        model = get_kge_model(
            "RotatE",
            num_nodes=1000,
            num_relations=50,
            hidden_channels=32
        )
        assert model is not None

    def test_unknown_kge_model_raises(self):
        """Should raise ValueError for unknown KGE model."""
        with pytest.raises(ValueError, match="Unknown KGE model"):
            get_kge_model("UnknownKGE", num_nodes=1000, num_relations=50)

    def test_case_insensitive_model_names(self):
        """Model names should be case-insensitive."""
        model1 = get_kge_model("transe", num_nodes=100, num_relations=10)
        model2 = get_kge_model("TRANSE", num_nodes=100, num_relations=10)
        assert type(model1) == type(model2)
