"""Tests for trainer classes."""

import pytest
import torch
from pathlib import Path

from src.models import GCN
from src.trainers import NodeClassificationTrainer
from src.datasets.elliptic import NodeClassificationData


class TestNodeClassificationTrainer:
    """Tests for NodeClassificationTrainer."""

    def test_trainer_initialization(
        self,
        mock_node_classification_data,
        sample_node_classification_config,
        device,
        temp_dir
    ):
        """Trainer should initialize correctly."""
        model = GCN(
            in_channels=32,
            hidden_channels=sample_node_classification_config.model.hidden_channels,
            out_channels=2
        ).to(device)

        trainer = NodeClassificationTrainer(
            model=model,
            dataset=mock_node_classification_data,
            config=sample_node_classification_config,
            device=device,
            results_dir=temp_dir
        )

        assert trainer.model is model
        assert trainer.device == device
        assert trainer.results_dir == temp_dir

    def test_train_epoch_returns_loss(
        self,
        mock_node_classification_data,
        sample_node_classification_config,
        device,
        temp_dir
    ):
        """train_epoch should return a loss value."""
        model = GCN(
            in_channels=32,
            hidden_channels=sample_node_classification_config.model.hidden_channels,
            out_channels=2
        ).to(device)

        trainer = NodeClassificationTrainer(
            model=model,
            dataset=mock_node_classification_data,
            config=sample_node_classification_config,
            device=device,
            results_dir=temp_dir
        )

        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert loss > 0  # Cross-entropy loss should be positive

    def test_evaluate_returns_metrics(
        self,
        mock_node_classification_data,
        sample_node_classification_config,
        device,
        temp_dir
    ):
        """evaluate should return train, val, and test accuracies."""
        model = GCN(
            in_channels=32,
            hidden_channels=sample_node_classification_config.model.hidden_channels,
            out_channels=2
        ).to(device)

        trainer = NodeClassificationTrainer(
            model=model,
            dataset=mock_node_classification_data,
            config=sample_node_classification_config,
            device=device,
            results_dir=temp_dir
        )

        metrics = trainer.evaluate()

        assert "train_acc" in metrics
        assert "val_acc" in metrics
        assert "test_acc" in metrics

        # Accuracies should be between 0 and 1
        for key in ["train_acc", "val_acc", "test_acc"]:
            assert 0 <= metrics[key] <= 1

    def test_full_training_loop(
        self,
        mock_node_classification_data,
        sample_node_classification_config,
        device,
        temp_dir
    ):
        """Full training loop should complete and return results."""
        model = GCN(
            in_channels=32,
            hidden_channels=sample_node_classification_config.model.hidden_channels,
            out_channels=2
        ).to(device)

        trainer = NodeClassificationTrainer(
            model=model,
            dataset=mock_node_classification_data,
            config=sample_node_classification_config,
            device=device,
            results_dir=temp_dir
        )

        results = trainer.train(log_interval=2)

        assert "final_loss" in results
        assert "train_acc" in results
        assert "val_acc" in results
        assert "test_acc" in results
        assert "best_val_acc" in results
        assert "training_time_seconds" in results

        assert results["training_time_seconds"] > 0


class TestNodeClassificationData:
    """Tests for NodeClassificationData dataclass."""

    def test_to_device(self, mock_node_classification_data, device):
        """Should move all tensors to device."""
        moved = mock_node_classification_data.to(device)

        assert moved.data.x.device == device
        assert moved.train_mask.device == device
        assert moved.val_mask.device == device
        assert moved.test_mask.device == device

    def test_properties(self, mock_node_classification_data):
        """Properties should return correct values."""
        data = mock_node_classification_data

        assert data.num_nodes == 100
        assert data.num_features == 32
        assert data.num_classes == 2

    def test_mask_coverage(self, mock_node_classification_data):
        """Train, val, and test masks should cover all nodes without overlap."""
        data = mock_node_classification_data

        # No overlap
        train_val_overlap = (data.train_mask & data.val_mask).sum()
        train_test_overlap = (data.train_mask & data.test_mask).sum()
        val_test_overlap = (data.val_mask & data.test_mask).sum()

        assert train_val_overlap == 0
        assert train_test_overlap == 0
        assert val_test_overlap == 0

        # Complete coverage
        total_covered = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
        assert total_covered == data.num_nodes
