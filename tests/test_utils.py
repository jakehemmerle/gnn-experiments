"""Tests for utility functions."""

import pytest
import torch
import numpy as np
import json
from pathlib import Path

from src.utils import set_seed, save_checkpoint, save_results


class TestSetSeed:
    """Tests for seed setting."""

    def test_torch_reproducibility(self):
        """Torch operations should be reproducible with same seed."""
        set_seed(42)
        a1 = torch.randn(10)

        set_seed(42)
        a2 = torch.randn(10)

        assert torch.allclose(a1, a2)

    def test_numpy_reproducibility(self):
        """Numpy operations should be reproducible with same seed."""
        set_seed(42)
        a1 = np.random.randn(10)

        set_seed(42)
        a2 = np.random.randn(10)

        assert np.allclose(a1, a2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        set_seed(42)
        a1 = torch.randn(10)

        set_seed(123)
        a2 = torch.randn(10)

        assert not torch.allclose(a1, a2)


class TestSaveCheckpoint:
    """Tests for checkpoint saving."""

    def test_saves_model_state(self, temp_dir):
        """Should save model state dict."""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = temp_dir / "checkpoint.pt"
        save_checkpoint(checkpoint_path, model, optimizer, epoch=5, metrics={"acc": 0.95})

        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["acc"] == 0.95

    def test_checkpoint_is_loadable(self, temp_dir):
        """Saved checkpoint should be loadable."""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Save
        checkpoint_path = temp_dir / "checkpoint.pt"
        save_checkpoint(checkpoint_path, model, optimizer, epoch=5, metrics={})

        # Load
        new_model = torch.nn.Linear(10, 2)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


class TestSaveResults:
    """Tests for results saving."""

    def test_saves_json(self, temp_dir):
        """Should save results as JSON."""
        results = {
            "run_id": "test_run",
            "accuracy": 0.95,
            "loss": 0.123
        }

        results_path = temp_dir / "results.json"
        save_results(results_path, results)

        assert results_path.exists()

        with open(results_path) as f:
            loaded = json.load(f)

        assert loaded["run_id"] == "test_run"
        assert loaded["accuracy"] == 0.95

    def test_handles_nested_dicts(self, temp_dir):
        """Should handle nested dictionaries."""
        results = {
            "hyperparameters": {
                "lr": 0.01,
                "epochs": 100
            },
            "metrics": {
                "train": {"acc": 0.95},
                "test": {"acc": 0.93}
            }
        }

        results_path = temp_dir / "results.json"
        save_results(results_path, results)

        with open(results_path) as f:
            loaded = json.load(f)

        assert loaded["hyperparameters"]["lr"] == 0.01
        assert loaded["metrics"]["test"]["acc"] == 0.93
