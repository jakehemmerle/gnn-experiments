"""Tests for configuration utilities."""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from omegaconf import OmegaConf

from src.config import (
    get_device,
    load_config,
    generate_run_id,
    validate_config,
    merge_with_defaults,
    get_config_defaults,
)


class TestGetDevice:
    """Tests for device selection."""

    def test_returns_torch_device(self):
        """Should return a torch.device object."""
        import torch
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_cuda_or_cpu(self):
        """Should return either cuda or cpu."""
        device = get_device()
        assert device.type in ("cuda", "cpu")


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_valid_config(self, sample_node_classification_config, temp_dir):
        """Should load a valid YAML config."""
        config_path = temp_dir / "test_config.yaml"
        OmegaConf.save(sample_node_classification_config, config_path)

        loaded = load_config(config_path)
        assert loaded.dataset.name == "elliptic"
        assert loaded.model.name == "GCN"

    def test_load_nonexistent_config_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_run_id_format(self):
        """Should generate properly formatted run ID."""
        run_id = generate_run_id("elliptic", "GCN")

        assert run_id.startswith("elliptic_GCN_")
        # Should have timestamp format YYYY-MM-DD_HH-MM-SS
        parts = run_id.split("_")
        assert len(parts) >= 4

    def test_run_ids_are_unique(self):
        """Sequential run IDs should be unique."""
        import time
        id1 = generate_run_id("test", "model")
        time.sleep(1.1)  # Delay to ensure different timestamp (second-level granularity)
        id2 = generate_run_id("test", "model")
        assert id1.startswith("test_model_")
        assert id2.startswith("test_model_")
        assert id1 != id2, "Run IDs should be unique"


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_valid_config_returns_empty_list(self, sample_node_classification_config):
        """Valid config should return no errors."""
        errors = validate_config(sample_node_classification_config)
        assert errors == []

    def test_missing_dataset_section(self):
        """Should report missing dataset section."""
        config = OmegaConf.create({
            "model": {"name": "GCN", "hidden_channels": 64},
            "training": {"lr": 0.01, "epochs": 100}
        })
        errors = validate_config(config)
        assert "Missing required section: dataset" in errors

    def test_missing_model_name(self):
        """Should report missing model name."""
        config = OmegaConf.create({
            "dataset": {"name": "elliptic"},
            "model": {"hidden_channels": 64},
            "training": {"lr": 0.01, "epochs": 100}
        })
        errors = validate_config(config)
        assert "Missing required field: model.name" in errors


class TestMergeWithDefaults:
    """Tests for configuration merging."""

    def test_preserves_user_values(self):
        """User values should override defaults."""
        user_config = OmegaConf.create({
            "seed": 123,
            "data": {"train_ratio": 0.8}
        })
        merged = merge_with_defaults(user_config)

        assert merged.seed == 123
        assert merged.data.train_ratio == 0.8

    def test_fills_missing_with_defaults(self):
        """Missing values should be filled from defaults."""
        user_config = OmegaConf.create({
            "seed": 42
        })
        merged = merge_with_defaults(user_config)

        defaults = get_config_defaults()
        assert merged.data.val_ratio == defaults["data"]["val_ratio"]
        assert merged.model.dropout == defaults["model"]["dropout"]
