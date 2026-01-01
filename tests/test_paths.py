"""Tests for path utilities."""

import pytest
from pathlib import Path

from src.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    get_data_dir,
    get_results_dir,
    ensure_dirs,
)


class TestProjectPaths:
    """Tests for project path constants."""

    def test_project_root_exists(self):
        """Project root should exist."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_src(self):
        """Project root should contain src directory."""
        assert (PROJECT_ROOT / "src").exists()

    def test_data_dir_is_under_project_root(self):
        """Data directory should be under project root."""
        assert DATA_DIR.is_relative_to(PROJECT_ROOT)

    def test_results_dir_is_under_project_root(self):
        """Results directory should be under project root."""
        assert RESULTS_DIR.is_relative_to(PROJECT_ROOT)


class TestGetDataDir:
    """Tests for get_data_dir function."""

    def test_returns_base_data_dir(self):
        """Should return base data directory when no dataset specified."""
        data_dir = get_data_dir()
        assert data_dir == DATA_DIR

    def test_returns_dataset_specific_dir(self):
        """Should return dataset-specific directory."""
        data_dir = get_data_dir("elliptic")
        assert data_dir == DATA_DIR / "elliptic"


class TestGetResultsDir:
    """Tests for get_results_dir function."""

    def test_returns_base_results_dir(self):
        """Should return base results directory when no run_id specified."""
        results_dir = get_results_dir()
        assert results_dir == RESULTS_DIR

    def test_returns_run_specific_dir(self):
        """Should return run-specific directory."""
        results_dir = get_results_dir("test_run_123")
        assert results_dir == RESULTS_DIR / "test_run_123"


class TestEnsureDirs:
    """Tests for ensure_dirs function."""

    def test_creates_single_directory(self, temp_dir):
        """Should create a single directory."""
        new_dir = temp_dir / "new_dir"
        ensure_dirs(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, temp_dir):
        """Should create nested directories."""
        nested_dir = temp_dir / "a" / "b" / "c"
        ensure_dirs(nested_dir)
        assert nested_dir.exists()

    def test_creates_multiple_directories(self, temp_dir):
        """Should create multiple directories."""
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        ensure_dirs(dir1, dir2)
        assert dir1.exists()
        assert dir2.exists()

    def test_handles_existing_directories(self, temp_dir):
        """Should not fail if directory already exists."""
        existing = temp_dir / "existing"
        existing.mkdir()

        # Should not raise
        ensure_dirs(existing)
        assert existing.exists()
