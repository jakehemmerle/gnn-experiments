"""Elliptic Bitcoin dataset loader with proper train/val/test splits."""

import torch
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.data import Data
from dataclasses import dataclass

from ..paths import get_data_dir


@dataclass
class NodeClassificationData:
    """Container for node classification data with proper splits.

    Provides a consistent interface for node classification datasets,
    ensuring train/val/test splits are always available.
    """
    data: Data
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int = 2

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_edges(self) -> int:
        return self.data.num_edges

    @property
    def num_features(self) -> int:
        return self.data.num_node_features

    def to(self, device: torch.device) -> "NodeClassificationData":
        """Move all tensors to the specified device."""
        return NodeClassificationData(
            data=self.data.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            num_classes=self.num_classes
        )


def load_elliptic(
    root: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> NodeClassificationData:
    """Load Elliptic Bitcoin dataset with train/val/test masks.

    Args:
        root: Path to store/load dataset (defaults to project data dir)
        train_ratio: Fraction of labeled nodes for training (default 0.7)
        val_ratio: Fraction of labeled nodes for validation (default 0.15)
            Note: test_ratio = 1 - train_ratio - val_ratio

    Returns:
        NodeClassificationData with properly split masks

    Raises:
        ValueError: If ratios don't sum to <= 1.0
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0"
        )

    if root is None:
        root = str(get_data_dir("elliptic"))

    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # Only labeled nodes (class 0=licit or 1=illicit) can be used
    # Class 2 = unknown, which we exclude from training/testing
    labeled_mask = (data.y == 0) | (data.y == 1)
    labeled_indices = labeled_mask.nonzero(as_tuple=True)[0]

    # Shuffle and split
    n_labeled = labeled_indices.size(0)
    perm = torch.randperm(n_labeled)

    n_train = int(train_ratio * n_labeled)
    n_val = int(val_ratio * n_labeled)

    train_idx = labeled_indices[perm[:n_train]]
    val_idx = labeled_indices[perm[n_train:n_train + n_val]]
    test_idx = labeled_indices[perm[n_train + n_val:]]

    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return NodeClassificationData(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=2
    )
