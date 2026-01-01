"""Elliptic Bitcoin dataset loader."""

import torch
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.data import Data
from pathlib import Path


def load_elliptic(
    root: str = './data/elliptic',
    train_ratio: float = 0.8
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Load Elliptic Bitcoin dataset with train/test masks.

    Args:
        root: Path to store/load dataset
        train_ratio: Fraction of labeled nodes for training

    Returns:
        data: PyG Data object
        train_mask: Boolean mask for training nodes
        test_mask: Boolean mask for test nodes
    """
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

    train_idx = labeled_indices[perm[:n_train]]
    test_idx = labeled_indices[perm[n_train:]]

    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return data, train_mask, test_mask
