"""FB15k-237 knowledge graph dataset loader."""

from torch_geometric.datasets import FB15k_237
from torch_geometric.data import Data
from pathlib import Path


def load_fb15k237(
    root: str = './data/fb15k237'
) -> tuple[Data, Data, Data, int, int]:
    """
    Load FB15k-237 knowledge graph dataset with train/val/test splits.

    Args:
        root: Path to store/load dataset

    Returns:
        train_data: Training triples as PyG Data object
        val_data: Validation triples as PyG Data object
        test_data: Test triples as PyG Data object
        num_entities: Total number of entities in the knowledge graph
        num_relations: Total number of relation types
    """
    # Load all splits
    train_data = FB15k_237(root=root, split='train')[0]
    val_data = FB15k_237(root=root, split='val')[0]
    test_data = FB15k_237(root=root, split='test')[0]

    # Get entity and relation counts from training data
    num_entities = train_data.num_nodes
    num_relations = train_data.num_edge_types

    return train_data, val_data, test_data, num_entities, num_relations


def get_fb15k237_info(root: str = './data/fb15k237') -> dict:
    """
    Get dataset statistics for FB15k-237.

    Args:
        root: Path to dataset

    Returns:
        Dictionary with dataset statistics
    """
    train_data, val_data, test_data, num_entities, num_relations = load_fb15k237(root)

    return {
        'name': 'FB15k-237',
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_triples': train_data.num_edges,
        'val_triples': val_data.num_edges,
        'test_triples': test_data.num_edges,
        'total_triples': train_data.num_edges + val_data.num_edges + test_data.num_edges
    }
