"""Knowledge Graph Embedding models: TransE, DistMult, RotatE."""

from torch_geometric.nn.kge import TransE, DistMult, RotatE
import torch.nn as nn


# Type alias for all KGE model types
KGEModel = TransE | DistMult | RotatE


def get_kge_model(
    model_name: str,
    num_nodes: int,
    num_relations: int,
    hidden_channels: int = 50,
    **kwargs
) -> KGEModel:
    """
    Factory function to create Knowledge Graph Embedding models.

    Args:
        model_name: One of 'TransE', 'DistMult', 'RotatE' (case-insensitive)
        num_nodes: Number of entities in the knowledge graph
        num_relations: Number of relation types
        hidden_channels: Embedding dimension (default 50 per literature)
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized KGE model

    Raises:
        ValueError: If model_name is not recognized
    """
    model_name_lower = model_name.lower()

    if model_name_lower == 'transe':
        return TransE(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **kwargs
        )
    elif model_name_lower == 'distmult':
        return DistMult(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **kwargs
        )
    elif model_name_lower == 'rotate':
        return RotatE(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown KGE model: {model_name}. "
            f"Supported models: TransE, DistMult, RotatE"
        )


# Model descriptions for reference
MODEL_INFO = {
    'TransE': {
        'description': 'Translation-based embedding (head + relation ≈ tail)',
        'paper': 'Bordes et al., 2013',
        'strengths': 'Simple, fast, good for 1-to-1 relations'
    },
    'DistMult': {
        'description': 'Bilinear diagonal model (head · relation · tail)',
        'paper': 'Yang et al., 2014',
        'strengths': 'Symmetric relations, efficient training'
    },
    'RotatE': {
        'description': 'Rotation-based embedding in complex space',
        'paper': 'Sun et al., 2019',
        'strengths': 'Handles symmetry, antisymmetry, inversion, composition'
    }
}
