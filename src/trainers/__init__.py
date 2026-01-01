"""Training loops for different task types."""

from .node_classifier import NodeClassificationTrainer
from .link_predictor import LinkPredictionTrainer

__all__ = ["NodeClassificationTrainer", "LinkPredictionTrainer"]
