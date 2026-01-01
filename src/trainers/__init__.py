"""Trainer classes for different ML tasks."""

from .base import BaseTrainer
from .node_classifier import NodeClassificationTrainer
from .link_predictor import LinkPredictionTrainer

__all__ = [
    "BaseTrainer",
    "NodeClassificationTrainer",
    "LinkPredictionTrainer",
]
