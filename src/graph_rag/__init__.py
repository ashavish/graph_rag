"""Graph RAG - A modular Graph-based Retrieval-Augmented Generation system."""

from .core import GraphRAGTrainer, GraphRAGInference
from .config import settings

__version__ = "1.0.0"
__all__ = ["GraphRAGTrainer", "GraphRAGInference", "settings"]