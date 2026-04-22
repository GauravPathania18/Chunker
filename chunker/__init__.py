"""
Chunker RAG System

A hierarchical document clustering and semantic RAG system with Ollama integration.

Main components:
- HierarchicalClusterSummarizer: Document chunking and clustering (mk3.py)
- RAGInterface: Conversational RAG interface (face.py)
"""

__version__ = "1.0.0"

from .mk3 import HierarchicalClusterSummarizer
from .face import RAGInterface
from .config import PATHS, MODELS, LIMITS, RAG, CLUSTERING, VALIDATION

__all__ = [
    "HierarchicalClusterSummarizer",
    "RAGInterface",
    "PATHS",
    "MODELS",
    "LIMITS",
    "RAG",
    "CLUSTERING",
    "VALIDATION",
]
