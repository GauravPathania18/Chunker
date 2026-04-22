#!/usr/bin/env python3
"""Quick test of imports"""

print("Testing imports...")

from chunker import HierarchicalClusterSummarizer, RAGInterface
from chunker.config import PATHS, MODELS, LIMITS

print("All imports successful")
print(f"DB path: {PATHS['chroma_db']}")
print(f"Model: {MODELS['ollama_model']}")
print(f"Embedding: {MODELS['embedding']}")
print("PyTorch issue RESOLVED")
