#!/usr/bin/env python3
"""Minimal test of HierarchicalClusterSummarizer initialization"""

print("TEST: Starting import...")
try:
    from chunker import HierarchicalClusterSummarizer
    print("TEST: Import successful")
except Exception as e:
    print(f"TEST: Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("TEST: Starting initialization...")
try:
    processor = HierarchicalClusterSummarizer()
    print("TEST: Initialization successful")
    print(f"TEST: ChromaDB path: {processor.chroma_path}")
except Exception as e:
    print(f"TEST: Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("TEST: All tests passed!")
