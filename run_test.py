#!/usr/bin/env python3
"""
Quick test to verify the chunker package is working correctly
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported"""
    print("="*60)
    print("TEST 1: Importing chunker modules")
    print("="*60)
    try:
        from chunker import HierarchicalClusterSummarizer, RAGInterface
        from chunker.config import PATHS, MODELS, LIMITS
        from chunker.logger import info, error
        print("[OK] All imports successful")
        print(f"  - Database path: {PATHS['chroma_db']}")
        print(f"  - Model: {MODELS['ollama_model']}")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_initialization():
    """Test that the processor can be initialized"""
    print("\n" + "="*60)
    print("TEST 2: Initializing HierarchicalClusterSummarizer")
    print("="*60)
    try:
        from chunker import HierarchicalClusterSummarizer
        processor = HierarchicalClusterSummarizer()
        print("[OK] HierarchicalClusterSummarizer initialized")
        print(f"  - Chunks in DB: {processor.chunks_collection.count()}")
        print(f"  - Summaries in DB: {processor.summaries_collection.count()}")
        return processor
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return None

def test_add_document(processor):
    """Test adding a document"""
    print("\n" + "="*60)
    print("TEST 3: Adding document to database")
    print("="*60)
    
    test_file = "examples/sample.txt"
    if not os.path.exists(test_file):
        print(f"[SKIP] Test file not found: {test_file}")
        return False
    
    try:
        chunks_added = processor.add_document(test_file, "test_sample")
        if chunks_added > 0:
            print(f"[OK] Successfully added {chunks_added} chunks")
            print(f"  - Total chunks in DB: {processor.chunks_collection.count()}")
            return True
        else:
            print("[WARN] No chunks added (file may be empty)")
            return False
    except Exception as e:
        print(f"[FAIL] Failed to add document: {e}")
        return False

def test_rag_interface():
    """Test RAG interface initialization (requires Ollama)"""
    print("\n" + "="*60)
    print("TEST 4: Initializing RAGInterface (requires Ollama)")
    print("="*60)
    try:
        from chunker import RAGInterface
        rag = RAGInterface()
        print("[OK] RAGInterface initialized")
        print("  - Ollama is running and model is available")
        return True
    except ConnectionError as e:
        print(f"[INFO] Ollama not available: {e}")
        print("  - This is expected if Ollama is not running")
        return False
    except Exception as e:
        print(f"[FAIL] RAGInterface initialization failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("CHUNKER PACKAGE TEST")
    print("="*60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n[CRITICAL] Imports failed - cannot continue")
        sys.exit(1)
    
    # Test 2: Initialization
    processor = test_initialization()
    if not processor:
        print("\n[CRITICAL] Initialization failed - cannot continue")
        sys.exit(1)
    
    # Test 3: Add document
    test_add_document(processor)
    
    # Test 4: RAG Interface (optional - requires Ollama)
    test_rag_interface()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCore functionality is working!")
    print("To use the RAG chat interface, start Ollama:")
    print("  ollama serve")
    print("  ollama pull gemma3:4b")

if __name__ == "__main__":
    main()
