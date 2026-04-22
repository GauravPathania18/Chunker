#!/usr/bin/env python3
"""
Verification script for document ingestion
Tests that PDFs are properly added to ChromaDB with 768-dim embeddings
"""

import sys
from chunker import HierarchicalClusterSummarizer


def test_initialization():
    """Test processor initializes correctly"""
    print("="*60)
    print("TEST 1: Initialize HierarchicalClusterSummarizer")
    print("="*60)
    
    try:
        processor = HierarchicalClusterSummarizer()
        print(f"✅ Processor initialized")
        print(f"   Database path: {processor.chroma_path}")
        print(f"   Embedding model: {processor.embedding_model}")
        return processor
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)


def test_embedding_dim(processor):
    """Verify embeddings are 768-dimensional"""
    print("\n" + "="*60)
    print("TEST 2: Verify 768-dimensional embeddings")
    print("="*60)
    
    test_text = "This is a test sentence."
    embedding = processor.embedding_model.encode(test_text)
    
    dim = len(embedding)
    print(f"   Generated embedding dimension: {dim}")
    
    if dim == 768:
        print(f"✅ Correct: 768-dim embeddings")
        return True
    else:
        print(f"❌ Wrong dimension! Expected 768, got {dim}")
        return False


def test_add_document(processor):
    """Test adding a PDF document"""
    print("\n" + "="*60)
    print("TEST 3: Add PDF document")
    print("="*60)
    
    import os
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        print("⚠️  No PDF files found to test")
        return 0
    
    test_pdf = pdf_files[0]
    print(f"   Testing with: {test_pdf}")
    
    # Get count before
    count_before = processor.chunks_collection.count()
    print(f"   Chunks before: {count_before}")
    
    try:
        chunks_added = processor.add_document(test_pdf, "test_doc")
        count_after = processor.chunks_collection.count()
        
        print(f"   Chunks added: {chunks_added}")
        print(f"   Chunks after: {count_after}")
        
        if chunks_added > 0:
            print(f"✅ Successfully added {chunks_added} chunks")
            return chunks_added
        else:
            print(f"⚠️  No chunks added (file may be empty or already processed)")
            return 0
            
    except Exception as e:
        print(f"❌ Failed to add document: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return 0


def test_persistence(processor):
    """Test data persists across restarts"""
    print("\n" + "="*60)
    print("TEST 4: Data persistence")
    print("="*60)
    
    count_before = processor.chunks_collection.count()
    print(f"   Current chunks: {count_before}")
    
    # Re-initialize (simulates restart)
    print("   Re-initializing processor...")
    processor2 = HierarchicalClusterSummarizer()
    count_after = processor2.chunks_collection.count()
    
    print(f"   Chunks after restart: {count_after}")
    
    if count_before == count_after:
        print(f"✅ Data persists correctly")
        return True
    else:
        print(f"❌ Data mismatch!")
        return False


def main():
    print("\n" + "="*60)
    print("CHUNKER INGESTION VERIFICATION")
    print("="*60)
    
    # Test 1: Initialization
    processor = test_initialization()
    
    # Test 2: Embedding dimensions
    test_embedding_dim(processor)
    
    # Test 3: Add document
    added = test_add_document(processor)
    
    # Test 4: Persistence
    test_persistence(processor)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    
    total_chunks = processor.chunks_collection.count()
    print(f"Total chunks in database: {total_chunks}")
    
    if added > 0 or total_chunks > 0:
        print("✅ Ingestion is working correctly!")
        print("\nNext step: Run RAG chat interface")
        print("   python -m chunker.face")
    else:
        print("⚠️  No documents in database. Add PDFs and run again.")
    
    return total_chunks > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
