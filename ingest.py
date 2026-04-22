#!/usr/bin/env python3
"""
Standalone Document Ingestion Script

Adds PDF documents to ChromaDB without requiring Ollama.
Generates real 768-dimensional embeddings using SentenceTransformer.
"""

import os
import sys
from chunker import HierarchicalClusterSummarizer
from chunker.config import PATHS
from chunker.logger import info, warning, error


def find_pdfs(directory='.'):
    """Find all PDF files in directory"""
    pdfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdfs.append(filename)
    return sorted(pdfs)


def main():
    print("="*60)
    print("DOCUMENT INGESTION - Chunker RAG System")
    print("="*60)
    print(f"Database: {PATHS['chroma_db']}")
    print(f"Embedding: 768-dimensional (all-mpnet-base-v2)")
    print()
    
    # Initialize processor (loads embedding model)
    print("Initializing processor...")
    try:
        processor = HierarchicalClusterSummarizer()
    except Exception as e:
        error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Show current database state
    current_count = processor.chunks_collection.count()
    print(f"✓ Processor ready (current DB has {current_count} chunks)")
    print()
    
    # Find PDFs to process
    pdfs = find_pdfs('.')
    
    if not pdfs:
        print("❌ No PDF files found in current directory.")
        print("   Please add PDF files and run again.")
        sys.exit(1)
    
    print(f"Found {len(pdfs)} PDF file(s):")
    for pdf in pdfs:
        print(f"  • {pdf}")
    print()
    
    # Process each PDF
    total_added = 0
    success_count = 0
    
    for pdf in pdfs:
        doc_id = pdf.replace('.pdf', '').replace(' ', '_').lower()
        print(f"Processing: {pdf}")
        
        try:
            chunks_added = processor.add_document(pdf, doc_id)
            
            if chunks_added > 0:
                print(f"  ✅ Added {chunks_added} chunks")
                total_added += chunks_added
                success_count += 1
            else:
                print(f"  ⚠️  No chunks added (file may be empty or already processed)")
                
        except Exception as e:
            error(f"  ❌ Failed: {e}")
            continue
    
    # Summary
    print()
    print("="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Successfully processed: {success_count}/{len(pdfs)} files")
    print(f"Total chunks added: {total_added}")
    print(f"Database now has: {processor.chunks_collection.count()} chunks")
    print()
    
    if total_added > 0:
        print("✅ Ready for RAG chat!")
        print("   Run: python -m chunker.face")
    else:
        print("⚠️  No new chunks added. Check PDFs or processed_chunks.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
