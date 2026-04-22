#!/usr/bin/env python3
"""Simple document ingestion - minimal version"""

import os
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np

DB_PATH = './hierarchical_chroma_db'
EMBEDDING_MODEL = 'all-mpnet-base-v2'

def main():
    print("="*60)
    print("SIMPLE DOCUMENT INGESTION")
    print("="*60)
    
    # Find PDFs
    pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if not pdfs:
        print("No PDF files found!")
        return
    
    print(f"Found {len(pdfs)} PDF(s): {pdfs}")
    
    # Initialize ChromaDB
    print(f"\nConnecting to ChromaDB at {DB_PATH}...")
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("document_chunks")
    print("Connected!")
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded!")
    
    # Process each PDF
    total_chunks = 0
    for pdf_file in pdfs:
        print(f"\nProcessing: {pdf_file}")
        try:
            # Read PDF
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            
            print(f"  Extracted {len(text)} characters")
            
            # Simple chunking
            chunk_size = 500
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i+chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'id': f"{pdf_file}_chunk_{i}"
                    })
            
            print(f"  Created {len(chunks)} chunks")
            
            # Add to ChromaDB
            for chunk in chunks:
                embedding = model.encode(chunk['text'])
                collection.add(
                    ids=[chunk['id']],
                    documents=[chunk['text']],
                    embeddings=[np.array(embedding, dtype=np.float32)],
                    metadatas=[{'source': pdf_file}]
                )
            
            total_chunks += len(chunks)
            print(f"  ✓ Added {len(chunks)} chunks to database")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"Total chunks added: {total_chunks}")
    print(f"{'='*60}")
    print("\nYou can now run: python -m chunker.face")

if __name__ == "__main__":
    main()
