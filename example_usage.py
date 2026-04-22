#!/usr/bin/env python3
"""
Example usage of the Chunker RAG system

This script demonstrates how to:
1. Import and initialize the HierarchicalClusterSummarizer
2. Add PDF documents to the database
3. Query the processed documents
"""

from chunker import HierarchicalClusterSummarizer

# Initialize the processor
processor = HierarchicalClusterSummarizer()

# Add a PDF document to the database
# Using one of the sample PDFs in the project
processor.add_document("The Odyssey.pdf", "the_odyssey")

print("Document added successfully!")
print(f"Total chunks in database: {processor.chunks_collection.count()}")
