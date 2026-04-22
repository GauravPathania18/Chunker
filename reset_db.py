#!/usr/bin/env python3
"""Reset ChromaDB database completely"""

import shutil
import os

print("Resetting ChromaDB database...")

# Remove database directory
if os.path.exists('hierarchical_chroma_db'):
    shutil.rmtree('hierarchical_chroma_db')
    print("✓ Removed hierarchical_chroma_db/")

# Remove tracking file
if os.path.exists('processed_chunks.json'):
    os.remove('processed_chunks.json')
    print("✓ Removed processed_chunks.json")

# Remove any lock files
for item in os.listdir('.'):
    if item.endswith('.lock') or item.startswith('chroma_'):
        if os.path.isfile(item):
            os.remove(item)
            print(f"✓ Removed {item}")

print("\n✅ Database reset complete")
print("You can now run: python ingest.py")
