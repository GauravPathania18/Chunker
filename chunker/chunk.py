from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import re

from .config import MODELS, LIMITS
from .logger import info, error, debug
from .validators import validate_file_readable

def semantic_chunk_ordered(text, n_chunks=None):
    """Semantic chunking that preserves story order
    
    Args:
        text: Input text to chunk
        n_chunks: Number of clusters (default: from config)
        
    Returns:
        (chunks, embeddings) tuple
    """
    n_chunks = n_chunks or 8
    
    info(f"Performing semantic chunking with {n_chunks} target clusters...")
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    debug(f"Split into {len(sentences)} sentences")
    
    if not sentences:
        error("No sentences found in text")
        return [], np.array([])
    
    # Get embeddings
    try:
        info(f"Generating embeddings with {MODELS['embedding']}...")
        model = SentenceTransformer(MODELS['embedding'])
        embeddings = model.encode(sentences)
        info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        error(f"Failed to generate embeddings: {e}")
        return [], np.array([])
    
    # Cluster sentences
    n_clusters = min(n_chunks, len(sentences))
    debug(f"Clustering into {n_clusters} clusters...")
    
    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
    except Exception as e:
        error(f"Clustering failed: {e}")
        return [], embeddings
    
    # Group by cluster while tracking original indices
    indexed_chunks = {}
    for idx, (sent, label) in enumerate(zip(sentences, labels)):
        if label not in indexed_chunks:
            indexed_chunks[label] = []
        indexed_chunks[label].append((idx, sent))
    
    # Sort by original position and create chunks
    chunks = []
    for label in sorted(indexed_chunks.keys()):
        sorted_sents = sorted(indexed_chunks[label], key=lambda x: x[0])
        chunk_text = ' '.join([s[1] for s in sorted_sents])
        chunks.append(chunk_text)
    
    info(f"Created {len(chunks)} semantic chunks")
    return chunks, embeddings

# Run on Cinderella
if __name__ == "__main__":
    try:
        if not validate_file_readable('Cinderella.sty'):
            error("Cannot read Cinderella.sty")
        else:
            with open('Cinderella.sty', 'r', encoding='utf-8') as f:
                text = f.read()

            chunks, embeddings = semantic_chunk_ordered(text, n_chunks=8)

            info(f"Created {len(chunks)} semantic chunks")

            for i, chunk in enumerate(chunks, 1):
                print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
                print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    except Exception as e:
        error(f"Error: {e}")