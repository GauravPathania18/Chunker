"""
Chunker Project Configuration
Centralized settings for all modules
"""

import os

# ============================================================
# DATABASE & STORAGE PATHS
# ============================================================
PATHS = {
    'chroma_db': './hierarchical_chroma_db', # Main ChromaDB persistent storage
    'output_dir': './output',                # Output reports and logs
    'logs_dir': './logs',                    # Log files
    'processed_chunks_db': './processed_chunks.json'  # Track which chunks were added
}

# ============================================================
# AI MODELS & EXTERNAL SERVICES
# ============================================================
MODELS = {
    'embedding': 'all-mpnet-base-v2',       # SentenceTransformer for embeddings
    'ollama_model': 'gemma3:4b',             # Local LLM model (4B params, fast & capable)
    'ollama_url': 'http://localhost:11434/api/generate',  # Ollama API endpoint
}

# ============================================================
# PROCESSING LIMITS & PARAMETERS
# ============================================================
LIMITS = {
    'chunk_size': 500,                       # Characters per chunk
    'chunk_overlap': 50,                     # Overlap between chunks
    'batch_size': 10,                      # Items per batch insert (reduced to prevent ChromaDB memory issues)
    'max_summary_length': 4000,              # Max chars sent to Ollama for summarization
    'max_clusters': 8,                       # Maximum hierarchical clustering levels
    'ollama_timeout': 60,                    # Seconds for Ollama API calls
    'pdf_extraction_timeout': 300,           # Seconds for PDF text extraction
    'connection_timeout': 10,                # Seconds for connection establishment
    'clustering_history_size': 10,           # Max iterations to keep in memory
    'max_retries': 3,                        # Retry attempts for failed batches
    'retry_backoff': [1, 2, 4],              # Backoff seconds: [attempt_1, attempt_2, attempt_3]
}

# ============================================================
# RAG & RETRIEVAL PARAMETERS
# ============================================================
RAG = {
    'max_context_chunks': 5,                 # Chunks to retrieve per query
    'include_summaries': True,               # Include cluster summaries in context
    'temperature': 0.3,                      # LLM temperature (0-1, lower=deterministic)
    'verbose': True,                         # Print detailed information
}

# ============================================================
# CLUSTERING PARAMETERS
# ============================================================
CLUSTERING = {
    'n_clusters': None,                      # Auto-calculate: max(2, min(8, len(embeddings) // 100))
    'covariance_type': 'diag',               # GMM covariance type
    'random_state': 42,                      # For reproducibility
    'silhouette_threshold': 0.05,            # Min improvement to continue hierarchical clustering
}

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOGGING = {
    'level': 'INFO',                         # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_to_file': True,
    'log_file': 'logs/chunker.log',
}

# ============================================================
# VALIDATION RULES
# ============================================================
VALIDATION = {
    'max_query_length': 5000,                # Max characters for query
    'allowed_file_types': ['.pdf'],
    'min_chunk_size': 100,                   # Minimum chunk size
    'max_chunk_size': 2000,                  # Maximum chunk size
}

# ============================================================
# SETUP: Create required directories
# ============================================================
for path_key, path_val in PATHS.items():
    if path_key.endswith('_dir') and path_val.startswith('./'):
        os.makedirs(path_val, exist_ok=True)
