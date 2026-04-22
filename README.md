# Chunker: Production-Ready RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for document processing, hierarchical clustering, and conversational retrieval using local LLMs.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Features

✅ **Document Processing**
- PDF, TXT, Markdown, LaTeX support
- Intelligent chunking with paragraph awareness
- Duplicate prevention with persistent tracking
- Configurable chunk sizes and overlap

✅ **Semantic Search & Retrieval**
- Vector embeddings via SentenceTransformers
- HNSW indexing for fast retrieval
- Multi-chunk context aggregation
- Relevance scoring

✅ **Hierarchical Clustering**
- Gaussian Mixture Model (GMM) clustering
- Multi-level hierarchical summaries
- Silhouette score-based convergence detection
- Automatic cluster optimization

✅ **Conversational RAG**
- Local LLM integration via Ollama
- Context-aware prompt building
- Conversation history tracking
- Input validation and error handling

✅ **Production-Ready**
- Structured logging (file + console)
- Comprehensive error handling with retries
- Timeout management
- Database connectivity validation
- API response validation

---

## Architecture

```
Documents (PDF/TXT)
    ↓
[mk3.py] HierarchicalClusterSummarizer
    ├── read_pdf()
    ├── create_chunks()
    ├── add_document() → ChromaDB
    ├── perform_clustering() → GMM
    ├── generate_summary_with_ollama()
    └── hierarchical_clustering()
    ↓
[ChromaDB] Storage
    ├── chunks_collection (embeddings + metadata)
    └── summaries_collection (cluster summaries)
    ↓
[face.py] RAGInterface
    ├── retrieve_context() → semantic search
    ├── build_prompt() → context augmentation
    ├── generate_response() → Ollama LLM
    └── query() → end-to-end RAG
    ↓
User (Conversational interface)
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized configuration (paths, models, limits) |
| `logger.py` | Structured logging with file + console output |
| `validators.py` | Input validation for files, queries, API responses |
| `mk3.py` | Main processor: document chunking, clustering, summaries |
| `face.py` | RAG interface: retrieval + generation |

---

## Installation

### Prerequisites
- Python 3.8+
- Ollama (for local LLM inference)
- 4GB+ RAM recommended

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
chromadb>=0.3.21
sentence-transformers>=2.2.0
scikit-learn>=1.2.0
numpy>=1.23.0
pandas>=1.5.0
PyPDF2>=3.0.0
requests>=2.28.0
```

### Step 2: Set Up Ollama

```bash
# Download and install Ollama from https://ollama.ai

# Pull the default model (gemma3:4b - fast & capable)
ollama pull gemma3:4b

# Or choose another model
ollama pull llama2
ollama pull mistral
```

### Step 3: Configure Project

Edit `config.py` if needed:
```python
PATHS = {
    'chroma_db': './hierarchical_chroma_db',  # Database location
    'logs_dir': './logs',                    # Logs location
}

MODELS = {
    'ollama_model': 'gemma3:4b',          # Your chosen model (4B params)
}

LIMITS = {
    'chunk_size': 500,                    # Characters per chunk
    'chunk_overlap': 50,                  # Overlap between chunks
}
```

---

## Configuration

All settings are centralized in `config.py`. Key sections:

### PATHS
```python
PATHS = {
    'chroma_db': './hierarchical_chroma_db',  # Persistent storage
    'output_dir': './output',                   # Report directory
    'logs_dir': './logs',                       # Log directory
    'processed_chunks_db': './processed_chunks.json'  # Duplicate tracking
}
```

### MODELS
```python
MODELS = {
    'embedding': 'all-mpnet-base-v2',     # SentenceTransformer model
    'ollama_model': 'gemma3:4b',           # Local LLM (4B params, fast & capable)
    'ollama_url': 'http://localhost:11434/api/generate'
}
```

### LIMITS
```python
LIMITS = {
    'chunk_size': 500,                     # Characters per chunk
    'chunk_overlap': 50,                   # Overlap
    'batch_size': 10,                      # Items per batch (reduced to prevent memory issues)
    'max_summary_length': 4000,            # Max text to summarize
    'ollama_timeout': 60,                  # LLM response timeout (seconds)
    'pdf_extraction_timeout': 300,         # PDF reading timeout
    'max_retries': 3,                      # Batch retry attempts
}
```

### CLUSTERING
```python
CLUSTERING = {
    'silhouette_threshold': 0.05,          # Convergence threshold
    'max_clusters': 8,                     # Maximum hierarchical levels
}
```

---

## Usage

### 1. Basic Document Addition (mk3.py)

```python
from chunker import HierarchicalClusterSummarizer

# Initialize processor
processor = HierarchicalClusterSummarizer()

# Add a document
chunks_added = processor.add_document("path/to/document.pdf", "my_document")
print(f"Added {chunks_added} chunks")

# Perform hierarchical clustering
history = processor.hierarchical_clustering()
print(f"Clustering converged at level {history['final_level']}")
```

### 2. Conversational RAG (face.py)

```python
from chunker import RAGInterface

# Initialize RAG interface (validates Ollama connectivity)
rag = RAGInterface()

# Query the system
response = rag.query("What are the main topics discussed?")

if response['success']:
    print(f"Answer: {response['answer']}")
else:
    print(f"Error: {response['error']}")

# Check conversation history
for turn in rag.conversation_history:
    print(f"Q: {turn['question']}")
    print(f"A: {turn['answer']}\n")
```

### 3. Retrieve Context Only

```python
from chunker import RAGInterface

rag = RAGInterface()

# Retrieve relevant context
context = rag.retrieve_context("specific question")

if context['success']:
    print(f"Found {len(context['chunks'])} relevant chunks")
    for i, chunk in enumerate(context['chunks'], 1):
        print(f"\n[{i}] {chunk[:200]}...")
```

### 4. Database Inspection

```python
# Use the ChromaDB client directly for inspection
from chunker import HierarchicalClusterSummarizer

processor = HierarchicalClusterSummarizer()
count = processor.chunks_collection.count()
print(f"Total chunks in database: {count}")
```

---

## API Reference

### HierarchicalClusterSummarizer (mk3.py)

#### `__init__(chroma_path=None, ollama_model=None, ollama_url=None)`
Initialize the processor with ChromaDB and Ollama.

#### `add_document(file_path, document_name=None) → int`
Add a document to the system. Returns number of chunks added.

**Features:**
- Automatic PDF/TXT/LaTeX detection
- Duplicate prevention via persistent tracking
- Retry logic with exponential backoff
- Timeout management

**Example:**
```python
chunks = processor.add_document("book.pdf", "my_book")
```

#### `hierarchical_clustering(max_levels=None, improvement_threshold=None) → Dict`
Perform hierarchical clustering with silhouette score-based convergence.

**Parameters:**
- `max_levels`: Maximum hierarchy depth (default: 8)
- `improvement_threshold`: Stop if improvement < threshold (default: 0.05)

**Returns:**
```python
{
    'levels': [...],          # Per-level results
    'convergence': True,       # Converged?
    'final_level': 2           # Last level performed
}
```

#### `query_by_level(query, level=None, n_results=5) → Dict`
Query summaries at a specific hierarchical level.

---

### RAGInterface (face.py)

#### `__init__(chroma_path=None, ollama_model=None, ...)`
Initialize RAG interface. **Raises `ConnectionError` if Ollama unavailable.**

#### `retrieve_context(query) → Dict`
Retrieve relevant chunks and summaries.

**Validation:**
- Query length ≤ 5000 chars
- Query not empty/whitespace

**Returns:**
```python
{
    'success': True,
    'chunks': [...],           # Retrieved chunks
    'scores': [...],           # Relevance scores (0-1)
    'cluster_summaries': [...], # High-level summaries
    'sources': [...]           # Source documents
}
```

#### `generate_response(prompt) → Dict`
Generate response using Ollama.

**Features:**
- Timeout enforcement (60s default)
- Response validation (structure, length)
- Graceful error handling

**Returns:**
```python
{
    'success': True,
    'answer': "...",           # Generated response
    'tokens': 150              # Token count
}
```

#### `query(question) → Dict`
Complete end-to-end RAG: retrieve → augment → generate.

**Example:**
```python
result = rag.query("What is RAG?")
# Automatically: retrieves context, builds prompt, generates response
```

---

## Validators (validators.py)

### Input Validation Functions

```python
from validators import (
    validate_file_path,      # File existence + type
    validate_query,          # Query length + content
    validate_chunk,          # Chunk structure
    validate_ollama_response, # API response format
    validate_chromadb_result  # Database result
)

# Example usage
is_valid, msg = validate_query("my question")
if not is_valid:
    print(f"Error: {msg}")
```

---

## Troubleshooting

### "Ollama is not available"
**Symptom:** `ConnectionError: Ollama not running or model not available`

**Solution:**
```bash
# 1. Start Ollama
ollama serve

# 2. Verify model is installed
ollama list

# 3. Pull model if missing
ollama pull gemma3:4b

# 4. Check connection
curl http://localhost:11434/api/tags
```

### "Database is empty"
**Symptom:** No results from retrieval

**Solution:**
```python
from mk3 import HierarchicalClusterSummarizer

processor = HierarchicalClusterSummarizer()
processor.add_document("your_file.pdf", "doc_name")
```

### "Timeout errors"
**Symptom:** Requests fail with timeout

**Solution (config.py):**
```python
LIMITS = {
    'ollama_timeout': 120,        # Increase from 60
    'pdf_extraction_timeout': 600, # Increase from 300
}
```

### "Out of memory"
**Symptom:** System crash during clustering

**Solution:**
```python
# Reduce batch size
LIMITS['batch_size'] = 500  # Down from 1000

# Or reduce max clusters
CLUSTERING['max_clusters'] = 4  # Down from 8
```

### "PDF extraction fails"
**Symptom:** `PyPDF2 not installed` or corrupt PDF

**Solution:**
```bash
# Install PyPDF2
pip install PyPDF2

# Verify PDF integrity
python -c "import PyPDF2; PyPDF2.PdfReader('file.pdf')"
```

---

## Logging

Logs are written to `./logs/chunker.log` and console:

```python
from logger import info, warning, error, debug

info("Normal operation")
warning("Potential issue")
error("Something failed")
debug("Detailed debug info")
```

**Log levels:**
- `DEBUG`: Detailed internal operations
- `INFO`: General information
- `WARNING`: Non-critical issues
- `ERROR`: Failed operations
- `CRITICAL`: System-level failures

---

## Performance Tips

1. **Chunk Size:** 300-500 chars optimal for most documents
2. **Batch Size:** 10-100 for stable ChromaDB operations
3. **Embedding Model:** `all-mpnet-base-v2` is fast; `all-MiniLM` is faster but lower quality
4. **LLM Model:** `gemma3:4b` balanced; `gemma3:12b` for higher quality; `gemma3:1b` for fastest
5. **Clustering:** Set `silhouette_threshold` high (0.1+) for faster convergence

---

## Contributing

To extend or modify:

1. **Add new document type:** Update `mk3.py:read_pdf()` or add new reader method
2. **Change embedding model:** Update `config.py:MODELS['embedding']`
3. **Add new validator:** Create function in `validators.py`
4. **Custom LLM:** Update `face.py:generate_response()`

---

## License

This project is provided as-is for educational and research purposes.

---

## Support

For issues, check:
1. Logs in `./logs/chunker.log`
2. Database in `./chroma_db/`
3. Configuration in `config.py`
4. Run tests: `python test_integration.py`
