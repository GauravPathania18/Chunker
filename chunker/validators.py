"""
Chunker Input Validators
Validation functions for files, queries, chunks, and API responses
"""

import os
from typing import List, Dict, Any
import inspect
from .config import VALIDATION, LIMITS
from .logger import warning, error

# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file_path(file_path: str) -> tuple[bool, str]:
    """
    Validate file path exists and is readable.
    
    Args:
        file_path: Path to file
        
    Returns:
        (is_valid, error_message)
    """
    if not file_path:
        return False, "File path is empty"
    
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    # Check file type
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALIDATION['allowed_file_types']:
        return False, f"File type {ext} not supported. Allowed: {VALIDATION['allowed_file_types']}"
    
    # Check readability
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"
    
    return True, ""


def validate_file_readable(file_path: str) -> bool:
    """Quick check if file is readable. Logs errors if not."""
    is_valid, msg = validate_file_path(file_path)
    if not is_valid:
        error(msg)
        return False
    return True


# ============================================================
# QUERY VALIDATION
# ============================================================

def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate user query.
    
    Args:
        query: User query string
        
    Returns:
        (is_valid, error_message)
    """
    if not query:
        return False, "Query is empty"
    
    if not isinstance(query, str):
        return False, f"Query must be string, got {type(query)}"
    
    # Check length
    if len(query) > VALIDATION['max_query_length']:
        return False, f"Query exceeds max length ({VALIDATION['max_query_length']} chars)"
    
    # Check for valid content (not just whitespace)
    if not query.strip():
        return False, "Query contains only whitespace"
    
    return True, ""


# ============================================================
# CHUNK VALIDATION
# ============================================================

def validate_chunk(chunk: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a single chunk dictionary.
    
    Args:
        chunk: Chunk dictionary
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(chunk, dict):
        return False, f"Chunk must be dict, got {type(chunk)}"
    
    # Required fields
    required = ['text', 'chunk_id']
    for field in required:
        if field not in chunk:
            return False, f"Chunk missing required field: {field}"
    
    if not chunk['text'] or not isinstance(chunk['text'], str):
        return False, "Chunk text must be non-empty string"
    
    # Validate chunk size
    text_len = len(chunk['text'])
    if text_len < VALIDATION['min_chunk_size']:
        return False, f"Chunk too small ({text_len} < {VALIDATION['min_chunk_size']} chars)"
    
    if text_len > VALIDATION['max_chunk_size']:
        return False, f"Chunk too large ({text_len} > {VALIDATION['max_chunk_size']} chars)"
    
    return True, ""


def validate_chunks(chunks: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """
    Validate list of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        (all_valid, list_of_error_messages)
    """
    if not isinstance(chunks, list):
        return False, [f"Chunks must be list, got {type(chunks)}"]
    
    if not chunks:
        return False, ["Chunks list is empty"]
    
    errors = []
    for i, chunk in enumerate(chunks):
        is_valid, msg = validate_chunk(chunk)
        if not is_valid:
            errors.append(f"Chunk {i}: {msg}")
    
    return len(errors) == 0, errors


# ============================================================
# API RESPONSE VALIDATION
# ============================================================

def validate_ollama_response(response: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate Ollama API response.
    
    Args:
        response: Response dictionary from Ollama
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response must be dict, got {type(response)}"
    
    # Check required fields
    if 'response' not in response:
        return False, "Response missing 'response' field"
    
    # Validate response content
    answer = response.get('response', '')
    if not isinstance(answer, str):
        return False, f"Response text must be string, got {type(answer)}"
    
    if not answer or not answer.strip():
        return False, "Response text is empty"
    
    return True, ""


def validate_chromadb_result(result: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate ChromaDB query result.
    
    Args:
        result: Result dictionary from ChromaDB query
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(result, dict):
        return False, f"Result must be dict, got {type(result)}"
    
    # Check required fields
    required = ['documents', 'metadatas']
    for field in required:
        if field not in result:
            return False, f"Result missing '{field}' field"
    
    if not result['documents'] or not result['documents'][0]:
        return False, "No documents retrieved"
    
    return True, ""


# ============================================================
# BATCH VALIDATION
# ============================================================

def validate_batch_size(batch_size: int) -> tuple[bool, str]:
    """
    Validate batch size.
    
    Args:
        batch_size: Size of batch
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(batch_size, int):
        return False, f"Batch size must be int, got {type(batch_size)}"
    
    if batch_size <= 0:
        return False, "Batch size must be positive"
    
    if batch_size > LIMITS['batch_size']:
        return False, f"Batch size exceeds limit ({batch_size} > {LIMITS['batch_size']})"
    
    return True, ""


# ============================================================
# UTILITY VALIDATION
# ============================================================

def validate_embeddings(embeddings: Any) -> tuple[bool, str]:
    """
    Validate embeddings array.
    
    Args:
        embeddings: Embeddings array (numpy)
        
    Returns:
        (is_valid, error_message)
    """
    try:
        import numpy as np
        if not isinstance(embeddings, np.ndarray):
            return False, f"Embeddings must be numpy array, got {type(embeddings)}"
        
        if len(embeddings.shape) != 2:
            return False, f"Embeddings must be 2D array, got shape {embeddings.shape}"
        
        if embeddings.shape[0] == 0:
            return False, "Embeddings array is empty"
        
        return True, ""
    except ImportError:
        return False, "NumPy not installed"


# ============================================================
# VALIDATION WRAPPER (Log errors automatically)
# ============================================================

def safe_validate(validator_func, *args, **kwargs) -> bool:
    """
    Wrapper to validate and log errors automatically.
    
    Args:
        validator_func: Validation function
        *args, **kwargs: Arguments to pass to validator
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = validator_func(*args, **kwargs)
        
        # Handle tuple returns (is_valid, message)
        if isinstance(result, tuple) and len(result) == 2:
            is_valid, msg = result
            if not is_valid:
                error(msg)
                return False
            return True
        
        # Handle single bool return
        if isinstance(result, bool):
            return result
            
        # Unexpected return type
        error(f"Validator returned unexpected type: {type(result)}")
        return False
        
    except Exception as e:
        error(f"Validation failed with exception: {e}")
        return False
