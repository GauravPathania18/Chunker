# ================================
# DOCUMENT CHUNKING PIPELINE (NO EMBEDDINGS)
# ================================

from .config import LIMITS
from .logger import info, error, debug
from .validators import validate_file_readable


# ----------------
# 1. READ DOCUMENT
# ----------------
def read_document(file_path):
    """
    Reads a text file and returns content as string.
    Validates file exists first.
    """
    # Validate file
    if not validate_file_readable(file_path):
        return ""
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            info(f"Read {len(text)} characters from {file_path}")
            return text
    except UnicodeDecodeError:
        debug("UTF-8 decoding failed, trying latin-1...")
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
                info(f"Read {len(text)} characters from {file_path} (latin-1 encoding)")
                return text
        except Exception as e:
            error(f"Error reading file with latin-1: {e}")
            return ""
    except Exception as e:
        error(f"Error reading file: {e}")
        return ""


# ----------------
# 2. CLEAN TEXT
# ----------------
def clean_text(text):
    """
    Normalize whitespace and remove unnecessary noise.
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ----------------
# 3. CHUNK TEXT (RAG-OPTIMIZED)
# ----------------
def chunk_text(text, chunk_size=None, overlap=None):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): Input text
        chunk_size (int): Characters per chunk (default: from config)
        overlap (int): Overlap between chunks (default: from config)

    Returns:
        list: List of chunk dictionaries
    """
    chunk_size = chunk_size or LIMITS['chunk_size']
    overlap = overlap or LIMITS['chunk_overlap']
    
    chunks = []
    step = chunk_size - overlap
    
    debug(f"Chunking with size={chunk_size}, overlap={overlap}")

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if not chunk:
            continue

        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk,
            "start_index": i,
            "end_index": i + len(chunk)
        })

    info(f"Created {len(chunks)} chunks")
    return chunks


# ----------------
# 4. FULL PIPELINE
# ----------------
def process_document(file_path, chunk_size=None, overlap=None):
    """
    Full pipeline:
    read -> clean -> chunk
    """
    chunk_size = chunk_size or LIMITS['chunk_size']
    overlap = overlap or LIMITS['chunk_overlap']
    
    info(f"Processing document: {file_path}")
    
    text = read_document(file_path)

    if not text:
        error("Empty or unreadable document.")
        return []

    info("Cleaning text...")
    text = clean_text(text)

    info("Chunking text...")
    chunks = chunk_text(text, chunk_size, overlap)

    return chunks


# ----------------
# 5. RUN
# ----------------
if __name__ == "__main__":
    file_path = "Cinderella.sty"  # change to your file

    chunks = process_document(file_path, chunk_size=LIMITS['chunk_size'], overlap=LIMITS['chunk_overlap'])

    # Preview chunks
    for chunk in chunks:
        print("\n----------------------------")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Start: {chunk['start_index']} | End: {chunk['end_index']}")
        print(f"Text Preview: {chunk['text']}")