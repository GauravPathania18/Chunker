import chromadb
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
from typing import List, Dict, Tuple, Optional
import uuid
import pandas as pd
import re
import json
import requests
import time
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from .config import PATHS, MODELS, LIMITS, CLUSTERING, RAG
from .logger import info, warning, error, debug, critical
from .validators import validate_file_readable, validate_chunks, validate_ollama_response

# Import sentence-transformers for real embeddings
from sentence_transformers import SentenceTransformer

class HierarchicalClusterSummarizer:
    def __init__(self, chroma_path: str = None, 
                 ollama_model: str = None,
                 ollama_url: str = None):
        """
        Initialize the hierarchical clustering system
        
        Args:
            chroma_path: Path to ChromaDB storage (default: from config)
            ollama_model: Ollama model name (default: from config)
            ollama_url: Ollama API endpoint (default: from config)
        """
        try:
            # Use config defaults if not provided
            self.chroma_path = chroma_path or PATHS['chroma_db']
            self.ollama_model = ollama_model or MODELS['ollama_model']
            self.ollama_url = ollama_url or MODELS['ollama_url']
            
            print(f"Initializing HierarchicalClusterSummarizer")
            print(f"  ChromaDB path: {self.chroma_path}")
            print(f"  Ollama model: {self.ollama_model}")
            
            # Create directories if needed
            print("DEBUG: Creating directories...", flush=True)
            os.makedirs(self.chroma_path, exist_ok=True)
            
            print("DEBUG: Creating ChromaDB client...", flush=True)
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            print("DEBUG: Client created", flush=True)
            
            # Main collection for chunks - use get_or_create to preserve existing data
            print("DEBUG: Getting chunks collection...", flush=True)
            self.chunks_collection = self.client.get_or_create_collection(name="document_chunks")
            print("DEBUG: Got chunks collection", flush=True)
            
            # Collection for hierarchical summaries
            print("DEBUG: Getting summaries collection...", flush=True)
            self.summaries_collection = self.client.get_or_create_collection(name="hierarchical_summaries")
            print("DEBUG: Got summaries collection", flush=True)
            
            # Skip count() calls which can hang - just show initialized
            print(f"  Chunks collection: initialized")
            print(f"  Summaries collection: initialized")
            
            self.batch_size = LIMITS['batch_size']
            self.clustering_history = []  # Track clustering iterations (max: LIMITS['clustering_history_size'])
            
            # Load persistent processed chunks tracking
            self.processed_chunks_file = PATHS['processed_chunks_db']
            self.processed_chunks = self._load_processed_chunks()
            
            # Initialize embedding model (cached for reuse)
            print(f"Loading embedding model: {MODELS['embedding']}...")
            try:
                self.embedding_model = SentenceTransformer(MODELS['embedding'])
                print(f"Embedding model loaded successfully (768-dim)")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                raise
            
            print(f"HierarchicalClusterSummarizer initialized successfully")
        except Exception as e:
            print(f"ERROR in __init__: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_processed_chunks(self) -> Dict[str, List[str]]:
        """Load processed chunks tracking from JSON file"""
        if os.path.exists(self.processed_chunks_file):
            try:
                with open(self.processed_chunks_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warning(f"Could not load processed_chunks.json: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save_processed_chunks(self):
        """Save processed chunks tracking to JSON file"""
        try:
            with open(self.processed_chunks_file, 'w') as f:
                json.dump(self.processed_chunks, f, indent=2)
            debug(f"Saved processed_chunks.json")
        except Exception as e:
            error(f"Failed to save processed_chunks.json: {e}")
    
    def _is_chunk_processed(self, document_name: str, chunk_id: str) -> bool:
        """Check if chunk was already processed"""
        if document_name not in self.processed_chunks:
            return False
        return chunk_id in self.processed_chunks[document_name]
    
    def _mark_chunk_processed(self, document_name: str, chunk_id: str):
        """Mark chunk as processed"""
        if document_name not in self.processed_chunks:
            self.processed_chunks[document_name] = []
        if chunk_id not in self.processed_chunks[document_name]:
            self.processed_chunks[document_name].append(chunk_id)
        
    def read_pdf(self, file_path: str) -> str:
        """Read PDF file and extract text with timeout"""
        if not validate_file_readable(file_path):
            raise FileNotFoundError(f"File not readable: {file_path}")
        
        try:
            import PyPDF2
            info(f"Reading PDF: {file_path}")
            text = ""
            
            start_time = time.time()
            timeout = LIMITS['pdf_extraction_timeout']
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                info(f"PDF has {len(reader.pages)} pages")
                
                for page_num, page in enumerate(reader.pages):
                    # Check timeout
                    if time.time() - start_time > timeout:
                        warning(f"PDF extraction timeout after {page_num} pages")
                        break
                    
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                        if (page_num + 1) % 50 == 0:
                            info(f"  Processed {page_num + 1}/{len(reader.pages)} pages")
            
            info(f"Extracted {len(text):,} characters in {time.time() - start_time:.1f}s")
            return text
            
        except ImportError:
            warning("PyPDF2 not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'PyPDF2'])
            import PyPDF2
            return self.read_pdf(file_path)
    
    def create_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
        """Create overlapping chunks from text"""
        chunk_size = chunk_size or LIMITS['chunk_size']
        overlap = overlap or LIMITS['chunk_overlap']
        
        chunks = []
        step = chunk_size - overlap
        
        # Split by paragraphs for better context
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) > 1 and len(text) > 10000:
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunk_id = f"chunk_{chunk_index:06d}"
                    
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_index,
                        'level': 0  # Base level
                    })
                    chunk_index += 1
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add last chunk
            if current_chunk:
                chunk_id = f"chunk_{chunk_index:06d}"
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'level': 0
                })
        else:
            # Simple chunking
            for i in range(0, len(text), step):
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():
                    chunk_id = f"chunk_{i:06d}"
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': chunk_id,
                        'position': i,
                        'chunk_index': len(chunks),
                        'level': 0
                    })
        
        info(f"Created {len(chunks)} base chunks")
        return chunks
    
    def add_document(self, file_path: str, document_name: str = None) -> int:
        """Add a document to ChromaDB with error handling and retry logic"""
        if document_name is None:
            document_name = os.path.basename(file_path)
        
        info(f"Adding document: {document_name}")
        
        # Read and chunk the document
        try:
            text = self.read_pdf(file_path)
        except Exception as e:
            error(f"Failed to read PDF: {e}")
            return 0
        
        try:
            chunks = self.create_chunks(text)
        except Exception as e:
            error(f"Failed to create chunks: {e}")
            return 0
        
        if not chunks:
            warning(f"No chunks created from {document_name}")
            return 0
        
        # Filter out already-processed chunks
        new_chunks = []
        for chunk in chunks:
            if not self._is_chunk_processed(document_name, chunk['chunk_id']):
                new_chunks.append(chunk)
        
        if not new_chunks:
            info(f"All chunks from {document_name} already processed")
            return 0
        
        if len(new_chunks) < len(chunks):
            info(f"Skipping {len(chunks) - len(new_chunks)} already-processed chunks")
        
        info(f"Adding {len(new_chunks)} new chunks to ChromaDB (one-by-one)...")
        all_ids = []
        failed_count = 0
        
        try:
            for idx, chunk in enumerate(new_chunks):
                if idx % 100 == 0:
                    info(f"  Processing chunk {idx}/{len(new_chunks)}...")
                
                chroma_id = f"{document_name}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}"
                
                metadata = {
                    'source_file': document_name,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk['chunk_index'],
                    'level': 0,
                    'chunk_preview': chunk['text'][:50].replace('\n', ' '),
                    'added_timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Generate real embedding using sentence-transformers (768-dim)
                embedding = self.embedding_model.encode(chunk['text'], show_progress_bar=False)
                
                # Add single item
                try:
                    self.chunks_collection.add(
                        ids=[chroma_id],
                        documents=[chunk['text']],
                        metadatas=[metadata],
                        embeddings=[np.array(embedding, dtype=np.float32)]
                    )
                    self._mark_chunk_processed(document_name, chunk['chunk_id'])
                    all_ids.append(chroma_id)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 5:
                        error(f"  Failed to add chunk {idx}: {e}")
                    elif failed_count == 6:
                        error(f"  ... suppressing further error messages")
            
            self._save_processed_chunks()
            info(f"Successfully added {len(all_ids)} chunks from {document_name}")
            if failed_count > 0:
                warning(f"Failed to add {failed_count} chunks")
            
        except Exception as e:
            error(f"Critical error during processing for {document_name}: {e}")
            import traceback
            error(f"Traceback: {traceback.format_exc()}")
            self._save_processed_chunks()
            info(f"Partial success: added {len(all_ids)} chunks before error")
        
        return len(all_ids)
    
    def get_embeddings_by_level(self, level: int = 0) -> Tuple[np.ndarray, List[str], List[Dict], List[str]]:
        """Get all embeddings and metadata for a specific level"""
        # Get all items from chunks collection
        total_count = self.chunks_collection.count()
        info(f"Retrieving embeddings from {total_count} items in collection (level {level})...")
        
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        batch_size = LIMITS['batch_size']
        for offset in range(0, total_count, batch_size):
            try:
                result = self.chunks_collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["embeddings", "documents", "metadatas"]
                )
                
                # Check if embeddings exist
                if result.get('embeddings') is not None and len(result['embeddings']) > 0:
                    # Filter by level if specified
                    for i, (emb, doc, meta, id_) in enumerate(zip(
                        result['embeddings'], 
                        result['documents'], 
                        result['metadatas'],
                        result['ids']
                    )):
                        if meta.get('level', 0) == level:
                            all_embeddings.append(emb)
                            all_documents.append(doc)
                            all_metadatas.append(meta)
                            all_ids.append(id_)
                            
            except Exception as e:
                error(f"Error retrieving batch at offset {offset}: {e}")
                continue
        
        if len(all_embeddings) == 0:
            warning(f"No embeddings found at level {level}")
            return None, None, None, None
        
        info(f"Retrieved {len(all_embeddings)} items at level {level}")
        return np.array(all_embeddings), all_documents, all_metadatas, all_ids
    
    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> Dict:
        """Perform GMM clustering on embeddings"""
        if len(embeddings) < 3:
            warning("Not enough embeddings for clustering (need at least 3)")
            return None
        
        info(f"Clustering {len(embeddings)} embeddings...")
        
        # Reduce dimensions with PCA
        if embeddings.shape[1] > 100 and len(embeddings) > 50:
            info(f"Reducing dimensions from {embeddings.shape[1]}...")
            n_components = min(50, len(embeddings) // 2)
            pca = PCA(n_components=n_components)
            embeddings_reduced = pca.fit_transform(embeddings)
            info(f"Reduced to {embeddings_reduced.shape[1]} dimensions")
            explained_variance = pca.explained_variance_ratio_.sum()
            info(f"Explained variance: {explained_variance:.2%}")
        else:
            embeddings_reduced = embeddings
        
        # Standardize
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_reduced)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(2, min(CLUSTERING['max_clusters'], len(embeddings) // 100))
        
        info(f"Running GMM with {n_clusters} clusters...")
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=CLUSTERING['covariance_type'],
            random_state=CLUSTERING['random_state'],
            n_init=5,
            max_iter=200
        )
        
        cluster_labels = gmm.fit_predict(embeddings_scaled)
        cluster_probs = gmm.predict_proba(embeddings_scaled)
        
        # Calculate silhouette score
        silhouette_avg = 0
        if len(set(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
                info(f"Silhouette Score: {silhouette_avg:.3f}")
            except Exception as e:
                warning(f"Could not calculate silhouette score: {e}")
        
        return {
            'labels': cluster_labels,
            'probabilities': cluster_probs,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'embeddings_scaled': embeddings_scaled
        }
    
    def generate_summary_with_ollama(self, texts: List[str], max_chunks: int = 15) -> str:
        """Generate summary using Ollama local LLM with timeout and validation"""
        # Take sample of chunks (avoid token limits)
        sample_size = min(max_chunks, len(texts))
        sampled_texts = texts[:sample_size]
        
        # Combine texts with separators
        combined_text = "\n\n---\n\n".join(sampled_texts)
        
        # Truncate if too long
        max_length = LIMITS['max_summary_length']
        if len(combined_text) > max_length:
            warning(f"Summary input truncated from {len(combined_text)} to {max_length} chars")
            combined_text = combined_text[:max_length] + "..."
        
        # Create prompt for summarization
        prompt = f"""You are a document summarizer. Summarize the following content into a concise, informative summary that captures the main themes and key points.

Content to summarize:
{combined_text}

Provide a clear summary (3-5 paragraphs) that:
1. Identifies the main topic or theme
2. Highlights key concepts discussed
3. Mentions important details
4. Uses clear, concise language

Summary:"""
        
        try:
            debug(f"Calling Ollama API with {self.ollama_model}")
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": RAG['temperature'],
                        "num_predict": 500
                    }
                },
                timeout=LIMITS['ollama_timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response
                is_valid, msg = validate_ollama_response(result)
                if is_valid:
                    summary = result.get('response', '').strip()
                    return summary if summary else self._generate_fallback_summary(texts)
                else:
                    warning(f"Invalid Ollama response: {msg}")
                    return self._generate_fallback_summary(texts)
            else:
                error(f"Ollama API error: {response.status_code}")
                return self._generate_fallback_summary(texts)
                
        except requests.exceptions.Timeout:
            error(f"Ollama API timeout after {LIMITS['ollama_timeout']}s")
            return self._generate_fallback_summary(texts)
        except Exception as e:
            error(f"Error calling Ollama: {e}")
            return self._generate_fallback_summary(texts)
    
    def _generate_fallback_summary(self, texts: List[str]) -> str:
        """Fallback summary generation if Ollama fails"""
        debug("Using fallback summary generation")
        # Extract keywords
        all_text = ' '.join(texts)
        words = all_text.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'is', 'are', 'was', 'were', 'of', 'with'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get top keywords
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(10)
        
        summary = f"""
Cluster Summary (Fallback):
- Total chunks: {len(texts)}
- Key themes: {', '.join([f"{word}" for word, _ in top_keywords[:8]])}
- Sample content: {texts[0][:200]}...

This cluster contains content related to these key concepts.
"""
        return summary.strip()
    
    def store_cluster_summaries(self, clustering_results: Dict, 
                                documents: List[str],
                                metadatas: List[Dict],
                                level: int) -> Dict:
        """Store summaries for each cluster at a given level"""
        labels = clustering_results['labels']
        n_clusters = clustering_results['n_clusters']
        
        cluster_summaries = {}
        
        info(f"Storing summaries for Level {level} ({n_clusters} clusters)")
        
        for cluster_id in range(n_clusters):
            # Get chunks in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [documents[i] for i in cluster_indices]
            cluster_metas = [metadatas[i] for i in cluster_indices]
            
            debug(f"Cluster {cluster_id}: {len(cluster_texts)} chunks")
            
            # Generate summary using Ollama
            info(f"Generating summary for cluster {cluster_id}...")
            summary = self.generate_summary_with_ollama(cluster_texts)
            
            # Store summary in summaries collection
            summary_id = f"level_{level}_cluster_{cluster_id}_{uuid.uuid4().hex[:8]}"
            
            # Get source files in this cluster
            sources = list(set([m.get('source_file', 'unknown') for m in cluster_metas]))
            
            # Calculate average confidence
            avg_confidence = float(np.mean([max(clustering_results['probabilities'][i]) 
                                           for i in cluster_indices]))
            
            summary_metadata = {
                'type': 'cluster_summary',
                'level': level,
                'cluster_id': cluster_id,
                'chunks_count': len(cluster_texts),
                'source_files': sources,
                'avg_confidence': avg_confidence,
                'silhouette_score': clustering_results.get('silhouette_score', 0),
                'created_at': pd.Timestamp.now().isoformat(),
                'summary_preview': summary[:100]
            }
            
            # Store in summaries collection
            try:
                self.summaries_collection.add(
                    ids=[summary_id],
                    documents=[summary],
                    metadatas=[summary_metadata]
                )
                
                # Also create a summary chunk for the next level
                summary_chunk_id = f"summary_level_{level}_cluster_{cluster_id}"
                
                # Store summary as a chunk in the chunks collection for next level
                self.chunks_collection.add(
                    ids=[summary_chunk_id],
                    documents=[summary],
                    metadatas=[{
                        'type': 'cluster_summary',
                        'level': level + 1,
                        'cluster_id': cluster_id,
                        'parent_cluster_id': cluster_id,
                        'chunks_count': len(cluster_texts),
                        'source_files': sources,
                        'is_summary': True
                    }]
                )
                
                cluster_summaries[cluster_id] = {
                    'summary': summary,
                    'summary_id': summary_id,
                    'chunks_count': len(cluster_texts),
                    'sources': sources,
                    'summary_chunk_id': summary_chunk_id
                }
                
                debug(f"Stored summary for cluster {cluster_id}")
                
            except Exception as e:
                error(f"Failed to store summary for cluster {cluster_id}: {e}")
        
        return cluster_summaries
    
    def hierarchical_clustering(self, max_levels: int = None, 
                                improvement_threshold: float = None) -> Dict:
        """
        Perform hierarchical clustering recursively until no improvement
        
        Args:
            max_levels: Maximum hierarchical levels (default: from config)
            improvement_threshold: Min silhouette score improvement to continue (default: from config)
        
        Returns:
            Dictionary with clustering history
        """
        max_levels = max_levels or CLUSTERING['max_clusters']
        improvement_threshold = improvement_threshold or CLUSTERING['silhouette_threshold']
        
        info(f"Starting hierarchical clustering (max_levels={max_levels}, threshold={improvement_threshold})")
        
        history = {
            'levels': [],
            'convergence': False,
            'final_level': 0
        }
        
        current_level = 0
        previous_silhouette = 0
        
        while current_level < max_levels:
            info(f"Hierarchical clustering: Level {current_level}")
            
            # Get embeddings for current level
            embeddings, documents, metadatas, ids = self.get_embeddings_by_level(level=current_level)
            
            if embeddings is None:
                warning(f"No embeddings found at level {current_level}")
                break
            
            debug(f"Level {current_level}: {len(embeddings)} items")
            
            # Determine number of clusters based on data size
            n_clusters = max(2, min(CLUSTERING['max_clusters'], len(embeddings) // 50))
            
            # Perform clustering
            clustering_results = self.perform_clustering(embeddings, n_clusters)
            
            if clustering_results is None:
                error("Clustering failed")
                break
            
            # Store summaries for this level
            cluster_summaries = self.store_cluster_summaries(
                clustering_results,
                documents,
                metadatas,
                current_level
            )
            
            # Track history (cap to LIMITS['clustering_history_size'])
            level_info = {
                'level': current_level,
                'n_clusters': clustering_results['n_clusters'],
                'silhouette_score': clustering_results['silhouette_score'],
                'n_items': len(embeddings),
                'summaries_count': len(cluster_summaries)
            }
            
            history['levels'].append(level_info)
            
            # Keep only recent history in memory
            if len(self.clustering_history) >= LIMITS['clustering_history_size']:
                self.clustering_history = self.clustering_history[-LIMITS['clustering_history_size'] + 1:]
            self.clustering_history.append(level_info)
            
            current_silhouette = clustering_results['silhouette_score']
            
            # Check for improvement
            improvement = current_silhouette - previous_silhouette if current_level > 0 else 1
            
            info(f"Level {current_level}: Silhouette={current_silhouette:.3f}, Improvement={improvement:.3f}")
            
            # Stop if no improvement or only one cluster
            if current_level > 0 and improvement < improvement_threshold:
                info(f"Convergence reached (improvement {improvement:.3f} < threshold {improvement_threshold})")
                history['convergence'] = True
                history['final_level'] = current_level
                break
            
            # Stop if only one cluster formed
            if clustering_results['n_clusters'] <= 1:
                info(f"Clustering converged to 1 cluster at level {current_level}")
                history['convergence'] = True
                history['final_level'] = current_level
                break
            
            previous_silhouette = current_silhouette
            current_level += 1
        
        if current_level >= max_levels:
            warning(f"Reached maximum hierarchical levels ({max_levels})")
            history['final_level'] = max_levels - 1
        
        info(f"Hierarchical clustering complete. Final level: {history['final_level']}")
        
        return history
    
    def get_summary_tree(self) -> pd.DataFrame:
        """Get all summaries organized as a tree"""
        result = self.summaries_collection.get(
            include=["documents", "metadatas"]
        )
        
        if not result['ids']:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'summary_id': result['ids'],
            'level': [m.get('level', -1) for m in result['metadatas']],
            'cluster_id': [m.get('cluster_id', -1) for m in result['metadatas']],
            'chunks_count': [m.get('chunks_count', 0) for m in result['metadatas']],
            'silhouette_score': [m.get('silhouette_score', 0) for m in result['metadatas']],
            'summary_preview': [doc[:100] for doc in result['documents']],
            'created_at': [m.get('created_at', '') for m in result['metadatas']]
        })
        
        return df.sort_values(['level', 'cluster_id'])
    
    def query_by_level(self, query: str, level: int = None, n_results: int = 5) -> Dict:
        """Query summaries at a specific level"""
        where_filter = {}
        if level is not None:
            where_filter = {"level": level}
        
        results = self.summaries_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def visualize_hierarchy(self) -> None:
        """Print hierarchical structure of summaries"""
        df = self.get_summary_tree()
        
        if df.empty:
            print("No summaries found")
            return
        
        print("\n" + "="*80)
        print("HIERARCHICAL SUMMARY STRUCTURE")
        print("="*80)
        
        for level in sorted(df['level'].unique()):
            level_df = df[df['level'] == level]
            print(f"\nLevel {level} ({len(level_df)} clusters):")
            print("-" * 60)
            
            for _, row in level_df.iterrows():
                print(f"  Cluster {row['cluster_id']}:")
                print(f"    - {row['chunks_count']} chunks")
                print(f"    - Silhouette: {row['silhouette_score']:.3f}")
                print(f"    - Preview: {row['summary_preview'][:80]}...")
                print()


# ========== MAIN EXECUTION ==========

def main():
    # Initialize the processor
    processor = HierarchicalClusterSummarizer(
        chroma_path="./hierarchical_chroma_db",
        ollama_model="gemma3:4b",
        ollama_url="http://localhost:11434/api/generate"
    )
    
    # Check if Ollama is running (optional for basic ingestion)
    ollama_available = False
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Ollama is running")
            ollama_available = True
        else:
            print("WARNING: Ollama not responding (clustering will be skipped)")
    except Exception as e:
        print(f"WARNING: Cannot connect to Ollama: {e}")
        print("   (Documents will be added, but clustering/summaries skipped)")
    
    if not ollama_available:
        print("   To enable full features:")
        print("     ollama serve")
        print("     ollama pull gemma3:4b")
    
    # List of files to add (modify as needed)
    files_to_add = [
        "The Odyssey.pdf",
        "The Book of Five Rings.pdf",
        "The Road to React - Robin Wieruch.pdf",
    ]
    
    # Add files (if they exist)
    for file_path in files_to_add:
        if os.path.exists(file_path):
            print(f"\nProcessing: {file_path}")
            try:
                processor.add_document(file_path)
            except Exception as e:
                error(f"Failed to process {file_path}: {e}")
                import traceback
                error(f"Traceback: {traceback.format_exc()}")
                print(f"\nWARNING: Error processing {file_path}, continuing to next file...")
        else:
            print(f"\nWARNING: File not found: {file_path}")
    
    # If no files were added, check if there's existing data
    total_chunks = processor.chunks_collection.count()
    if total_chunks == 0:
        print("\nWARNING: No documents found. Please add PDF files to process.")
        return
    
    print(f"\nTotal chunks in database: {total_chunks}")
    
    # Perform hierarchical clustering only if Ollama is available
    if ollama_available:
        history = processor.hierarchical_clustering(
            max_levels=3,  # Maximum depth of hierarchy
            improvement_threshold=0.05  # Stop if improvement < 5%
        )
    else:
        print("\nSkipping hierarchical clustering (Ollama not available)")
        history = {'levels': [], 'convergence': False, 'final_level': 0}
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\nClustering History:")
    for level_info in history['levels']:
        print(f"  Level {level_info['level']}: {level_info['n_clusters']} clusters, "
              f"Silhouette: {level_info['silhouette_score']:.3f}, "
              f"Items: {level_info['n_items']}")
    
    print(f"\nConvergence reached: {history['convergence']}")
    print(f"Final level: {history['final_level']}")
    
    # Display summary tree
    processor.visualize_hierarchy()
    
    # Export summaries to CSV
    summaries_df = processor.get_summary_tree()
    if not summaries_df.empty:
        output_file = "hierarchical_summaries.csv"
        summaries_df.to_csv(output_file, index=False)
        print(f"\nSummaries exported to: {output_file}")
    
    # Export history
    history_df = pd.DataFrame(history['levels'])
    if not history_df.empty:
        history_file = "clustering_history.csv"
        history_df.to_csv(history_file, index=False)
        print(f"Clustering history exported to: {history_file}")

if __name__ == "__main__":
    main()