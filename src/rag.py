"""
RAG (Retrieval-Augmented Generation) module for the Solar Energy Grid Optimization Assistant.

This module loads knowledge documents from the `knowledge/` directory,
splits them into chunks, embeds them using sentence-transformers,
and stores them for fast similarity search using sklearn NearestNeighbors.
"""

import os
import glob
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Paths
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge')
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rag_index.pkl')

# Embedding model (small, fast, runs locally — no API needed)
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_documents(knowledge_dir: str = KNOWLEDGE_DIR) -> list[dict]:
    """Load all markdown files from the knowledge directory."""
    documents = []
    for filepath in sorted(glob.glob(os.path.join(knowledge_dir, '*.md'))):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append({
            'source': os.path.basename(filepath),
            'content': content
        })
    return documents


def chunk_documents(documents: list[dict], chunk_size: int = 300) -> list[dict]:
    """
    Split documents into smaller chunks for better retrieval precision.
    Each chunk is roughly `chunk_size` words.
    """
    chunks = []
    for doc in documents:
        lines = doc['content'].split('\n')
        current_chunk = []
        current_word_count = 0

        for line in lines:
            word_count = len(line.split())
            if current_word_count + word_count > chunk_size and current_chunk:
                chunks.append({
                    'source': doc['source'],
                    'text': '\n'.join(current_chunk)
                })
                current_chunk = [line]
                current_word_count = word_count
            else:
                current_chunk.append(line)
                current_word_count += word_count

        # Add the last chunk
        if current_chunk:
            chunks.append({
                'source': doc['source'],
                'text': '\n'.join(current_chunk)
            })
    return chunks


class RAGRetriever:
    """Sklearn-based retrieval system for grid management knowledge."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.nn_index: NearestNeighbors | None = None

    def build_index(self, knowledge_dir: str = KNOWLEDGE_DIR):
        """Load documents, chunk them, embed, and build the search index."""
        documents = load_documents(knowledge_dir)
        if not documents:
            raise FileNotFoundError(f"No .md files found in {knowledge_dir}")

        self.chunks = chunk_documents(documents)
        texts = [chunk['text'] for chunk in self.chunks]

        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=False)
        self.embeddings = np.array(self.embeddings, dtype='float32')

        # Build NearestNeighbors index
        self.nn_index = NearestNeighbors(n_neighbors=min(5, len(self.chunks)), metric='cosine')
        self.nn_index.fit(self.embeddings)

        # Save the index and chunks
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)

        return len(self.chunks)

    def _load_index(self):
        """Load a previously saved index from disk."""
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.nn_index = NearestNeighbors(n_neighbors=min(5, len(self.chunks)), metric='cosine')
            self.nn_index.fit(self.embeddings)
        else:
            self.build_index()

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a given query.
        
        Returns a list of dicts with 'source', 'text', and 'score' keys.
        """
        if self.nn_index is None:
            self._load_index()

        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype='float32')

        # Search
        distances, indices = self.nn_index.kneighbors(query_embedding, n_neighbors=min(k, len(self.chunks)))

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'source': self.chunks[idx]['source'],
                'text': self.chunks[idx]['text'],
                'score': float(distances[0][i])
            })
        return results


# Singleton instance for use across the app
_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Get or create the singleton RAGRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.build_index()
    return _retriever


if __name__ == '__main__':
    # Quick test
    retriever = get_retriever()
    print(f"Built index with {len(retriever.chunks)} chunks")

    results = retriever.retrieve("How should I handle battery storage during peak hours?")
    for r in results:
        print(f"\n--- Source: {r['source']} (score: {r['score']:.4f}) ---")
        print(r['text'][:200])
