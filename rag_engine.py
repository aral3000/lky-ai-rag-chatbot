"""
RAG Engine for Lee Kuan Yew Persona Chatbot (LKY-AI)
Vectorized Cosine Similarity Search & Semantic Grounding over Singapore Historical Speeches.
Author: M. Athallah Rafif Aldera, S.Kom.
"""

import json
import os
import re
import math
from typing import List, Dict, Any, Tuple
import numpy as np


class LKYVectorStore:
    """
    Lightweight, deterministic vectorized vector store with BM25 / Cosine Similarity
    built using NumPy. Zero external vector database overhead.
    """

    def __init__(self, corpus_path: str = None):
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])
        
        if corpus_path and os.path.exists(corpus_path):
            self.load_corpus(corpus_path)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        return re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())

    def load_corpus(self, corpus_path: str):
        """Loads and indexes JSON corpus."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Build vocabulary
        doc_tokens = []
        df: Dict[str, int] = {}

        for doc in self.documents:
            full_text = f"{doc.get('title', '')} {doc.get('theme', '')} {doc.get('content', '')}"
            tokens = set(self._tokenize(full_text))
            doc_tokens.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        # Sorted vocab for deterministic index
        self.vocabulary = {t: i for i, t in enumerate(sorted(df.keys()))}
        n_docs = len(self.documents)
        vocab_size = len(self.vocabulary)

        # Calculate IDF: log((N + 1) / (df + 1)) + 1
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for t, idx in self.vocabulary.items():
            self.idf[idx] = math.log((n_docs + 1) / (df[t] + 1)) + 1.0

        # Build TF-IDF document vectors
        self.doc_vectors = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for i, doc in enumerate(self.documents):
            full_text = f"{doc.get('title', '')} {doc.get('theme', '')} {doc.get('content', '')}"
            tokens = self._tokenize(full_text)
            for t in tokens:
                if t in self.vocabulary:
                    self.doc_vectors[i, self.vocabulary[t]] += 1.0

            # Multiply by IDF and L2-normalize
            self.doc_vectors[i] *= self.idf
            norm = np.linalg.norm(self.doc_vectors[i])
            if norm > 0:
                self.doc_vectors[i] /= norm

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes vectorized Cosine Similarity search for the given query.
        Returns top-k documents sorted by descending similarity score.
        """
        if len(self.documents) == 0 or len(self.vocabulary) == 0:
            return []

        tokens = self._tokenize(query)
        q_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for t in tokens:
            if t in self.vocabulary:
                q_vec[self.vocabulary[t]] += 1.0

        q_vec *= self.idf
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
        else:
            # Fallback if no exact vocab hit
            return [(self.documents[0], 0.1)]

        # Vectorized dot product for all documents simultaneously
        scores = np.dot(self.doc_vectors, q_vec)

        # Rank documents
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score >= min_score or len(results) == 0:
                results.append((self.documents[idx], round(score, 4)))

        return results

    def format_retrieved_context(self, matches: List[Tuple[Dict[str, Any], float]]) -> str:
        """Formats retrieved documents into grounded citations for LLM prompt context."""
        if not matches:
            return "No historical speeches matched this exact query."

        context_blocks = []
        for i, (doc, score) in enumerate(matches, 1):
            block = (
                f"[Source #{i}]: {doc.get('title')} ({doc.get('year')}) | Relevance: {score:.2f}\n"
                f"Citation: {doc.get('source')}\n"
                f"Excerpt: \"{doc.get('content')}\""
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
