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

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


ACRONYM_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "ai": ("artificial", "intelligence", "automation", "technology"),
    "hdb": ("housing", "development", "board", "home", "ownership", "flats"),
    "cpf": ("central", "provident", "fund", "savings", "retirement"),
    "cpib": ("corrupt", "practices", "investigation", "corruption"),
    "eip": ("ethnic", "integration", "policy", "enclaves", "multiracial"),
    "mnc": ("multinational", "corporations", "industrial"),
    "mncs": ("multinational", "corporations", "industrial"),
    "bribery": ("corruption", "corrupt", "cpib", "honest"),
    "graft": ("corruption", "corrupt", "cpib", "honest"),
    "segregation": ("enclaves", "ethnic", "integration", "quotas"),
    "ghettos": ("enclaves", "ethnic", "integration", "quotas"),
    "wife": ("marriage", "kwa", "geok", "choo", "family", "partnership"),
    "spouse": ("marriage", "kwa", "geok", "choo", "family", "partnership"),
    "workout": ("physical", "exercise", "swimming", "running", "regimen"),
    "fitness": ("physical", "exercise", "swimming", "running", "regimen"),
    "burnout": ("physical", "exercise", "sleep", "meditation", "stamina"),
    "death": ("mortality", "grief", "bereavement", "aging", "stoic"),
    "dying": ("mortality", "grief", "bereavement", "aging", "stoic"),
    "career": ("young", "vocation", "competence", "resilience", "character"),
}


def _porter_lite_stem(token: str) -> str:
    """Lightweight morphological stemmer unifying inflections (automating/automation, enclaves/enclave)."""
    if len(token) <= 3:
        return token
    for suffix, repl in (
        ("ations", "at"), ("ation", "at"), ("ments", "ment"),
        ("ities", "iti"), ("izing", "iz"), ("ising", "is"),
        ("ized", "iz"), ("ised", "is"), ("ting", "t"),
        ("ning", "n"), ("ling", "l"), ("ing", ""),
        ("ies", "i"), ("ers", "er"), ("ed", ""), ("es", ""), ("s", ""),
    ):
        if token.endswith(suffix) and (len(token) - len(suffix) + len(repl)) >= 4:
            token = token[: -len(suffix)] + repl
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("y"):
        token = token[:-1] + "i"
    return token


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

    def _tokenize(self, text: str, filter_stops: bool = True) -> List[str]:
        """Tokenize text into stemmed alphanumeric terms with domain acronym expansion."""
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        expanded: List[str] = []
        for tok in raw_tokens:
            expanded.append(tok)
            if tok in ACRONYM_EXPANSIONS:
                expanded.extend(ACRONYM_EXPANSIONS[tok])
        if filter_stops:
            filtered = [_porter_lite_stem(t) for t in expanded if t not in STOP_WORDS]
            return filtered if filtered else [_porter_lite_stem(t) for t in expanded]
        return [_porter_lite_stem(t) for t in expanded]

    def load_corpus(self, corpus_path: str):
        """Loads and indexes JSON corpus."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Build vocabulary
        doc_tokens = []
        df: Dict[str, int] = {}

        for doc in self.documents:
            full_text = f"{doc.get('title', '')} {doc.get('theme', '')} {doc.get('content', '')}"
            tokens = set(self._tokenize(full_text, filter_stops=True))
            doc_tokens.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        # Sorted vocab for deterministic index
        self.vocabulary = {t: i for i, t in enumerate(sorted(df.keys()))}
        n_docs = len(self.documents)
        vocab_size = len(self.vocabulary)

        if vocab_size == 0 or n_docs == 0:
            return

        # Calculate IDF: log((N + 1) / (df + 1)) + 1
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for t, idx in self.vocabulary.items():
            self.idf[idx] = math.log((n_docs + 1) / (df[t] + 1)) + 1.0

        # Build TF-IDF document vectors
        self.doc_vectors = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for i, doc in enumerate(self.documents):
            full_text = f"{doc.get('title', '')} {doc.get('theme', '')} {doc.get('content', '')}"
            tokens = self._tokenize(full_text, filter_stops=True)
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

        tokens = self._tokenize(query, filter_stops=True)
        q_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        matched_any = False
        oov_count = 0
        for t in tokens:
            if t in self.vocabulary:
                q_vec[self.vocabulary[t]] += 1.0
                matched_any = True
            else:
                oov_count += 1

        if not matched_any:
            # No vocabulary overlap found
            return []

        q_vec *= self.idf
        oov_idf = math.log((len(self.documents) + 1) / 1.0) + 1.0
        q_norm = math.sqrt(float(np.dot(q_vec, q_vec)) + oov_count * (oov_idf ** 2))
        if q_norm > 0:
            q_vec /= q_norm
        else:
            return []

        # Vectorized dot product for all documents simultaneously
        scores = np.dot(self.doc_vectors, q_vec)

        # Rank documents
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self.documents[idx], round(score, 4)))

        return results

    def format_retrieved_context(self, matches: List[Tuple[Dict[str, Any], float]]) -> str:
        """Formats retrieved documents into grounded citations for LLM prompt context."""
        if not matches:
            return "No historical speeches met the relevance threshold for this query."

        context_blocks = []
        for i, (doc, score) in enumerate(matches, 1):
            block = (
                f"[Source #{i}]: {doc.get('title')} ({doc.get('year')}) | Relevance: {score:.2f}\n"
                f"Citation: {doc.get('source')}\n"
                f"Excerpt: \"{doc.get('content')}\""
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
