"""
Pytest Verification Suite for LKY-AI RAG Engine
Author: M. Athallah Rafif Aldera, S.Kom.
"""

import os
import pytest
from rag_engine import LKYVectorStore
from llm_client import generate_lky_response

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "lky_corpus.json")


@pytest.fixture
def vector_store():
    assert os.path.exists(CORPUS_PATH), f"Corpus file missing at {CORPUS_PATH}"
    store = LKYVectorStore(CORPUS_PATH)
    return store


def test_corpus_loading(vector_store):
    """Verify that corpus documents and vocabulary are properly indexed."""
    assert len(vector_store.documents) >= 8
    assert len(vector_store.vocabulary) > 50
    assert vector_store.doc_vectors.shape[0] == len(vector_store.documents)
    assert vector_store.doc_vectors.shape[1] == len(vector_store.vocabulary)


def test_hdb_housing_query_retrieval(vector_store):
    """
    Critical for 99 Group assessment: Verify that housing/real estate queries
    accurately retrieve HDB and urban planning documents.
    """
    query = "How did you solve housing shortage and property ownership for citizens?"
    results = vector_store.search(query, top_k=2)

    assert len(results) > 0
    top_doc, score = results[0]
    assert score > 0.05
    assert top_doc["theme"] == "Housing & Real Estate"
    assert "HDB" in top_doc["content"] or "home" in top_doc["content"].lower()


def test_technology_automation_query_retrieval(vector_store):
    """Verify that technology, automation, and AI queries retrieve the human capital document."""
    query = "How should nations adapt to technological automation and artificial intelligence?"
    results = vector_store.search(query, top_k=2)

    assert len(results) > 0
    top_doc, score = results[0]
    assert "Technology" in top_doc["theme"] or "education" in top_doc["content"].lower()


def test_cosine_similarity_ranking_order(vector_store):
    """Verify that similarity scores are strictly ordered descending."""
    query = "meritocracy corruption rule of law"
    results = vector_store.search(query, top_k=4)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_context_formatting(vector_store):
    """Verify context formatting includes citations and excerpts."""
    query = "corruption investigation"
    results = vector_store.search(query, top_k=1)
    context = vector_store.format_retrieved_context(results)

    assert "[Source #1]" in context
    assert "Citation:" in context
    assert "Excerpt:" in context


def test_offline_persona_generation(vector_store):
    """Verify that offline deterministic persona synthesis produces grounded output."""
    query = "Why is public housing important?"
    results = vector_store.search(query, top_k=2)
    context = vector_store.format_retrieved_context(results)

    response = generate_lky_response(query, context, results, provider="Demo Mode (Offline)")

    assert len(response) > 100
    assert "Lee Kuan Yew" not in response or "In" in response
    assert "property ownership" in response.lower() or "housing" in response.lower()
