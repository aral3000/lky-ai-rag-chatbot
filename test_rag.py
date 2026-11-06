"""
Automated Verification Suite for LKY-AI RAG Engine.
Compatible with both `python -m unittest test_rag.py -v` and `pytest`.
"""

import os
import unittest
from rag_engine import LKYVectorStore
from llm_client import generate_lky_response
from eval_rag import run_evaluation

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "lky_corpus.json")


class TestLKYRagEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = LKYVectorStore(CORPUS_PATH)

    def test_corpus_loading(self) -> None:
        self.assertGreaterEqual(len(self.store.documents), 12)
        self.assertGreater(len(self.store.vocabulary), 100)
        self.assertEqual(self.store.doc_vectors.shape, (len(self.store.documents), len(self.store.vocabulary)))

    def test_hdb_housing_query_retrieval(self) -> None:
        query = "How did you solve housing shortage and property ownership for citizens?"
        results = self.store.search(query, top_k=2)
        self.assertGreater(len(results), 0)
        top_doc, score = results[0]
        self.assertGreater(score, 0.05)
        self.assertEqual(top_doc["theme"], "Housing & Real Estate")

    def test_ethnic_integration_housing_retrieval(self) -> None:
        query = "How do you prevent racial enclaves in public housing neighborhoods?"
        results = self.store.search(query, top_k=2)
        self.assertEqual(results[0][0]["id"], "hdb_03")

    def test_technology_automation_query_retrieval(self) -> None:
        query = "How should nations adapt to technological automation and artificial intelligence?"
        results = self.store.search(query, top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("Technology", results[0][0]["theme"])

    def test_cosine_similarity_ranking_order(self) -> None:
        query = "meritocracy corruption rule of law"
        results = self.store.search(query, top_k=4)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for score in scores:
            self.assertTrue(0.0 <= score <= 1.0)

    def test_context_formatting(self) -> None:
        results = self.store.search("corruption investigation", top_k=1)
        context = self.store.format_retrieved_context(results)
        self.assertIn("[Source #1]", context)
        self.assertIn("Citation:", context)
        self.assertIn("Excerpt:", context)

    def test_offline_persona_generation(self) -> None:
        query = "Why is public housing important?"
        results = self.store.search(query, top_k=2)
        context = self.store.format_retrieved_context(results)
        response = generate_lky_response(query, context, results, provider="Demo Mode (Offline)")
        self.assertGreater(len(response), 100)
        self.assertTrue("property ownership" in response.lower() or "housing" in response.lower())

    def test_strict_min_score_filtering(self) -> None:
        query = "quantum entanglement astrophysics galaxy blackhole"
        results = self.store.search(query, top_k=2, min_score=0.12)
        self.assertEqual(len(results), 0)

    def test_quantitative_rag_evaluation_thresholds(self) -> None:
        report = run_evaluation(top_k=3, ood_min_score=0.12)
        self.assertGreaterEqual(report["metrics"]["hit_rate_at_1"], 0.85)
        self.assertGreaterEqual(report["metrics"]["hit_rate_at_3"], 0.95)
        self.assertGreaterEqual(report["metrics"]["mrr"], 0.90)
        self.assertEqual(report["metrics"]["ood_rejection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
