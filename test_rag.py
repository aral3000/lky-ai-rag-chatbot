"""
Automated Verification Suite for LKY-AI RAG Engine.
Compatible with both `python -m unittest test_rag.py -v` and `pytest`.
"""

import json
import os
import unittest
from rag_engine import LKYVectorStore
from llm_client import generate_lky_response
from eval_rag import run_evaluation, audit_response_quality

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "lky_corpus.json")
FLOWISE_PATH = os.path.join(os.path.dirname(__file__), "flowise_chatflow.json")


class TestLKYRagEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = LKYVectorStore(CORPUS_PATH)

    def test_corpus_loading(self) -> None:
        self.assertGreaterEqual(len(self.store.documents), 16)
        self.assertGreater(len(self.store.vocabulary), 140)
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

    def test_personal_life_philosophy_retrieval(self) -> None:
        query = "What made your marriage to Kwa Geok Choo endure and how did you raise your children?"
        results = self.store.search(query, top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0]["id"], "life_01")

    def test_paraphrased_low_lexical_overlap_retrieval(self) -> None:
        query = "How do you stop bribery and graft among government officials?"
        results = self.store.search(query, top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0]["id"], "econ_02")

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

    def test_generation_temporal_and_citation_guardrails(self) -> None:
        docs = [({"id": "tech_01", "year": 1999, "title": "Lifelong Education"}, 0.31)]
        good_reply = "In 1999, I warned that technology moves relentlessly and we must upgrade our workforce."
        bad_anachronism = "In 1999 and later in 2022, I launched a generative AI subsidy."
        bad_ungrounded_year = "In 1999 and 2006, I launched the SkillsFuture agency."
        self.assertTrue(audit_response_quality(good_reply, docs)["temporal_guardrail_pass"])
        self.assertFalse(audit_response_quality(bad_anachronism, docs)["temporal_guardrail_pass"])
        self.assertFalse(audit_response_quality(bad_ungrounded_year, docs)["temporal_guardrail_pass"])

    def test_flowise_chatflow_json_schema(self) -> None:
        self.assertTrue(os.path.exists(FLOWISE_PATH))
        with open(FLOWISE_PATH, "r", encoding="utf-8") as fh:
            flow = json.load(fh)
        self.assertEqual(len(flow.get("nodes", [])), 6)
        self.assertEqual(len(flow.get("edges", [])), 5)

    def test_quantitative_rag_evaluation_thresholds(self) -> None:
        report = run_evaluation(top_k=3, ood_min_score=0.12)
        self.assertGreaterEqual(report["metrics"]["hit_rate_at_1"], 0.85)
        self.assertGreaterEqual(report["metrics"]["hit_rate_at_3"], 0.95)
        self.assertGreaterEqual(report["metrics"]["mrr"], 0.90)
        self.assertEqual(report["metrics"]["ood_rejection_rate"], 1.0)
        self.assertEqual(report["metrics"]["temporal_guardrail_rate"], 1.0)
        self.assertEqual(report["metrics"]["live_transcript_guardrail_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
