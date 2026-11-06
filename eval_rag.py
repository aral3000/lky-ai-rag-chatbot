"""
Quantitative RAG Evaluation Suite for LKY-AI Chatbot.
Measures Hit Rate@1, Hit Rate@3, Mean Reciprocal Rank (MRR),
Out-of-Domain Abstention Rate, and Grounded Citation Faithfulness.
"""

import json
import os
from typing import Any
from rag_engine import LKYVectorStore
from llm_client import generate_lky_response

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "lky_corpus.json")

IN_DOMAIN_BENCHMARK: list[dict[str, str]] = [
    {"query": "How did Singapore enable 90% public home ownership through HDB flats?", "expected_id": "hdb_01"},
    {"query": "Why did you plant trees and reclaim land in a dense high-rise city?", "expected_id": "hdb_02"},
    {"query": "How do you prevent racial enclaves in public housing neighborhoods?", "expected_id": "hdb_03"},
    {"query": "How can a small island with no hinterland industrialize using MNCs?", "expected_id": "econ_01"},
    {"query": "How do you eradicate civil service corruption and enforce rule of law?", "expected_id": "econ_02"},
    {"query": "Should governments fund unemployment handouts or compulsory CPF savings?", "expected_id": "cpf_01"},
    {"query": "Why choose English as the working language alongside Mother Tongue?", "expected_id": "lang_01"},
    {"query": "How should workers adapt to automation and artificial intelligence?", "expected_id": "tech_01"},
    {"query": "How should a country recruit and promote top talent into public leadership?", "expected_id": "gov_01"},
    {"query": "How did Singapore overcome water supply vulnerability with NEWater?", "expected_id": "water_01"},
    {"query": "How does a small state deter larger neighbors with a poison shrimp defense?", "expected_id": "geo_01"},
    {"query": "How should ASEAN balance the rise of China and American presence in Asia?", "expected_id": "asean_01"},
]

OUT_OF_DOMAIN_TRAPS: list[str] = [
    "What is the best Solidity smart contract for meme coin yield farming?",
    "Who won the UEFA Champions League penalty shootout last Tuesday?",
    "Explain quark-gluon plasma phase transitions in quantum chromodynamics.",
]


def run_evaluation(top_k: int = 3, ood_min_score: float = 0.12) -> dict[str, Any]:
    store = LKYVectorStore(CORPUS_PATH)
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    faithfulness_hits = 0
    per_query: list[dict[str, Any]] = []

    for case in IN_DOMAIN_BENCHMARK:
        results = store.search(case["query"], top_k=top_k, min_score=0.05)
        retrieved_ids = [doc["id"] for doc, _ in results]
        expected = case["expected_id"]

        rank = (retrieved_ids.index(expected) + 1) if expected in retrieved_ids else 0
        if rank == 1:
            hits_at_1 += 1
        if rank > 0:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        context_str = store.format_retrieved_context(results)
        reply = generate_lky_response(case["query"], context_str, results, provider="Demo Mode (Offline)")
        top_year = str(results[0][0]["year"]) if results else ""
        top_title = results[0][0]["title"] if results else ""
        is_faithful = bool(results and (top_year in reply or top_title in reply))
        if is_faithful:
            faithfulness_hits += 1

        per_query.append({
            "query": case["query"],
            "expected_id": expected,
            "retrieved_ids": retrieved_ids,
            "top_score": results[0][1] if results else 0.0,
            "rank": rank,
            "citation_faithful": is_faithful,
        })

    ood_rejected = 0
    for trap in OUT_OF_DOMAIN_TRAPS:
        trap_results = store.search(trap, top_k=top_k, min_score=ood_min_score)
        if len(trap_results) == 0:
            ood_rejected += 1

    total = len(IN_DOMAIN_BENCHMARK)
    summary = {
        "corpus_size": len(store.documents),
        "vocabulary_size": len(store.vocabulary),
        "in_domain_queries": total,
        "out_of_domain_queries": len(OUT_OF_DOMAIN_TRAPS),
        "metrics": {
            "hit_rate_at_1": round(hits_at_1 / total, 4),
            f"hit_rate_at_{top_k}": round(hits_at_k / total, 4),
            "mrr": round(sum(reciprocal_ranks) / total, 4),
            "ood_rejection_rate": round(ood_rejected / len(OUT_OF_DOMAIN_TRAPS), 4),
            "citation_faithfulness_rate": round(faithfulness_hits / total, 4),
        },
        "details": per_query,
    }
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    report = run_evaluation()
    print("LKY-AI RAG Quantitative Evaluation Summary")
    print("-" * 46)
    print(f"Corpus Documents : {report['corpus_size']} | Vocabulary: {report['vocabulary_size']} terms")
    for k, v in report["metrics"].items():
        print(f"{k:28s}: {v:.2%}" if k != "mrr" else f"{k:28s}: {v:.4f}")
