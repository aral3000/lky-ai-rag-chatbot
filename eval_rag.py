"""
Quantitative RAG & Persona Faithfulness Evaluation Suite for LKY-AI Chatbot.
Measures Retrieval Metrics (Hit Rate@1, Hit Rate@3, MRR, Slice Breakdowns, OOD Abstention)
and Generation Guardrails (Citation Grounding, Temporal Anachronism Guard <=2015, Persona Voice).
"""

import json
import os
import re
from typing import Any
from rag_engine import LKYVectorStore
from llm_client import generate_lky_response

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "lky_corpus.json")
TRANSCRIPT_PATH = os.path.join(os.path.dirname(__file__), "sample_chat_session.json")

IN_DOMAIN_BENCHMARK: list[dict[str, str]] = [
    # Slice 1: Policy, Housing, Economy & Geopolitics (12 queries)
    {"slice": "policy_geopolitics", "query": "How did Singapore enable 90% public home ownership through HDB flats?", "expected_id": "hdb_01"},
    {"slice": "policy_geopolitics", "query": "Why did you plant trees and reclaim land in a dense high-rise city?", "expected_id": "hdb_02"},
    {"slice": "policy_geopolitics", "query": "How do you prevent racial enclaves in public housing neighborhoods?", "expected_id": "hdb_03"},
    {"slice": "policy_geopolitics", "query": "How can a small island with no hinterland industrialize using MNCs?", "expected_id": "econ_01"},
    {"slice": "policy_geopolitics", "query": "How do you eradicate civil service corruption and enforce rule of law?", "expected_id": "econ_02"},
    {"slice": "policy_geopolitics", "query": "Should governments fund unemployment handouts or compulsory CPF savings?", "expected_id": "cpf_01"},
    {"slice": "policy_geopolitics", "query": "Why choose English as the working language alongside Mother Tongue?", "expected_id": "lang_01"},
    {"slice": "policy_geopolitics", "query": "How should workers adapt to automation and artificial intelligence?", "expected_id": "tech_01"},
    {"slice": "policy_geopolitics", "query": "How should a country recruit and promote top talent into public leadership?", "expected_id": "gov_01"},
    {"slice": "policy_geopolitics", "query": "How did Singapore overcome water supply vulnerability with NEWater?", "expected_id": "water_01"},
    {"slice": "policy_geopolitics", "query": "How does a small state deter larger neighbors with a poison shrimp defense?", "expected_id": "geo_01"},
    {"slice": "policy_geopolitics", "query": "How should ASEAN balance the rise of China and American presence in Asia?", "expected_id": "asean_01"},
    # Slice 2: Life, Family & Personal Philosophy (4 queries)
    {"slice": "life_philosophy", "query": "What made your marriage to Kwa Geok Choo last and how did you raise your children?", "expected_id": "life_01"},
    {"slice": "life_philosophy", "query": "What was your daily physical exercise, sleep, and meditation routine?", "expected_id": "life_02"},
    {"slice": "life_philosophy", "query": "How should a person face grief, bereavement, aging, and mortality?", "expected_id": "life_03"},
    {"slice": "life_philosophy", "query": "What is your advice to young people on ambition, character, and overcoming failure?", "expected_id": "life_04"},
    # Slice 3: Paraphrased / Low-Lexical-Overlap Queries (4 queries)
    {"slice": "paraphrased_low_overlap", "query": "How do you stop bribery and graft among government officials?", "expected_id": "econ_02"},
    {"slice": "paraphrased_low_overlap", "query": "What is the secret to a lifelong partnership with your wife?", "expected_id": "life_01"},
    {"slice": "paraphrased_low_overlap", "query": "How does a leader avoid burnout and stay fit under decades of stress?", "expected_id": "life_02"},
    {"slice": "paraphrased_low_overlap", "query": "How do you stop minority segregation and ghettos in apartment blocks?", "expected_id": "hdb_03"},
]

OUT_OF_DOMAIN_TRAPS: list[str] = [
    "What is the best Solidity smart contract for meme coin yield farming?",
    "Who won the UEFA Champions League penalty shootout last Tuesday?",
    "Explain quark-gluon plasma phase transitions in quantum chromodynamics.",
    "How do I configure a Kubernetes ingress controller for gRPC load balancing?",
]

ALLOWED_HISTORICAL_ANCHOR_YEARS = {1923, 1959, 1965, 2015}


def audit_response_quality(
    reply: str,
    retrieved_docs: list[tuple[dict[str, Any], float]]
) -> dict[str, Any]:
    """
    Audits a generated response across 3 deterministic RAG + Persona guardrails:
    1. citation_grounded: References at least one retrieved document year or title.
    2. temporal_guardrail_pass: Contains zero post-2015 years and zero ungrounded years.
    3. persona_voice_pass: Uses first-person LKY voice and zero forbidden punctuation (em-dashes).
    """
    if not retrieved_docs:
        abstains = "outside my historical archival record" in reply.lower()
        return {
            "citation_grounded": abstains,
            "temporal_guardrail_pass": True,
            "persona_voice_pass": "—" not in reply,
        }

    retrieved_years = {int(doc["year"]) for doc, _ in retrieved_docs if "year" in doc}
    retrieved_titles = [str(doc.get("title", "")).lower() for doc, _ in retrieved_docs]

    # 1. Citation grounding check
    has_year = any(str(y) in reply for y in retrieved_years)
    has_title = any(t and t in reply.lower() for t in retrieved_titles)
    citation_grounded = bool(has_year or has_title)

    # 2. Temporal guardrail check (no years > 2015, and all cited years must be grounded)
    mentioned_years = {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", reply)}
    post_2015_years = {y for y in mentioned_years if y > 2015}
    ungrounded_years = mentioned_years - retrieved_years - ALLOWED_HISTORICAL_ANCHOR_YEARS
    temporal_guardrail_pass = len(post_2015_years) == 0 and len(ungrounded_years) == 0

    # 3. First-person persona voice & formatting check
    has_first_person = bool(re.search(r"\b(i|my|we|our)\b", reply.lower()))
    no_em_dash = "—" not in reply
    persona_voice_pass = has_first_person and no_em_dash

    return {
        "citation_grounded": citation_grounded,
        "temporal_guardrail_pass": temporal_guardrail_pass,
        "persona_voice_pass": persona_voice_pass,
        "mentioned_years": sorted(mentioned_years),
        "ungrounded_years": sorted(ungrounded_years),
    }


def run_evaluation(top_k: int = 3, ood_min_score: float = 0.12) -> dict[str, Any]:
    store = LKYVectorStore(CORPUS_PATH)
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    faithfulness_hits = 0
    temporal_guard_hits = 0
    persona_voice_hits = 0
    per_query: list[dict[str, Any]] = []
    slice_totals: dict[str, dict[str, float]] = {}

    for case in IN_DOMAIN_BENCHMARK:
        sl = case["slice"]
        if sl not in slice_totals:
            slice_totals[sl] = {"count": 0, "hits_at_1": 0, "hits_at_k": 0, "rr_sum": 0.0}
        slice_totals[sl]["count"] += 1

        results = store.search(case["query"], top_k=top_k, min_score=0.05)
        retrieved_ids = [doc["id"] for doc, _ in results]
        expected = case["expected_id"]

        rank = (retrieved_ids.index(expected) + 1) if expected in retrieved_ids else 0
        if rank == 1:
            hits_at_1 += 1
            slice_totals[sl]["hits_at_1"] += 1
        if rank > 0:
            hits_at_k += 1
            slice_totals[sl]["hits_at_k"] += 1
            reciprocal_ranks.append(1.0 / rank)
            slice_totals[sl]["rr_sum"] += 1.0 / rank
        else:
            reciprocal_ranks.append(0.0)

        context_str = store.format_retrieved_context(results)
        reply = generate_lky_response(case["query"], context_str, results, provider="Demo Mode (Offline)")
        audit = audit_response_quality(reply, results)

        if audit["citation_grounded"]:
            faithfulness_hits += 1
        if audit["temporal_guardrail_pass"]:
            temporal_guard_hits += 1
        if audit["persona_voice_pass"]:
            persona_voice_hits += 1

        per_query.append({
            "slice": sl,
            "query": case["query"],
            "expected_id": expected,
            "retrieved_ids": retrieved_ids,
            "top_score": results[0][1] if results else 0.0,
            "rank": rank,
            "citation_faithful": audit["citation_grounded"],
            "temporal_guardrail_pass": audit["temporal_guardrail_pass"],
            "persona_voice_pass": audit["persona_voice_pass"],
        })

    ood_rejected = 0
    ood_abstention_passed = 0
    for trap in OUT_OF_DOMAIN_TRAPS:
        trap_results = store.search(trap, top_k=top_k, min_score=ood_min_score)
        if len(trap_results) == 0:
            ood_rejected += 1
        empty_ctx = store.format_retrieved_context([])
        ood_reply = generate_lky_response(trap, empty_ctx, [], provider="Demo Mode (Offline)")
        if audit_response_quality(ood_reply, [])["citation_grounded"]:
            ood_abstention_passed += 1

    # Audit live multi-turn transcript if present
    transcript_turns_passed = 0
    transcript_total_turns = 0
    if os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as tf:
            tdata = json.load(tf)
        for turn in tdata.get("turns", []):
            transcript_total_turns += 1
            fake_results = [
                ({"year": s["year"], "title": s["title"]}, s.get("cosine_score", 0.2))
                for s in turn.get("retrieved_sources", [])
            ]
            t_audit = audit_response_quality(turn.get("lky_response", ""), fake_results)
            if (
                t_audit["citation_grounded"]
                and t_audit["temporal_guardrail_pass"]
                and t_audit["persona_voice_pass"]
            ):
                transcript_turns_passed += 1

    total = len(IN_DOMAIN_BENCHMARK)
    slice_summary = {
        sl: {
            "queries": int(vals["count"]),
            "hit_rate_at_1": round(vals["hits_at_1"] / vals["count"], 4),
            f"hit_rate_at_{top_k}": round(vals["hits_at_k"] / vals["count"], 4),
            "mrr": round(vals["rr_sum"] / vals["count"], 4),
        }
        for sl, vals in slice_totals.items()
    }

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
            "ood_abstention_rate": round(ood_abstention_passed / len(OUT_OF_DOMAIN_TRAPS), 4),
            "citation_faithfulness_rate": round(faithfulness_hits / total, 4),
            "temporal_guardrail_rate": round(temporal_guard_hits / total, 4),
            "persona_voice_rate": round(persona_voice_hits / total, 4),
            "live_transcript_guardrail_rate": (
                round(transcript_turns_passed / transcript_total_turns, 4)
                if transcript_total_turns > 0 else 1.0
            ),
        },
        "slice_metrics": slice_summary,
        "details": per_query,
    }
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    report = run_evaluation()
    print("LKY-AI RAG Quantitative Evaluation Summary")
    print("-" * 52)
    print(f"Corpus Documents : {report['corpus_size']} | Vocabulary: {report['vocabulary_size']} terms")
    for k, v in report["metrics"].items():
        print(f"{k:32s}: {v:.2%}" if k != "mrr" else f"{k:32s}: {v:.4f}")
    print("\nBreakdown by Evaluation Slice:")
    for sl, s_metrics in report["slice_metrics"].items():
        print(f"  - {sl:24s}: Hit@1={s_metrics['hit_rate_at_1']:.0%}, Hit@3={s_metrics['hit_rate_at_3']:.0%}, MRR={s_metrics['mrr']:.4f} (n={s_metrics['queries']})")
