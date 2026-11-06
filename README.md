# "What Would Lee Kuan Yew Do?" (LKY-AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Store-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-13%20Passed-green.svg)](test_rag.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Language**: **English** | [Versi Bahasa Indonesia](README_ID.md)

An interactive **Retrieval-Augmented Generation (RAG)** conversational AI chatbot grounded in the speeches, public policy records, and personal life memoirs of **Lee Kuan Yew**, Founding Prime Minister of Singapore (1923-2015).

Built for the **99 Group AI Aptitude Challenge (Section 3)**, this engine implements domain corpus curation across **life, geopolitics, history, and public policy**, vectorized TF-IDF + OOV-penalized Cosine Similarity search in pure NumPy, source citation verification, multi-slice RAG + Generation Faithfulness evaluation (`eval_rag.py`), an importable **Flowise v2 Chatflow JSON (`flowise_chatflow.json`)**, and strict persona prompt engineering.

---

## Key Engineering Highlights

1. **Zero-Database Vector Store (`rag_engine.py`)**:
   - Implements smoothed **TF-IDF + OOV-Penalized L2-Normalized Cosine Similarity** vector search using pure **NumPy**, with morphological stemming (`_porter_lite_stem`) and domain acronym/synonym expansion (`ACRONYM_EXPANSIONS`).
   - Deterministic, sub-millisecond retrieval with zero external vector database dependencies.
2. **16-Document Historical & Personal Memoir Corpus (`data/lky_corpus.json`)**:
   - **Policy, Housing & Geopolitics (12 docs)**: **HDB Home Ownership Scheme (1964)**, **Urban Density & Land Reclamation (1978)**, **Ethnic Integration Policy (1989)**, **MNC Industrialization (1970)**, **CPIB Anti-Corruption (1986)**, **CPF vs Welfare Entitlements (1972)**, **Bilingual Education (1966)**, **Lifelong Learning & Automation (1999)**, **Ruthless Meritocracy (1971)**, **NEWater Strategic Autonomy (2002)**, **SAF Poison Shrimp Deterrence (2005)**, and **US-China Balance in ASEAN (2009)**.
   - **Personal Life, Family & Philosophy (4 docs)**: **Marriage & Equal Partnership with Kwa Geok Choo (2010)**, **Daily Physical Regimen, Sleep & Mental Stamina (2013)**, **Stoic Realism on Grief, Aging & Mortality (2013)**, and **Advice to the Young on Ambition, Integrity & Resilience (1998)**.
3. **Multi-Slice RAG + Generation Guardrail Evaluation Suite (`eval_rag.py`)**:
   - Benchmarks **Hit Rate@1**, **Hit Rate@3**, **Mean Reciprocal Rank (MRR)** across **3 evaluation slices** (`policy_geopolitics`, `life_philosophy`, `paraphrased_low_overlap` — 20 in-domain queries) + **4 Out-of-Domain Traps**, plus automated generation guardrails (**Citation Grounding**, **Temporal Anachronism Guard <= 2015**, **Persona Voice Guard**, and **4-Turn Live Transcript Audit**).
4. **Dual-Track Code + No-Code (`llm_client.py` + `flowise_chatflow.json`)**:
   - Supports **Google Gemini (`gemini-3-flash-preview` + `gemini-3.5-flash-lite` fallback)**, **OpenAI (`gpt-4o-mini`)**, optional environment-configured multi-provider failover (`LLM_VAULT_DIR`), an **Offline Grounded Persona Engine**, and an importable **Flowise v2 Chatflow (`flowise_chatflow.json`)**.
5. **Automated Test Suite (`test_rag.py`)**:
   - 13 passing automated unit and evaluation tests runnable with Python's built-in `unittest` or `pytest`.

---

## System Architecture

```text
+-----------------------+
| User Query / Multi-   |  e.g. "How did Singapore achieve 90% home ownership?"
| Turn Chat Input       |
+-----------------------+
            |
            v
+-----------------------+
| Stopword-Filtered     |  --> Smoothed IDF: log((N + 1) / (df + 1)) + 1
| TF-IDF Tokenizer      |  --> L2-Normalized Query Vector q_vec
+-----------------------+
            |
            v
+-----------------------+
| NumPy Vector Store    |  --> Cosine Similarity: dot(doc_vectors, q_vec)
| (Top-k + Threshold)   |  --> Rejects off-topic noise below min_score
+-----------------------+
            |
            v
+-----------------------+
| Grounded Citation     |  --> [Source #1]: Home Ownership Scheme (1964) | Score: 0.38
| Context Builder       |      Citation: Speech at NTUC, 1964
+-----------------------+
            |
            v
+-----------------------+
| LLM / Offline Persona |  --> Online: Gemini / OpenAI / Groq
| Dispatcher            |  --> Offline: Deterministic Grounded Synthesis
+-----------------------+
            |
            v
+-----------------------+
| Streamlit Chat UI &   |  --> Pragmatic Lee Kuan Yew response
| Evaluation Suite      |      + Expandable Source Excerpt Accordion
+-----------------------+
```

---

## Quick Start

```bash
git clone https://github.com/aral3000/lky-ai-rag-chatbot.git
cd lky-ai-rag-chatbot
pip install -r requirements.txt

# Run 13 unit + RAG evaluation tests (stdlib unittest or pytest)
python -m unittest test_rag.py -v

# Run quantitative RAG benchmark report (writes eval_results.json)
python eval_rag.py

# Launch interactive Streamlit app
streamlit run app.py
```

---

## Quantitative RAG & Persona Evaluation Results (`eval_rag.py`)

Evaluated over **20 in-domain queries across 3 slices** (`policy_geopolitics`, `life_philosophy`, `paraphrased_low_overlap`) and **4 adversarial out-of-domain queries** (`ood_min_score = 0.12`) across a 16-document, 551-term indexed corpus:

| Metric | Score | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Hit Rate@1 (Top-1 Accuracy)** | `100.00%` (20/20) | `>= 85.0%` | `PASS` |
| **Hit Rate@3 (Top-3 Recall)** | `100.00%` (20/20) | `>= 95.0%` | `PASS` |
| **Mean Reciprocal Rank (MRR)** | `1.0000` | `>= 0.9000` | `PASS` |
| **Out-of-Domain Rejection Rate** | `100.00%` (4/4) | `100.0%` | `PASS` |
| **Citation Faithfulness Rate** | `100.00%` (20/20) | `>= 90.0%` | `PASS` |
| **Temporal Guardrail Rate (<= 2015)** | `100.00%` (20/20) | `100.0%` | `PASS` |
| **Persona Voice & Formatting Rate** | `100.00%` (20/20) | `100.0%` | `PASS` |
| **4-Turn Live Transcript Audit** | `100.00%` (4/4) | `100.0%` | `PASS` |

---

## Author

- **Author**: M. Athallah Rafif Aldera, S.Kom.
- **Education**: B.S. in Computer Systems, Universitas Sriwijaya (GPA: 3.77 / 4.00)
- **Portfolio**: [https://aral3000.github.io](https://aral3000.github.io)
- **GitHub**: [@aral3000](https://github.com/aral3000)
- **License**: MIT
