# "What Would Lee Kuan Yew Do?" (LKY-AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Store-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-9%20Passed-green.svg)](test_rag.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Language**: **English** | [Versi Bahasa Indonesia](README_ID.md)

An interactive **Retrieval-Augmented Generation (RAG)** conversational AI chatbot grounded in the speeches, public policy records, and memoirs of **Lee Kuan Yew**, Founding Prime Minister of Singapore (1923-2015).

Built for the **99 Group AI Aptitude Challenge (Section 3)**, this engine implements domain corpus curation, vectorized TF-IDF + Cosine Similarity search in pure NumPy, source citation verification, quantitative RAG evaluation (`eval_rag.py`), and strict persona prompt engineering, with dedicated coverage of Singapore's **Housing & Development Board (HDB)** home ownership and urban planning policies.

---

## Key Engineering Highlights

1. **Zero-Database Vector Store (`rag_engine.py`)**:
   - Implements smoothed **TF-IDF + L2-normalized Cosine Similarity** vector search using pure **NumPy**.
   - Deterministic, sub-millisecond retrieval with zero external vector database dependencies.
2. **12-Document Historical Policy Corpus (`data/lky_corpus.json`)**:
   - Covers **HDB Home Ownership Scheme (1964)**, **Urban Density & Land Reclamation (1978)**, **Ethnic Integration Policy (1989)**, **MNC Industrialization (1970)**, **CPIB Anti-Corruption (1986)**, **CPF vs Welfare Entitlements (1972)**, **Bilingual Education (1966)**, **Lifelong Learning & Automation (1999)**, **Ruthless Meritocracy (1971)**, **NEWater Strategic Autonomy (2002)**, **SAF Poison Shrimp Deterrence (2005)**, and **US-China Balance in ASEAN (2009)**.
3. **Quantitative RAG Evaluation Suite (`eval_rag.py`)**:
   - Benchmarks **Hit Rate@1**, **Hit Rate@3**, **Mean Reciprocal Rank (MRR)**, **Out-of-Domain Rejection Rate**, and **Citation Faithfulness** across 12 in-domain policy queries and 3 adversarial out-of-domain trap queries.
4. **Multi-Provider & Deterministic Offline Fallback (`llm_client.py`)**:
   - Supports **Google Gemini (`gemini-1.5-flash`)**, **OpenAI (`gpt-4o-mini`)**, optional environment-configured multi-provider failover (`LLM_VAULT_DIR`), and an **Offline Grounded Persona Engine** that runs without API keys.
5. **Automated Test Suite (`test_rag.py`)**:
   - 9 passing automated unit and evaluation tests runnable with Python's built-in `unittest` or `pytest`.

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

# Run 9 unit + RAG evaluation tests (stdlib unittest or pytest)
python -m unittest test_rag.py -v

# Run quantitative RAG benchmark report (writes eval_results.json)
python eval_rag.py

# Launch interactive Streamlit app
streamlit run app.py
```

---

## Quantitative RAG Evaluation Results (`eval_rag.py`)

Evaluated over **12 historical policy queries** and **3 adversarial out-of-domain queries** (`ood_min_score = 0.12`) across a 12-document, 445-term indexed corpus:

| Metric | Score | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Hit Rate@1 (Top-1 Accuracy)** | `100.00%` (12/12) | `>= 85.0%` | `PASS` |
| **Hit Rate@3 (Top-3 Recall)** | `100.00%` (12/12) | `>= 95.0%` | `PASS` |
| **Mean Reciprocal Rank (MRR)** | `1.0000` | `>= 0.9000` | `PASS` |
| **Out-of-Domain Rejection Rate** | `100.00%` (3/3) | `100.0%` | `PASS` |
| **Citation Faithfulness Rate** | `100.00%` (12/12) | `>= 90.0%` | `PASS` |

---

## Author

- **Author**: M. Athallah Rafif Aldera, S.Kom.
- **Education**: B.S. in Computer Systems, Universitas Sriwijaya (GPA: 3.77 / 4.00)
- **Portfolio**: [https://aral3000.github.io](https://aral3000.github.io)
- **GitHub**: [@aral3000](https://github.com/aral3000)
- **License**: MIT
