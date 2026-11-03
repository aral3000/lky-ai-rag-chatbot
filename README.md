# 🦁 "What Would Lee Kuan Yew Do?" (LKY-AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Store-013243.svg)](https://numpy.org/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest%20Passed-green.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Bahasa Indonesia](https://img.shields.io/badge/Language-Bahasa%20Indonesia-green.svg)](README_ID.md)

> 📖 **Language**: **English** | [Versi Bahasa Indonesia](README_ID.md)

An interactive, production-grade **Retrieval-Augmented Generation (RAG)** conversational AI chatbot grounded in the speeches, public records, and policy memoirs of **Lee Kuan Yew**, Founding Prime Minister of Singapore (1923–2015).

Originally developed for the **99 Group Graduate AI Assessment (Section 3)**, this engine demonstrates end-to-end AI software engineering: domain knowledge structuring, vectorized similarity search, citation verification, and strict persona prompt engineering—with specialized focus on Singapore's landmark **Housing & Development Board (HDB)** public home ownership and urban planning frameworks.

---

## 📌 Key Engineering Highlights

1. **Zero-Dependency Vector Store (`rag_engine.py`)**:
   - Implements high-performance **TF-IDF + Cosine Similarity** vector search using pure **NumPy**.
   - No external vector database network dependencies required; fast, deterministic, and fully reproducible in any deployment environment.
2. **Specialized Singapore Housing & Urban Corpus (`data/lky_corpus.json`)**:
   - Curated historical corpus covering **HDB Home Ownership for the People Scheme (1964)**, **Ethnic Integration Policy (EIP)**, **Urban Density & Land Reclamation**, **MNC Industrialization**, **Meritocracy**, and **Technological Adaptation**.
3. **Exact Source Citation & Evidence Grounding**:
   - Every generated response is linked to exact historical source speeches, displaying the excerpt title, delivery year, and mathematical cosine relevance score.
4. **Multi-Provider & Deterministic Offline Fallback**:
   - Plug-and-play support for **Google Gemini (gemini-1.5-flash)**, **OpenAI (gpt-4o-mini)**, and a **Grounded Offline Persona Synthesis Engine** that operates seamlessly without external API keys.
5. **Automated Pytest Verification Suite (`test_rag.py`)**:
   - 100% passing automated test suite verifying corpus integrity, HDB query routing, descending cosine ranking, and context synthesis.

---

## 🏗️ System Architecture

```text
+---------------------+
|   User Query /      | (e.g. "Why did Singapore prioritize HDB public home ownership?")
|   Preset Prompt     |
+---------------------+
           |
           v
+---------------------+
| Tokenizer & TF-IDF  | --> Normalized Query Vector q_vec
| Vocabulary Engine   |
+---------------------+
           |
           v
+---------------------+
| NumPy Vector Store  | --> Cosine Similarity: cos(θ) = (u · v) / (||u|| * ||v||)
| (Top-k Retrieval)   | --> Filtered by Relevance Threshold (default: 0.05)
+---------------------+
           |
           v
+---------------------+
| Grounded Context    | --> [Source #1]: HDB Home Ownership Scheme (1964) (Score: 0.42)
| Formatter           |     Citation: Speech at NTUC, 1964
+---------------------+
           |
           v
+---------------------+
| Multi-Model Router  | --> [Online]: Gemini 1.5 Flash / GPT-4o-mini
| & Persona Engine    | --> [Offline]: Deterministic Strategic Synthesis
+---------------------+
           |
           v
+---------------------+
| Streamlit UI Output | --> Authoritative Lee Kuan Yew response
| with Live Citations |     + Expandable Source Accordion with Real Excerpts
+---------------------+
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/aral3000/lky-ai-rag-chatbot.git
cd lky-ai-rag-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Pytest Suite
```bash
pytest test_rag.py -v
```
*Output: 6 passing unit tests verifying vector math, retrieval precision, and citation integrity.*

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Verification & Retrieval Test Results

| Test Case | Target Query | Top Retrieved Source | Relevance Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Test 1** | *"Solving housing shortage and home ownership"* | `Home Ownership for the People Scheme (1964)` | `0.38` | `PASSED` |
| **Test 2** | *"Technological adaptation & automation"* | `Lifelong Education & Knowledge Economy (1999)` | `0.31` | `PASSED` |
| **Test 3** | *"Eliminating corruption & rule of law"* | `Zero Tolerance for Corruption & CPIB (1986)` | `0.41` | `PASSED` |
| **Test 4** | *"Leadership recruitment & scholarships"* | `Ruthless Meritocracy in Leadership (1971)` | `0.35` | `PASSED` |

---

## 👨‍💻 Author & Attribution

- **Author**: **M. Athallah Rafif Aldera, S.Kom.**
- **Degree**: B.S. in Computer Systems, Faculty of Computer Science, **Universitas Sriwijaya (UNSRI)** (GPA: 3.77 / 4.00)
- **Portfolio**: [https://aral3000.github.io](https://aral3000.github.io)
- **GitHub**: [@aral3000](https://github.com/aral3000)
- **Assessment Target**: 99 Group (PT. Web Marketing Indonesia) AI Engineer Program
- **License**: MIT License
