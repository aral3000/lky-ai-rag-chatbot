# "What Would Lee Kuan Yew Do?" (LKY-AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Store-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Pengujian-9%20Lolos-green.svg)](test_rag.py)
[![Lisensi: MIT](https://img.shields.io/badge/Lisensi-MIT-yellow.svg)](LICENSE)

> **Bahasa**: **Bahasa Indonesia** | [English Version](README.md)

Aplikasi chatbot AI percakapan berbasis **Retrieval-Augmented Generation (RAG)** yang berakar pada pidato, arsip kebijakan publik, dan memoar **Lee Kuan Yew**, Perdana Menteri Pendiri Singapura (1923-2015).

Dirancang untuk **99 Group AI Aptitude Challenge (Bagian 3)**, sistem ini mencakup penataan 12 dokumen korpus historis, pencarian kemiripan vektor **TF-IDF + Stemming + Cosine Similarity** murni dengan NumPy, verifikasi kutipan sumber, evaluasi kuantitatif RAG (`eval_rag.py`), serta rekayasa prompt persona pada topik **Housing & Development Board (HDB)**, CPF, tata kota, ekonomi, dan geopolitik.

---

## Sorotan Rekayasa Teknis

1. **Vector Store Mandiri Tanpa Database Eksternal (`rag_engine.py`)**:
   - Mengimplementasikan tokenisasi *stopword*, *Porter-lite stemming*, ekspansi akronim domain (`AI`, `HDB`, `CPF`, `EIP`, `CPIB`), serta pencarian **TF-IDF + Cosine Similarity** menggunakan **NumPy**.
2. **12 Dokumen Korpus Kebijakan Singapura (`data/lky_corpus.json`)**:
   - Mencakup skema **HDB Home Ownership (1964)**, **Kepadatan Urban & Reklamasi Lahan (1978)**, **Ethnic Integration Policy (1989)**, **Industrialisasi MNC (1970)**, **Pemberantasan Korupsi CPIB (1986)**, **CPF vs Negara Kesejahteraan (1972)**, **Pendidikan Bilingual (1966)**, **Adaptasi Otomasi & AI (1999)**, **Meritokrasi Kepemimpinan (1971)**, **Kemandirian Air NEWater (2002)**, **Pertahanan SAF (2005)**, dan **Keseimbangan AS-Tiongkok di ASEAN (2009)**.
3. **Evaluasi Kuantitatif RAG (`eval_rag.py`)**:
   - Mengukur **Hit Rate@1 (`100%`)**, **Hit Rate@3 (`100%`)**, **MRR (`1.0000`)**, **Out-of-Domain Rejection Rate (`100%`)**, dan **Citation Faithfulness (`100%`)**.
4. **Multi-Provider & Mesin Persona Offline (`llm_client.py`)**:
   - Mendukung **Google Gemini (`gemini-3-flash-preview` + `gemini-3.5-flash-lite` fallback)**, **OpenAI (`gpt-4o-mini`)**, failover multi-provider via variabel lingkungan `LLM_VAULT_DIR`, serta mode offline deterministik tanpa API key.

---

## Panduan Memulai Cepat

```bash
git clone https://github.com/aral3000/lky-ai-rag-chatbot.git
cd lky-ai-rag-chatbot
pip install -r requirements.txt

# Jalankan 9 unit & RAG evaluation test
python -m unittest test_rag.py -v

# Jalankan laporan benchmark evaluasi RAG
python eval_rag.py

# Jalankan aplikasi Streamlit
streamlit run app.py
```

---

## Penulis

- **Penulis**: M. Athallah Rafif Aldera, S.Kom.
- **Pendidikan**: S1 Sistem Komputer, Universitas Sriwijaya (IPK: 3,77 / 4,00)
- **Portofolio**: [https://aral3000.github.io](https://aral3000.github.io)
- **GitHub**: [@aral3000](https://github.com/aral3000)
- **Lisensi**: MIT
