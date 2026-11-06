# "What Would Lee Kuan Yew Do?" (LKY-AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Store-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Pengujian-13%20Lolos-green.svg)](test_rag.py)
[![Lisensi: MIT](https://img.shields.io/badge/Lisensi-MIT-yellow.svg)](LICENSE)

> **Bahasa**: **Bahasa Indonesia** | [English Version](README.md)

Aplikasi chatbot AI percakapan berbasis **Retrieval-Augmented Generation (RAG)** yang berakar pada pidato, arsip kebijakan publik, dan memoar kehidupan pribadi **Lee Kuan Yew**, Perdana Menteri Pendiri Singapura (1923-2015).

Dirancang untuk **99 Group AI Aptitude Challenge (Bagian 3)**, sistem ini mencakup penataan 16 dokumen korpus historis & memoar kehidupan (`1964-2013`), pencarian kemiripan vektor **TF-IDF + Stemming + OOV-Penalized Cosine Similarity** murni dengan NumPy, verifikasi kutipan sumber, evaluasi kuantitatif multi-slice RAG + Guardrail Persona (`eval_rag.py`), ekspor **Flowise v2 Chatflow (`flowise_chatflow.json`)**, serta rekayasa prompt persona pada topik kehidupan, keluarga, disiplin diri, **Housing & Development Board (HDB)**, CPF, tata kota, ekonomi, dan geopolitik.

---

## Sorotan Rekayasa Teknis

1. **Vector Store Mandiri Tanpa Database Eksternal (`rag_engine.py`)**:
   - Mengimplementasikan tokenisasi *stopword*, *Porter-lite stemming*, ekspansi akronim/sinonim domain (`AI`, `HDB`, `CPF`, `EIP`, `CPIB`), serta pencarian **TF-IDF + OOV-Penalized Cosine Similarity** menggunakan **NumPy**.
2. **16 Dokumen Korpus Kebijakan & Memoar Kehidupan (`data/lky_corpus.json`)**:
   - **Kebijakan, Perumahan & Geopolitik (12 dokumen)**: **HDB Home Ownership (1964)**, **Kepadatan Urban & Reklamasi Lahan (1978)**, **Ethnic Integration Policy (1989)**, **Industrialisasi MNC (1970)**, **Pemberantasan Korupsi CPIB (1986)**, **CPF vs Negara Kesejahteraan (1972)**, **Pendidikan Bilingual (1966)**, **Adaptasi Otomasi & AI (1999)**, **Meritokrasi Kepemimpinan (1971)**, **Kemandirian Air NEWater (2002)**, **Pertahanan SAF (2005)**, dan **Keseimbangan AS-Tiongkok di ASEAN (2009)**.
   - **Kehidupan Pribadi, Keluarga & Filosofi (4 dokumen)**: **Pernikahan & Kemitraan Setara dengan Kwa Geok Choo (2010)**, **Disiplin Fisik Harian, Tidur & Stamina Mental (2013)**, **Realisme Stoik menghadapi Duka, Penuaan & Mortalitas (2013)**, serta **Nasihat bagi Generasi Muda tentang Ambisi, Integritas & Ketangguhan (1998)**.
3. **Evaluasi Kuantitatif Multi-Slice RAG & Guardrail Persona (`eval_rag.py`)**:
   - Mengukur **Hit Rate@1 (`100%`)**, **Hit Rate@3 (`100%`)**, **MRR (`1.0000`)** di 3 slice evaluasi (20 kueri *in-domain*), **Out-of-Domain Rejection Rate (`100%`)**, **Citation Faithfulness (`100%`)**, **Temporal Guardrail <= 2015 (`100%`)**, dan **Audit Transkrip 4-Turn (`100%`)**.
4. **Multi-Provider, Flowise Chatflow & Mesin Persona Offline (`llm_client.py` + `flowise_chatflow.json`)**:
   - Mendukung **Google Gemini (`gemini-3-flash-preview` + `gemini-3.5-flash-lite` fallback)**, **OpenAI (`gpt-4o-mini`)**, failover multi-provider via variabel lingkungan `LLM_VAULT_DIR`, mode offline deterministik tanpa API key, serta ekspor **Flowise v2 Chatflow JSON (`flowise_chatflow.json`)**.

---

## Panduan Memulai Cepat

```bash
git clone https://github.com/aral3000/lky-ai-rag-chatbot.git
cd lky-ai-rag-chatbot
pip install -r requirements.txt

# Jalankan 13 unit & RAG evaluation test
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
