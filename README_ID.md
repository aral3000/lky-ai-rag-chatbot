# 🦁 "What Would Lee Kuan Yew Do?" (LKY-AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Store-013243.svg)](https://numpy.org/)
[![Pytest](https://img.shields.io/badge/Pengujian-Pytest%20Lolos-green.svg)](https://docs.pytest.org/)
[![Lisensi: MIT](https://img.shields.io/badge/Lisensi-MIT-yellow.svg)](LICENSE)
[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)

> 📖 **Bahasa**: **Bahasa Indonesia** | [English Version](README.md)

Aplikasi chatbot AI percakapan berbasis **Retrieval-Augmented Generation (RAG)** tingkat produksi yang berakar pada pidato, arsip kebijakan publik, dan memoar **Lee Kuan Yew**, Perdana Menteri Pendiri Singapura (1923–2015).

Awalnya dirancang untuk **99 Group Graduate AI Assessment (Bagian 3)**, sistem ini mendemonstrasikan keahlian rekayasa perangkat lunak AI *end-to-end*: penataan korpus data domain, pencarian kemiripan vektor (*vectorized similarity search*), verifikasi kutipan sumber (*citations*), serta perancangan prompt persona yang ketat—dengan penekanan khusus pada kebijakan perumahan publik **Housing & Development Board (HDB)** dan perencanaan perkotaan Singapura.

---

## 📌 Sorotan Rekayasa Teknis

1. **Vector Store Mandiri Tanpa Dependensi Berat (`rag_engine.py`)**:
   - Mengimplementasikan pencarian vektor **TF-IDF + Cosine Similarity** tervektorisasi menggunakan **NumPy**.
   - Tanpa dependensi database vektor eksternal; cepat, deterministik, dan dapat dijalankan di lingkungan mana pun.
2. **Korpus Khusus Perumahan & Tata Kota Singapura (`data/lky_corpus.json`)**:
   - Korpus historis terkurasi yang memuat skema **HDB Home Ownership for the People (1964)**, **Ethnic Integration Policy (EIP)**, **Kepadatan Urban & Reklamasi Lahan**, **Industrialisasi MNC**, **Meritokrasi**, serta **Adaptasi Teknologi**.
3. **Kutipan Sumber Otentik & Bukti Historis**:
   - Setiap respon yang dihasilkan langsung terhubung ke kutipan pidato asli, menampilkan judul dokumen, tahun penyampaian, dan skor relevansi kosinus matematika.
4. **Multi-Model & Mesin Persona Offline**:
   - Mendukung **Google Gemini (gemini-1.5-flash)**, **OpenAI (gpt-4o-mini)**, serta **Mesin Sintesis Persona Deterministik** yang dapat berjalan lancar tanpa memerlukan API key eksternal.
5. **Suite Pengujian Otomatis Pytest (`test_rag.py`)**:
   - 100% lolos uji otomatis untuk integritas korpus, akurasi pencarian topik HDB, urutan ranking kosinus, dan penyusunan prompt terverifikasi.

---

## 🚀 Panduan Memulai Cepat

### 1. Kloning Repositori
```bash
git clone https://github.com/aral3000/lky-ai-rag-chatbot.git
cd lky-ai-rag-chatbot
```

### 2. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Pengujian Otomatis
```bash
pytest test_rag.py -v
```

### 4. Jalankan Aplikasi Web Streamlit
```bash
streamlit run app.py
```
Buka `http://localhost:8501` di peramban Anda.

---

## 👨‍💻 Penulis & Pengembang

- **Penulis**: **M. Athallah Rafif Aldera, S.Kom.**
- **Pendidikan**: S1 Sistem Komputer, Fakultas Ilmu Komputer, **Universitas Sriwijaya (UNSRI)** (IPK: 3,77 / 4,00)
- **Portofolio**: [https://aral3000.github.io](https://aral3000.github.io)
- **GitHub**: [@aral3000](https://github.com/aral3000)
- **Target Asesmen**: 99 Group (PT. Web Marketing Indonesia) AI Engineer Program
- **Lisensi**: MIT License
