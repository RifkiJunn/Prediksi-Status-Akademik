# Prediksi Status Akademik dengan LLM

Aplikasi Streamlit untuk memprediksi status akademik mahasiswa (**Aman**, **Waspada**, atau **Berisiko**) menggunakan LLM melalui AI model service **Google Gemini** (`gemini-3.1-flash-lite`).

Proyek ini adalah Modul 2B mata kuliah *Kecerdasan Artifisial dan Penerapannya*. Konfigurasi prompt (`config.json`) berasal dari hasil evaluasi Modul 2A.

## Fitur
- Prediksi satu mahasiswa lewat form (IPK, kehadiran, jam belajar, tugas terkumpul, nilai UTS, keaktifan organisasi)
- Prediksi banyak data dari file CSV (maksimal 30 baris), lalu hasilnya bisa diunduh
- Sidebar berisi kinerja model pada data uji dan perbandingannya dengan DNN Modul 1A

## Hasil evaluasi (200 data uji yang sama dengan Modul 1A)

| Model | Accuracy | F1 macro |
|---|---|---|
| DNN (Modul 1A) | 0,84 | 0,8269 |
| LLM Gemini (Modul 2A) | 0,63 | 0,6286 |

Konfigurasi LLM ini **belum lolos** gerbang evaluasi (target F1 ≥ 0,70). Aplikasi di-deploy untuk keperluan praktikum dan perbandingan.

## Struktur
```
app.py               aplikasi Streamlit
classifier.py        fungsi klasifikasi (sama persis dengan saat evaluasi)
config.json          model, prompt, contoh few-shot, dan hasil evaluasi
requirements.txt
contoh_data_baru.csv contoh input CSV
.streamlit/config.toml  tema tampilan
```

## Menjalankan secara lokal
1. Install library:
   ```
   python -m pip install -r requirements.txt
   ```
2. Buat file `.streamlit/secrets.toml` (file ini **jangan** di-commit):
   ```toml
   GEMINI_API_KEY = "api_key_anda"
   ```
   API key bisa dibuat di [Google AI Studio](https://aistudio.google.com).
3. Jalankan aplikasi:
   ```
   python -m streamlit run app.py
   ```
   Lalu buka http://localhost:8501.

## Deploy ke Streamlit Community Cloud
1. Push repository ini ke GitHub.
2. Di [share.streamlit.io](https://share.streamlit.io), klik **Create app**, lalu pilih repository ini dengan main file `app.py`.
3. Di **Advanced settings → Secrets**, isi `GEMINI_API_KEY = "api_key_anda"`, lalu klik **Deploy**.

## Catatan
- Setiap prediksi memanggil Gemini API, jadi aplikasi butuh internet dan dibatasi kuota free tier. Data yang sama di-cache supaya tidak memanggil API dua kali.
- Jangan mengubah prompt langsung di aplikasi. Setiap perubahan `config.json` harus dievaluasi ulang di Modul 2A.
