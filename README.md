# Aplikasi Streamlit Klasifikasi Tulang Belakang

## Isi folder
- `app.py`: antarmuka Streamlit, evaluasi, dan prediksi data baru.
- `train_model.py`: melatih KNN, Decision Tree, dan Gaussian Naive Bayes serta menyimpan model `.pkl`.
- `dataset_pasien_tulang.csv`: dataset 3 kelas.
- `requirements.txt`: pustaka Python yang dibutuhkan.
- `models/`: akan dibuat otomatis setelah pelatihan, berisi file model `.pkl` dan `evaluation.json`.

## Menjalankan di Windows
Buka terminal/Command Prompt di folder proyek, lalu jalankan:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Jika perintah `python` tidak dikenali, coba gunakan `py` sebagai penggantinya.

## Fitur yang digunakan
Model menggunakan 3 fitur biomekanik:
- `pelvic_incidence`
- `sacral_slope`
- `degree_spondylolisthesis`

Target klasifikasi adalah kolom `class` yang terdiri dari 3 kategori: **Rendah**, **Sedang**, dan **Tinggi**.
Kategori ini merupakan pengelompokan statistik, bukan diagnosis medis.