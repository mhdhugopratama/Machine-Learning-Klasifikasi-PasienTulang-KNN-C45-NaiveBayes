from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import naive_bayes_text
from naive_bayes_text import CategoricalTextNaiveBayes

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_pasien_tulang.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = ["pelvic_incidence", "sacral_slope", "degree_spondylolisthesis"]
TARGET = "class"
RANDOM_STATE = 42
MODEL_OPTIONS = {
    "Naïve Bayes (Categorical / Laplace Smoothing)": "naive_bayes",
    "K-Nearest Neighbors (KNN)": "knn",
    "Decision Tree": "decision_tree",
}

st.set_page_config(page_title="Klasifikasi Tulang Belakang", page_icon="🦴", layout="wide")
st.title("🦴 Klasifikasi Kategori Tulang Belakang")
st.write("Aplikasi klasifikasi kategori pergeseran tulang belakang: **Rendah, Sedang, dan Tinggi**.")

st.info(
    "Aplikasi menggunakan 3 fitur biomekanik: **Sudut Panggul** (*pelvic_incidence*), **Kemiringan Sakrum** "
    "(*sacral_slope*), dan **Pergeseran Tulang** (*degree_spondylolisthesis*). Pada metode **Naïve Bayes**, "
    "data kontinu didiskritisasi menjadi data teks diskret dan dihitung menggunakan frekuensi peluang dengan "
    "koreksi **Laplace Smoothing** sesuai dokumen acuan tugas."
)

@st.cache_data
def _load_dataset_cached(mtime):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH.name}")
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing}")
    data = data[required].dropna().copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES)
    data[TARGET] = data[TARGET].astype(str).str.strip()
    return data

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH.name}")
    return _load_dataset_cached(DATA_PATH.stat().st_mtime)

@st.cache_resource
def _load_model_cached(model_key, mtime):
    path = MODEL_DIR / f"{model_key}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

def load_model(model_key):
    path = MODEL_DIR / f"{model_key}.pkl"
    if not path.exists():
        return None
    return _load_model_cached(model_key, path.stat().st_mtime)

try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    st.stop()

if not all((MODEL_DIR / f"{key}.pkl").exists() for key in MODEL_OPTIONS.values()):
    st.warning("File model belum lengkap. Jalankan `python train_model.py` terlebih dahulu di terminal.")
    st.stop()

class_counts = df[TARGET].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah data", f"{len(df)}")
col2.metric("Jumlah fitur", f"{len(FEATURES)}")
col3.metric("Jumlah kelas", f"{df[TARGET].nunique()}")

# Binning transformer instance
binner = CategoricalTextNaiveBayes()
df_binned = binner.bin_dataframe(df[FEATURES])
df_binned_with_target = pd.concat([df_binned, df[[TARGET]]], axis=1)

with st.expander("📊 Lihat Dataset Lengkap & Distribusi Kelas", expanded=True):
    tab_data1, tab_data2, tab_data3 = st.tabs([
        "Seluruh Dataset Numerik (310 Baris)",
        "Dataset Teks Diskret (Hasil Binning)",
        "Aturan Binning & Distribusi Kelas",
    ])
    with tab_data1:
        st.write(f"Menampilkan seluruh data asli numerik ({len(df)} baris) dengan 3 fitur dan label kelas:")
        st.dataframe(df, width="stretch", hide_index=False)
    with tab_data2:
        st.write(f"Menampilkan seluruh data setelah ditransformasikan ke teks kategori diskret ({len(df)} baris):")
        st.dataframe(df_binned_with_target, width="stretch", hide_index=False)
    with tab_data3:
        st.write("##### Aturan Rentang Pengelompokan (Binning Rules):")
        bin_rules_df = pd.DataFrame([
            {"Variabel Fitur": "Sudut Panggul (X1 / pelvic_incidence)", "Kategori Rendah": "< 55.00°", "Kategori Sedang": "55.00° s.d. 72.00°", "Kategori Tinggi": "> 72.00°"},
            {"Variabel Fitur": "Kemiringan Sakrum (X2 / sacral_slope)", "Kategori Rendah": "< 36.00°", "Kategori Sedang": "36.00° s.d. 46.00°", "Kategori Tinggi": "> 46.00°"},
            {"Variabel Fitur": "Pergeseran Tulang (X3 / degree_spondylolisthesis)", "Kategori Rendah": "≤ 4.00°", "Kategori Sedang": "4.01° s.d. 25.00°", "Kategori Tinggi": "> 25.00°"},
        ])
        st.dataframe(bin_rules_df, width="stretch", hide_index=True)

        st.write("##### Jumlah Data per Kelas:")
        summary_class = class_counts.rename_axis("Kelas").reset_index(name="Jumlah data")
        summary_class["Persentase"] = (summary_class["Jumlah data"] / len(df) * 100).round(2).astype(str) + "%"
        st.dataframe(summary_class, width="stretch", hide_index=True)

st.sidebar.header("Pengaturan model")
selected_label = st.sidebar.selectbox("Pilih metode klasifikasi", list(MODEL_OPTIONS.keys()))
model_key = MODEL_OPTIONS[selected_label]
model_bundle = load_model(model_key)
model = model_bundle["model"]

st.header("1. Evaluasi model")
X = df[FEATURES]
y = df[TARGET]
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi data uji", f"{accuracy * 100:.2f}%")
    m2.metric("Data latih", len(X_train))
    m3.metric("Data uji", len(X_test))

    left, right = st.columns([1, 1])
    labels = sorted(y.unique().tolist())
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Aktual: {x}" for x in labels], columns=[f"Prediksi: {x}" for x in labels])
        st.dataframe(cm_df, width="stretch")
        st.caption("Baris menunjukkan kelas aktual; kolom menunjukkan kelas prediksi.")
    with right:
        st.subheader("Classification report")
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, width="stretch")

    st.subheader("Data Partisi (Latih & Uji)")
    tab_train, tab_test = st.tabs([
        f"Data Latih (Training Set - {len(X_train)} baris)",
        f"Data Uji & Hasil Prediksi (Testing Set - {len(X_test)} baris)"
    ])
    with tab_train:
        st.write(f"Seluruh {len(X_train)} data latih (80%) yang digunakan untuk melatih model:")
        if model_key == "naive_bayes":
            train_binned = binner.bin_dataframe(X_train)
            train_df = pd.concat([X_train, train_binned.add_suffix(" (Teks)"), y_train], axis=1)
        else:
            train_df = pd.concat([X_train, y_train], axis=1)
        st.dataframe(train_df, width="stretch")

    with tab_test:
        st.write(f"Seluruh {len(X_test)} data uji (20%) beserta perbandingan kelas aktual vs prediksi {selected_label}:")
        if model_key == "naive_bayes":
            test_binned = binner.bin_dataframe(X_test)
            test_df = pd.DataFrame({
                "X1 (Angka)": X_test["pelvic_incidence"],
                "X1 (Teks)": test_binned["pelvic_incidence"],
                "X2 (Angka)": X_test["sacral_slope"],
                "X2 (Teks)": test_binned["sacral_slope"],
                "X3 (Angka)": X_test["degree_spondylolisthesis"],
                "X3 (Teks)": test_binned["degree_spondylolisthesis"],
                "Kelas Aktual": y_test,
                "Prediksi Model": y_pred,
                "Status Prediksi": [
                    "✅ Benar" if akt == pred else "❌ Salah"
                    for akt, pred in zip(y_test, y_pred)
                ]
            })
        else:
            test_df = X_test.copy()
            test_df["Kelas Aktual"] = y_test
            test_df["Prediksi Model"] = y_pred
            test_df["Status Prediksi"] = [
                "✅ Benar" if akt == pred else "❌ Salah"
                for akt, pred in zip(y_test, y_pred)
            ]
        st.dataframe(test_df, width="stretch")
except Exception as exc:
    st.error(f"Evaluasi model gagal: {exc}")

st.divider()
st.header("2. Prediksi data baru")
st.write("Masukkan nilai tiga fitur biomekanik untuk mendapatkan prediksi dari model yang dipilih.")

# Opsi untuk mengisi nilai contoh dari PDF
with st.expander("💡 Contoh Data Uji dari Dokumen PDF"):
    st.markdown(
        "Data uji pada PDF: **Sudut Panggul ($X_1$) = 52.00°**, **Kemiringan Sakrum ($X_2$) = 38.00°**, "
        "**Pergeseran Tulang ($X_3$) = 6.00°** (Hasil prediksi: **Sedang**)."
    )
    if st.button("Terapkan Contoh Nilai PDF ke Form"):
        st.session_state["p_input"] = 52.0
        st.session_state["s_input"] = 38.0
        st.session_state["d_input"] = 6.0
        st.rerun()

default_p = st.session_state.get("p_input", float(df["pelvic_incidence"].median()))
default_s = st.session_state.get("s_input", float(df["sacral_slope"].median()))
default_d = st.session_state.get("d_input", float(df["degree_spondylolisthesis"].median()))

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        pelvic_incidence = st.number_input(
            "Sudut Panggul (Pelvic incidence - X1)", min_value=-1000.0, max_value=1000.0,
            value=default_p, step=0.1, format="%.4f"
        )
    with c2:
        sacral_slope = st.number_input(
            "Kemiringan Sakrum (Sacral slope - X2)", min_value=-1000.0, max_value=1000.0,
            value=default_s, step=0.1, format="%.4f"
        )
    with c3:
        degree_spondylolisthesis = st.number_input(
            "Pergeseran Tulang (Degree spondylolisthesis - X3)", min_value=-1000.0, max_value=1000.0,
            value=default_d, step=0.1, format="%.4f"
        )
    submitted = st.form_submit_button("Prediksi kelas", type="primary", width="stretch")

if submitted:
    if model_key == "naive_bayes" and hasattr(model, "predict_single_details"):
        details = model.predict_single_details(pelvic_incidence, sacral_slope, degree_spondylolisthesis)
        best_cls = details["best_class"]
        st.success(f"Hasil Prediksi Akhir: **{best_cls.upper()}**")

        st.subheader("1. Konversi Nilai Angka ke Kategori Teks (Aturan Binning PDF)")
        b1, b2, b3 = st.columns(3)
        b1.info(f"**Sudut Panggul ($X_1$)**: `{pelvic_incidence:.2f}°`  \n⟹ **{details['binned_text']['pelvic_incidence']}**")
        b2.info(f"**Kemiringan Sakrum ($X_2$)**: `{sacral_slope:.2f}°`  \n⟹ **{details['binned_text']['sacral_slope']}**")
        b3.info(f"**Pergeseran Tulang ($X_3$)**: `{degree_spondylolisthesis:.2f}°`  \n⟹ **{details['binned_text']['degree_spondylolisthesis']}**")

        st.subheader("2. Tabel Rekapitulasi Peluang & Keputusan Akhir (Laplace Smoothing)")
        table_disp = details["table_details"].copy()
        table_disp["Prior P(Y)"] = table_disp["Prior P(Y)"].apply(lambda x: f"{x:.4f}")
        table_disp["P(X1|Y)"] = table_disp["P(X1|Y)"].apply(lambda x: f"{x:.4f}")
        table_disp["P(X2|Y)"] = table_disp["P(X2|Y)"].apply(lambda x: f"{x:.4f}")
        table_disp["P(X3|Y)"] = table_disp["P(X3|Y)"].apply(lambda x: f"{x:.4f}")
        table_disp["Nilai Posterior Akhir"] = table_disp["Nilai Posterior Akhir"].apply(lambda x: f"{x:.6f}")
        st.dataframe(table_disp, width="stretch", hide_index=True)

        max_post = max(details["posteriors"].values())
        st.markdown(
            f"**Keputusan Akhir:** Nilai posterior terbesar dimenangkan oleh kelas **{best_cls.upper()}** "
            f"({max_post:.6f})."
        )

        st.subheader("3. Probabilitas Normalisasi per Kelas")
        prob_df = pd.DataFrame({
            "Kelas": list(details["norm_probs"].keys()),
            "Probabilitas (%)": [round(v * 100, 2) for v in details["norm_probs"].values()]
        }).sort_values("Probabilitas (%)", ascending=False)
        st.dataframe(prob_df, width="stretch", hide_index=True)
    else:
        input_df = pd.DataFrame([[pelvic_incidence, sacral_slope, degree_spondylolisthesis]], columns=FEATURES)
        prediction = model.predict(input_df)[0]
        st.success(f"Hasil prediksi: **{prediction}**")
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame({"Kelas": model.classes_, "Probabilitas (%)": (probabilities * 100).round(2)})
                st.subheader("Probabilitas prediksi per kelas")
                st.dataframe(prob_df.sort_values("Probabilitas (%)", ascending=False), width="stretch", hide_index=True)
            except Exception:
                pass

    st.caption("Hasil ini adalah prediksi model berdasarkan fitur yang dimasukkan, bukan penilaian atau diagnosis medis.")

st.divider()
st.caption("Catatan: hasil evaluasi dihitung pada pembagian data uji 20% dengan random_state=42. Model harus dilatih ulang jika dataset berubah.")
