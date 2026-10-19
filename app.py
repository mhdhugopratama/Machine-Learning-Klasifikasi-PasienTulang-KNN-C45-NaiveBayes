from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_pasien_tulang.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = ["pelvic_incidence", "sacral_slope", "degree_spondylolisthesis"]
TARGET = "class"
RANDOM_STATE = 42
MODEL_OPTIONS = {
    "K-Nearest Neighbors (KNN)": "knn",
    "Decision Tree": "decision_tree",
    "Naive Bayes (GaussianNB)": "naive_bayes",
}

st.set_page_config(page_title="Klasifikasi Tulang Belakang", page_icon="🦴", layout="wide")
st.title("🦴 Klasifikasi Kategori Tulang Belakang")
st.write("Aplikasi klasifikasi tiga kategori: **Rendah, Sedang, dan Tinggi**.")
st.info(
    "Model menggunakan 3 fitur biomekanik: pelvic_incidence, sacral_slope, dan degree_spondylolisthesis "
    "untuk memprediksi kategori class (Rendah, Sedang, Tinggi)."
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

with st.expander("📊 Lihat Dataset Lengkap & Distribusi Kelas", expanded=True):
    tab_data1, tab_data2 = st.tabs(["Seluruh Dataset (310 Baris)", "Distribusi Kelas"])
    with tab_data1:
        st.write(f"Menampilkan seluruh data ({len(df)} baris) dengan 3 fitur dan label kelas:")
        st.dataframe(df, width="stretch", hide_index=False)
    with tab_data2:
        st.write("Jumlah data per kelas:")
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
        train_df = pd.concat([X_train, y_train], axis=1)
        st.dataframe(train_df, width="stretch")
    with tab_test:
        st.write(f"Seluruh {len(X_test)} data uji (20%) beserta perbandingan kelas aktual vs prediksi {selected_label}:")
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
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        pelvic_incidence = st.number_input(
            "Pelvic incidence", min_value=-1000.0, max_value=1000.0,
            value=float(df["pelvic_incidence"].median()), step=0.1, format="%.4f"
        )
    with c2:
        sacral_slope = st.number_input(
            "Sacral slope", min_value=-1000.0, max_value=1000.0,
            value=float(df["sacral_slope"].median()), step=0.1, format="%.4f"
        )
    with c3:
        degree_spondylolisthesis = st.number_input(
            "Degree spondylolisthesis", min_value=-1000.0, max_value=1000.0,
            value=float(df["degree_spondylolisthesis"].median()), step=0.1, format="%.4f"
        )
    submitted = st.form_submit_button("Prediksi kelas", type="primary", width="stretch")

if submitted:
    input_df = pd.DataFrame([[pelvic_incidence, sacral_slope, degree_spondylolisthesis]], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil prediksi: **{prediction}**")
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({"Kelas": model.classes_, "Probabilitas": probabilities})
            prob_df["Probabilitas"] = (prob_df["Probabilitas"] * 100).round(2)
            st.subheader("Probabilitas prediksi per kelas")
            st.dataframe(prob_df.sort_values("Probabilitas", ascending=False), width="stretch", hide_index=True)
        except Exception:
            pass
    st.caption("Hasil ini adalah prediksi model berdasarkan fitur yang dimasukkan, bukan penilaian atau diagnosis medis.")

st.divider()
st.caption("Catatan: hasil evaluasi dihitung pada pembagian data uji 20% dengan random_state=42. Model harus dilatih ulang jika dataset berubah.")
