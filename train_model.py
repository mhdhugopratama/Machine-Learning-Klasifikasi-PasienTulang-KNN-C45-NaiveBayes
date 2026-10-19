from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_pasien_tulang.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = ["pelvic_incidence", "sacral_slope", "degree_spondylolisthesis"]
TARGET = "class"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Letakkan dataset_pasien_tulang.csv di folder yang sama dengan train_model.py."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df[required].dropna().copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = df.loc[valid_rows, TARGET]

    if y.nunique() != 3:
        raise ValueError(f"Target harus memiliki 3 kelas. Kelas ditemukan: {sorted(y.unique())}")
    if y.value_counts().min() < 2:
        raise ValueError("Setiap kelas minimal harus memiliki 2 data untuk pembagian train-test.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "naive_bayes": GaussianNB(),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    summary = {
        "dataset": DATA_PATH.name,
        "total_data_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "class_names": sorted(y.unique().tolist()),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        joblib.dump({
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "class_names": sorted(y.unique().tolist()),
            "model_name": name,
        }, MODEL_DIR / f"{name}.pkl")
        summary["models"][name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "classification_report": classification_report(
                y_test, predictions, labels=summary["class_names"], output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=summary["class_names"]
            ).tolist(),
        }
        print(f"{name}: akurasi test = {summary['models'][name]['accuracy']:.4f}")
        print(f"Model tersimpan: {MODEL_DIR / f'{name}.pkl'}")

    with open(MODEL_DIR / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nPelatihan selesai.")
    print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")
    print(f"Fitur yang digunakan ({len(FEATURES)}): {', '.join(FEATURES)}")


if __name__ == "__main__":
    main()
