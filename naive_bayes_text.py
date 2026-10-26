from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class CategoricalTextNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Categorical Naive Bayes dengan diskritisasi data kontinu ke teks kategori
    (Rendah, Sedang, Tinggi) dan koreksi Laplace Smoothing (alpha=1, K=3),
    sesuai dokumen tugas klasifikasi pergeseran tulang belakang.
    """
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.categories = ["Rendah", "Sedang", "Tinggi"]
        self.features = ["pelvic_incidence", "sacral_slope", "degree_spondylolisthesis"]

    @staticmethod
    def bin_pelvic_incidence(val):
        """
        Sudut Panggul (X1):
        - < 55.00: Rendah
        - 55.00 s.d. 72.00: Sedang
        - > 72.00: Tinggi
        """
        v = float(val)
        if v < 55.0:
            return "Rendah"
        elif v <= 72.0:
            return "Sedang"
        else:
            return "Tinggi"

    @staticmethod
    def bin_sacral_slope(val):
        """
        Kemiringan Sakrum (X2):
        - < 36.00: Rendah
        - 36.00 s.d. 46.00: Sedang
        - > 46.00: Tinggi
        """
        v = float(val)
        if v < 36.0:
            return "Rendah"
        elif v <= 46.0:
            return "Sedang"
        else:
            return "Tinggi"

    @staticmethod
    def bin_degree_spondylolisthesis(val):
        """
        Pergeseran Tulang (X3):
        - <= 4.00: Rendah
        - 4.01 s.d. 25.00: Sedang
        - > 25.00: Tinggi
        """
        v = float(val)
        if v <= 4.0:
            return "Rendah"
        elif v <= 25.0:
            return "Sedang"
        else:
            return "Tinggi"

    def bin_dataframe(self, X):
        """
        Mengonversi nilai numerik pada setiap fitur menjadi teks kategori diskret.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.features)

        first_val = df.iloc[0, 0]
        if isinstance(first_val, str) and first_val in self.categories:
            return df

        binned = pd.DataFrame(index=df.index)
        col_names = list(df.columns)

        if "pelvic_incidence" in col_names:
            binned["pelvic_incidence"] = df["pelvic_incidence"].apply(self.bin_pelvic_incidence)
        else:
            binned["pelvic_incidence"] = df.iloc[:, 0].apply(self.bin_pelvic_incidence)

        if "sacral_slope" in col_names:
            binned["sacral_slope"] = df["sacral_slope"].apply(self.bin_sacral_slope)
        else:
            binned["sacral_slope"] = df.iloc[:, 1].apply(self.bin_sacral_slope)

        if "degree_spondylolisthesis" in col_names:
            binned["degree_spondylolisthesis"] = df["degree_spondylolisthesis"].apply(self.bin_degree_spondylolisthesis)
        else:
            binned["degree_spondylolisthesis"] = df.iloc[:, 2].apply(self.bin_degree_spondylolisthesis)

        return binned

    def fit(self, X, y):
        X_binned = self.bin_dataframe(X)
        y_series = pd.Series(y).astype(str).str.strip()

        self.classes_ = np.array(sorted(y_series.unique().tolist()))
        self.class_counts_ = {c: int((y_series == c).sum()) for c in self.classes_}
        self.total_samples_ = int(len(y_series))

        # Peluang Awal (Prior) P(Y)
        self.prior_ = {c: self.class_counts_[c] / self.total_samples_ for c in self.classes_}

        # Peluang Bersyarat (Likelihood) dengan Laplace Smoothing (alpha=1, K=3)
        # P(Xi = k | Y = c) = (count(Xi=k, Y=c) + alpha) / (count(Y=c) + K * alpha)
        self.likelihood_ = {}
        for col in self.features:
            self.likelihood_[col] = {}
            for c in self.classes_:
                sub = X_binned[y_series == c][col]
                count_c = len(sub)
                denom = count_c + len(self.categories) * self.alpha
                self.likelihood_[col][c] = {}
                for cat in self.categories:
                    count_cat = int((sub == cat).sum())
                    self.likelihood_[col][c][cat] = (count_cat + self.alpha) / denom

        return self

    def predict_single_details(self, x1_num, x2_num, x3_num):
        """
        Menghasilkan rincian perhitungan manual langkah demi langkah
        sebagaimana disajikan dalam laporan PDF tugas.
        """
        cat1 = self.bin_pelvic_incidence(x1_num)
        cat2 = self.bin_sacral_slope(x2_num)
        cat3 = self.bin_degree_spondylolisthesis(x3_num)
        binned_text = {
            "pelvic_incidence": cat1,
            "sacral_slope": cat2,
            "degree_spondylolisthesis": cat3
        }

        details = []
        posteriors = {}
        for c in self.classes_:
            p_prior = self.prior_[c]
            p_x1 = self.likelihood_["pelvic_incidence"][c][cat1]
            p_x2 = self.likelihood_["sacral_slope"][c][cat2]
            p_x3 = self.likelihood_["degree_spondylolisthesis"][c][cat3]
            post = p_prior * p_x1 * p_x2 * p_x3
            posteriors[c] = post
            details.append({
                "Kelas Target": c,
                "Prior P(Y)": p_prior,
                "P(X1|Y)": p_x1,
                "P(X2|Y)": p_x2,
                "P(X3|Y)": p_x3,
                "Nilai Posterior Akhir": post
            })

        best_class = max(posteriors, key=posteriors.get)
        details_df = pd.DataFrame(details).sort_values("Nilai Posterior Akhir", ascending=False).reset_index(drop=True)
        details_df["Peringkat Keputusan"] = [
            f"{i + 1} (Terpilih)" if i == 0 else str(i + 1)
            for i in range(len(details_df))
        ]

        total_post = sum(posteriors.values())
        norm_probs = {
            c: (posteriors[c] / total_post if total_post > 0 else 1.0 / len(self.classes_))
            for c in self.classes_
        }

        return {
            "binned_text": binned_text,
            "best_class": best_class,
            "posteriors": posteriors,
            "norm_probs": norm_probs,
            "table_details": details_df
        }

    def predict(self, X):
        X_binned = self.bin_dataframe(X)
        preds = []
        for _, row in X_binned.iterrows():
            posteriors = {}
            for c in self.classes_:
                post = self.prior_[c]
                for col in self.features:
                    post *= self.likelihood_[col][c][row[col]]
                posteriors[c] = post
            preds.append(max(posteriors, key=posteriors.get))
        return np.array(preds)

    def predict_proba(self, X):
        X_binned = self.bin_dataframe(X)
        proba_list = []
        for _, row in X_binned.iterrows():
            posteriors = []
            for c in self.classes_:
                post = self.prior_[c]
                for col in self.features:
                    post *= self.likelihood_[col][c][row[col]]
                posteriors.append(post)
            total = sum(posteriors)
            if total > 0:
                proba_list.append([p / total for p in posteriors])
            else:
                proba_list.append([1.0 / len(self.classes_)] * len(self.classes_))
        return np.array(proba_list)
