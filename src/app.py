from utils import db_connect
engine = db_connect()

# src/app.py

import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


URL = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"

ALPHA = 0.5
THRESHOLD = 0.3

ARTIFACTS_DIR = "models"
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "count_vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "multinomial_nb_alpha_0_5.pkl")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "model_config.json")


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df = df.drop(columns=["package_name"])
    df["review"] = df["review"].astype(str).str.strip().str.lower()
    df = df[df["review"].str.len() > 0].copy()
    return df


def train_and_save(df: pd.DataFrame) -> None:
    X = df["review"]
    y = df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = CountVectorizer(stop_words="english")
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = MultinomialNB(alpha=ALPHA)
    model.fit(X_train_vec, y_train)

    # Predicción con threshold custom
    y_proba = model.predict_proba(X_test_vec)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_1": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_1": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_1": float(f1_score(y_test, y_pred, pos_label=1)),
        "alpha": ALPHA,
        "threshold": THRESHOLD,
        "n_rows": int(df.shape[0])
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vec, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"alpha": ALPHA, "threshold": THRESHOLD, "metrics": metrics}, f, indent=2)

    print("✅ Artefactos guardados en /models")
    print("📊 Métricas:", metrics)


if __name__ == "__main__":
    df = load_data(URL)
    train_and_save(df)
