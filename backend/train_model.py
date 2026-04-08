"""
train_model.py — Train the fake review classifier.

Usage:
    python train_model.py --csv path/to/fake_reviews_dataset.csv

Outputs model.pkl in the same directory.
"""

import argparse, re, os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def train(csv_path: str, model_path: str = "model.pkl"):
    print(f"Loading dataset from {csv_path}…")
    df = pd.read_csv(csv_path)
    if "text_" in df.columns and "label" in df.columns:
        df = df.dropna(subset=["text_", "label"])
        df["label_bin"] = (df["label"] == "CG").astype(int)
        text_col = "text_"
    elif "reviewContent" in df.columns and "flagged" in df.columns:
        df = df.dropna(subset=["reviewContent", "flagged"])
        df["label_bin"] = (df["flagged"] == "Y").astype(int)
        text_col = "reviewContent"
    else:
        raise ValueError(f"Unknown CSV format. Columns found: {df.columns.tolist()}")
        
    df["clean"] = df[text_col].apply(clean)
    print(f"Dataset: {len(df):,} rows | Fake: {df.label_bin.sum():,} | Genuine: {(df.label_bin==0).sum():,}")

    X, y = list(df["clean"]), list(df["label_bin"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=60_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs", class_weight="balanced")),
    ])

    print("Training…")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fake"]))

    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="df.csv")
    parser.add_argument("--out", default="model.pkl")
    args = parser.parse_args()
    train(args.csv, args.out)