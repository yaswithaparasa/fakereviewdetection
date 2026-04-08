import argparse, re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers torch")
    import sys
    sys.exit(1)

def clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def train_sbert(csv_path: str, model_path: str = "sbert_model.pkl"):
    print(f"Loading dataset from {csv_path}...")
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
        raise ValueError("Unknown CSV format.")
        
    df["clean"] = df[text_col].apply(clean)
    print(f"Total valid rows: {len(df)}")
    
    print("Loading SBERT model (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    X_texts = list(df["clean"])
    y = list(df["label_bin"])
    print(f"Extracting embeddings for {len(X_texts)} reviews (this might take a bit)...")
    X_embeddings = sbert_model.encode(X_texts, show_progress_bar=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Logistic Regression on SBERT embeddings...")
    clf = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fake"]))
    
    joblib.dump(clf, model_path)
    print(f"SBERT Classifier saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="df.csv")
    parser.add_argument("--out", default="sbert_model.pkl")
    args = parser.parse_args()
    train_sbert(args.csv, args.out)
