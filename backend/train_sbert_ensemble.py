import argparse, re, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers torch")
    import sys
    sys.exit(1)

# Import exact logic functions from current API
from main import compute_fuzzy_features, clean_text, g

def train_ensemble(csv_path: str, bundle_out: str = "sbert_ensemble_bundle.pkl"):
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
        
    print(f"Total valid rows: {len(df)}")
    
    print("Loading existing scalers and column maps from model_bundle.pkl...")
    old_bundle = joblib.load("model_bundle.pkl")
    behavioral_cols = old_bundle["behavioral_cols"]
    scaler_behav = old_bundle["scaler_behav"]
    scaler_fuzzy = old_bundle["scaler_fuzzy"]
    
    print("\nComputing Behavioral and Fuzzy features based on ReviewScan guidelines...")
    df["clean_text"] = df[text_col].apply(clean_text)
    
    behav_list = []
    fuzzy_list = []
    records = df.to_dict("records")
    
    for row in records:
        text = row["clean_text"]
        row["review_len"] = len(text.split())
        
        # 1. Behavioral Features
        b = [g(row, c) for c in behavioral_cols]
        behav_list.append(b)
        
        # 2. Fuzzy Features
        f = compute_fuzzy_features(row)
        fuzzy_list.append(f)
        
    X_behav = np.array(behav_list)
    X_behav_scaled = scaler_behav.transform(X_behav)
    
    X_fuzzy_arr = np.array(fuzzy_list)
    X_fuzzy_scaled = scaler_fuzzy.transform(X_fuzzy_arr)
    
    print("\nLoading SBERT model (all-MiniLM-L6-v2) for deep semantics...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df["clean_text"].tolist()
    print(f"Extracting dense embeddings for {len(texts)} reviews (approx. 12-15 mins with CPU)...")
    X_sbert = sbert_model.encode(texts, show_progress_bar=True)
    
    print("\nCombining SBERT + Behavioral + Fuzzy into hybrid Unified Feature Space...")
    X_all = np.hstack([X_sbert, X_behav_scaled, X_fuzzy_scaled])
    y = df["label_bin"].values
    
    print(f"Feature Space Dimensions: {X_all.shape[1]} features securely mapped.")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining the Ultimate Random Forest Ensemble...")
    # Utilizing an intense tree depth to maximize relations between fuzzy and nuerals
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print("=" * 60)
    print(f"FINAL SBERT + HYBRID ENSEMBLE TEST ACCURACY   : {acc*100:.2f}%")
    print(f"FINAL SBERT + HYBRID ENSEMBLE TEST AUC-ROC    : {auc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fake"]))
    
    # Export securely mapping
    new_bundle = {
        "sbert_rf": rf,
        "scaler_behav": scaler_behav,
        "scaler_fuzzy": scaler_fuzzy,
        "behavioral_cols": behavioral_cols
    }
    joblib.dump(new_bundle, bundle_out)
    print(f"\n🚀 Hybrid SBERT Ensemble successfully mapped and saved to {bundle_out}")

if __name__ == "__main__":
    train_ensemble("df.csv")
