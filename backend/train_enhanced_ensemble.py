"""
train_enhanced_ensemble.py
Enhanced SBERT + Behavioral (with 10 new features) + Fuzzy + Random Forest
Target: Push accuracy from 88.55% → 90%+
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("pip install sentence-transformers torch")
    import sys; sys.exit(1)

from main import compute_fuzzy_features, clean_text, g

# ── Parse date helper ─────────────────────────────────────────────────────────
def parse_date(s):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except:
            pass
    return None

# ── Engineer new behavioral features ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Parse dates
    df["_review_date"] = df["date"].apply(parse_date)
    df["_join_date"]   = df["yelpJoinDate"].apply(parse_date)

    # 1. account_age_days: how old account was when review was posted
    def acct_age(row):
        if row["_review_date"] and row["_join_date"]:
            delta = (row["_review_date"] - row["_join_date"]).days
            return max(delta, 0)
        return 0
    df["account_age_days"] = df.apply(acct_age, axis=1)

    # 2. rating_deviation: |rating - restaurant avg rating|
    df["rating_deviation"] = (df["rating"] - df["restaurantRating"]).abs()

    # 3. influence_score: engagement per review
    df["influence_score"] = (
        (df["usefulCount"] + df["coolCount"] + df["funnyCount"])
        / (df["reviewCount"] + 1)
    )

    # 4. social_ratio: friends per review
    df["social_ratio"] = df["friendCount"] / (df["reviewCount"] + 1)

    # 5. elite_signal: prestige composite
    df["elite_signal"] = df["complimentCount"] + df["fanCount"] + df["tipCount"]

    # 6. burst_intensity: relative burst (mnr / total reviews)
    df["burst_intensity"] = df["mnr"] * (1.0 / (df["reviewCount"] + 1))

    # 7. review_productivity: reviews per day over lifetime
    df["review_productivity"] = df["reviewCount"] / (df["account_age_days"] + 1)

    # 8. content_sim_x_rating: interaction — copy-paste + extreme rating
    df["content_sim_x_rating"] = (
        df["Maximum Content Similarity"].fillna(0) * df["rating"]
    )

    # 9. lonely_reviewer: completely isolated (no friends, no fans)
    df["lonely_reviewer"] = (
        (df["friendCount"] == 0) & (df["fanCount"] == 0)
    ).astype(int)

    # 10. vote_diversity: balance of useful/cool/funny
    total_votes = df["usefulCount"] + df["coolCount"] + df["funnyCount"] + 1
    df["vote_diversity"] = df["usefulCount"] / total_votes

    print("✅ Engineered 10 new behavioral features.")
    return df

NEW_BEHAVIORAL_COLS = [
    "account_age_days", "rating_deviation", "influence_score",
    "social_ratio", "elite_signal", "burst_intensity",
    "review_productivity", "content_sim_x_rating",
    "lonely_reviewer", "vote_diversity"
]

OLD_BEHAVIORAL_COLS = [
    "rating", "reviewUsefulCount", "friendCount", "reviewCount",
    "firstCount", "usefulCount", "coolCount", "funnyCount",
    "complimentCount", "tipCount", "fanCount", "restaurantRating",
    "mnr", "rl", "rd", "Maximum Content Similarity", "review_len"
]

ALL_BEHAVIORAL_COLS = OLD_BEHAVIORAL_COLS + NEW_BEHAVIORAL_COLS  # 27 total

# ── Main Training ─────────────────────────────────────────────────────────────
def train(csv_path="df.csv", bundle_out="enhanced_ensemble_bundle.pkl"):
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if "reviewContent" in df.columns and "flagged" in df.columns:
        df = df.dropna(subset=["reviewContent", "flagged"])
        df["label_bin"] = (df["flagged"] == "Y").astype(int)
        text_col = "reviewContent"
    elif "text_" in df.columns and "label" in df.columns:
        df = df.dropna(subset=["text_", "label"])
        df["label_bin"] = (df["label"] == "CG").astype(int)
        text_col = "text_"
    else:
        raise ValueError("Unknown CSV format.")

    print(f"Rows: {len(df)} | Fake: {df.label_bin.sum()} | Genuine: {(df.label_bin==0).sum()}")

    # Engineer new features
    df = engineer_features(df)
    df["clean_text"] = df[text_col].apply(clean_text)
    df["review_len"] = df["clean_text"].apply(lambda t: len(t.split()))

    # ── Stream A: Behavioral (old + new) ────────────────────────────────────
    print(f"\nBuilding behavioral feature matrix ({len(ALL_BEHAVIORAL_COLS)} cols)...")
    records = df.to_dict("records")
    behav_list = []
    fuzzy_list = []
    for row in records:
        b = [g(row, c) for c in ALL_BEHAVIORAL_COLS]
        behav_list.append(b)
        fuzzy_list.append(compute_fuzzy_features(row))

    X_behav = np.array(behav_list, dtype=float)
    X_fuzzy = np.array(fuzzy_list, dtype=float)

    scaler_behav = StandardScaler()
    scaler_fuzzy  = StandardScaler()
    X_behav_s = scaler_behav.fit_transform(X_behav)
    X_fuzzy_s  = scaler_fuzzy.fit_transform(X_fuzzy)

    # ── Stream B: SBERT embeddings ───────────────────────────────────────────
    print("\nLoading SBERT model (all-MiniLM-L6-v2)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Encoding {len(df)} reviews (this takes ~12 mins on CPU)...")
    X_sbert = sbert.encode(df["clean_text"].tolist(), show_progress_bar=True)

    # ── Combine all streams ──────────────────────────────────────────────────
    X_all = np.hstack([X_sbert, X_behav_s, X_fuzzy_s])
    y = df["label_bin"].values
    print(f"\nFinal Feature Space: {X_all.shape[1]} dims "
          f"(384 SBERT + {len(ALL_BEHAVIORAL_COLS)} Behavioral + 32 Fuzzy)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train Random Forest ──────────────────────────────────────────────────
    print("\nTraining Enhanced Random Forest (300 trees, max_depth=25)...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_leaf=2,
        n_jobs=-1, class_weight="balanced", random_state=42
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    print("\n" + "=" * 65)
    print(f"  ENHANCED HYBRID ENSEMBLE ACCURACY  : {acc*100:.2f}%")
    print(f"  ENHANCED HYBRID ENSEMBLE AUC-ROC   : {auc:.4f}")
    print(f"  Feature Dimensions                 : {X_all.shape[1]}")
    print("=" * 65)
    print(classification_report(y_te, y_pred, target_names=["Genuine", "Fake"]))

    # ── Save bundle ──────────────────────────────────────────────────────────
    bundle = {
        "sbert_rf":          rf,
        "scaler_behav":      scaler_behav,
        "scaler_fuzzy":      scaler_fuzzy,
        "behavioral_cols":   ALL_BEHAVIORAL_COLS,
        "new_feature_cols":  NEW_BEHAVIORAL_COLS,
        "accuracy":          round(acc, 4),
        "auc_roc":           round(auc, 4),
    }
    joblib.dump(bundle, bundle_out)
    print(f"\n🚀 Saved enhanced bundle → {bundle_out}")

if __name__ == "__main__":
    train()
