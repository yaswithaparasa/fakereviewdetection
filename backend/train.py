"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             ReviewGuard AI  —  Model Training Script  v4.0                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Compatible: Python 3.9-3.14 · sentence-transformers ≥3.0 · numpy ≥2.0   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  New in v4.0:                                                               ║
║    • all-mpnet-base-v2  (768-dim, +2-4% vs MiniLM)                        ║
║    • Removed StandardScaler  (no effect on Random Forest)                  ║
║    • Selective MinMaxScaler on 29 hand-crafted dims only                   ║
║    • 5 new features: VADER sentiment, stopword ratio, repeated-word ratio, ║
║      consecutive-exclamation count, dramatic-word ratio                    ║
║    • Improved clean_text: emoji + unicode noise removal                    ║
║    • Deduplication before training                                          ║
║    • RandomizedSearchCV for fast hyperparameter tuning                     ║
║    • class_weight {0:1, 1:2} — penalises fake misses twice as hard        ║
║  Run ONCE before starting the API:                                          ║
║      pip install vaderSentiment nltk                                        ║
║      python train.py                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, re, sys, time, logging, warnings, unicodedata
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "Balanced_Final_Dataset.csv"
MODEL_PATH    = "rf_model.joblib"
SBERT_NAME    = "all-mpnet-base-v2"        # upgraded from all-MiniLM-L6-v2
SBERT_DIM     = 768                         # mpnet output dim (was 384)
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
MIN_TEXT_LEN  = 15
MAX_TEXT_LEN  = 8000
RF_N_TREES    = 500
TARGET_ACC    = 0.90
SBERT_BATCH   = 128    # mpnet is heavier; lower to 64 if OOM
CHUNK_SIZE    = 10_000
CLASS_WEIGHT  = {0: 1, 1: 2}   # penalise fake misses 2x more

# ── VADER (lazy-loaded once) ──────────────────────────────────────────────────
_vader = None
def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            log.warning("vaderSentiment not installed — VADER score will be 0.")
            log.warning("Run:  pip install vaderSentiment")
    return _vader

# ── NLTK stopwords (lazy-loaded once) ────────────────────────────────────────
_stopwords = None
def _get_stopwords():
    global _stopwords
    if _stopwords is None:
        try:
            from nltk.corpus import stopwords as sw
            import nltk
            try:
                _stopwords = set(sw.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                _stopwords = set(sw.words("english"))
        except ImportError:
            log.warning("nltk not installed — stopword ratio will be 0.")
            log.warning("Run:  pip install nltk")
            _stopwords = set()
    return _stopwords


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    text = re.sub(r'"+', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT_LEN]


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log.error("Dataset not found: %s", path)
        sys.exit(1)

    log.info("Loading: %s", path)
    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        log.error("CSV must have columns: text, label  (found: %s)", df.columns.tolist())
        sys.exit(1)

    df["text"]  = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LEN].reset_index(drop=True)
    log.info("Removed %d short rows.", before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    log.info("Removed %d duplicate rows. Remaining: %d", before - len(df), len(df))

    log.info("Label counts:\n%s", df["label"].value_counts().to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  HAND-CRAFTED FEATURES  (29 dims — was 24)
# ══════════════════════════════════════════════════════════════════════════════

_POS = {
    "excellent","amazing","outstanding","wonderful","fantastic","superb",
    "incredible","extraordinary","exceptional","magnificent","brilliant",
    "flawless","perfect","spectacular","phenomenal","unbelievable","remarkable",
    "breathtaking","divine","immaculate","impeccable","transcendent","unmatched",
    "incomparable","greatest","finest","best","ultimate","definitive","supreme",
}
_NEG = {
    "terrible","awful","horrible","disgusting","atrocious","dreadful","appalling",
    "abysmal","pathetic","catastrophic","disastrous","unacceptable","deplorable",
    "worst","useless","broken","defective","damaged","fraud","scam",
    "disappointing","mediocre","poor","cheap",
}
_SPAM = [
    "buy now","order now","act fast","limited time","selling fast",
    "while stocks last","do not miss","don't miss","book now","book today",
    "order today","buy today","buy immediately","purchase now","get it now",
    "hurry","click here","limited offer","shop now","deal ends",
]
_CRED = [
    "years of experience","years experience","as a professional","in my career",
    "as an expert","as a certified","as a former","i have a degree","i have a phd",
    "as a doctor","as a chef","as an engineer","as a consultant","as a blogger",
    "in my professional","i rarely write","i never write","i had to write",
    "i created an account","real customer","real review","not a fake",
    "not sponsored","genuine review","not exaggerating","verified purchase",
]
_DRAMATIC_WORDS = {
    "absolutely","completely","totally","utterly","literally","honestly",
    "seriously","insanely","ridiculously","insane","crazy","unreal",
    "mindblowing","mind-blowing","gamechanging","game-changing","lifechanging",
    "life-changing","unbelievably","extraordinarily","phenomenally",
}

FEAT_NAMES = [
    "word_count","char_count","avg_word_length","avg_sentence_length",
    "lexical_diversity","repetition_ratio","exclamation_marks","question_marks",
    "allcaps_words","caps_ratio","positive_keywords","negative_keywords",
    "spam_phrases","credibility_claims","superlatives",
    "first_person_ratio","second_person_ratio","digit_ratio",
    "sentiment_skew","sentence_count","excl_capped","allcaps_capped",
    "pos_capped","html_tags",
    # 5 new
    "vader_compound",
    "stopword_ratio",
    "repeated_word_ratio",
    "consec_excl",
    "dramatic_word_ratio",
]
N_HAND = len(FEAT_NAMES)   # 29


def extract_features(text: str) -> np.ndarray:
    from collections import Counter

    t       = text.lower()
    words   = text.split()
    words_l = [w.lower() for w in words]
    sents   = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    nw      = max(len(words), 1)
    nc      = max(len(text),  1)
    ns      = max(len(sents), 1)

    avg_wl = float(np.mean([len(w.rstrip(".,!?;:\"'")) for w in words]))
    avg_sl = nw / ns
    ttr    = len(set(words_l)) / nw
    rep    = 1.0 - ttr

    excl   = text.count("!")
    quest  = text.count("?")
    caps_w = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_r = sum(1 for c in text if c.isupper()) / nc

    pos    = sum(1 for w in _POS  if w in t)
    neg    = sum(1 for w in _NEG  if w in t)
    spam   = sum(1 for p in _SPAM if p in t)
    cred   = sum(1 for c in _CRED if c in t)
    sups   = len(re.findall(
        r"\b(best|greatest|finest|worst|most|least|only|ever|never|always|"
        r"perfect|flawless|ideal|ultimate|definitive|absolute|complete|total|"
        r"entire|every|all|nothing|everything|everyone|nobody|anyone)\b", t))

    fp_r  = sum(1 for w in words_l if w in {"i","me","my","myself","we","our","us"}) / nw
    tp_r  = sum(1 for w in words_l if w in {"you","your","everyone","everybody","anyone","all"}) / nw
    dig_r = sum(1 for c in text if c.isdigit()) / nc
    skew  = pos / max(pos + neg + 1, 1)
    html  = len(re.findall(r"<[^>]+>", text))

    # ── 5 new features ────────────────────────────────────────────────────────
    vader = _get_vader()
    vader_compound = vader.polarity_scores(text)["compound"] if vader else 0.0

    sw = _get_stopwords()
    stopword_ratio = sum(1 for w in words_l if w in sw) / nw if sw else 0.0

    word_counts = Counter(words_l)
    unique_words = max(len(word_counts), 1)
    repeated_word_ratio = sum(1 for w, c in word_counts.items() if c >= 3) / unique_words

    consec_excl = max((len(m.group()) for m in re.finditer(r"!+", text)), default=0)

    dramatic_word_ratio = sum(1 for w in words_l if w in _DRAMATIC_WORDS) / nw

    return np.array([
        nw, nc, avg_wl, avg_sl, ttr, rep,
        excl, quest, caps_w, caps_r,
        pos, neg, spam, cred, sups,
        fp_r, tp_r, dig_r, skew, ns,
        min(excl, 10), min(caps_w, 15), min(pos, 20), html,
        vader_compound, stopword_ratio, repeated_word_ratio,
        consec_excl, dramatic_word_ratio,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTIVE FEATURE TRANSFORMER
#  SBERT dims (0..767) — passed through unchanged (already unit-normalized)
#  Hand dims (768..796) — MinMaxScaler applied
# ══════════════════════════════════════════════════════════════════════════════

class SelectiveScaler(BaseEstimator, TransformerMixin):
    """
    Leaves SBERT embeddings untouched.
    Applies MinMaxScaler only to the trailing N_HAND hand-crafted dims.
    """
    def __init__(self):
        self.scaler_ = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler_.fit(X[:, SBERT_DIM:])
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out[:, SBERT_DIM:] = self.scaler_.transform(X[:, SBERT_DIM:])
        return X_out


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(texts: list, sbert_model) -> np.ndarray:
    n = len(texts)
    log.info("  Encoding %d texts with '%s' (batch=%d, chunks=%d) …",
             n, SBERT_NAME, SBERT_BATCH, CHUNK_SIZE)

    emb_chunks = []
    for start in range(0, n, CHUNK_SIZE):
        chunk = texts[start : start + CHUNK_SIZE]
        log.info("    SBERT chunk %d–%d …", start, min(start + CHUNK_SIZE, n) - 1)
        chunk_emb = sbert_model.encode(
            chunk,
            batch_size=SBERT_BATCH,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        emb_chunks.append(np.asarray(chunk_emb, dtype=np.float32))
    emb = np.vstack(emb_chunks)                        # (n, 768)

    log.info("  Extracting %d hand-crafted features (parallel) …", N_HAND)
    from joblib import Parallel, delayed
    hand_list = Parallel(n_jobs=-1, prefer="threads")(
        delayed(extract_features)(t) for t in texts
    )
    hand = np.vstack(hand_list)                         # (n, 29)

    X = np.hstack([emb, hand])                          # (n, 797)
    log.info("  Feature matrix: %s  dtype=%s  RAM~%.0f MB",
             X.shape, X.dtype, X.nbytes / 1024 / 1024)
    return X


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(n_estimators=RF_N_TREES, max_depth=None,
                   min_samples_split=2, min_samples_leaf=2) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight=CLASS_WEIGHT,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    return Pipeline([
        ("scaler", SelectiveScaler()),
        ("rf",     rf),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def tune_hyperparams(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    log.info("RandomizedSearchCV (n_iter=12, cv=3) — finding best RF params …")
    log.info("This may take 30-90 minutes on CPU with 100k samples.")

    param_dist = {
        # prefix 'rf__' because of Pipeline wrapper — IMPORTANT
        "rf__n_estimators":      [300, 500, 700],
        "rf__max_depth":         [10, 20, None],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf":  [1, 2],
    }

    base_pipe = build_pipeline()
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=12,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",      # optimise for fake/genuine balance, not just accuracy
        n_jobs=1,          # RF already uses n_jobs=-1 inside
        random_state=RANDOM_STATE,
        verbose=1,
        refit=False,
    )
    search.fit(X_tr, y_tr)
    log.info("Best params : %s", search.best_params_)
    log.info("Best CV f1  : %.4f", search.best_score_)
    return search.best_params_


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> Pipeline:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Split → Train: %d  |  Test: %d", len(y_tr), len(y_te))

    best_params = tune_hyperparams(X_tr, y_tr)

    pipe = build_pipeline()
    pipe.set_params(**best_params)

    # 5-fold CV
    log.info("5-fold cross-validation (f1) …")
    cv_scores = cross_val_score(
        pipe, X_tr, y_tr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1", n_jobs=1,
    )
    log.info("CV f1: %.4f +/- %.4f  (folds: %s)",
             cv_scores.mean(), cv_scores.std(),
             " | ".join(f"{s:.4f}" for s in cv_scores))

    # Fit on train split
    log.info("Training final Random Forest …")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    log.info("Training done in %.1f s", time.time() - t0)

    # Evaluate
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    acc     = accuracy_score(y_te, y_pred)
    auc     = roc_auc_score(y_te, y_proba)

    log.info("=" * 62)
    log.info("TEST RESULTS")
    log.info("  Accuracy : %.4f  (%.1f%%)", acc, acc * 100)
    log.info("  ROC-AUC  : %.4f", auc)
    log.info("  Confusion matrix:\n%s", confusion_matrix(y_te, y_pred))
    log.info("Classification report:\n%s",
             classification_report(y_te, y_pred,
                                   target_names=["Genuine (0)", "Fake (1)"]))
    log.info("  *** Focus on Fake (1) recall — target >= 0.80 ***")

    if acc >= TARGET_ACC:
        log.info("TARGET MET — accuracy %.1f%% >= %.0f%%", acc * 100, TARGET_ACC * 100)
    else:
        log.warning("Accuracy %.1f%% < %.0f%% target.", acc * 100, TARGET_ACC * 100)

    # Checkpoint
    ckpt = MODEL_PATH.replace(".joblib", "_checkpoint.joblib")
    joblib.dump(pipe, ckpt)
    log.info("Checkpoint saved -> %s", ckpt)

    # Re-fit on 100% data
    log.info("Re-fitting on 100%% of data …")
    pipe.fit(X, y)
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 62)
    log.info("ReviewGuard AI — Training Pipeline v4.0")
    log.info("Python %s | numpy %s", sys.version.split()[0], np.__version__)
    log.info("SBERT : %s  (%d-dim)", SBERT_NAME, SBERT_DIM)
    log.info("Dims  : %d SBERT + %d hand = %d total", SBERT_DIM, N_HAND, SBERT_DIM + N_HAND)
    log.info("=" * 62)

    df = load_dataset(DATASET_PATH)

    # Warm up lazy resources before parallel extraction
    _get_vader()
    _get_stopwords()

    sbert = SentenceTransformer(SBERT_NAME)
    log.info("SBERT loaded.")

    X = build_feature_matrix(df["text"].tolist(), sbert)
    y = df["label"].values

    pipe = train_and_evaluate(X, y)

    joblib.dump(pipe, MODEL_PATH)
    log.info("Saved -> %s  (%.1f MB)",
             MODEL_PATH, os.path.getsize(MODEL_PATH) / 1024 / 1024)
    log.info("Total time: %.1f s", time.time() - t0)
    log.info("Done!  Run:  python app.py")


if __name__ == "__main__":
    main()
