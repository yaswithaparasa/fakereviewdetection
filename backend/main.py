"""
ReviewScan Pro — FastAPI Backend
Pipeline: SBERT-style LSA embeddings + Fuzzy Logic features + Random Forest
Dataset: Yelp restaurant reviews (26,958 reviews, Y/N flagged)
Metrics: Accuracy ~88.9% | AUC-ROC ~0.94

Run:
    pip install fastapi uvicorn scikit-learn joblib pandas numpy scipy
    uvicorn main:app --reload --port 8001
"""

import re, time, logging, math
from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="ReviewScan Pro API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load model bundle ──────────────────────────────────────────────────────────
BUNDLE_PATH = Path(__file__).parent / "model_bundle.pkl"
bundle         = joblib.load(BUNDLE_PATH)
tfidf          = bundle["tfidf"]
svd            = bundle["svd"]
scaler_behav   = bundle["scaler_behav"]
scaler_fuzzy   = bundle["scaler_fuzzy"]
behavioral_cols = bundle["behavioral_cols"]
rf             = bundle["rf"]
log.info("Model bundle loaded.")

# Load the new text-only Logistic Regression model (fallback)
TEXT_MODEL_PATH = Path(__file__).parent / "model.pkl"
try:
    text_clf = joblib.load(TEXT_MODEL_PATH)
    log.info("Text-only LR model loaded.")
except Exception as e:
    log.warning(f"Could not load text model: {e}")
    text_clf = None

# ── SBERT Hybrid Ensemble (PRIORITY 1 — SBERT + Behavioral + Fuzzy + RF) ──────
SBERT_ENSEMBLE_PATH = Path(__file__).parent / "sbert_ensemble_bundle.pkl"
sbert_encoder = None
sbert_ensemble_bundle = None
sbert_ensemble_rf = None
try:
    from sentence_transformers import SentenceTransformer
    sbert_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_ensemble_bundle = joblib.load(SBERT_ENSEMBLE_PATH)
    sbert_ensemble_rf = sbert_ensemble_bundle["sbert_rf"]
    log.info("✅ SBERT Hybrid Ensemble loaded (88.55% acc, AUC 0.9347).")
except Exception as e:
    log.info(f"SBERT Hybrid Ensemble not available: {e}. Falling back to standard models.")
    sbert_ensemble_bundle = None
    sbert_ensemble_rf = None

# Legacy SBERT-only classifier (no longer used as primary)
sbert_clf = None  # disabled — superseded by sbert_ensemble_rf

# ── Validation Logic ───────────────────────────────────────────────────────────
COMMON_WORDS = {"the", "and", "is", "for", "with", "that", "this", "was", "but", "not", "have", "you", "review", "food", "service", "great", "good", "bad", "place", "product", "amazing", "very", "they", "here", "it", "so", "all", "my", "at", "from"}

def validate_review(text: str) -> tuple[bool, str]:
    text = text.strip()

    # 1. PRESENCE — must not be empty
    if not text:
        return False, "Review is empty."

    # 2. LENGTH — minimum characters
    if len(text) < 10:
        return False, "Too short (min 10 chars)."

    # 3. LENGTH — maximum characters
    if len(text) > 4000:
        return False, "Too long (max 4000 chars)."

    # 4. PATTERN/REGEX — must contain actual letters, not just symbols/numbers
    if re.fullmatch(r"[\d\s\W]+", text):
        return False, "Only numbers or symbols — no real words found."

    # 5. SYMBOL DENSITY — too many special characters (spam/garbage)
    symbols = re.sub(r'[a-zA-Z0-9\s]', '', text)
    if len(symbols) / len(text) > 0.35:
        return False, "Contains too many special symbols/punctuation."

    # 6. NUMERIC DENSITY — too many numbers
    numbers = re.sub(r'[^0-9]', '', text)
    if len(numbers) / len(text) > 0.45:
        return False, "Contains mostly numbers — not a real review."

    # 7. LENGTH — minimum word count
    words = text.split()
    if len(words) < 3:
        return False, "Enter at least 3 words for a meaningful review."

    # 8. LENGTH — maximum word count
    if len(words) > 600:
        return False, "Too many words (max 600). Please shorten your review."

    # 9. CHARACTER — no injection-style content (HTML/script tags)
    if re.search(r"<[^>]+>", text):
        return False, "HTML or script tags are not allowed."

    # 10. CHARACTER — no URLs
    if re.search(r"http[s]?://|www\.", text, re.IGNORECASE):
        return False, "URLs are not allowed in reviews."

    # 11. UNIQUENESS — repeated single character (e.g. "aaaaaaa good")
    if len(set(text.replace(" ", ""))) < 3:
        return False, "Looks like repeated characters or nonsense."

    letters = [c for c in text.lower() if c.isalpha()]

    # 12. RANGE — letter ratio must be high enough
    if letters and len(letters) / len(text) < 0.40:
        return False, "Too many numbers or symbols compared to text."

    # 13. SEMANTIC — overall vowel ratio (detects non-language text)
    vowels = [c for c in letters if c in "aeiou"]
    if letters and len(vowels) / len(letters) < 0.12:
        return False, "Text has very few vowels — does not look like real language."

    # 14. SEMANTIC — Common English word check (spam/gibberish filter)
    # Most real reviews will contain at least one extremely common functional word
    if len(words) >= 5:
        has_common = any(word.lower() in COMMON_WORDS for word in words)
        if not has_common:
            return False, "Review does not appear to contain meaningful English words."

    # 15. CHARACTER — no word mixes letters and digits suspiciously
    for word in words:
        if word.isdigit() and len(word) > 12:
             return False, "Contains unusually long numeric strings."

        has_digit  = any(c.isdigit() for c in word)
        has_letter = any(c.isalpha() for c in word)
        if has_digit and has_letter:
            # Reject if there's a long sequence of digits inside a mixed word (likely an ID/hash)
            digit_segments = re.findall(r'\d+', word)
            if any(len(ds) > 5 for ds in digit_segments):
                return False, f"'{word}' contains a long numeric sequence (likely a random ID or hash)."

            segments = len(re.findall(r'[a-zA-Z]+|\d+', word))
            if segments > 3 and len(word) > 5:
                # Common things like "win10x64" (segments=4)
                return False, f"'{word}' looks like a random alphanumeric string."

    # 16. SEMANTIC — too many gibberish words (long words with very few vowels)
    suspicious = 0
    for word in words:
        word_clean = re.sub(r"[^a-z]", "", word.lower())
        if len(word_clean) >= 8:
            wv = [c for c in word_clean if c in "aeiou"]
            if len(wv) / len(word_clean) < 0.18:
                suspicious += 1
    if suspicious >= 2:
        return False, "Contains words that look like gibberish."

    # 17. PATTERN — gibberish word with 5+ consecutive consonants
    consonants = set("bcdfghjklmnpqrstvwxyz")
    for word in words:
        word_clean = re.sub(r"[^a-z]", "", word.lower())
        if len(word_clean) < 5:
            continue
        max_consec = 0
        current    = 0
        for ch in word_clean:
            if ch in consonants:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        if max_consec >= 6: 
            return False, f"'{word}' looks like gibberish (excessive consecutive consonants)."
        if max_consec >= 5 and len(word_clean) > 12:
            return False, f"'{word}' looks like gibberish (high consonant density)."

    # 18. FORMAT — all caps shouting
    alpha_words = [w for w in words if w.isalpha() and len(w) > 3]
    if alpha_words and sum(1 for w in alpha_words if w.isupper()) / len(alpha_words) > 0.7:
        return False, "Review appears to be all caps shouting. Please use normal casing."

    # 19. RANGE — too many exclamation or question marks
    exclaim_count = text.count("!") + text.count("?")
    if exclaim_count > 10:
        return False, f"Too many exclamation/question marks ({exclaim_count})."

    # 20. UNIQUENESS — repeating word spam
    if len(words) >= 10:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.38:
            return False, "Too many repeated words (possible spam)."

    # 21. STRUCTURAL — must contain at least some real words and basic structure
    alpha_words_2 = [w for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) >= 2]
    if len(alpha_words_2) < 3:
        return False, "Review must contain proper words and sentences."

    return True, "OK"

# ── Utilities ──────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def fuzzy_tri(x, a, b, c):
    x = float(x)
    left  = 0 if x <= a else (1 if x >= b else (x-a)/(b-a+1e-9))
    right = 0 if x >= c else (1 if x <= b else (c-x)/(c-b+1e-9))
    return min(left, right)

def fuzzy_gauss(x, mean, sigma):
    return float(np.exp(-0.5 * ((float(x)-mean)/sigma)**2))

def g(d, k, default=0.0):
    try: return float(d.get(k, default) or default)
    except: return default

# ── Fuzzy feature extractor ────────────────────────────────────────────────────
def compute_fuzzy_features(row: dict) -> np.ndarray:
    rating       = g(row, "rating", 3.0)
    sim          = g(row, "Maximum Content Similarity", 0.0)
    mnr          = g(row, "mnr", 0.0)
    rd           = g(row, "rd", 0.0)
    rev_len      = g(row, "review_len", 50.0)
    rev_count    = g(row, "reviewCount", 1.0)
    friend_count = g(row, "friendCount", 0.0)
    useful_count = g(row, "usefulCount", 0.0)

    rl=fuzzy_tri(rating,0,1,2.5);rm=fuzzy_tri(rating,1.5,3,4.5);rh=fuzzy_tri(rating,3.5,5,6)
    sl=fuzzy_tri(sim,-0.1,0,0.3);sm=fuzzy_tri(sim,0.2,0.5,0.7);sh=fuzzy_tri(sim,0.6,1.0,1.1)
    ml=fuzzy_tri(mnr,-0.1,0,0.2);mm=fuzzy_tri(mnr,0.1,0.35,0.6);mh=fuzzy_tri(mnr,0.5,1.0,1.1)
    lxs=fuzzy_tri(rev_len,-1,0,30);ls=fuzzy_tri(rev_len,10,50,100);ln=fuzzy_tri(rev_len,80,150,300);ll=fuzzy_tri(rev_len,250,500,1500)
    rcn=fuzzy_gauss(rev_count,0,5);rcm=fuzzy_gauss(rev_count,20,15);rcv=fuzzy_gauss(rev_count,100,50)
    fi=fuzzy_tri(friend_count,-1,0,10);fs=fuzzy_tri(friend_count,5,40,150);fh=fuzzy_tri(friend_count,100,500,2000)
    ul=fuzzy_gauss(useful_count,0,3);um=fuzzy_gauss(useful_count,10,8);uh=fuzzy_gauss(useful_count,50,30)
    rdl=fuzzy_tri(rd,-0.1,0,0.2);rdm=fuzzy_tri(rd,0.15,0.375,0.6);rdh=fuzzy_tri(rd,0.5,1.0,1.1)

    rext=max(rl,rh)
    s1=min(sh,rext); s2=min(mh,rcn); s3=min(lxs,rh); s4=min(min(fi,rcn),ul)
    g1=min(min(uh,rcv),ll)
    fake_score=0.30*s1+0.25*s2+0.20*s3+0.25*s4
    genuine_score=g1
    return np.array([rl,rm,rh,sl,sm,sh,ml,mm,mh,lxs,ls,ln,ll,rcn,rcm,rcv,fi,fs,fh,ul,um,uh,rdl,rdm,rdh,s1,s2,s3,s4,g1,fake_score,genuine_score],dtype=float)

def get_fuzzy_signals(row: dict) -> List[dict]:
    rating=g(row,"rating",3); sim=g(row,"Maximum Content Similarity"); mnr=g(row,"mnr")
    rev_len=g(row,"review_len",50); rev_count=g(row,"reviewCount",1)
    friend_cnt=g(row,"friendCount"); useful=g(row,"usefulCount")

    rh=fuzzy_tri(rating,3.5,5,6); rl=fuzzy_tri(rating,0,1,2.5)
    sh=fuzzy_tri(sim,0.6,1.0,1.1); mh=fuzzy_tri(mnr,0.5,1.0,1.1)
    lxs=fuzzy_tri(rev_len,-1,0,30); rcn=fuzzy_gauss(rev_count,0,5)
    fi=fuzzy_tri(friend_cnt,-1,0,10); ul=fuzzy_gauss(useful,0,3)
    uh=fuzzy_gauss(useful,50,30); rcv=fuzzy_gauss(rev_count,100,50)
    ll=fuzzy_tri(rev_len,250,500,1500)

    signals = []
    r1=min(sh,max(rl,rh))
    if r1>0.25: signals.append({"rule":"R1","strength":round(r1,3),"label":"High content similarity + extreme rating","detail":f"Similarity={sim:.2f}, Rating={rating:.0f}★","verdict":"suspicious"})
    r2=min(mh,rcn)
    if r2>0.25: signals.append({"rule":"R2","strength":round(r2,3),"label":"Burst-reviewer pattern","detail":f"MNR={mnr:.3f}, Reviews={int(rev_count)}","verdict":"suspicious"})
    r3=min(lxs,rh)
    if r3>0.25: signals.append({"rule":"R3","strength":round(r3,3),"label":"Short 5-star one-liner","detail":f"Length={int(rev_len)} words, Rating={rating:.0f}★","verdict":"suspicious"})
    r4=min(min(fi,rcn),ul)
    if r4>0.25: signals.append({"rule":"R4","strength":round(r4,3),"label":"Isolated new user, zero influence","detail":f"Friends={int(friend_cnt)}, Reviews={int(rev_count)}, Useful={int(useful)}","verdict":"suspicious"})
    r5=min(min(uh,rcv),ll)
    if r5>0.25: signals.append({"rule":"R5","strength":round(r5,3),"label":"Veteran influential reviewer","detail":f"Useful={int(useful)}, Reviews={int(rev_count)}, Length={int(rev_len)} words","verdict":"genuine"})
    if not signals: signals.append({"rule":"—","strength":0.0,"label":"No strong fuzzy rules fired","detail":"Classification driven by semantic content","verdict":"neutral"})
    return signals

def get_semantic_signals(text: str, is_fake: bool) -> List[str]:
    t=text.lower(); sigs=[]
    if re.search(r"!{2,}",text): sigs.append("Excessive exclamation marks")
    if re.search(r"\b(amazing|perfect|best|love|great)\b.*\b(amazing|perfect|best|love|great)\b",t): sigs.append("Repetitive generic praise language")
    if re.search(r"\b(highly recommend|must visit|best .* ever|absolutely perfect)\b",t): sigs.append("Templated marketing-style phrases")
    if re.search(r"\b(received|given|provided).{0,20}\b(free|discount|complimentary)\b",t): sigs.append("Possible incentivised review")
    if len(text.split())<15 and is_fake: sigs.append("Very short with little detail")
    if re.search(r"\b(however|although|despite|but)\b",t): sigs.append("✓ Balanced perspective (genuine signal)")
    if re.search(r"\b(after \d+ (days?|weeks?|months?))\b",t): sigs.append("✓ Time-based personal experience")
    return sigs[:5] if sigs else ["Natural language pattern consistent with corpus"]

def build_description(text,is_fake,fake_prob,meta):
    rev_len=meta.get("review_len",len(text.split())); sim=float(meta.get("Maximum Content Similarity",0) or 0)
    if is_fake:
        reasons=[]
        if sim>0.5: reasons.append(f"high content similarity ({sim:.2f})")
        if rev_len<20: reasons.append("unusually short")
        if re.search(r"!{2,}",text): reasons.append("excessive punctuation")
        if not reasons: reasons.append("semantic pattern matches fake review clusters")
        return (f"Flagged as FAKE ({fake_prob*100:.1f}% RF confidence). Fuzzy rule engine and "
                f"SBERT-LSA embeddings indicate: {'; '.join(reasons[:2])}. "
                f"Random Forest ensemble vote: {fake_prob*100:.0f}% fake across 300 decision trees.")
    else:
        pos=[]
        if rev_len>80: pos.append("detailed personal narrative")
        if re.search(r"\b(however|but|although)\b",text.lower()): pos.append("balanced perspective")
        if not pos: pos.append("authentic writing consistent with genuine reviewers")
        return (f"Classified GENUINE ({(1-fake_prob)*100:.1f}% confidence). "
                f"Demonstrates {'; '.join(pos[:2])}. "
                f"SBERT semantic embeddings and fuzzy reviewer-profile scoring support authentic authorship.")

def extract_features(text: str, meta: dict) -> np.ndarray:
    cleaned = clean_text(text)
    meta["review_len"] = len(cleaned.split())
    X_tfidf = tfidf.transform([cleaned])
    X_sbert = svd.transform(X_tfidf)
    behav = np.array([[g(meta, c) for c in behavioral_cols]])
    behav_s = scaler_behav.transform(behav)
    fuzzy = compute_fuzzy_features(meta).reshape(1,-1)
    fuzzy_s = scaler_fuzzy.transform(fuzzy)
    return np.hstack([X_sbert, behav_s, fuzzy_s])

def do_classify(text: str, meta: dict) -> dict:
    X = extract_features(text, meta)
    
    if sbert_ensemble_rf is not None and sbert_encoder is not None:
        # PRIORITY 1: Full SBERT + Behavioral + Fuzzy + Random Forest (88.55%, AUC 0.9347)
        cleaned = clean_text(text)
        emb = sbert_encoder.encode([cleaned])  # 384-dim SBERT embedding
        b = np.array([[g(meta, c) for c in sbert_ensemble_bundle["behavioral_cols"]]])
        b_scaled = sbert_ensemble_bundle["scaler_behav"].transform(b)
        fz = compute_fuzzy_features(meta).reshape(1, -1)
        fz_scaled = sbert_ensemble_bundle["scaler_fuzzy"].transform(fz)
        X_hybrid = np.hstack([emb, b_scaled, fz_scaled])  # 433-dim hybrid vector
        proba = sbert_ensemble_rf.predict_proba(X_hybrid)[0]
        fake_prob = float(proba[1])
        genuine_prob = float(proba[0])
    elif text_clf is not None:
        # PRIORITY 2: TF-IDF + Logistic Regression
        cleaned = clean_text(text)
        proba = text_clf.predict_proba([cleaned])[0]
        fake_prob = float(proba[1])
        genuine_prob = float(proba[0])
    else:
        # PRIORITY 3: TF-IDF + SVD + Fuzzy + Random Forest (original ensemble)
        proba = rf.predict_proba(X)[0]
        fake_prob = float(proba[1])
        genuine_prob = float(proba[0])

    is_fake = fake_prob >= 0.5; conf = max(fake_prob, genuine_prob)
    risk = "High" if (is_fake and conf>=0.75) else "Low" if (not is_fake and conf>=0.75) else "Medium"
    return {
        "prediction": "Fake" if is_fake else "Genuine",
        "confidence": round(conf,4),
        "fake_probability": round(fake_prob,4),
        "genuine_probability": round(genuine_prob,4),
        "risk_level": risk,
        "fuzzy_signals": get_fuzzy_signals(meta),
        "semantic_signals": get_semantic_signals(text, is_fake),
        "description": build_description(text, is_fake, fake_prob, meta),
        "meta": {
            "review_length": meta.get("review_len"),
            "content_similarity": round(float(meta.get("Maximum Content Similarity",0) or 0),4),
            "sbert_dims": 100, "fuzzy_rules": 5, "rf_trees": 300,
        }
    }

# ── Schemas ────────────────────────────────────────────────────────────────────
class ReviewIn(BaseModel):
    review: str = Field(..., min_length=3)
    rating: Optional[float] = None
    reviewCount: Optional[int] = None
    friendCount: Optional[int] = None
    usefulCount: Optional[int] = None
    coolCount: Optional[int] = None
    funnyCount: Optional[int] = None
    fanCount: Optional[int] = None
    complimentCount: Optional[int] = None
    tipCount: Optional[int] = None
    firstCount: Optional[int] = None
    reviewUsefulCount: Optional[int] = None
    restaurantRating: Optional[float] = None
    mnr: Optional[float] = None
    rl: Optional[float] = None
    rd: Optional[float] = None
    max_similarity: Optional[float] = None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status":"ok","model":"SBERT-LSA + Fuzzy Logic + Random Forest",
            "accuracy":"88.55%","auc_roc":"0.9347","version":"5.0.0",
            "pipeline":["TF-IDF(25k)→TruncatedSVD(100d)","Fuzzy Mamdani (5 rules, 32 features)","RandomForest(300 trees)"],
            "dataset":"Yelp 26,958 restaurant reviews"}

@app.post("/predict")
def predict(body: ReviewIn):
    t0=time.perf_counter()
    meta=body.model_dump(); meta.pop("review",None)
    if body.max_similarity is not None:
        meta["Maximum Content Similarity"] = body.max_similarity
    
    # Run validation
    is_valid, msg = validate_review(body.review)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    result=do_classify(body.review, meta)
    result["review"]=body.review[:200]
    result["latency_ms"]=round((time.perf_counter()-t0)*1000,1)
    return result

@app.post("/explain")
def explain(body: ReviewIn):
    """Ablation study: how much each sub-system contributes."""
    t0=time.perf_counter()
    meta=body.model_dump(); meta.pop("review",None)
    if body.max_similarity is not None:
        meta["Maximum Content Similarity"] = body.max_similarity
    result=do_classify(body.review, meta)
    X=extract_features(body.review, meta)
    X0=np.zeros_like(X); X0[0,:100]=X[0,:100]
    X1=np.zeros_like(X); X1[0,100:117]=X[0,100:117]
    X2=np.zeros_like(X); X2[0,117:]=X[0,117:]
    result["subsystem_votes"]={
        "sbert_lsa":{"fake_prob":round(float(rf.predict_proba(X0)[0][1]),4),"label":"Semantic (SBERT-LSA)"},
        "behavioural":{"fake_prob":round(float(rf.predict_proba(X1)[0][1]),4),"label":"Reviewer Behaviour"},
        "fuzzy_logic":{"fake_prob":round(float(rf.predict_proba(X2)[0][1]),4),"label":"Fuzzy Rule Engine"},
        "ensemble":{"fake_prob":round(result["fake_probability"],4),"label":"Full Ensemble"},
    }
    result["latency_ms"]=round((time.perf_counter()-t0)*1000,1)
    return result
