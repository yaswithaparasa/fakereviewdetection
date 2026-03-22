import os, re, logging, warnings, asyncio, hashlib, unicodedata, time, socket, subprocess, sys
from urllib.parse import urlparse
from typing import List, Optional, Tuple
from collections import Counter
from contextlib import asynccontextmanager
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_PATH     = "rf_model.joblib"
SBERT_NAME     = "all-mpnet-base-v2"
SBERT_DIM      = 768
N_HAND         = 29
FAKE_THRESHOLD = 0.45
MAX_TEXT       = 10_000
MAX_URL_REVIEWS= 30
MIN_REVIEW_LEN = 20
MAX_REVIEW_LEN = 5000

_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
_ua_idx = 0
def _next_ua():
    global _ua_idx; ua = _UA_LIST[_ua_idx % len(_UA_LIST)]; _ua_idx += 1; return ua

_vader = None
def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError: pass
    return _vader

_stopwords = None
def _get_stopwords():
    global _stopwords
    if _stopwords is None:
        try:
            from nltk.corpus import stopwords as sw
            import nltk
            try: _stopwords = set(sw.words("english"))
            except LookupError: nltk.download("stopwords", quiet=True); _stopwords = set(sw.words("english"))
        except ImportError: _stopwords = set()
    return _stopwords

_POS = {"excellent","amazing","outstanding","wonderful","fantastic","superb","incredible","extraordinary",
        "exceptional","magnificent","brilliant","flawless","perfect","spectacular","phenomenal",
        "unbelievable","remarkable","breathtaking","divine","immaculate","impeccable","transcendent",
        "unmatched","incomparable","greatest","finest","best","ultimate","definitive","supreme"}
_NEG = {"terrible","awful","horrible","disgusting","atrocious","dreadful","appalling","abysmal",
        "pathetic","catastrophic","disastrous","unacceptable","deplorable","worst","useless","broken",
        "defective","damaged","fraud","scam","disappointing","mediocre","poor","cheap"}
_SPAM = ["buy now","order now","act fast","limited time","selling fast","while stocks last",
         "do not miss","don't miss","book now","book today","order today","buy today",
         "buy immediately","purchase now","get it now","hurry","click here","limited offer",
         "shop now","deal ends"]
_CRED = ["years of experience","years experience","as a professional","in my career","as an expert",
         "as a certified","as a former","i have a degree","i have a phd","as a doctor","as a chef",
         "as an engineer","as a consultant","as a blogger","in my professional","i rarely write",
         "i never write","i had to write","i created an account","real customer","real review",
         "not a fake","not sponsored","genuine review","not exaggerating","verified purchase"]
_DRAMATIC = {"absolutely","completely","totally","utterly","literally","honestly","seriously",
             "insanely","ridiculously","insane","crazy","unreal","mindblowing","mind-blowing",
             "gamechanging","game-changing","lifechanging","life-changing","unbelievably",
             "extraordinarily","phenomenally"}
FEAT_NAMES = [
    "word_count","char_count","avg_word_length","avg_sentence_length","lexical_diversity",
    "repetition_ratio","exclamation_marks","question_marks","allcaps_words","caps_ratio",
    "positive_keywords","negative_keywords","spam_phrases","credibility_claims","superlatives",
    "first_person_ratio","second_person_ratio","digit_ratio","sentiment_skew","sentence_count",
    "excl_capped","allcaps_capped","pos_capped","html_tags",
    "vader_compound","stopword_ratio","repeated_word_ratio","consec_excl","dramatic_word_ratio"]

_EMOJI_RE = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
                       "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+", re.UNICODE)

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    text = re.sub(r'"+', " ", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_TEXT]

def extract_features(text: str) -> np.ndarray:
    t, words = text.lower(), text.split()
    wl = [w.lower() for w in words]
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    nw, nc, ns = max(len(words),1), max(len(text),1), max(len(sents),1)
    ttr = len(set(wl))/nw; excl = text.count("!"); quest = text.count("?")
    caps_w = sum(1 for w in words if w.isupper() and len(w)>1)
    pos = sum(1 for w in _POS if w in t); neg = sum(1 for w in _NEG if w in t)
    sups = len(re.findall(r"\b(best|greatest|finest|worst|most|least|only|ever|never|always|"
        r"perfect|flawless|ideal|ultimate|definitive|absolute|complete|total|"
        r"entire|every|all|nothing|everything|everyone|nobody|anyone)\b", t))
    fp_r = sum(1 for w in wl if w in {"i","me","my","myself","we","our","us"})/nw
    tp_r = sum(1 for w in wl if w in {"you","your","everyone","everybody","anyone","all"})/nw
    vader = _get_vader(); vc = vader.polarity_scores(text)["compound"] if vader else 0.0
    sw = _get_stopwords(); stop_r = sum(1 for w in wl if w in sw)/nw if sw else 0.0
    wc = Counter(wl); rep_r = sum(1 for _,c in wc.items() if c>=3)/max(len(wc),1)
    return np.array([
        nw, nc, float(np.mean([len(w.rstrip(".,!?;:\"'")) for w in words])), nw/ns,
        ttr, 1.0-ttr, excl, quest, caps_w, sum(1 for c in text if c.isupper())/nc,
        pos, neg, sum(1 for p in _SPAM if p in t), sum(1 for c in _CRED if c in t), sups,
        fp_r, tp_r, sum(1 for c in text if c.isdigit())/nc, pos/max(pos+neg+1,1), ns,
        min(excl,10), min(caps_w,15), min(pos,20), len(re.findall(r"<[^>]+>",text)),
        vc, stop_r, rep_r,
        max((len(m.group()) for m in re.finditer(r"!+",text)), default=0),
        sum(1 for w in wl if w in _DRAMATIC)/nw,
    ], dtype=np.float32)

class SelectiveScaler(BaseEstimator, TransformerMixin):
    def __init__(self): self.scaler_ = MinMaxScaler()
    def fit(self, X, y=None): self.scaler_.fit(X[:, SBERT_DIM:]); return self
    def transform(self, X):
        X_out = X.copy(); X_out[:, SBERT_DIM:] = self.scaler_.transform(X[:, SBERT_DIM:]); return X_out

_HARD_REJECT_RE = re.compile(
    r"(©|\bcopyright\b|all rights reserved|amazon\.com,\s*inc|"
    r"\b(privacy policy|terms of use|terms & conditions|cookie policy)\b)", re.I)
_NOISE_RE = [re.compile(p, re.I) for p in [
    r"^(home|about|contact|faq|help|login|sign\s*in|register|cart|checkout|wishlist)$",
    r"^\d+(\.\d+)?[%\u20b9$\u20ac\xa3\xa5]",
    r"^(read more|see more|show more|load more|view all|show all)$",
    r"^[\d\s\-\+\(\)]+$", r"^[\u2605\u2606\u2713\u2717]+\s*\d*$",
    r"^\d+\s*(star|stars|rating|ratings|out of).*$",
    r"^(sort by|filter|showing|results|page \d+).*$"]]
_SIGNALS = {"good","great","nice","love","hate","buy","bought","purchase","product","quality",
            "price","worth","recommend","happy","satisfied","disappointed","arrived","delivery",
            "shipping","packed","works","broke","returned","refund","seller","item","star",
            "review","verified","customer","order","received","excellent","terrible","amazing",
            "awful","best","worst","fast","slow","easy","hard","perfect","bad","cheap",
            "expensive","material","color","colour","size","fit","comfortable","soft","durable"}

def _is_valid_review(text: str) -> bool:
    t = text.strip()
    if not t or len(t)<MIN_REVIEW_LEN or len(t)>MAX_REVIEW_LEN: return False
    if _HARD_REJECT_RE.search(t): return False
    if any(p.match(t) for p in _NOISE_RE): return False
    tl, words = t.lower(), t.lower().split()
    if len(words)<3: return False
    if sum(1 for c in t if c.isalnum())/max(len(t),1)<0.40: return False
    return any(w in _SIGNALS for w in words) or bool(re.search(r"\b(i|we|my|our|me)\b", tl))

def _resolve_short_url(url: str) -> str:
    parsed = urlparse(url); h = (parsed.hostname or "").lower()
    if not (h in ("amzn.in","amzn.to","amzn.eu","a.co") or
            (re.search(r"amazon\.",h) and re.match(r"^/d/",parsed.path))):
        return url
    for method in ("head", "get"):
        try:
            with httpx.Client(follow_redirects=True, timeout=10, headers={"User-Agent": _next_ua()}) as c:
                resolved = str(getattr(c, method)(url).url)
                log.info("Short URL resolved: %s → %s", url, resolved); return resolved
        except Exception: pass
    return url

def _extract_asin(url: str) -> Optional[str]:
    for pat in [r"/(?:dp|gp/product|product)/([A-Z0-9]{10})",
                r"[?&]asin=([A-Z0-9]{10})", r"/([A-Z0-9]{10})(?:[/?#]|$)"]:
        m = re.search(pat, url)
        if m: return m.group(1)
    return None

def _build_review_url(url: str, page: int = 1) -> Tuple[str, str]:
    parsed = urlparse(url); h = (parsed.hostname or "").lower()
    path = parsed.path; site = re.sub(r"^www\.", "", h)
    if re.search(r"amazon\.", h):
        asin = _extract_asin(url)
        if asin:
            return (f"https://{parsed.hostname}/product-reviews/{asin}"
                    f"?pageNumber={page}&sortBy=recent&reviewerType=all_reviews", site)
    if "flipkart" in h:
        m = re.search(r"/p/([^/?#]+)", path) or re.search(r"/(itm[a-z0-9]+)", path, re.I)
        if m: return f"https://www.flipkart.com/product-reviews/{m.group(1)}?page={page}", "flipkart.com"
    if "walmart" in h:
        m = re.search(r"/ip/(?:[^/]+/)?(\d+)", path)
        if m: return f"https://www.walmart.com/reviews/product/{m.group(1)}?page={page}", "walmart.com"
    if "bestbuy" in h:
        m = re.search(r"/(\d{7,})/", path)
        if m: return f"https://www.bestbuy.com/site/reviews/{m.group(1)}?page={page}", "bestbuy.com"
    if "ebay" in h:
        m = re.search(r"/itm/(?:[^/]+/)?(\d{10,})", path)
        if m: return f"https://www.ebay.com/urw/{m.group(1)}/product-reviews", "ebay.com"
    return url, site

_BLOCK_SIGNALS = ["enter the characters you see below","api-services-support@amazon",
                  "robot check","verify you are a human","access denied",
                  "please enable cookies","checking your browser","cloudflare","ddos-guard"]

def _fetch_page(url: str, retries: int = 3) -> Optional[str]:
    hdrs = {"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language":"en-IN,en-US;q=0.9,en;q=0.8","Accept-Encoding":"gzip, deflate, br",
            "Connection":"keep-alive","Upgrade-Insecure-Requests":"1","DNT":"1"}
    for attempt in range(retries):
        try:
            resp = httpx.Client(follow_redirects=True, timeout=20,
                                headers={**hdrs,"User-Agent":_next_ua()}).get(url)
            if resp.status_code == 200:
                html = resp.text
                if any(s in html.lower() for s in _BLOCK_SIGNALS):
                    log.warning("  Attempt %d: blocked", attempt+1); time.sleep(1.5); continue
                return html
            elif resp.status_code in (429, 503): time.sleep(2); continue
            else: return None
        except Exception as e:
            log.warning("  Attempt %d error: %s", attempt+1, e); time.sleep(1)
    return None

def _parse_amazon(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml"); reviews = []
    for el in soup.select("[data-hook='review-body']"):
        span = el.find("span", recursive=False) or el.find("span") or el
        t = re.sub(r"\s+", " ", span.get_text(" ", strip=True)).strip()
        if _is_valid_review(t): reviews.append(t)
    if not reviews:
        for el in soup.select(".review-text-content span, .review-text-content"):
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if _is_valid_review(t): reviews.append(t)
    return reviews

def _parse_flipkart(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml"); reviews = []
    for sel in [".ZmyHeo","._6K-7Co",".t-ZTKy","._2-N8zT",".DByaG","[class*='review']"]:
        for el in soup.select(sel):
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if _is_valid_review(t): reviews.append(t)
        if reviews: break
    return reviews

def _parse_generic(html: str, hostname: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml"); reviews = []
    for sel in ["[itemprop='reviewBody']","[itemprop='description']",".review-text",
                ".review-body",".review-content",".reviewText",".reviewBody",
                "[data-testid='review-body']","[class*='review-text']","[class*='reviewText']",
                "[class*='review-body']","[class*='reviewBody']","[class*='review-content']",
                "[class*='comment-body']","[class*='feedback-content']"]:
        for el in soup.select(sel):
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if _is_valid_review(t): reviews.append(t)
        if reviews: break
    if not reviews:
        for container in soup.select("[class*='review'],[class*='Review'],[id*='review'],[id*='Review']"):
            for p in container.find_all("p"):
                t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
                if _is_valid_review(t): reviews.append(t)
    return reviews

def _dedup(reviews: List[str]) -> List[str]:
    seen, out = set(), []
    for r in reviews:
        h = hashlib.md5(r.encode()).hexdigest()
        if h not in seen: seen.add(h); out.append(r)
    return out

async def scrape_reviews(url: str) -> Tuple[List[str], str]:
    def _sync(url: str) -> Tuple[List[str], str]:
        url = _resolve_short_url(url)
        h = (urlparse(url).hostname or "").lower()
        all_reviews: List[str] = []
        for pg in range(1, 5):
            review_url, site_name = _build_review_url(url, page=pg)
            log.info("  Fetching page %d: %s", pg, review_url)
            html = _fetch_page(review_url)
            if not html: break
            found = _dedup(_parse_amazon(html) if re.search(r"amazon\.",h)
                           else _parse_flipkart(html) if "flipkart" in h
                           else _parse_generic(html, h))
            log.info("  Page %d → %d reviews", pg, len(found))
            all_reviews.extend(found)
            if not found or len(all_reviews) >= MAX_URL_REVIEWS: break
            time.sleep(0.8)
        all_reviews = _dedup(all_reviews)
        log.info("Total scraped: %d from %s", len(all_reviews), site_name)
        return all_reviews[:MAX_URL_REVIEWS], site_name
    return await asyncio.get_event_loop().run_in_executor(None, _sync, url)

sbert: SentenceTransformer = None
model = None

def _load_models():
    global sbert, model
    log.info("Loading SBERT '%s' …", SBERT_NAME)
    try: sbert = SentenceTransformer(SBERT_NAME, local_files_only=True); log.info("SBERT loaded from cache.")
    except Exception: sbert = SentenceTransformer(SBERT_NAME); log.info("SBERT ready.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"'{MODEL_PATH}' not found. Run: python train.py")
    model = joblib.load(MODEL_PATH)
    log.info("Model loaded: %s", type(model).__name__)

def _build_X(texts: list) -> np.ndarray:
    emb = np.asarray(sbert.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    return np.hstack([emb, np.vstack([extract_features(t) for t in texts])])

def _classify_one(text: str, proba: np.ndarray):
    gp, fp = float(proba[0]), float(proba[1])
    pred = "Fake" if fp >= FAKE_THRESHOLD else "Genuine"
    return (pred, round(max(gp,fp),4), round(fp,4), round(gp,4),
            {k: round(float(v),4) for k,v in zip(FEAT_NAMES, extract_features(text).tolist())},
            _signals(text), _build_description(text, pred, fp, gp, []))

def _classify(text: str):
    c = clean_text(text); return _classify_one(c, model.predict_proba(_build_X([c]))[0])

def _classify_batch(texts: list) -> list:
    cleaned = [clean_text(t) for t in texts]; X = _build_X(cleaned)
    return [_classify_one(c,p) for c,p in zip(cleaned, model.predict_proba(X))]

def _risk(fp: float) -> str:
    return "High" if fp>=0.75 else "Medium" if fp>=0.45 else "Low"

def _signals(text: str) -> List[str]:
    t, words, out = text.lower(), text.split(), []
    ex = text.count("!"); cw = sum(1 for w in words if w.isupper() and len(w)>1)
    if ex>=3: out.append(f"{ex} exclamation marks — high emotional intensity")
    if cw>=2: out.append(f"{cw} ALL-CAPS word(s) — urgency / spam signal")
    if sum(1 for w in _POS if w in t)>=4: out.append(f"{sum(1 for w in _POS if w in t)} extreme positive keywords — possible exaggeration")
    if sum(1 for p in _SPAM if p in t): out.append(f"{sum(1 for p in _SPAM if p in t)} spam / urgency phrase(s) detected")
    if sum(1 for c in _CRED if c in t): out.append("Unsolicited credibility claim — common deception tactic")
    div = len(set(w.lower() for w in words))/max(len(words),1)
    if div<0.55: out.append(f"Low lexical diversity ({div:.0%}) — repetitive language")
    sups = len(re.findall(r"\b(best|greatest|finest|perfect|flawless|never|always|everyone|everything)\b",t))
    if sups>=5: out.append(f"{sups} superlative / absolute terms — over-claiming pattern")
    return out or ["No strong deception signals detected"]

def _build_description(text: str, pred: str, fp: float, gp: float, _) -> str:
    words, nw, t = text.split(), len(text.split()), text.lower()
    if pred == "Fake":
        r = []
        ex = text.count("!"); cw = sum(1 for w in words if w.isupper() and len(w)>1)
        ph = sum(1 for w in _POS if w in t); sp = sum(1 for p in _SPAM if p in t)
        cr = sum(1 for c in _CRED if c in t); div = len(set(w.lower() for w in words))/max(nw,1)
        if ex>=3: r.append(f"excessive punctuation ({ex} exclamation marks)")
        if cw>=2: r.append(f"ALL-CAPS language ({cw} words)")
        if ph>=3: r.append(f"high concentration of superlative praise ({ph} terms)")
        if sp: r.append(f"commercial spam phrases ({sp} found)")
        if cr: r.append("unprompted credibility claims")
        if div<0.55: r.append(f"low lexical diversity ({div:.0%})")
        if not r: r.append("an unnatural writing pattern detected by semantic analysis")
        rs = ", ".join(r[:-1]) + (f", and {r[-1]}" if len(r)>1 else r[0])
        return (f"This review is classified as FAKE with {fp*100:.1f}% probability. "
                f"The model identified {rs}. These patterns are statistically associated with "
                f"inauthentic, incentivised, or bot-generated reviews.")
    else:
        s = []; div = len(set(w.lower() for w in words))/max(nw,1)
        if div>=0.65: s.append("rich, natural vocabulary")
        if text.count("!")<=1: s.append("measured, calm tone")
        if nw>=30: s.append("sufficient detail")
        if sum(1 for w in _NEG if w in t)>=1: s.append("balanced perspective including negatives")
        if not s: s.append("natural language patterns consistent with authentic experience")
        ss = ", ".join(s[:-1]) + (f", and {s[-1]}" if len(s)>1 else s[0])
        return (f"This review appears GENUINE with {gp*100:.1f}% confidence. "
                f"The model found {ss}. No significant deception signals were detected.")

def _url_verdict(avg_fp: float, fk: int, total: int) -> Tuple[str, str]:
    if total == 0: return "Inconclusive", "No reviews could be scraped from this URL."
    pct = avg_fp*100
    if avg_fp>=0.60: return "Highly Suspicious", (f"{fk}/{total} reviews classified as fake (avg {pct:.1f}%). Exercise extreme caution.")
    if avg_fp>=0.40: return "Moderately Suspicious", (f"{fk}/{total} reviews suspicious (avg {pct:.1f}%). Cross-reference before purchasing.")
    if avg_fp>=0.25: return "Mostly Trustworthy", (f"{fk}/{total} flagged (avg {pct:.1f}%). Majority appear genuine.")
    return "Trustworthy", (f"Only {fk}/{total} flagged (avg {pct:.1f}%). Review section appears largely authentic.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models(); yield

app = FastAPI(title="ReviewGuard AI", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ReviewIn(BaseModel):  review: str
class BatchItem(BaseModel): review: str
class ReviewOut(BaseModel):
    prediction: str; confidence: float; genuine_probability: float; fake_probability: float
    risk_level: str; features: dict; signals: List[str]; description: str
class BatchOut(BaseModel):
    total: int; fake: int; genuine: int; results: List[dict]
class UrlIn(BaseModel): url: str
class UrlAnalysisOut(BaseModel):
    url: str; site: str; total_scraped: int; fake: int; genuine: int; fake_rate: float
    overall_verdict: str; verdict_description: str; results: List[dict]

@app.get("/")
def root(): return {"service":"ReviewGuard AI","version":"5.0.0","status":"running"}

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/model-info")
def model_info():
    return {"sbert_model":SBERT_NAME,"embedding_dim":SBERT_DIM,"hand_features":N_HAND,
            "total_dims":SBERT_DIM+N_HAND,"fake_threshold":FAKE_THRESHOLD,
            "classifier":type(model).__name__,"scraper":"httpx + BeautifulSoup"}

@app.post("/predict", response_model=ReviewOut)
async def predict(body: ReviewIn):
    text = body.review.strip()
    if len(text)<10: raise HTTPException(400, "Review must be at least 10 characters.")
    if len(text)>MAX_TEXT: raise HTTPException(400, f"Review must not exceed {MAX_TEXT} characters.")
    pred,conf,fp,gp,feats,sigs,desc = _classify(text)
    return ReviewOut(prediction=pred,confidence=conf,genuine_probability=gp,fake_probability=fp,
                     risk_level=_risk(fp),features=feats,signals=sigs,description=desc)

@app.post("/batch-predict", response_model=BatchOut)
async def batch_predict(items: List[BatchItem]):
    if len(items)>30: raise HTTPException(400, "Maximum 30 reviews per batch.")
    valid = [i.review.strip() for i in items if len(i.review.strip())>=10]
    if not valid: raise HTTPException(422, "All reviews were too short.")
    results = [{"review":t[:100]+("…" if len(t)>100 else ""),"prediction":pred,
                "confidence":conf,"fake_probability":fp,"risk_level":_risk(fp),"description":desc}
               for t,(pred,conf,fp,gp,_,sigs,desc) in zip(valid,
               await asyncio.get_event_loop().run_in_executor(None,_classify_batch,valid))]
    fk = sum(1 for r in results if r["prediction"]=="Fake")
    return BatchOut(total=len(results),fake=fk,genuine=len(results)-fk,results=results)

@app.post("/analyze-url", response_model=UrlAnalysisOut)
async def analyze_url(body: UrlIn):
    url = body.url.strip()
    if not url.startswith(("http://","https://")): raise HTTPException(400,"URL must start with http:// or https://")
    raw, site_name = await scrape_reviews(url)
    if not raw:
        raise HTTPException(422, f"No reviews extracted from {site_name}. Causes: (1) login required "
            "(2) anti-bot protection — retry in seconds (3) no reviews yet "
            "(4) not a product page. For Amazon use: https://www.amazon.in/dp/XXXXXXXXXX")
    valid = [t.strip() for t in raw if len(t.strip())>=MIN_REVIEW_LEN]
    if not valid: raise HTTPException(422,"Scraped content had no readable reviews.")
    log.info("Classifying %d reviews from %s…", len(valid), site_name)
    batch = await asyncio.get_event_loop().run_in_executor(None, _classify_batch, valid)
    results = [{"review":t[:200]+("…" if len(t)>200 else ""),"full_review":t,
                "prediction":pred,"confidence":conf,"fake_probability":fp,
                "genuine_probability":gp,"risk_level":_risk(fp),"signals":sigs,"description":desc}
               for t,(pred,conf,fp,gp,_,sigs,desc) in zip(valid,batch)]
    fk = sum(1 for r in results if r["prediction"]=="Fake")
    total = len(results)
    avg_fp = round(sum(r["fake_probability"] for r in results)/max(total,1),4)
    verdict, vdesc = _url_verdict(avg_fp, fk, total)
    return UrlAnalysisOut(url=url,site=site_name,total_scraped=total,fake=fk,genuine=total-fk,
                          fake_rate=avg_fp,overall_verdict=verdict,verdict_description=vdesc,results=results)

if __name__ == "__main__":
    import uvicorn
    PORT = 8000
    def _port_in_use(p): 
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: return s.connect_ex(("127.0.0.1",p))==0
    if _port_in_use(PORT):
        log.warning("Port %d in use — killing existing process…", PORT)
        try:
            if sys.platform=="win32":
                for line in subprocess.check_output(f"netstat -ano | findstr :{PORT}",shell=True).decode().splitlines():
                    parts=line.strip().split()
                    if parts and parts[-1].isdigit():
                        subprocess.call(f"taskkill /F /PID {parts[-1]}",shell=True); break
            else: subprocess.call(f"fuser -k {PORT}/tcp",shell=True)
        except Exception as e:
            log.warning("Could not kill port %d: %s — close it manually and retry.",PORT,e); sys.exit(1)
        time.sleep(1.5)
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False, log_level="info", workers=1)
