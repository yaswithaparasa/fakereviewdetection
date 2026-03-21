"""
ReviewGuard AI — FastAPI Backend v5.0
Scraper: httpx + BeautifulSoup (NO Playwright — faster, more reliable)
Supports: Amazon (.in/.com/.co.uk etc.), Flipkart, Walmart, BestBuy, eBay + generic
"""

import os, re, logging, warnings, asyncio, json, hashlib, unicodedata, time
from urllib.parse import urlparse
from typing import List, Optional, Tuple
from collections import Counter
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = "rf_model.joblib"
SBERT_NAME      = "all-mpnet-base-v2"
SBERT_DIM       = 768
N_HAND          = 29
FAKE_THRESHOLD  = 0.40
MAX_TEXT        = 10_000
MAX_URL_REVIEWS = 30        # 30 reviews per URL — fast enough, representative enough
MIN_REVIEW_LEN  = 20
MAX_REVIEW_LEN  = 5000

# Browser-like headers — rotate User-Agents per request to reduce blocking
_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
_ua_idx = 0
def _next_ua() -> str:
    global _ua_idx
    ua = _UA_LIST[_ua_idx % len(_UA_LIST)]
    _ua_idx += 1
    return ua

# ── VADER (lazy) ──────────────────────────────────────────────────────────────
_vader = None
def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            pass
    return _vader

# ── Stopwords (lazy) ──────────────────────────────────────────────────────────
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
            _stopwords = set()
    return _stopwords


# ══════════════════════════════════════════════════════════════════════════════
#  HAND-CRAFTED FEATURES  (29 dims — must match train.py exactly)
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
    "vader_compound","stopword_ratio","repeated_word_ratio",
    "consec_excl","dramatic_word_ratio",
]

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
    return text[:MAX_TEXT]


def extract_features(text: str) -> np.ndarray:
    t       = text.lower()
    words   = text.split()
    words_l = [w.lower() for w in words]
    sents   = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    nw      = max(len(words), 1)
    nc      = max(len(text),  1)
    ns      = max(len(sents), 1)
    avg_wl  = float(np.mean([len(w.rstrip(".,!?;:\"'")) for w in words]))
    avg_sl  = nw / ns
    ttr     = len(set(words_l)) / nw
    rep     = 1.0 - ttr
    excl    = text.count("!")
    quest   = text.count("?")
    caps_w  = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_r  = sum(1 for c in text if c.isupper()) / nc
    pos     = sum(1 for w in _POS  if w in t)
    neg     = sum(1 for w in _NEG  if w in t)
    spam    = sum(1 for p in _SPAM if p in t)
    cred    = sum(1 for c in _CRED if c in t)
    sups    = len(re.findall(
        r"\b(best|greatest|finest|worst|most|least|only|ever|never|always|"
        r"perfect|flawless|ideal|ultimate|definitive|absolute|complete|total|"
        r"entire|every|all|nothing|everything|everyone|nobody|anyone)\b", t))
    fp_r    = sum(1 for w in words_l if w in {"i","me","my","myself","we","our","us"}) / nw
    tp_r    = sum(1 for w in words_l if w in {"you","your","everyone","everybody","anyone","all"}) / nw
    dig_r   = sum(1 for c in text if c.isdigit()) / nc
    skew    = pos / max(pos + neg + 1, 1)
    html    = len(re.findall(r"<[^>]+>", text))
    vader   = _get_vader()
    vader_c = vader.polarity_scores(text)["compound"] if vader else 0.0
    sw      = _get_stopwords()
    stop_r  = sum(1 for w in words_l if w in sw) / nw if sw else 0.0
    wc      = Counter(words_l)
    rep_r   = sum(1 for w, c in wc.items() if c >= 3) / max(len(wc), 1)
    c_excl  = max((len(m.group()) for m in re.finditer(r"!+", text)), default=0)
    dram_r  = sum(1 for w in words_l if w in _DRAMATIC_WORDS) / nw
    return np.array([
        nw, nc, avg_wl, avg_sl, ttr, rep,
        excl, quest, caps_w, caps_r,
        pos, neg, spam, cred, sups,
        fp_r, tp_r, dig_r, skew, ns,
        min(excl, 10), min(caps_w, 15), min(pos, 20), html,
        vader_c, stop_r, rep_r, c_excl, dram_r,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SelectiveScaler — MUST be defined before joblib.load
# ══════════════════════════════════════════════════════════════════════════════

class SelectiveScaler(BaseEstimator, TransformerMixin):
    """MinMaxScaler on hand-crafted dims only; SBERT embeddings left untouched."""
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
#  SCRAPER  — httpx + BeautifulSoup (NO Playwright)
#  Strategy:
#   1. Resolve short links (amzn.in/d/XXXX)
#   2. Extract ASIN / product ID
#   3. Build direct review-page URL
#   4. Fetch with httpx (retry up to 3 times with different User-Agents)
#   5. Parse with site-specific BeautifulSoup selectors
#   6. Paginate up to 4 pages
# ══════════════════════════════════════════════════════════════════════════════

# Hard-reject: legal / copyright / nav text
_HARD_REJECT_RE = re.compile(
    r"(©|\bcopyright\b|all rights reserved|amazon\.com,\s*inc|"
    r"\b(privacy policy|terms of use|terms & conditions|cookie policy)\b)",
    re.I,
)

_NOISE_RE = [re.compile(p, re.I) for p in [
    r"^(home|about|contact|faq|help|login|sign\s*in|register|cart|checkout|wishlist)$",
    r"^\d+(\.\d+)?[%\u20b9$\u20ac\xa3\xa5]",
    r"^(read more|see more|show more|load more|view all|show all)$",
    r"^[\d\s\-\+\(\)]+$",
    r"^[\u2605\u2606\u2713\u2717]+\s*\d*$",
    r"^\d+\s*(star|stars|rating|ratings|out of).*$",
    r"^(sort by|filter|showing|results|page \d+).*$",
]]

_REVIEW_SIGNALS = {
    "good","great","nice","love","hate","buy","bought","purchase","product",
    "quality","price","worth","recommend","happy","satisfied","disappointed",
    "arrived","delivery","shipping","packed","works","broke","returned",
    "refund","seller","item","star","review","verified","customer",
    "order","received","excellent","terrible","amazing","awful","best","worst",
    "fast","slow","easy","hard","perfect","bad","cheap","expensive",
    "material","color","colour","size","fit","comfortable","soft","durable",
}


def _is_valid_review(text: str) -> bool:
    """Return True if text looks like a real review."""
    t = text.strip()
    if not t or len(t) < MIN_REVIEW_LEN or len(t) > MAX_REVIEW_LEN:
        return False
    if _HARD_REJECT_RE.search(t):
        return False
    for pat in _NOISE_RE:
        if pat.match(t):
            return False
    tl    = t.lower()
    words = tl.split()
    if len(words) < 3:
        return False
    alnum = sum(1 for c in t if c.isalnum())
    if alnum / max(len(t), 1) < 0.40:
        return False
    # Must have at least 1 review signal word OR first-person pronoun
    has_signal = any(w in _REVIEW_SIGNALS for w in words)
    has_fp     = bool(re.search(r"\b(i|we|my|our|me)\b", tl))
    return has_signal or has_fp


def _resolve_short_url(url: str) -> str:
    """Follow redirects on short links like amzn.in/d/XXXX."""
    parsed   = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    is_short = (
        hostname in ("amzn.in", "amzn.to", "amzn.eu", "a.co") or
        (re.search(r"amazon\.", hostname) and re.match(r"^/d/", parsed.path))
    )
    if not is_short:
        return url
    try:
        with httpx.Client(follow_redirects=True, timeout=10,
                          headers={"User-Agent": _next_ua()}) as c:
            r = c.head(url)
            resolved = str(r.url)
            log.info("Short URL resolved: %s → %s", url, resolved)
            return resolved
    except Exception as e:
        log.warning("Short URL resolution failed (%s): %s", url, e)
        # Try GET if HEAD fails
        try:
            with httpx.Client(follow_redirects=True, timeout=10,
                              headers={"User-Agent": _next_ua()}) as c:
                r = c.get(url)
                return str(r.url)
        except Exception:
            return url


def _extract_asin(url: str) -> Optional[str]:
    """Extract 10-char ASIN from any Amazon URL."""
    for pat in [
        r"/(?:dp|gp/product|product)/([A-Z0-9]{10})",
        r"[?&]asin=([A-Z0-9]{10})",
        r"/([A-Z0-9]{10})(?:[/?#]|$)",
    ]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _build_review_url(url: str, page: int = 1) -> Tuple[str, str]:
    """
    Returns (review_page_url, site_name).
    Builds the most direct URL to the reviews page for each site.
    """
    parsed   = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path     = parsed.path
    site     = re.sub(r"^www\.", "", hostname)

    # Amazon (all domains: .in, .com, .co.uk, .de, .jp, etc.)
    if re.search(r"amazon\.", hostname):
        asin = _extract_asin(url)
        if asin:
            base = f"https://{parsed.hostname}"
            return (
                f"{base}/product-reviews/{asin}"
                f"?pageNumber={page}&sortBy=recent&reviewerType=all_reviews",
                site
            )

    # Flipkart
    if "flipkart" in hostname:
        pid = re.search(r"/p/([^/?#]+)", path)
        if pid:
            return (
                f"https://www.flipkart.com/product-reviews/{pid.group(1)}?page={page}",
                "flipkart.com"
            )
        itm = re.search(r"/(itm[a-z0-9]+)", path, re.I)
        if itm:
            return (
                f"https://www.flipkart.com/product-reviews/{itm.group(1)}?page={page}",
                "flipkart.com"
            )

    # Walmart
    if "walmart" in hostname:
        item = re.search(r"/ip/(?:[^/]+/)?(\d+)", path)
        if item:
            return (
                f"https://www.walmart.com/reviews/product/{item.group(1)}?page={page}",
                "walmart.com"
            )

    # BestBuy
    if "bestbuy" in hostname:
        sku = re.search(r"/(\d{7,})/", path)
        if sku:
            return (
                f"https://www.bestbuy.com/site/reviews/{sku.group(1)}?page={page}",
                "bestbuy.com"
            )

    # eBay
    if "ebay" in hostname:
        item_id = re.search(r"/itm/(?:[^/]+/)?(\d{10,})", path)
        if item_id:
            return (
                f"https://www.ebay.com/urw/{item_id.group(1)}/product-reviews",
                "ebay.com"
            )

    # Generic fallback — use URL as-is
    return url, site


def _fetch_page(url: str, retries: int = 3) -> Optional[str]:
    """
    Fetch a URL with httpx, rotating User-Agents on retry.
    Returns HTML string or None if all attempts fail.
    """
    headers_base = {
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT":             "1",
    }

    for attempt in range(retries):
        try:
            headers = {**headers_base, "User-Agent": _next_ua()}
            with httpx.Client(
                follow_redirects=True,
                timeout=20,
                headers=headers,
            ) as client:
                resp = client.get(url)

                if resp.status_code == 200:
                    html = resp.text
                    # Detect soft-blocks
                    if _is_blocked(html):
                        log.warning("  Attempt %d: soft-block detected", attempt + 1)
                        time.sleep(1.5)
                        continue
                    return html

                elif resp.status_code in (429, 503):
                    log.warning("  Attempt %d: HTTP %d — rate limited", attempt + 1, resp.status_code)
                    time.sleep(2)
                    continue

                else:
                    log.warning("  Attempt %d: HTTP %d", attempt + 1, resp.status_code)
                    return None

        except Exception as e:
            log.warning("  Attempt %d fetch error: %s", attempt + 1, e)
            time.sleep(1)

    return None


def _is_blocked(html: str) -> bool:
    """Detect CAPTCHA / robot-check pages."""
    lower = html.lower()
    signals = [
        "enter the characters you see below",
        "type the characters you see in this image",
        "api-services-support@amazon",
        "robot check",
        "verify you are a human",
        "access denied",
        "please enable cookies",
        "checking your browser",
        "cloudflare",
        "ddos-guard",
    ]
    return any(s in lower for s in signals)


def _parse_amazon(html: str) -> List[str]:
    """Extract reviews from Amazon product-reviews page."""
    soup    = BeautifulSoup(html, "lxml")
    reviews = []

    # Primary: data-hook="review-body"
    for el in soup.select("[data-hook='review-body']"):
        # The actual text is in the inner <span>
        span = el.find("span", recursive=False) or el.find("span") or el
        text = re.sub(r"\s+", " ", span.get_text(" ", strip=True)).strip()
        if _is_valid_review(text):
            reviews.append(text)

    # Fallback: .review-text-content
    if not reviews:
        for el in soup.select(".review-text-content span, .review-text-content"):
            text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if _is_valid_review(text):
                reviews.append(text)

    return reviews


def _parse_flipkart(html: str) -> List[str]:
    """Extract reviews from Flipkart product-reviews page."""
    soup    = BeautifulSoup(html, "lxml")
    reviews = []

    # Flipkart uses obfuscated class names — try multiple known patterns
    selectors = [
        ".ZmyHeo",           # v3 review body
        "._6K-7Co",          # v3 alternate
        ".t-ZTKy",           # v2
        "._2-N8zT",          # older
        ".DByaG",            # mobile
        "[class*='review']", # generic fallback
    ]
    for sel in selectors:
        for el in soup.select(sel):
            text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if _is_valid_review(text):
                reviews.append(text)
        if reviews:
            break

    return reviews


def _parse_generic(html: str, hostname: str) -> List[str]:
    """Generic review extractor for Walmart, BestBuy, eBay, and unknown sites."""
    soup    = BeautifulSoup(html, "lxml")
    reviews = []

    # Priority order: schema.org → named selectors → class fragments
    for sel in [
        "[itemprop='reviewBody']",
        "[itemprop='description']",
        ".review-text",
        ".review-body",
        ".review-content",
        ".reviewText",
        ".reviewBody",
        "[data-testid='review-body']",
        "[class*='review-text']",
        "[class*='reviewText']",
        "[class*='review-body']",
        "[class*='reviewBody']",
        "[class*='review-content']",
        "[class*='comment-body']",
        "[class*='feedback-content']",
    ]:
        for el in soup.select(sel):
            text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if _is_valid_review(text):
                reviews.append(text)
        if reviews:
            break

    # Last resort: <p> tags inside review containers
    if not reviews:
        for container in soup.select(
            "[class*='review'], [class*='Review'], [id*='review'], [id*='Review']"
        ):
            for p in container.find_all("p"):
                text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
                if _is_valid_review(text):
                    reviews.append(text)

    return reviews


def _deduplicate(reviews: List[str]) -> List[str]:
    seen, out = set(), []
    for r in reviews:
        h = hashlib.md5(r.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


async def scrape_reviews(url: str) -> Tuple[List[str], str]:
    """
    Main scrape entry point.
    Runs synchronous httpx fetching in a thread pool so FastAPI stays async.
    """
    def _sync_scrape(url: str) -> Tuple[List[str], str]:
        # Step 1: resolve short links
        url = _resolve_short_url(url)
        log.info("Scraping URL: %s", url)

        parsed   = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        all_reviews: List[str] = []

        for page_num in range(1, 5):   # up to 4 pages
            review_url, site_name = _build_review_url(url, page=page_num)
            log.info("  Fetching page %d: %s", page_num, review_url)

            html = _fetch_page(review_url)
            if not html:
                log.warning("  Page %d: fetch returned nothing", page_num)
                break

            # Site-specific parsers
            if re.search(r"amazon\.", hostname):
                found = _parse_amazon(html)
            elif "flipkart" in hostname:
                found = _parse_flipkart(html)
            else:
                found = _parse_generic(html, hostname)

            found = _deduplicate(found)
            log.info("  Page %d → %d reviews found", page_num, len(found))
            all_reviews.extend(found)

            if not found:
                break   # no reviews on this page — stop paginating
            if len(all_reviews) >= MAX_URL_REVIEWS:
                break

            time.sleep(0.8)  # polite delay between pages

        all_reviews = _deduplicate(all_reviews)
        log.info("Total scraped: %d from %s", len(all_reviews), site_name)
        return all_reviews[:MAX_URL_REVIEWS], site_name

    # Run blocking httpx calls in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_scrape, url)


# ══════════════════════════════════════════════════════════════════════════════
#  SBERT + MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

log.info("Loading SBERT '%s' …", SBERT_NAME)
try:
    sbert = SentenceTransformer(SBERT_NAME, local_files_only=True)
    log.info("SBERT loaded from local cache.")
except Exception:
    log.info("Downloading SBERT model from HuggingFace…")
    sbert = SentenceTransformer(SBERT_NAME)
    log.info("SBERT ready.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n\n  ❌  '{MODEL_PATH}' not found.\n"
        f"  Run:  python train.py\n"
    )

log.info("Loading RF model from '%s' …", MODEL_PATH)
model = joblib.load(MODEL_PATH)
log.info("Model loaded: %s", type(model).__name__)


def _build_X(texts: list) -> np.ndarray:
    emb  = sbert.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    emb  = np.asarray(emb, dtype=np.float32)
    hand = np.vstack([extract_features(t) for t in texts])
    return np.hstack([emb, hand])


def _classify_one(text: str, proba: np.ndarray):
    gp, fp = float(proba[0]), float(proba[1])
    pred   = "Fake" if fp >= FAKE_THRESHOLD else "Genuine"
    conf   = max(gp, fp)
    feats  = {k: round(float(v), 4) for k, v in zip(FEAT_NAMES, extract_features(text).tolist())}
    sigs   = _signals(text)
    desc   = _build_description(text, pred, fp, gp, sigs)
    return pred, round(conf, 4), round(fp, 4), round(gp, 4), feats, sigs, desc


def _classify(text: str):
    cleaned = clean_text(text)
    X       = _build_X([cleaned])
    proba   = model.predict_proba(X)[0]
    return _classify_one(cleaned, proba)


def _classify_batch(texts: list) -> list:
    """Classify all reviews in ONE SBERT call + ONE RF call."""
    cleaned = [clean_text(t) for t in texts]
    X       = _build_X(cleaned)
    probas  = model.predict_proba(X)
    return [_classify_one(c, p) for c, p in zip(cleaned, probas)]


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ReviewGuard AI",
    version="5.0.0",
    description="Fake review detection — SBERT + Random Forest. httpx scraper.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewIn(BaseModel):   review: str
class BatchItem(BaseModel):  review: str

class ReviewOut(BaseModel):
    prediction:          str
    confidence:          float
    genuine_probability: float
    fake_probability:    float
    risk_level:          str
    features:            dict
    signals:             List[str]
    description:         str

class BatchOut(BaseModel):
    total:   int
    fake:    int
    genuine: int
    results: List[dict]

class UrlIn(BaseModel):      url: str

class UrlAnalysisOut(BaseModel):
    url:                 str
    site:                str
    total_scraped:       int
    fake:                int
    genuine:             int
    fake_rate:           float
    overall_verdict:     str
    verdict_description: str
    results:             List[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk(fp: float) -> str:
    if fp >= 0.75: return "High"
    if fp >= 0.45: return "Medium"
    return "Low"


def _signals(text: str) -> List[str]:
    t, words, out = text.lower(), text.split(), []
    ex = text.count("!")
    if ex >= 3: out.append(f"{ex} exclamation marks — high emotional intensity")
    cw = sum(1 for w in words if w.isupper() and len(w) > 1)
    if cw >= 2: out.append(f"{cw} ALL-CAPS word(s) — urgency / spam signal")
    ph = sum(1 for w in _POS if w in t)
    if ph >= 4: out.append(f"{ph} extreme positive keywords — possible exaggeration")
    sp = sum(1 for p in _SPAM if p in t)
    if sp: out.append(f"{sp} spam / urgency phrase(s) detected")
    cr = sum(1 for c in _CRED if c in t)
    if cr: out.append("Unsolicited credibility claim — common deception tactic")
    div = len(set(w.lower() for w in words)) / max(len(words), 1)
    if div < 0.55: out.append(f"Low lexical diversity ({div:.0%}) — repetitive language")
    sups = len(re.findall(r"\b(best|greatest|finest|perfect|flawless|never|always|everyone|everything)\b", t))
    if sups >= 5: out.append(f"{sups} superlative / absolute terms — over-claiming pattern")
    return out or ["No strong deception signals detected"]


def _build_description(text: str, pred: str, fp: float, gp: float, signals: List[str]) -> str:
    words, nw, t = text.split(), len(text.split()), text.lower()
    if pred == "Fake":
        reasons = []
        ex = text.count("!")
        if ex >= 3:  reasons.append(f"excessive punctuation ({ex} exclamation marks)")
        cw = sum(1 for w in words if w.isupper() and len(w) > 1)
        if cw >= 2:  reasons.append(f"ALL-CAPS language ({cw} words)")
        ph = sum(1 for w in _POS if w in t)
        if ph >= 3:  reasons.append(f"high concentration of superlative praise ({ph} extreme positive terms)")
        sp = sum(1 for p in _SPAM if p in t)
        if sp:       reasons.append(f"commercial spam phrases ({sp} found)")
        cr = sum(1 for c in _CRED if c in t)
        if cr:       reasons.append("unprompted credibility claims (a common fake-review tactic)")
        div = len(set(w.lower() for w in words)) / max(nw, 1)
        if div < 0.55: reasons.append(f"low lexical diversity ({div:.0%}) suggesting templated text")
        if not reasons: reasons.append("an unnatural writing pattern detected by semantic analysis")
        rs = ", ".join(reasons[:-1]) + (f", and {reasons[-1]}" if len(reasons) > 1 else reasons[0])
        return (f"This review is classified as FAKE with {fp*100:.1f}% probability. "
                f"The model identified {rs}. "
                f"These patterns are statistically associated with inauthentic, incentivised, "
                f"or bot-generated reviews rather than genuine customer experiences.")
    else:
        strengths = []
        div = len(set(w.lower() for w in words)) / max(nw, 1)
        if div >= 0.65: strengths.append("rich, natural vocabulary")
        if text.count("!") <= 1: strengths.append("measured, calm tone")
        if nw >= 30:  strengths.append("sufficient detail")
        neg = sum(1 for w in _NEG if w in t)
        if neg >= 1:  strengths.append("balanced perspective including negatives")
        if not strengths: strengths.append("natural language patterns consistent with authentic experience")
        ss = ", ".join(strengths[:-1]) + (f", and {strengths[-1]}" if len(strengths) > 1 else strengths[0])
        return (f"This review appears GENUINE with {gp*100:.1f}% confidence. "
                f"The model found {ss}. "
                f"No significant deception signals were detected, and the semantic "
                f"embedding matches authentic review patterns in the training data.")


def _url_verdict(fake: int, total: int) -> Tuple[str, str]:
    if total == 0:
        return "Inconclusive", "No reviews could be scraped from this URL."
    rate = fake / total
    if rate >= 0.60:
        return "Highly Suspicious", (
            f"{fake}/{total} scraped reviews ({rate*100:.0f}%) were classified as fake. "
            f"This product has an abnormally high proportion of inauthentic reviews. "
            f"Exercise extreme caution — ratings may be artificially inflated.")
    elif rate >= 0.35:
        return "Moderately Suspicious", (
            f"{fake}/{total} scraped reviews ({rate*100:.0f}%) appear fake. "
            f"A notable portion show signs of inauthenticity. "
            f"Consider cross-referencing with other sources before purchasing.")
    elif rate >= 0.15:
        return "Mostly Trustworthy", (
            f"{fake}/{total} scraped reviews ({rate*100:.0f}%) were flagged. "
            f"The majority appear genuine. A small proportion may be inauthentic.")
    else:
        return "Trustworthy", (
            f"Only {fake}/{total} scraped reviews ({rate*100:.0f}%) were flagged as fake. "
            f"This product's review section appears largely authentic.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "ReviewGuard AI", "version": "5.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "sbert_model": SBERT_NAME, "embedding_dim": SBERT_DIM,
        "hand_features": N_HAND, "total_dims": SBERT_DIM + N_HAND,
        "fake_threshold": FAKE_THRESHOLD,
        "classifier": type(model).__name__,
        "scraper": "httpx + BeautifulSoup (no browser required)",
    }

@app.post("/predict", response_model=ReviewOut)
async def predict(body: ReviewIn):
    text = body.review.strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Review must be at least 10 characters.")
    if len(text) > MAX_TEXT:
        raise HTTPException(status_code=400, detail=f"Review must not exceed {MAX_TEXT} characters.")
    pred, conf, fp, gp, feats, sigs, desc = _classify(text)
    return ReviewOut(prediction=pred, confidence=conf, genuine_probability=gp,
                     fake_probability=fp, risk_level=_risk(fp),
                     features=feats, signals=sigs, description=desc)

@app.post("/batch-predict", response_model=BatchOut)
async def batch_predict(items: List[BatchItem]):
    if len(items) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 reviews per batch.")
    valid = [item.review.strip() for item in items if len(item.review.strip()) >= 10]
    if not valid:
        raise HTTPException(status_code=422, detail="All reviews were too short (min 10 chars).")
    batch_results = await asyncio.get_event_loop().run_in_executor(None, _classify_batch, valid)
    results = []
    for t, (pred, conf, fp, gp, _, sigs, desc) in zip(valid, batch_results):
        results.append({
            "review": t[:100] + ("…" if len(t) > 100 else ""),
            "prediction": pred, "confidence": conf,
            "fake_probability": fp, "risk_level": _risk(fp), "description": desc,
        })
    fk = sum(1 for r in results if r["prediction"] == "Fake")
    return BatchOut(total=len(results), fake=fk, genuine=len(results) - fk, results=results)

@app.post("/analyze-url", response_model=UrlAnalysisOut)
async def analyze_url(body: UrlIn):
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    raw_reviews, site_name = await scrape_reviews(url)

    if not raw_reviews:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No reviews could be extracted from {site_name}. "
                "Possible causes: (1) page requires login, "
                "(2) heavy anti-bot protection active (try again in a few seconds), "
                "(3) product has no reviews yet, "
                "(4) URL is a category/search page rather than a product page. "
                "For Amazon: paste the full product URL like https://www.amazon.in/dp/XXXXXXXXXX"
            ),
        )

    valid = [t.strip() for t in raw_reviews if len(t.strip()) >= MIN_REVIEW_LEN]
    if not valid:
        raise HTTPException(status_code=422, detail="Scraped content had no readable reviews.")

    log.info("Batch-classifying %d reviews from %s…", len(valid), site_name)
    batch_results = await asyncio.get_event_loop().run_in_executor(None, _classify_batch, valid)

    results = []
    for t, (pred, conf, fp, gp, _, sigs, desc) in zip(valid, batch_results):
        results.append({
            "review":              t[:200] + ("…" if len(t) > 200 else ""),
            "full_review":         t,
            "prediction":          pred,
            "confidence":          conf,
            "fake_probability":    fp,
            "genuine_probability": gp,
            "risk_level":          _risk(fp),
            "signals":             sigs,
            "description":         desc,
        })

    fk    = sum(1 for r in results if r["prediction"] == "Fake")
    total = len(results)
    verdict, verdict_desc = _url_verdict(fk, total)

    return UrlAnalysisOut(
        url=url, site=site_name, total_scraped=total,
        fake=fk, genuine=total - fk,
        fake_rate=round(fk / max(total, 1), 4),
        overall_verdict=verdict, verdict_description=verdict_desc,
        results=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")