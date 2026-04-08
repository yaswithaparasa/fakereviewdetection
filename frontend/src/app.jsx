import React, { useState, useCallback, useEffect } from "react";

/* ─── Config ──────────────────────────────────────────────────────────────── */
const API = "http://localhost:8000";

/* ─── Helpers ─────────────────────────────────────────────────────────────── */
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ─── Theme ───────────────────────────────────────────────────────────────── */
const T = {
  bg0: "#ffffff",
  bg1: "#f8f9fa",
  bg2: "#e9ecef",
  bg3: "#dee2e6",
  border: "#adb5bd",
  accent: "#00d4ff",
  accentDim: "#0090bb",
  fake: "#ff4d6d",
  genuine: "#00e5a0",
  warn: "#ffb347",
  muted: "#6c757d",
  text: "#212529",
  textDim: "#495057",
};

/* ─── Validation ─────────────────────────────────────────────────────────── */
const COMMON_WORDS = [
  "the",
  "and",
  "is",
  "for",
  "with",
  "that",
  "this",
  "was",
  "but",
  "not",
  "have",
  "you",
  "review",
  "food",
  "service",
  "great",
  "good",
  "bad",
  "place",
  "product",
  "amazing",
  "very",
  "they",
  "here",
  "it",
  "so",
  "all",
  "my",
  "at",
  "from",
];

const validateReview = (text) => {
  const t = text.trim();
  if (!t) return { ok: false, msg: "Review is empty." };
  if (t.length < 10) return { ok: false, msg: "Too short (min 10 chars)." };
  if (t.length > 10000)
    return { ok: false, msg: "Too long (max 10,000 chars)." };
  if (/^[\d\s\W]+$/.test(t))
    return { ok: false, msg: "Only numbers or symbols — no real words found." };
  const symbols = t.replace(/[a-zA-Z0-9\s]/g, "");
  if (symbols.length / t.length > 0.35)
    return { ok: false, msg: "Contains too many special symbols/punctuation." };
  const numbers = t.replace(/[^0-9]/g, "");
  if (numbers.length / t.length > 0.45)
    return { ok: false, msg: "Contains mostly numbers — not a real review." };
  const words = t.split(/\s+/).filter((w) => w.length > 0);
  if (words.length < 3)
    return {
      ok: false,
      msg: "Enter at least 3 words for a meaningful review.",
    };
  if (words.length >= 5) {
    const hasCommon = words.some((w) => COMMON_WORDS.includes(w.toLowerCase()));
    if (!hasCommon)
      return {
        ok: false,
        msg: "Review does not appear to contain meaningful English words.",
      };
  }
  for (const word of words) {
    if (/^\d+$/.test(word) && word.length > 12)
      return { ok: false, msg: "Contains unusually long numeric strings." };
    const hasDigit = /\d/.test(word);
    const hasLetter = /[a-zA-Z]/.test(word);
    if (hasDigit && hasLetter) {
      const digitMatches = word.match(/\d+/g) || [];
      if (digitMatches.some((dm) => dm.length > 5))
        return {
          ok: false,
          msg: `'${word}' contains a long numeric sequence (likely a random ID or hash).`,
        };
      const segments = word.match(/[a-zA-Z]+|\d+/g) || [];
      if (segments.length > 3 && word.length > 5)
        return {
          ok: false,
          msg: `'${word}' looks like a random alphanumeric string.`,
        };
    }
    const cleanWord = word.toLowerCase().replace(/[^a-z]/g, "");
    if (cleanWord.length >= 8) {
      const vowels = cleanWord.replace(/[^aeiou]/g, "").length;
      if (vowels / cleanWord.length < 0.18)
        return { ok: false, msg: "Contains words that look like gibberish." };
    }
    const cleanAlpha = word.replace(/[^a-zA-Z]/g, "");
    const maxConsecConsonants = (
      cleanAlpha.match(/[bcdfghjklmnpqrstvwxyz]{5,}/gi) || []
    ).reduce((max, m) => Math.max(max, m.length), 0);
    if (maxConsecConsonants >= 6)
      return {
        ok: false,
        msg: "Contains words with too many consecutive consonants.",
      };
    if (maxConsecConsonants >= 5 && cleanAlpha.length > 12)
      return { ok: false, msg: "Contains suspicious long words." };
  }
  const alphaWords2 = words.filter(
    (w) => w.replace(/[^a-zA-Z]/g, "").length >= 2,
  );
  if (alphaWords2.length < 3)
    return {
      ok: false,
      msg: "Review must contain proper words and sentences.",
    };
  if (/<[^>]+>/.test(t))
    return { ok: false, msg: "HTML or script tags are not allowed." };
  if (/https?:\/\/|www\./i.test(t))
    return { ok: false, msg: "URLs are not allowed in reviews." };
  const uniqueChars = new Set(t.replace(/\s/g, "")).size;
  if (uniqueChars < 3)
    return { ok: false, msg: "Looks like repeated characters or nonsense." };
  const letters = t.replace(/[^a-zA-Z]/g, "");
  if (letters.length / t.length < 0.35)
    return { ok: false, msg: "Too many numbers or symbols compared to text." };
  const vowels = letters.replace(/[^aeiouAEIOU]/g, "").length;
  if (letters.length > 0 && vowels / letters.length < 0.1)
    return {
      ok: false,
      msg: "Content does not look like real language (very few vowels).",
    };
  return { ok: true };
};

/* ─── Animated bar ────────────────────────────────────────────────────────── */
function Bar({ value, color }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const id = setTimeout(() => setW(clamp(value, 0, 1) * 100), 60);
    return () => clearTimeout(id);
  }, [value]);
  return (
    <div
      style={{
        height: 5,
        background: T.bg0,
        borderRadius: 3,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${w}%`,
          background: color,
          borderRadius: 3,
          boxShadow: `0 0 10px ${color}88`,
          transition: "width 1.3s cubic-bezier(0.34,1.56,0.64,1)",
        }}
      />
    </div>
  );
}

/* ─── Spinner ─────────────────────────────────────────────────────────────── */
function Spin({ c = T.accent, s = 18 }) {
  return (
    <span
      style={{
        width: s,
        height: s,
        display: "inline-block",
        border: `2px solid ${c}30`,
        borderTopColor: c,
        borderRadius: "50%",
        animation: "spin .65s linear infinite",
        flexShrink: 0,
      }}
    />
  );
}

/* ─── Hex badge ───────────────────────────────────────────────────────────── */
function HexBadge({ label, value, color }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          width: 68,
          height: 68,
          margin: "0 auto 5px",
          background: `${color}18`,
          border: `2px solid ${color}55`,
          borderRadius: "30% 70% 70% 30% / 30% 30% 70% 70%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: `0 0 20px ${color}30`,
        }}
      >
        <span
          style={{
            fontSize: 18,
            fontWeight: 900,
            color,
            fontFamily: "monospace",
          }}
        >
          {value}
        </span>
      </div>
      <div
        style={{
          fontSize: 8,
          color: T.textDim,
          letterSpacing: 2,
          textTransform: "uppercase",
          fontFamily: "monospace",
        }}
      >
        {label}
      </div>
    </div>
  );
}

/* ─── Prob row ────────────────────────────────────────────────────────────── */
function ProbRow({ label, value, color }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 7,
        }}
      >
        <span
          style={{
            fontSize: 9,
            color: T.textDim,
            fontFamily: "monospace",
            letterSpacing: 2.5,
            textTransform: "uppercase",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: 14,
            color,
            fontWeight: 700,
            fontFamily: "monospace",
          }}
        >
          {pct(value)}
        </span>
      </div>
      <Bar value={value} color={color} />
    </div>
  );
}

/* ─── Fuzzy signal chip ───────────────────────────────────────────────────── */
function FuzzyChip({ signal }) {
  const c =
    signal.verdict === "suspicious"
      ? T.fake
      : signal.verdict === "genuine"
        ? T.genuine
        : T.muted;
  return (
    <div
      style={{
        background: `${c}12`,
        border: `1px solid ${c}40`,
        borderLeft: `3px solid ${c}`,
        borderRadius: "0 8px 8px 0",
        padding: "10px 14px",
        marginBottom: 8,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 5,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span
            style={{
              fontSize: 8,
              color: c,
              fontFamily: "monospace",
              fontWeight: 700,
              letterSpacing: 1.5,
              background: `${c}22`,
              padding: "2px 6px",
              borderRadius: 3,
            }}
          >
            {signal.rule}
          </span>
          <span style={{ fontSize: 12, color: T.text, fontWeight: 600 }}>
            {signal.label}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <div
            style={{ width: 44, height: 3, background: T.bg0, borderRadius: 2 }}
          >
            <div
              style={{
                height: "100%",
                width: `${(signal.strength || 0) * 100}%`,
                background: c,
                borderRadius: 2,
                boxShadow: `0 0 5px ${c}`,
              }}
            />
          </div>
          <span
            style={{
              fontSize: 9,
              color: c,
              fontFamily: "monospace",
              fontWeight: 700,
            }}
          >
            {((signal.strength || 0) * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <p
        style={{
          fontSize: 10,
          color: T.textDim,
          margin: 0,
          fontFamily: "monospace",
        }}
      >
        {signal.detail}
      </p>
    </div>
  );
}

/* ─── Semantic tag ────────────────────────────────────────────────────────── */
function SigTag({ text }) {
  const isGood = text.startsWith("✓");
  const c = isGood ? T.genuine : T.fake;
  return (
    <span
      style={{
        fontSize: 10,
        color: c,
        background: `${c}15`,
        border: `1px solid ${c}40`,
        borderRadius: 4,
        padding: "3px 9px",
        fontFamily: "monospace",
      }}
    >
      {text}
    </span>
  );
}

/* ─── Subsystem bar ───────────────────────────────────────────────────────── */
function SubBar({ label, fakeProbability }) {
  const isFake = fakeProbability >= 0.5;
  const c = isFake ? T.fake : T.genuine;
  return (
    <div style={{ marginBottom: 13 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 5,
        }}
      >
        <span
          style={{
            fontSize: 9,
            color: T.textDim,
            fontFamily: "monospace",
            letterSpacing: 1.5,
            textTransform: "uppercase",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: 11,
            color: c,
            fontFamily: "monospace",
            fontWeight: 700,
          }}
        >
          {isFake ? "FAKE" : "GENUINE"}{" "}
          {pct(isFake ? fakeProbability : 1 - fakeProbability)}
        </span>
      </div>
      <Bar value={fakeProbability} color={T.fake} />
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [review, setReview] = useState("");
  const [meta, setMeta] = useState({
    rating: "",
    reviewCount: "",
    usefulCount: "",
    friendCount: "",
    mnr: "",
    rd: "",
    max_similarity: "",
  });
  const [result, setResult] = useState(null);
  const [explainData, setExplain] = useState(null);
  const [loading, setLoading] = useState(false);
  const [explLoading, setExplLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) })
      .then((r) => (r.ok ? setApiStatus("online") : setApiStatus("error")))
      .catch(() => setApiStatus("offline"));
  }, []);

  const buildBody = () => ({
    review: review.trim(),
    ...(meta.rating ? { rating: parseFloat(meta.rating) } : {}),
    ...(meta.reviewCount ? { reviewCount: parseInt(meta.reviewCount) } : {}),
    ...(meta.usefulCount ? { usefulCount: parseInt(meta.usefulCount) } : {}),
    ...(meta.friendCount ? { friendCount: parseInt(meta.friendCount) } : {}),
    ...(meta.mnr ? { mnr: parseFloat(meta.mnr) } : {}),
    ...(meta.rd ? { rd: parseFloat(meta.rd) } : {}),
    ...(meta.max_similarity
      ? { max_similarity: parseFloat(meta.max_similarity) }
      : {}),
  });

  const analyze = useCallback(async () => {
    if (loading) return;
    setError("");
    setResult(null);
    setExplain(null);
    if (review.trim().length < 5) return;
    const v = validateReview(review);
    if (!v.ok) {
      setError(v.msg);
      return;
    }
    setLoading(true);
    try {
      const r = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildBody()),
      });
      if (!r.ok) {
        const e = await r.json();
        throw new Error(e.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [review, meta, loading]);

  const explainReview = useCallback(async () => {
    if (!review.trim() || explLoading) return;
    setExplLoading(true);
    setExplain(null);
    try {
      const r = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildBody()),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setExplain(await r.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setExplLoading(false);
    }
  }, [review, meta, explLoading]);

  const isFake = result?.prediction === "Fake";
  const vColor = result ? (isFake ? T.fake : T.genuine) : T.accent;
  const statusColors = {
    online: T.genuine,
    offline: T.fake,
    checking: T.warn,
    error: T.fake,
  };

  const css = `
    @import url('https://fonts.googleapis.com/css2?family=Syne+Mono&family=Syne:wght@400;600;700;800;900&display=swap');
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    html,body{background:${T.bg0};color:${T.text};font-family:'Syne',sans-serif;min-height:100vh;}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
    @keyframes fadeIn{from{opacity:0}to{opacity:1}}
    @keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
    @keyframes glow{0%,100%{opacity:.6}50%{opacity:1}}
    .fadeup{animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both}
    .fadein{animation:fadeIn .3s ease both}
    textarea,input{outline:none;font-family:'Syne',sans-serif;color:${T.text};}
    textarea:focus,input:focus{border-color:${T.accent}!important;box-shadow:0 0 0 2px ${T.accent}22!important;}
    ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:${T.bg0}}::-webkit-scrollbar-thumb{background:${T.bg3};border-radius:2px}
    .btn{transition:all .15s;border:none;cursor:pointer;font-family:'Syne',sans-serif;font-weight:700;}
    .btn:hover:not(:disabled){filter:brightness(1.15);transform:translateY(-1px)}
    .btn:active:not(:disabled){transform:scale(.98)}
    .card{background:${T.bg1};border:1px solid ${T.border};border-radius:12px;}
    .inp{background:${T.bg0}!important;border:1px solid ${T.border}!important;transition:border-color .2s,box-shadow .2s!important;}
    .inp::placeholder{color:${T.muted}}
    .mi{background:${T.bg2}!important;border:1px solid ${T.border}!important;color:${T.text}!important;border-radius:6px;padding:7px 10px;font-size:11px;font-family:'Syne Mono',monospace;width:100%;}
    .mi:focus{border-color:${T.accent}!important;outline:none}
    .mi::placeholder{color:${T.muted}}
    .topbar{position:fixed;top:0;left:0;right:0;height:2px;z-index:1000;background:linear-gradient(90deg,transparent 0%,${T.accent} 40%,${T.fake} 60%,transparent 100%);animation:glow 3s ease-in-out infinite;}
  `;

  // The 3 key behavioral inputs
  const behavioralInputs = [
    {
      key: "rating",
      label: "Star Rating",
      ph: "1 – 5",
      icon: "★",
      hint: "Rating given by the reviewer",
      min: 1,
      max: 5,
      step: 0.5,
      type: "number",
    },
    {
      key: "reviewCount",
      label: "Reviewer's Total Reviews",
      ph: "e.g. 12",
      icon: "✍",
      hint: "Total reviews ever written by this reviewer",
      min: 0,
      max: 9999,
      step: 1,
      type: "number",
    },
    {
      key: "usefulCount",
      label: "Useful Votes on Reviewer",
      ph: "e.g. 0",
      icon: "👍",
      hint: "Total 'useful' votes the reviewer has ever received",
      min: 0,
      max: 99999,
      step: 1,
      type: "number",
    },
  ];

  return (
    <>
      <style>{css}</style>
      <div className="topbar" />
      <div style={{ minHeight: "100vh", position: "relative", zIndex: 1 }}>
        {/* HEADER */}
        <header
          style={{
            position: "sticky",
            top: 0,
            zIndex: 100,
            background: "rgba(255,255,255,0.95)",
            backdropFilter: "blur(20px)",
            borderBottom: `1px solid ${T.border}`,
            height: 66,
            padding: "0 36px",
            display: "flex",
            alignItems: "center",
            gap: 18,
          }}
        >
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 7,
              background: `linear-gradient(135deg,${T.accentDim},${T.accent})`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 17,
              boxShadow: `0 0 16px ${T.accent}44`,
            }}
          >
            ⬡
          </div>
          <div>
            <div
              style={{
                fontSize: 17,
                fontWeight: 900,
                letterSpacing: -0.5,
                lineHeight: 1,
              }}
            >
              Review<span style={{ color: T.accent }}>Scan</span>
              <span
                style={{
                  fontSize: 8,
                  color: T.muted,
                  marginLeft: 5,
                  fontFamily: "monospace",
                  letterSpacing: 2,
                  verticalAlign: "middle",
                }}
              >
                PRO v5
              </span>
            </div>
            <div
              style={{
                fontSize: 7,
                color: T.muted,
                fontFamily: "monospace",
                letterSpacing: 3.5,
                marginTop: 3,
              }}
            >
              SBERT · RANDOM FOREST · FUZZY LOGIC
            </div>
          </div>
          <div
            style={{
              marginLeft: "auto",
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: T.bg2,
              border: `1px solid ${T.border}`,
              borderRadius: 20,
              padding: "5px 12px",
            }}
          >
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: statusColors[apiStatus] || T.warn,
                boxShadow: `0 0 8px ${statusColors[apiStatus] || T.warn}`,
                animation:
                  apiStatus === "checking"
                    ? "pulse 1.4s ease infinite"
                    : "none",
              }}
            />
            <span
              style={{
                fontSize: 8,
                color: T.textDim,
                fontFamily: "monospace",
                letterSpacing: 1.5,
              }}
            >
              {apiStatus.toUpperCase()}
            </span>
          </div>
        </header>

        <main
          style={{ maxWidth: 1500, margin: "0 auto", padding: "26px 18px" }}
        >
          {/* ── SINGLE ANALYSER ── */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "400px 1fr",
              gap: 18,
              alignItems: "start",
            }}
          >
            {/* Left */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Review input */}
              <div className="card fadeup" style={{ padding: 20 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: 10,
                  }}
                >
                  <span
                    style={{
                      fontSize: 8,
                      color: T.muted,
                      fontFamily: "monospace",
                      letterSpacing: 3,
                      textTransform: "uppercase",
                    }}
                  >
                    ⬡ Review Text
                  </span>
                  <span
                    style={{
                      fontSize: 8,
                      color: T.muted,
                      fontFamily: "monospace",
                    }}
                  >
                    {review.length}/20000
                  </span>
                </div>
                <textarea
                  id="review-input"
                  value={review}
                  onChange={(e) => {
                    setReview(e.target.value);
                    if (result) setResult(null);
                    if (error) setError("");
                    if (explainData) setExplain(null);
                  }}
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === "Enter")
                      analyze();
                  }}
                  placeholder="Paste a restaurant review to classify as genuine or fake..."
                  rows={9}
                  maxLength={20000}
                  className="inp"
                  style={{
                    width: "100%",
                    borderRadius: 8,
                    padding: "12px 13px",
                    fontSize: 12,
                    lineHeight: 1.8,
                    resize: "none",
                    background: T.bg0,
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    gap: 7,
                    marginTop: 11,
                    justifyContent: "flex-end",
                  }}
                >
                  <button
                    id="clear-btn"
                    className="btn"
                    onClick={() => {
                      setReview("");
                      setResult(null);
                      setExplain(null);
                      setError("");
                    }}
                    style={{
                      background: T.bg2,
                      border: `1px solid ${T.border}`,
                      borderRadius: 7,
                      padding: "8px 16px",
                      color: T.textDim,
                      fontSize: 9,
                      fontFamily: "monospace",
                      letterSpacing: 1,
                    }}
                  >
                    CLR
                  </button>
                  <button
                    id="analyse-btn"
                    className="btn"
                    onClick={analyze}
                    disabled={loading || review.trim().length < 5}
                    style={{
                      background:
                        loading || review.trim().length < 5
                          ? T.bg2
                          : `linear-gradient(135deg,${T.accentDim},${T.accent})`,
                      border: `1px solid ${loading || review.trim().length < 5 ? T.border : T.accent}`,
                      borderRadius: 7,
                      padding: "8px 22px",
                      color:
                        loading || review.trim().length < 5 ? T.muted : T.bg0,
                      fontSize: 9,
                      fontFamily: "monospace",
                      letterSpacing: 1.5,
                      cursor:
                        loading || review.trim().length < 5
                          ? "not-allowed"
                          : "pointer",
                      boxShadow:
                        loading || review.trim().length < 5
                          ? "none"
                          : `0 0 16px ${T.accent}44`,
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    {loading ? <Spin c={T.bg0} s={13} /> : null}
                    {loading ? "SCANNING..." : "⬡ ANALYSE"}
                  </button>
                </div>
              </div>

              {/* Behavioral Signals Card */}
              <div className="card fadeup" style={{ padding: 20 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 14,
                  }}
                >
                  <div
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: 5,
                      background: `linear-gradient(135deg,${T.accentDim},${T.accent})`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                    }}
                  >
                    ⬡
                  </div>
                  <span
                    style={{
                      fontSize: 8,
                      color: T.muted,
                      fontFamily: "monospace",
                      letterSpacing: 3,
                      textTransform: "uppercase",
                    }}
                  >
                    User Behavioral Signals
                  </span>
                  <span
                    style={{
                      marginLeft: "auto",
                      fontSize: 8,
                      color: T.muted,
                      fontFamily: "monospace",
                      background: T.bg2,
                      border: `1px solid ${T.border}`,
                      borderRadius: 4,
                      padding: "2px 7px",
                    }}
                  >
                    Optional - Improves Accuracy
                  </span>
                </div>
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 10 }}
                >
                  {behavioralInputs.map(
                    ({ key, label, ph, icon, hint, min, max, step, type }) => (
                      <div key={key}>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                            marginBottom: 5,
                          }}
                        >
                          <span style={{ fontSize: 13 }}>{icon}</span>
                          <label
                            htmlFor={`behavioral-${key}`}
                            style={{
                              fontSize: 9,
                              color: T.textDim,
                              fontFamily: "monospace",
                              letterSpacing: 1.5,
                              textTransform: "uppercase",
                              fontWeight: 700,
                            }}
                          >
                            {label}
                          </label>
                        </div>
                        <input
                          id={`behavioral-${key}`}
                          type={type}
                          min={min}
                          max={max}
                          step={step}
                          value={meta[key]}
                          onChange={(e) =>
                            setMeta((m) => ({ ...m, [key]: e.target.value }))
                          }
                          placeholder={ph}
                          className="mi"
                          style={{
                            borderRadius: 7,
                            padding: "9px 12px",
                            fontSize: 12,
                            width: "100%",
                          }}
                        />
                        <div
                          style={{
                            fontSize: 8,
                            color: T.muted,
                            fontFamily: "monospace",
                            marginTop: 4,
                            paddingLeft: 2,
                          }}
                        >
                          {hint}
                        </div>
                      </div>
                    ),
                  )}
                </div>
                <div
                  style={{
                    marginTop: 14,
                    padding: "9px 12px",
                    background: `${T.accent}0a`,
                    border: `1px solid ${T.accent}22`,
                    borderRadius: 7,
                  }}
                >
                  <p
                    style={{
                      fontSize: 8,
                      color: T.muted,
                      fontFamily: "monospace",
                      lineHeight: 1.7,
                      margin: 0,
                    }}
                  >
                    These signals are fed into the{" "}
                    <span style={{ color: T.accent }}>Fuzzy Rule Engine</span>{" "}
                    alongside SBERT embeddings. Providing them improves fake
                    detection accuracy by up to{" "}
                    <span style={{ color: T.genuine }}>+3%</span>.
                  </p>
                </div>
              </div>
            </div>

            {/* Right - results */}
            <div
              style={{
                maxHeight: "calc(100vh - 110px)",
                overflowY: "auto",
                paddingRight: 5,
              }}
            >
              {error && (
                <div
                  className="fadeup"
                  style={{
                    background: `${T.fake}12`,
                    border: `1px solid ${T.fake}44`,
                    borderRadius: 10,
                    padding: 14,
                    marginBottom: 14,
                    color: T.fake,
                    fontSize: 12,
                  }}
                >
                  x {error}
                </div>
              )}

              {loading && !result && (
                <div
                  className="card fadein"
                  style={{ padding: "80px 32px", textAlign: "center" }}
                >
                  <Spin c={T.accent} s={34} />
                  <p
                    style={{
                      marginTop: 16,
                      fontSize: 10,
                      color: T.textDim,
                      fontFamily: "monospace",
                      letterSpacing: 2,
                    }}
                  >
                    EXTRACTING SBERT EMBEDDINGS...
                  </p>
                  <p
                    style={{
                      marginTop: 7,
                      fontSize: 9,
                      color: T.muted,
                      fontFamily: "monospace",
                    }}
                  >
                    COMPUTING FUZZY MEMBERSHIPS - RUNNING RF FOREST...
                  </p>
                </div>
              )}

              {result && (
                <div className="fadeup">
                  {/* Hero verdict */}
                  <div
                    className="card"
                    style={{
                      border: `1px solid ${vColor}44`,
                      borderTop: `3px solid ${vColor}`,
                      padding: 26,
                      marginBottom: 14,
                      boxShadow: `0 0 60px ${vColor}08`,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        gap: 16,
                        alignItems: "center",
                        marginBottom: 22,
                        paddingBottom: 18,
                        borderBottom: `1px solid ${T.border}`,
                      }}
                    >
                      <div
                        style={{
                          width: 68,
                          height: 68,
                          borderRadius: 13,
                          flexShrink: 0,
                          background: `${vColor}15`,
                          border: `2px solid ${vColor}55`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 28,
                          boxShadow: `0 0 28px ${vColor}30`,
                        }}
                      >
                        {isFake ? "" : ""}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            fontSize: 28,
                            fontWeight: 900,
                            color: vColor,
                            letterSpacing: -1,
                            lineHeight: 1,
                          }}
                        >
                          {result.prediction.toUpperCase()}
                        </div>
                        <div
                          style={{
                            fontSize: 9,
                            color: T.textDim,
                            fontFamily: "monospace",
                            marginTop: 6,
                            letterSpacing: 1.5,
                          }}
                        >
                          CONFIDENCE {pct(result.confidence)} -{" "}
                          {result.risk_level.toUpperCase()} RISK
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 12 }}>
                        <HexBadge
                          label="Fake Prob"
                          value={`${(result.fake_probability * 100).toFixed(0)}%`}
                          color={T.fake}
                        />
                        <HexBadge
                          label="Genuine"
                          value={`${(result.genuine_probability * 100).toFixed(0)}%`}
                          color={T.genuine}
                        />
                      </div>
                    </div>

                    <ProbRow
                      label="Genuine Probability"
                      value={result.genuine_probability}
                      color={T.genuine}
                    />
                    <ProbRow
                      label="Fake Probability"
                      value={result.fake_probability}
                      color={T.fake}
                    />

                    <div
                      style={{
                        background: `${vColor}10`,
                        border: `1px solid ${vColor}30`,
                        borderLeft: `3px solid ${vColor}`,
                        borderRadius: "0 8px 8px 0",
                        padding: "12px 15px",
                        marginTop: 16,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 7,
                          color: vColor,
                          fontFamily: "monospace",
                          letterSpacing: 2.5,
                          textTransform: "uppercase",
                          marginBottom: 6,
                          fontWeight: 700,
                        }}
                      >
                        {isFake ? "WHY FAKE" : "WHY GENUINE"}
                      </div>
                      <p
                        style={{
                          fontSize: 12,
                          color: T.text,
                          lineHeight: 1.75,
                          margin: 0,
                        }}
                      >
                        {result.description}
                      </p>
                    </div>
                  </div>

                  {/* Fuzzy + Semantic */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: 12,
                      marginBottom: 12,
                    }}
                  >
                    <div className="card" style={{ padding: 16 }}>
                      <div
                        style={{
                          fontSize: 8,
                          color: T.muted,
                          fontFamily: "monospace",
                          letterSpacing: 3,
                          textTransform: "uppercase",
                          marginBottom: 13,
                        }}
                      >
                        ⬡ Fuzzy Rule Engine
                      </div>
                      {result.fuzzy_signals?.map((s, i) => (
                        <FuzzyChip key={i} signal={s} />
                      ))}
                    </div>
                    <div className="card" style={{ padding: 16 }}>
                      <div
                        style={{
                          fontSize: 8,
                          color: T.muted,
                          fontFamily: "monospace",
                          letterSpacing: 3,
                          textTransform: "uppercase",
                          marginBottom: 13,
                        }}
                      >
                        SBERT Semantic Signals
                      </div>
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 5,
                          marginBottom: 14,
                        }}
                      >
                        {result.semantic_signals?.map((s, i) => (
                          <SigTag key={i} text={s} />
                        ))}
                      </div>
                      <div
                        style={{
                          borderTop: `1px solid ${T.border}`,
                          paddingTop: 12,
                        }}
                      >
                        {[
                          ["SBERT Dims", "384d"],
                          ["Fuzzy Rules", "5 (Mamdani)"],
                          ["RF Trees", "300"],
                          [
                            "Review Length",
                            `${result.meta?.review_length} words`,
                          ],
                          [
                            "Content Sim",
                            `${result.meta?.content_similarity?.toFixed(4)}`,
                          ],
                        ].map(([k, v]) => (
                          <div
                            key={k}
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              marginBottom: 5,
                            }}
                          >
                            <span
                              style={{
                                fontSize: 8,
                                color: T.muted,
                                fontFamily: "monospace",
                                letterSpacing: 1.5,
                              }}
                            >
                              {k}
                            </span>
                            <span
                              style={{
                                fontSize: 8,
                                color: T.accent,
                                fontFamily: "monospace",
                                fontWeight: 700,
                              }}
                            >
                              {v}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Explain ablation */}
                  <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                    <button
                      id="explain-btn"
                      className="btn"
                      onClick={explainReview}
                      disabled={explLoading}
                      style={{
                        background: explLoading ? T.bg2 : `${T.accent}18`,
                        border: `1px solid ${T.accent}44`,
                        borderRadius: 7,
                        padding: "8px 16px",
                        color: T.accent,
                        fontSize: 9,
                        fontFamily: "monospace",
                        letterSpacing: 1.5,
                        cursor: explLoading ? "not-allowed" : "pointer",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      {explLoading ? <Spin c={T.accent} s={11} /> : null}
                      {explLoading ? "COMPUTING..." : "EXPLAIN SUBSYSTEMS"}
                    </button>
                  </div>

                  {explainData?.subsystem_votes && (
                    <div
                      className="card fadeup"
                      style={{ padding: 18, border: `1px solid ${T.accent}33` }}
                    >
                      <div
                        style={{
                          fontSize: 8,
                          color: T.accent,
                          fontFamily: "monospace",
                          letterSpacing: 3,
                          textTransform: "uppercase",
                          marginBottom: 14,
                        }}
                      >
                        Subsystem Ablation - How Each Layer Voted
                      </div>
                      {Object.entries(explainData.subsystem_votes).map(
                        ([k, v]) => (
                          <SubBar
                            key={k}
                            label={v.label}
                            fakeProbability={v.fake_prob}
                          />
                        ),
                      )}
                    </div>
                  )}
                </div>
              )}

              {!result && !loading && !error && (
                <div
                  className="card"
                  style={{
                    border: `1px dashed ${T.border}`,
                    padding: "80px 32px",
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{ fontSize: 48, marginBottom: 13, opacity: 0.12 }}
                  >
                    ⬡
                  </div>
                  <p
                    style={{
                      fontSize: 10,
                      color: T.muted,
                      fontFamily: "monospace",
                      letterSpacing: 2,
                      textTransform: "uppercase",
                    }}
                  >
                    Paste a review - Enter behavioral signals - Press Analyse
                  </p>
                  <p
                    style={{
                      fontSize: 8,
                      color: T.muted,
                      fontFamily: "monospace",
                      marginTop: 7,
                    }}
                  >
                    Powered by SBERT (384d) - Random Forest (300 trees) - Fuzzy
                    Mamdani Logic
                  </p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
