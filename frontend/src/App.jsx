import { useState, useRef, useCallback, useEffect } from "react";

/* ─── Config ──────────────────────────────────────────────────────────────── */
const API = "http://localhost:8000";

/* ─── Helpers ─────────────────────────────────────────────────────────────── */
const pct   = (v) => `${(v * 100).toFixed(1)}%`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ─── Risk/verdict palette (Light Mode Optimized) ─────────────────────────── */
const RISK_META = {
  Low:    { color: "#059669", bg: "#ecfdf5", border: "#a7f3d0" },
  Medium: { color: "#d97706", bg: "#fffbeb", border: "#fde68a" },
  High:   { color: "#dc2626", bg: "#fef2f2", border: "#fecaca" },
};

const VERDICT_META = {
  "Trustworthy":          { color: "#059669", icon: "✅", bg: "#ecfdf5", border: "#a7f3d0" },
  "Mostly Trustworthy":   { color: "#0284c7", icon: "🔵", bg: "#f0f9ff", border: "#bae6fd" },
  "Moderately Suspicious":{ color: "#d97706", icon: "⚠️",  bg: "#fffbeb", border: "#fde68a" },
  "Highly Suspicious":    { color: "#dc2626", icon: "🚨", bg: "#fef2f2", border: "#fecaca" },
  "Inconclusive":         { color: "#475569", icon: "❓", bg: "#f8fafc", border: "#e2e8f0" },
};

/* ─── Animated bar ────────────────────────────────────────────────────────── */
function AnimatedBar({ value, color }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const id = setTimeout(() => setW(clamp(value, 0, 1) * 100), 80);
    return () => clearTimeout(id);
  }, [value]);
  return (
    <div style={{ height: 8, background: "#e2e8f0", borderRadius: 4, overflow: "hidden", position: "relative" }}>
      <div style={{
        height: "100%",
        width: `${w}%`,
        background: color,
        borderRadius: 4,
        transition: "width 1.1s cubic-bezier(0.34, 1.56, 0.64, 1)",
      }} />
    </div>
  );
}

function ProbRow({ label, value, color }) {
  return (
    <div style={{ marginBottom: 17 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
        <span style={{ fontSize: 14, color: "#64748b", fontFamily: "monospace", letterSpacing: 1.2, textTransform: "uppercase" }}>
          {label}
        </span>
        <span style={{ fontSize: 18, color, fontWeight: 700, fontFamily: "monospace" }}>
          {pct(value)}
        </span>
      </div>
      <AnimatedBar value={value} color={color} />
    </div>
  );
}

/* ─── Spinner ─────────────────────────────────────────────────────────────── */
function Spinner({ color = "#2563eb", size = 22 }) {
  return (
    <span style={{
      width: size, height: size, display: "inline-block",
      border: `2px solid ${color}40`,
      borderTopColor: color,
      borderRadius: "50%",
      animation: "spin .7s linear infinite",
      flexShrink: 0,
    }} />
  );
}

/* ─── Verdict badge ───────────────────────────────────────────────────────── */
function VerdictBadge({ verdict }) {
  const m = VERDICT_META[verdict] || VERDICT_META["Inconclusive"];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 7,
      fontSize: 16, fontWeight: 700, color: m.color,
      background: m.bg, border: `1px solid ${m.border}`,
      borderRadius: 8, padding: "4px 14px",
    }}>
      {m.icon} {verdict}
    </span>
  );
}

/* ─── Signal item ─────────────────────────────────────────────────────────── */
function SignalItem({ text }) {
  const isOk = text.includes("No strong");
  return (
    <div style={{
      display: "flex", gap: 12, alignItems: "flex-start",
      background: isOk ? "#ecfdf5" : "#fef2f2",
      border: `1px solid ${isOk ? "#a7f3d0" : "#fecaca"}`,
      borderRadius: 11, padding: "12px 16px", marginBottom: 8,
    }}>
      <span style={{ flexShrink: 0, fontSize: 16, marginTop: 1 }}>{isOk ? "✅" : "⚠️"}</span>
      <span style={{ fontSize: 16, color: isOk ? "#047857" : "#be123c", lineHeight: 1.6 }}>{text}</span>
    </div>
  );
}

/* ─── Description card ────────────────────────────────────────────────────── */
function DescriptionCard({ description, isFake }) {
  const color = isFake ? "#dc2626" : "#059669";
  const bg = isFake ? "#fef2f2" : "#ecfdf5";
  const border = isFake ? "#fecaca" : "#a7f3d0";
  return (
    <div style={{
      background: bg,
      border: `1px solid ${border}`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 14, padding: "20px 22px", marginBottom: 22,
    }}>
      <div style={{ fontSize: 14, color: color, fontFamily: "monospace", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 11, fontWeight: 700 }}>
        {isFake ? "🔍 Why this review is FAKE" : "🔍 Why this review is GENUINE"}
      </div>
      <p style={{ fontSize: 18, color: "#334155", lineHeight: 1.7, margin: 0 }}>{description}</p>
    </div>
  );
}

/* ─── Summary card ────────────────────────────────────────────────────────── */
function SummaryCard({ label, value, color }) {
  return (
    <div className="glass-panel" style={{
      flex: 1, borderRadius: 16, padding: "22px 25px", textAlign: "center",
    }}>
      <div style={{ fontSize: 39, fontWeight: 900, color, fontFamily: "monospace" }}>{value}</div>
      <div style={{ fontSize: 14, color: "#64748b", marginTop: 7, letterSpacing: 0.5, fontWeight: 600 }}>{label}</div>
    </div>
  );
}

/* ─── URL Result row ──────────────────────────────────────────────────────── */
function UrlResultRow({ r, idx }) {
  const [open, setOpen] = useState(false);
  const fc = r.prediction === "Fake" ? "#dc2626" : "#059669";
  
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        className="glass-panel url-row"
        onClick={() => setOpen(o => !o)}
        style={{
          borderLeft: `4px solid ${fc}`,
          borderRadius: open ? "14px 14px 0 0" : 14,
          padding: "15px 20px",
          display: "flex", justifyContent: "space-between", alignItems: "center", gap: 20,
          cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 15, color: "#94a3b8", fontFamily: "monospace", flexShrink: 0, fontWeight: 600 }}>
          #{String(idx + 1).padStart(2, "0")}
        </span>
        <p style={{ margin: 0, fontSize: 16, color: "#1e293b", flex: 1, lineHeight: 1.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {r.review}
        </p>
        <div style={{ textAlign: "right", flexShrink: 0, minWidth: 160 }}>
          <div style={{ fontSize: 16, fontWeight: 800, color: fc }}>{r.prediction === "Fake" ? "🚨 FAKE" : "✅ GENUINE"}</div>
          <div style={{ fontSize: 14, color: "#64748b", fontFamily: "monospace", marginTop: 3 }}>{pct(r.confidence)} conf.</div>
        </div>
        <span style={{ color: "#94a3b8", fontSize: 15, flexShrink: 0 }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{
          background: "rgba(255, 255, 255, 0.7)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          border: `1px solid #e2e8f0`,
          borderTop: "none",
          borderLeft: `4px solid ${fc}`,
          borderRadius: "0 0 14px 14px",
          padding: "16px 20px",
        }}>
          <div style={{ fontSize: 16, color: "#334155", lineHeight: 1.7, marginBottom: 14, maxHeight: 160, overflowY: "auto" }}>
            {r.full_review || r.review}
          </div>
          <div style={{
            background: r.prediction === "Fake" ? "#fef2f2" : "#ecfdf5",
            border: `1px solid ${fc}30`,
            borderLeft: `3px solid ${fc}`,
            borderRadius: 11, padding: "14px 16px",
          }}>
            <div style={{ fontSize: 13, color: fc, fontFamily: "monospace", letterSpacing: 1.3, textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
              {r.prediction === "Fake" ? "Why FAKE" : "Why GENUINE"}
            </div>
            <p style={{ fontSize: 16, color: "#334155", lineHeight: 1.65, margin: 0 }}>{r.description}</p>
          </div>
          {r.signals && r.signals.length > 0 && (
            <div style={{ marginTop: 11, display: "flex", flexWrap: "wrap", gap: 7 }}>
              {r.signals.map((s, si) => (
                <span key={si} style={{
                  fontSize: 14, color: s.includes("No strong") ? "#047857" : "#b45309",
                  background: s.includes("No strong") ? "#ecfdf5" : "#fffbeb",
                  border: `1px solid ${s.includes("No strong") ? "#a7f3d0" : "#fde68a"}`,
                  borderRadius: 7, padding: "3px 11px", fontWeight: 500
                }}>
                  {s}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MAIN APP                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [tab,       setTab]       = useState("single");
  const [inputMode, setInputMode] = useState("text");   // "text" | "url"
  const [review,    setReview]    = useState("");
  const [urlInput,  setUrlInput]  = useState("");
  const [result,    setResult]    = useState(null);
  const [urlResult, setUrlResult] = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");
  const [batchText, setBatchText] = useState("");
  const [batchRes,  setBatchRes]  = useState(null);
  const [history,   setHistory]   = useState([]);
  const [apiStatus, setApiStatus] = useState("checking");
  const [scrapeStep, setScrapeStep] = useState("");
  const areaRef = useRef();

  /* ── API health check ──────────────────────────────────────────────────── */
  useEffect(() => {
    fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) })
      .then(r => r.ok ? setApiStatus("online") : setApiStatus("error"))
      .catch(() => setApiStatus("offline"));
  }, []);

  /* ── Single predict ─────────────────────────────────────────────────────── */
  const analyze = useCallback(async () => {
    const text = review.trim();
    if (text.length < 10 || loading) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review: text }),
      });
      if (!r.ok) {
        const e = await r.json();
        throw new Error(e.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      setResult(data);
      setHistory(h => [
        { snippet: text.slice(0, 70) + (text.length > 70 ? "…" : ""), ...data, ts: Date.now() },
        ...h,
      ].slice(0, 20));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [review, loading]);

  /* ── URL analyze ────────────────────────────────────────────────────────── */
  const analyzeUrl = useCallback(async () => {
    const url = urlInput.trim();
    if (!url || loading) return;
    setLoading(true); setError(""); setUrlResult(null);
    setScrapeStep("🌐 Connecting to page…");
    try {
      setScrapeStep("🔍 Scraping reviews…");
      const r = await fetch(`${API}/analyze-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      setScrapeStep("🧠 Classifying reviews…");
      if (!r.ok) {
        const e = await r.json();
        throw new Error(e.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      setUrlResult(data);
      setScrapeStep("");
    } catch (e) {
      setError(e.message);
      setScrapeStep("");
    } finally {
      setLoading(false);
    }
  }, [urlInput, loading]);

  /* ── Batch predict ──────────────────────────────────────────────────────── */
  const runBatch = useCallback(async () => {
    const lines = batchText.split("\n").map(l => l.trim()).filter(l => l.length >= 10);
    if (!lines.length || loading) return;
    setLoading(true); setError(""); setBatchRes(null);
    try {
      const r = await fetch(`${API}/batch-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lines.slice(0, 30).map(review => ({ review }))),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setBatchRes(await r.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [batchText, loading]);

  /* ── Derived ────────────────────────────────────────────────────────────── */
  const isFake = result?.prediction === "Fake";
  const vColor = result ? (isFake ? "#dc2626" : "#059669") : "#2563eb";
  const rMeta  = result ? RISK_META[result.risk_level] : null;

  const statusColor = { online:"#10b981", offline:"#dc2626", checking:"#f59e0b", error:"#dc2626" };
  const statusLabel = { online:"Online", offline:"Offline", checking:"Checking…", error:"Error" };

  /* ── CSS ─────────────────────────────────────────────────────────────────── */
  const globalCSS = `
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    
    html, body { 
      background-color: #f8fafc; 
      color: #1e293b; 
      font-family: 'Syne', sans-serif; 
    }

    /* ─── LIGHT GLASSMORPHISM UTILITY CLASS ─── */
    .glass-panel {
      background: rgba(255, 255, 255, 0.8) !important;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid #ffffff !important;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
    }

    @keyframes fadeUp    { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
    @keyframes fadeIn    { from { opacity:0 } to { opacity:1 } }
    @keyframes spin      { to { transform: rotate(360deg); } }
    @keyframes pulse     { 0%,100% { opacity:.4; } 50% { opacity:1; } }

    .anim-up  { animation: fadeUp  .55s cubic-bezier(.22,1,.36,1) both; }
    .anim-in  { animation: fadeIn  .35s ease both; }

    textarea, input { outline: none; font-family: 'Syne', sans-serif; }
    textarea:focus  { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important; }
    input:focus     { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important; }

    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

    .btn       { transition: all .15s; border: none; cursor: pointer; font-family: 'Syne', sans-serif; }
    .btn:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.05); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
    .btn:active:not(:disabled){ transform: translateY(0px) scale(.98); }

    .tab-pill  { transition: all .2s; cursor: pointer; user-select: none; }
    .tab-pill:hover:not(.active-tab) { background: #f1f5f9 !important; }

    .mode-btn  { transition: all .2s; cursor: pointer; user-select: none; }
    .mode-btn:hover:not(.active-mode) { background: #f1f5f9 !important; }

    .hist-row  { transition: background .12s, transform .1s; cursor: default; }
    .hist-row:hover { background: #ffffff !important; transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,0.06); }

    .url-row   { transition: background .12s, transform .1s; cursor: pointer; }
    .url-row:hover { background: #ffffff !important; transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,0.06); }

    .cyber-input {
      background: #ffffff !important;
      border: 1px solid #cbd5e1 !important;
      color: #1e293b !important;
      transition: border-color .2s, box-shadow .2s !important;
    }
    .cyber-input::placeholder { color: #94a3b8; }
  `;

  return (
    <>
      <style>{globalCSS}</style>
      
      {/* Main UI Container */}
      <div style={{ minHeight: "100vh", background: "transparent", position: "relative", zIndex: 2 }}>

        {/* ═══════════════════ HEADER ═══════════════════ */}
        <header style={{
          position: "sticky", top: 0, zIndex: 100,
          background: "rgba(255, 255, 255, 0.8)", 
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          borderBottom: "1px solid #e2e8f0",
          height: 86, padding: "0 56px",
          display: "flex", alignItems: "center", gap: 20,
        }}>
          {/* Logo */}
          <div style={{
            width: 53, height: 53, borderRadius: 12, flexShrink: 0,
            background: "linear-gradient(135deg, #1d4ed8, #3b82f6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 24, boxShadow: "0 4px 14px rgba(37, 99, 235, 0.3)",
            border: "1px solid #60a5fa",
          }}>🛡️</div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", letterSpacing: -0.3, lineHeight: 1 }}>
              Review<span style={{ color: "#2563eb" }}>Analyzer</span>
              <span style={{ fontSize: 10, color: "#64748b", marginLeft: 10, fontFamily: "monospace", letterSpacing: 2, verticalAlign: "middle", fontWeight: 600 }}>AI v3</span>
            </div>
            <div style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace", letterSpacing: 3.5, marginTop: 4, fontWeight: 600 }}>
              FAKE REVIEW DETECTION SYSTEM
            </div>
          </div>

          <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ background: "#ffffff", border: "1px solid #e2e8f0", display: "flex", alignItems: "center", gap: 7, marginLeft: 8, padding: "8px 16px", borderRadius: 20, boxShadow: "0 2px 8px rgba(0,0,0,0.02)" }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: statusColor[apiStatus],
                boxShadow: `0 0 8px ${statusColor[apiStatus]}80`,
                animation: apiStatus === "checking" ? "pulse 1.4s ease infinite" : "none",
              }} />
              <span style={{ fontSize: 11, color: "#475569", fontFamily: "monospace", fontWeight: 600 }}>
                {statusLabel[apiStatus]}
              </span>
            </div>
          </div>
        </header>

        {/* ═══════════════════ MAIN ═══════════════════ */}
        <main style={{ maxWidth: 1700, margin: "0 auto", padding: "36px 28px" }}>

          {/* Tab bar */}
          <div className="glass-panel" style={{
            display: "flex", gap: 3, 
            borderRadius: 15, padding: "5px", marginBottom: 30, width: "fit-content",
          }}>
            {[["single","🔍 Analyse"], ["batch","📋 Batch"], ["history","🕑 History"]].map(([id, lbl]) => (
              <div key={id} className={`tab-pill ${tab === id ? 'active-tab' : ''}`}
                onClick={() => setTab(id)}
                style={{
                  padding: "10px 24px", borderRadius: 11,
                  fontSize: 14, fontWeight: 700, letterSpacing: 0.3,
                  background: tab === id ? "linear-gradient(135deg, #1d4ed8, #3b82f6)" : "transparent",
                  color: tab === id ? "#ffffff" : "#475569",
                  boxShadow: tab === id ? "0 4px 12px rgba(37, 99, 235, 0.25)" : "none",
                }}
              >{lbl}</div>
            ))}
          </div>

          {/* ════ SINGLE TAB ════ */}
          {tab === "single" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28, alignItems: "start" }}>

              {/* ── Left: input ── */}
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

                {/* Mode switcher */}
                <div className="glass-panel anim-up" style={{
                  borderRadius: 20, padding: 6, display: "flex", gap: 4, width: "fit-content",
                }}>
                  {[
                    { id: "text", icon: "📝", label: "Single Review" },
                    { id: "url",  icon: "🔗", label: "Product URL" },
                  ].map(m => (
                    <div
                      key={m.id}
                      className={`mode-btn ${inputMode === m.id ? 'active-mode' : ''}`}
                      onClick={() => { setInputMode(m.id); setError(""); setResult(null); setUrlResult(null); }}
                      style={{
                        padding: "10px 20px", borderRadius: 14,
                        background: inputMode === m.id ? "linear-gradient(135deg, #1d4ed8, #3b82f6)" : "transparent",
                        border: `1px solid ${inputMode === m.id ? "#60a5fa" : "transparent"}`,
                        color: inputMode === m.id ? "#ffffff" : "#475569",
                        fontSize: 12, fontWeight: 700, textAlign: "center",
                        cursor: "pointer",
                        boxShadow: inputMode === m.id ? "0 4px 12px rgba(37, 99, 235, 0.2)" : "none",
                      }}
                    >
                      {m.icon} {m.label}
                    </div>
                  ))}
                </div>

                {/* ── TEXT mode ── */}
                {inputMode === "text" && (
                  <>
                    <div className="glass-panel anim-up" style={{ borderRadius: 20, padding: 28 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
                        <span style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", letterSpacing: 2.5, textTransform: "uppercase", fontWeight: 700 }}>
                          Review Text
                        </span>
                        <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", fontWeight: 600 }}>
                          {review.length} / 10 000
                        </span>
                      </div>
                      <textarea
                        ref={areaRef}
                        value={review}
                        onChange={e => setReview(e.target.value)}
                        onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); }}
                        placeholder="Paste a product review here to analyse authenticity…"
                        maxLength={10000}
                        rows={9}
                        className="cyber-input"
                        style={{
                          width: "100%", borderRadius: 12,
                          padding: "16px 20px",
                          fontSize: 14, lineHeight: 1.75, resize: "none",
                        }}
                      />
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16 }}>
                        <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", fontWeight: 600 }}>
                          {review.split(/\s+/).filter(Boolean).length} words · ⌘↵ to run
                        </span>
                        <div style={{ display: "flex", gap: 10 }}>
                          <button className="btn" onClick={() => { setReview(""); setResult(null); setError(""); }}
                            style={{
                              background: "#f1f5f9", border: "1px solid #e2e8f0",
                              borderRadius: 10, padding: "12px 24px",
                              color: "#475569", fontSize: 12, fontWeight: 700,
                              display: "inline-flex", alignItems: "center", justifyContent: "center"
                            }}>Clear</button>
                          <button className="btn" onClick={analyze}
                            disabled={loading || review.trim().length < 10}
                            style={{
                              background: (loading || review.trim().length < 10)
                                ? "#e2e8f0" : "linear-gradient(135deg, #1d4ed8, #3b82f6)",
                              borderRadius: 10, padding: "12px 32px",
                              border: (loading || review.trim().length < 10) ? "1px solid #cbd5e1" : "1px solid #3b82f6",
                              color: (loading || review.trim().length < 10) ? "#94a3b8" : "#ffffff",
                              fontSize: 12, fontWeight: 700,
                              cursor: (loading || review.trim().length < 10) ? "not-allowed" : "pointer",
                              boxShadow: (loading || review.trim().length < 10) ? "none" : "0 4px 14px rgba(37, 99, 235, 0.3)",
                              display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 10,
                            }}>
                            {loading ? <Spinner color="#ffffff" /> : null}
                            {loading ? "Analysing…" : "🔍 Analyse"}
                          </button>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {/* ── URL mode ── */}
                {inputMode === "url" && (
                  <div className="glass-panel anim-up" style={{
                    borderRadius: 20, padding: 28, minHeight: 320, display: "flex", flexDirection: "column"
                  }}>
                    <div style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", letterSpacing: 2.5, textTransform: "uppercase", marginBottom: 16, fontWeight: 700 }}>
                      Product URL
                    </div>

                    <div style={{ position: "relative", marginBottom: 16 }}>
                      <span style={{
                        position: "absolute", left: 16, top: "50%", transform: "translateY(-50%)",
                        fontSize: 20, pointerEvents: "none",
                      }}>🔗</span>
                      <input
                        type="url"
                        value={urlInput}
                        onChange={e => setUrlInput(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter") analyzeUrl(); }}
                        placeholder="Paste any Amazon product URL…"
                        className="cyber-input"
                        style={{
                          width: "100%", borderRadius: 12,
                          padding: "15px 20px 15px 50px",
                          fontSize: 14, lineHeight: 1.5,
                        }}
                      />
                    </div>

                    {/* Supported Sites Indicator */}
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
                      <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>
                        Supports
                      </span>
                      <div style={{ display: "flex", gap: 6 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: "#d97706", background: "#fffbeb", border: "1px solid #fde68a", padding: "4px 10px", borderRadius: 6 }}>
                          Amazon
                        </span>
                      </div>
                    </div>

                    {/* Button Container - Centers button */}
                    <div style={{ marginTop: "auto", display: "flex", justifyContent: "center" }}>
                      <button className="btn" onClick={analyzeUrl}
                        disabled={loading || !urlInput.trim()}
                        style={{
                          width: "fit-content",
                          background: (loading || !urlInput.trim())
                            ? "#e2e8f0" : "linear-gradient(135deg, #1d4ed8, #3b82f6)",
                          border: (loading || !urlInput.trim()) ? "1px solid #cbd5e1" : "1px solid #3b82f6",
                          borderRadius: 10, padding: "12px 32px", 
                          color: (loading || !urlInput.trim()) ? "#94a3b8" : "#ffffff",
                          fontSize: 12, fontWeight: 700,
                          cursor: (loading || !urlInput.trim()) ? "not-allowed" : "pointer",
                          boxShadow: (loading || !urlInput.trim()) ? "none" : "0 4px 14px rgba(37, 99, 235, 0.3)",
                          display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 10,
                        }}>
                        {loading ? <Spinner color="#ffffff" /> : null}
                        {loading ? "Processing…" : "🌐 Analyse"}
                      </button>
                    </div>

                    {loading && scrapeStep && (
                      <div style={{
                        marginTop: 16, textAlign: "center",
                        fontSize: 15, color: "#2563eb", fontFamily: "monospace", fontWeight: 600,
                        animation: "pulse 1.5s ease infinite",
                      }}>
                        {scrapeStep}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── Right: results (WITH FIXED SCROLLING BOUNDARY) ── */}
              <div style={{ maxHeight: 630, overflowY: "auto", paddingRight: 11, overflowX: "hidden" }}>
                {/* TEXT mode result */}
                {inputMode === "text" && (
                  <>
                    {error && (
                      <div className="glass-panel anim-up" style={{
                        background: "#fef2f2 !important",
                        border: "1px solid #fecaca !important",
                        borderRadius: 16, padding: 20, marginBottom: 20,
                        color: "#dc2626", fontSize: 16, fontWeight: 500
                      }}>⚠️ {error}</div>
                    )}

                    {loading && !result && (
                      <div className="glass-panel anim-in" style={{
                        borderRadius: 22, padding: "84px 34px", textAlign: "center",
                      }}>
                        <Spinner color="#2563eb" size={36} />
                        <p style={{ marginTop: 22, fontSize: 16, color: "#475569", fontFamily: "monospace", fontWeight: 600 }}>
                          Embedding + classifying…
                        </p>
                      </div>
                    )}

                    {result && (
                      <div className="glass-panel anim-up" style={{
                        border: `1px solid ${vColor}40 !important`,
                        borderRadius: 22, padding: 31, overflow: "hidden",
                        boxShadow: `0 8px 32px ${vColor}15`
                      }}>
                        {/* Verdict header */}
                        <div style={{
                          display: "flex", alignItems: "center", gap: 17, marginBottom: 25,
                          paddingBottom: 22, borderBottom: "1px solid #e2e8f0",
                        }}>
                          <div style={{
                            width: 73, height: 73, borderRadius: 17, flexShrink: 0,
                            background: isFake ? "#fef2f2" : "#ecfdf5",
                            border: `2px solid ${isFake ? "#fecaca" : "#a7f3d0"}`,
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 31,
                          }}>
                            {isFake ? "🚨" : "✅"}
                          </div>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 28, fontWeight: 800, color: vColor, lineHeight: 1 }}>
                              {result.prediction}
                            </div>
                            <div style={{ fontSize: 14, color: "#64748b", fontFamily: "monospace", marginTop: 6, fontWeight: 600 }}>
                              {pct(result.confidence)} confidence · {result.risk_level} Risk
                            </div>
                          </div>
                          <div style={{
                            fontSize: 14, fontWeight: 700, color: rMeta.color,
                            background: rMeta.bg, border: `1px solid ${rMeta.border}`,
                            borderRadius: 8, padding: "6px 15px",
                          }}>
                            {result.risk_level} Risk
                          </div>
                        </div>

                        {/* Description */}
                        <DescriptionCard description={result.description} isFake={isFake} />

                        {/* Prob bars */}
                        <ProbRow label="Genuine Probability" value={result.genuine_probability} color="#059669" />
                        <ProbRow label="Fake Probability"    value={result.fake_probability}    color="#dc2626" />

                        {/* Signals */}
                        <p style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", letterSpacing: 2, textTransform: "uppercase", margin: "24px 0 11px", fontWeight: 700 }}>
                          Signals Detected
                        </p>
                        {result.signals.map((s, i) => <SignalItem key={i} text={s} />)}

                      </div>
                    )}

                    {!result && !loading && !error && (
                      <div className="glass-panel" style={{
                        border: "2px dashed #cbd5e1 !important",
                        background: "rgba(255,255,255,0.5) !important",
                        borderRadius: 22, padding: "84px 34px", textAlign: "center",
                        boxShadow: "none"
                      }}>
                        <div style={{ fontSize: 50, marginBottom: 17 }}>🛡️</div>
                        <p style={{ fontSize: 18, color: "#64748b", fontWeight: 500 }}>
                          Paste a review and click Analyse to get started.
                        </p>
                      </div>
                    )}
                  </>
                )}

                {/* URL mode result */}
                {inputMode === "url" && (
                  <>
                    {error && (
                      <div className="glass-panel anim-up" style={{
                        background: "#fef2f2 !important",
                        border: "1px solid #fecaca !important",
                        borderRadius: 16, padding: 20, marginBottom: 20,
                        color: "#dc2626", fontSize: 16, fontWeight: 500
                      }}>⚠️ {error}</div>
                    )}

                    {loading && (
                      <div className="glass-panel anim-in" style={{
                        borderRadius: 22, padding: "84px 34px", textAlign: "center",
                      }}>
                        <Spinner color="#2563eb" size={36} />
                        <p style={{ marginTop: 22, fontSize: 16, color: "#475569", fontFamily: "monospace", fontWeight: 600 }}>
                          {scrapeStep || "Processing…"}
                        </p>
                        <p style={{ marginTop: 11, fontSize: 14, color: "#94a3b8", fontFamily: "monospace", fontWeight: 500 }}>
                          This may take sometime....
                        </p>
                      </div>
                    )}

                    {urlResult && !loading && (
                      <div className="anim-up">
                        {/* Overall verdict */}
                        <div className="glass-panel" style={{
                          border: `1px solid ${(VERDICT_META[urlResult.overall_verdict] || VERDICT_META["Inconclusive"]).color}40 !important`,
                          borderRadius: 22, padding: 28, marginBottom: 20,
                          boxShadow: `0 8px 32px ${(VERDICT_META[urlResult.overall_verdict] || VERDICT_META["Inconclusive"]).color}10`
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                            <div>
                              <div style={{ fontSize: 14, color: "#64748b", fontFamily: "monospace", letterSpacing: 2, textTransform: "uppercase", marginBottom: 7, fontWeight: 700 }}>
                                {urlResult.site} Product Verdict
                              </div>
                              <VerdictBadge verdict={urlResult.overall_verdict} />
                            </div>
                          </div>

                          {/* Mini summary bars */}
                          <div style={{ marginTop: 26 }}>
                            <ProbRow label="Fake Probability"    value={urlResult.fake_rate}          color="#dc2626" />
                            <ProbRow label="Genuine Probability" value={1 - urlResult.fake_rate}      color="#059669" />
                          </div>
                        </div>

                        {/* Per-Review Results */}
                        <p style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", letterSpacing: 2, textTransform: "uppercase", margin: "0 0 13px", fontWeight: 700 }}>
                          Per-Review Results
                        </p>
                        <div style={{ maxHeight: 320, overflowY: "auto", paddingRight: 4 }}>
                          {urlResult.results.slice(0, 5).map((r, i) => (
                            <UrlResultRow key={i} r={r} idx={i} />
                          ))}
                        </div>
                      </div>
                    )}

                    {!urlResult && !loading && !error && (
                      <div className="glass-panel" style={{
                        border: "2px dashed #cbd5e1 !important",
                        background: "rgba(255,255,255,0.5) !important",
                        borderRadius: 22, padding: "84px 34px", textAlign: "center",
                        boxShadow: "none"
                      }}>
                        <div style={{ fontSize: 50, marginBottom: 17 }}>🔗</div>
                        <p style={{ fontSize: 18, color: "#64748b", fontWeight: 500 }}>
                          Paste a product URL to analyse reviews.
                        </p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* ════ BATCH TAB ════ */}
          {tab === "batch" && (
            <div style={{ maxWidth: 1090 }}>
              <div className="glass-panel anim-up" style={{
                borderRadius: 20, padding: 28, marginBottom: 22,
              }}>
                <div style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", letterSpacing: 2.5, textTransform: "uppercase", marginBottom: 14, fontWeight: 700 }}>
                  Batch Reviews — one per line (max 30)
                </div>
                <textarea
                  value={batchText}
                  onChange={e => setBatchText(e.target.value)}
                  placeholder={"Review one\nReview two\nReview three…"}
                  rows={8}
                  className="cyber-input"
                  style={{ width: "100%", borderRadius: 12, padding: "17px 20px", fontSize: 14, lineHeight: 1.7, resize: "vertical" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 17 }}>
                  <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", fontWeight: 600 }}>
                    {batchText.split("\n").filter(l => l.trim().length >= 10).length} valid lines
                  </span>
                  <button className="btn" onClick={runBatch}
                    disabled={loading || !batchText.trim()}
                    style={{
                      background: (loading || !batchText.trim()) ? "#e2e8f0" : "linear-gradient(135deg, #1d4ed8, #3b82f6)",
                      border: (loading || !batchText.trim()) ? "1px solid #cbd5e1" : "1px solid #3b82f6",
                      borderRadius: 11, padding: "12px 32px",
                      color: (loading || !batchText.trim()) ? "#94a3b8" : "#ffffff",
                      fontSize: 12, fontWeight: 700,
                      cursor: (loading || !batchText.trim()) ? "not-allowed" : "pointer",
                      display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 10,
                    }}>
                    {loading ? <Spinner color="#ffffff" /> : null}
                    {loading ? "Processing…" : "📋 Run Batch"}
                  </button>
                </div>
              </div>

              {error && (
                <div className="glass-panel" style={{
                  background: "#fef2f2 !important",
                  border: "1px solid #fecaca !important",
                  borderRadius: 15, padding: 18, marginBottom: 18,
                  color: "#dc2626", fontSize: 16, fontWeight: 500
                }}>⚠️ {error}</div>
              )}

              {batchRes && (
                <div className="anim-up">
                  <div style={{ display: "flex", gap: 14, marginBottom: 22 }}>
                    <SummaryCard label="Total Analysed" value={batchRes.total}   color="#2563eb" />
                    <SummaryCard label="🚨 Fake"         value={batchRes.fake}    color="#dc2626" />
                    <SummaryCard label="✅ Genuine"       value={batchRes.genuine} color="#059669" />
                    <SummaryCard
                      label="Fake Rate"
                      value={batchRes.total > 0 ? `${Math.round(batchRes.fake / batchRes.total * 100)}%` : "—"}
                      color="#d97706"
                    />
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                    {batchRes.results.map((r, i) => {
                      const fc = r.prediction === "Fake" ? "#dc2626" : "#059669";
                      const bg = r.prediction === "Fake" ? "#fef2f2" : "#ecfdf5";
                      const border = r.prediction === "Fake" ? "#fecaca" : "#a7f3d0";
                      
                      return (
                        <div key={i} className="glass-panel anim-up" style={{
                          borderLeft: `4px solid ${fc} !important`,
                          borderRadius: 14, padding: "15px 20px",
                          animationDelay: `${i * 0.04}s`,
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 20, marginBottom: 10 }}>
                            <p style={{ margin: 0, fontSize: 16, color: "#1e293b", flex: 1, lineHeight: 1.5 }}>{r.review}</p>
                            <div style={{ textAlign: "right", flexShrink: 0, minWidth: 160 }}>
                              <div style={{ fontSize: 16, fontWeight: 800, color: fc }}>{r.prediction === "Fake" ? "🚨 FAKE" : "✅ GENUINE"}</div>
                              <div style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", marginTop: 3 }}>{pct(r.confidence)} conf.</div>
                            </div>
                          </div>
                          {/* Description for each batch result */}
                          <div style={{
                            background: bg,
                            border: `1px solid ${border}`,
                            borderLeft: `3px solid ${fc}`,
                            borderRadius: 10, padding: "11px 15px",
                          }}>
                            <p style={{ margin: 0, fontSize: 15, color: "#334155", lineHeight: 1.6 }}>{r.description}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ════ HISTORY TAB ════ */}
          {tab === "history" && (
            <div style={{ maxWidth: 1090 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <p style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", letterSpacing: 2.5, textTransform: "uppercase", fontWeight: 700 }}>
                  Recent Analyses ({history.length})
                </p>
                {history.length > 0 && (
                  <button className="btn" onClick={() => setHistory([])} style={{
                    background: "#f1f5f9", border: "1px solid #e2e8f0",
                    borderRadius: 10, padding: "8px 20px",
                    color: "#475569", fontSize: 11, fontFamily: "inherit", fontWeight: 700,
                    display: "inline-flex", alignItems: "center", justifyContent: "center"
                  }}>Clear History</button>
                )}
              </div>
              {!history.length ? (
                <div className="glass-panel" style={{
                  border: "2px dashed #cbd5e1 !important",
                  background: "rgba(255,255,255,0.5) !important",
                  borderRadius: 20, padding: "60px 34px", textAlign: "center",
                  boxShadow: "none"
                }}>
                  <p style={{ color: "#64748b", fontSize: 18, fontWeight: 500 }}>No analyses yet.</p>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                  {history.map((h, i) => {
                    const fc = h.prediction === "Fake" ? "#dc2626" : "#059669";
                    return (
                      <div key={h.ts} className="glass-panel hist-row anim-up" style={{
                        borderLeft: `4px solid ${fc} !important`,
                        borderRadius: 14, padding: "15px 20px",
                        display: "flex", justifyContent: "space-between",
                        alignItems: "center", gap: 20,
                        animationDelay: `${i * 0.03}s`,
                      }}>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <p style={{ fontSize: 16, color: "#1e293b", margin: 0,
                            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{h.snippet}</p>
                          <p style={{ fontSize: 13, color: "#64748b", margin: "6px 0 0", fontFamily: "monospace", fontWeight: 600 }}>
                            {new Date(h.ts).toLocaleTimeString()}
                          </p>
                        </div>
                        <div style={{ textAlign: "right", flexShrink: 0 }}>
                          <div style={{ fontSize: 16, fontWeight: 800, color: fc }}>{h.prediction === "Fake" ? "🚨" : "✅"} {h.prediction}</div>
                          <div style={{ fontSize: 13, color: "#64748b", fontFamily: "monospace", marginTop: 2 }}>{pct(h.confidence)} conf.</div>
                          <div style={{ fontSize: 13, color: RISK_META[h.risk_level].color, marginTop: 4, fontWeight: 600 }}>{h.risk_level} Risk</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

        </main>
      </div>
    </>
  );
}