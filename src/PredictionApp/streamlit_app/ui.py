"""Shared Streamlit pieces: theme, cached model/predictions, common sidebar."""

import datetime as dt
import os

import pandas as pd
import streamlit as st

import engine

# ------------------------------------------------------------------
# Techy theme (injected as CSS so it applies regardless of CWD/config)
# ------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@600;800&display=swap');

:root{
  --cyan:#00e5ff; --green:#2ecc71; --red:#ff3b6b; --amber:#f1c40f;
  --bg0:#05080f; --bg1:#0a0f1a;
  --line:rgba(0,229,255,.16); --line-strong:rgba(0,229,255,.45);
  --txt:#cdd6e4; --txt-dim:#8a93a6;
}

.stApp{
  background:
    radial-gradient(1100px 520px at 85% -10%, rgba(0,229,255,.08), transparent 60%),
    radial-gradient(900px 460px at -10% 110%, rgba(46,204,113,.05), transparent 55%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
  background-attachment: fixed;
  color:var(--txt);
  font-family:'JetBrains Mono', ui-monospace, "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", monospace;
}
/* faint grid, fading out toward the bottom */
.stApp::before{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(0,229,255,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,229,255,.03) 1px,transparent 1px);
  background-size:36px 36px;
  mask-image:radial-gradient(1200px 800px at 50% 0%, black, transparent 85%); }

h1,h2,h3{ font-family:'Orbitron', "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
  letter-spacing:.05em; color:#eaf6ff !important; }
h2,h3{ text-shadow:0 0 18px rgba(0,229,255,.22); }

/* gradient hero page title (used by ui.page_header) */
.mv-hero{ font-family:'Orbitron',sans-serif; font-weight:800; font-size:2.1rem;
  letter-spacing:.06em; padding-bottom:.1em; margin-bottom:.4rem;
  background:linear-gradient(90deg,#eaf6ff 0%, var(--cyan) 55%, #7df9ff 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  filter:drop-shadow(0 0 14px rgba(0,229,255,.30)); }
.mv-hero::after{ content:""; display:block; height:2px; margin-top:.4rem;
  background:linear-gradient(90deg, var(--cyan), transparent 70%);
  box-shadow:0 0 12px rgba(0,229,255,.55); }

/* metric cards — glass, lift + glow on hover */
[data-testid="stMetric"]{
  background:linear-gradient(160deg, rgba(16,23,38,.85), rgba(8,12,20,.85));
  backdrop-filter:blur(8px);
  border:1px solid var(--line); border-radius:14px; padding:14px 16px;
  box-shadow:0 10px 30px -18px rgba(0,0,0,.9);
  transition:transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}
[data-testid="stMetric"]:hover{
  transform:translateY(-3px); border-color:var(--line-strong);
  box-shadow:0 14px 34px -16px rgba(0,229,255,.35); }
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace; color:var(--cyan);
  text-shadow:0 0 14px rgba(0,229,255,.35); }
[data-testid="stMetricLabel"]{ text-transform:uppercase; letter-spacing:.09em;
  font-size:.7rem; opacity:.7; }

/* sidebar — mono text comes from the .stApp cascade; do NOT use a `*` override,
   it clobbers Streamlit's Material icon font and prints ligatures as raw text
   (e.g. "arrow_right" over an expander). */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#060a12 0%, #0a1120 100%);
  border-right:1px solid var(--line); }

/* keep Streamlit's Material icons intact everywhere */
[data-testid="stIconMaterial"], span.material-symbols-rounded,
span.material-symbols-outlined, [class*="material-symbols"]{
  font-family:'Material Symbols Rounded','Material Symbols Outlined' !important; }

/* inputs — subtle focus glow */
.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea,
.stMultiSelect div[data-baseweb="select"]>div, .stSelectbox div[data-baseweb="select"]>div{
  background:#0b111c !important; border:1px solid var(--line) !important;
  color:var(--txt) !important; border-radius:10px !important;
  transition:border-color .2s ease, box-shadow .2s ease; }
.stTextInput input:focus, .stNumberInput input:focus,
.stDateInput input:focus, .stTextArea textarea:focus{
  border-color:var(--line-strong) !important;
  box-shadow:0 0 0 3px rgba(0,229,255,.12) !important; }

/* buttons */
.stButton>button, .stDownloadButton>button{
  background:rgba(0,229,255,.04); border:1px solid var(--cyan); color:var(--cyan);
  border-radius:10px; font-family:'JetBrains Mono';
  transition:background .2s ease, box-shadow .2s ease, transform .2s ease; }
.stButton>button:hover, .stDownloadButton>button:hover{
  background:rgba(0,229,255,.14); box-shadow:0 0 18px rgba(0,229,255,.35);
  transform:translateY(-1px); }

/* expanders */
[data-testid="stExpander"]{
  border:1px solid var(--line); border-radius:12px;
  background:rgba(10,15,26,.55); overflow:hidden; }
[data-testid="stExpander"] summary{ transition:color .2s ease; }
[data-testid="stExpander"] summary:hover{ color:var(--cyan); }

/* dataframes */
[data-testid="stDataFrame"]{
  border:1px solid var(--line); border-radius:12px; overflow:hidden;
  box-shadow:0 10px 26px -20px rgba(0,0,0,.9); }

[data-testid="stHeader"]{ background:transparent; }
hr{ border-color:var(--line) !important; }

/* terminal-style banner with a pulsing status dot */
.tk-banner{ font-family:'JetBrains Mono'; border:1px solid var(--line); border-radius:10px;
  background:linear-gradient(90deg, rgba(0,229,255,.10), rgba(0,229,255,.02) 55%, transparent);
  padding:7px 14px; margin:2px 0 12px 0; color:var(--cyan); font-size:.8rem; letter-spacing:.06em; }
.tk-dot{ height:9px; width:9px; border-radius:50%; display:inline-block; margin-right:8px;
  box-shadow:0 0 10px currentColor; animation:tk-pulse 2.2s ease-in-out infinite; }
@keyframes tk-pulse{ 0%,100%{ opacity:1; box-shadow:0 0 10px currentColor; }
                     50%{ opacity:.45; box-shadow:0 0 3px currentColor; } }

/* small ticker/legend chips */
.mv-chip{ display:inline-block; padding:3px 10px; margin:3px 4px 3px 0;
  border:1px solid var(--line); border-radius:999px; font-size:.75rem;
  color:var(--txt); background:rgba(0,229,255,.05); }

/* scrollbar */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:rgba(0,229,255,.18); border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(0,229,255,.35); }
</style>
"""


# Named theme colors — use these instead of raw hex so a color change is one edit.
CYAN = "#00e5ff"     # up candles, primary accent
RED = "#ff3b6b"      # down candles, sells/shorts
GREEN = "#2ecc71"    # buys/longs
AMBER = "#f1c40f"    # benchmark line (S&P 500)
GRAY = "#8a93a6"     # muted reference lines
GRID = "rgba(0,229,255,.07)"   # chart grid lines
MONO = "JetBrains Mono"        # chart font


# Tooltip text for every metric/indicator, shown on the (?) hover. Kept in one
# place so the same metric reads the same on every page. Each ends with rough
# "fair vs excellent" guideposts.
HELP = {
    "ic":
        "**Rank IC (Spearman).** Rank correlation between the signal "
        "(P(Long)−P(Short)) and the realized forward return — how well the model "
        "*orders* names from worst to best. 0 = no skill.\n\n"
        "Fair ≈ 0.02 · Good ≈ 0.03–0.05 · Excellent ≥ 0.05 · "
        "🚀 World-class ≥ 0.10 (Medallion-tier — a sustained daily IC this high prints "
        "money; on a backtest, suspect a data leak *before* believing it)",
    "icir":
        "**ICIR (annualized).** mean(daily IC) ÷ std(daily IC), annualized. Rewards "
        "*consistency*, not just size — a small steady edge beats a big erratic one.\n\n"
        "Fair ≈ 0.5 · Good ≈ 1.0 · Excellent ≥ 1.5 · "
        "🚀 World-class ≥ 2.5 (an edge that shows up almost every day)",
    "ic_hit_rate":
        "**IC > 0 days.** Share of days the cross-sectional IC was positive — how "
        "*reliably* the edge shows up, ignoring its size. 50% = no edge.\n\n"
        "Fair ≈ 55% · Excellent ≥ 60% · 🚀 World-class ≥ 65% (the edge is there almost daily)",
    "decile":
        "**Decile spread.** Mean forward return of the top signal decile minus the "
        "bottom. Pair it with the decile chart — you want a clean upward staircase, "
        "not just a positive number.\n\n"
        "Fair = clearly positive · Excellent = large *and* monotone across deciles · "
        "🚀 World-class = huge, perfectly monotone staircase every decile",
    "ls_sharpe":
        "**Long-short Sharpe (annualized).** Risk-adjusted return of a book that longs "
        "the top-signal names and shorts the bottom. Gross of costs and overlapping, "
        "so read it as optimistic/relative.\n\n"
        "Fair ≈ 1.0 · Good ≈ 1.5 · Excellent ≥ 2.0 · "
        "🚀 World-class ≥ 4.0 gross (a net Sharpe ~3 is the ~40%/yr, hedge-fund-legend "
        "territory you're after — and the level where leakage is the likeliest explanation)",
    "hit_rate":
        "**Directional hit rate.** Of the *confident* calls (|signal| > 0.1), how often "
        "the signal's sign matched the realized move. 50% = a coin flip.\n\n"
        "Fair ≈ 53% · Good ≈ 54% · Excellent ≥ 55% · "
        "🚀 World-class ≥ 58% (tiny on paper, enormous compounded across many bets)",
    "volatility":
        "**Annualized volatility.** Std of daily returns × √252 — how jumpy the stock "
        "is (context, not a quality score).\n\n"
        "Calm < 20% · Typical large-cap 20–30% · Volatile > 40%",
    "max_dd":
        "**Max drawdown (buy & hold).** Worst peak-to-trough drop from simply holding "
        "the stock over the window. Less negative is better.\n\n"
        "Mild > −20% · Painful −20% to −50% · Severe < −50%",
    "vs_spx":
        "Did the stock *itself* beat the S&P 500 this window — no strategy involved. "
        "Positive = outperformed the market.",
    "signal":
        "**Signal = P(Long) − P(Short)**, range −1…+1. Above 0 leans long, below 0 "
        "leans short, near 0 = no opinion. The continuous score everything ranks on.",
    "confidence":
        "Highest of the three class probabilities. ~0.33 = unsure (coin-flip across 3 "
        "classes); higher = more decisive.",
    "signal_thr":
        "Only take a position when |P(Long)−P(Short)| clears this. Higher = fewer, more "
        "confident trades. Typical 0.1–0.3.",
    "prob_thr":
        "Go long when P(Long) ≥ this, sell when P(Short) ≥ this. With 3 classes the "
        "probabilities rarely top ~0.5, so 0.4–0.5 is already selective.",
    "ma_window":
        "Days in the moving average. Shorter = more reactive but more whipsaws; "
        "50 and 200 are the classic windows.",
    "rsi_period":
        "Lookback for RSI. 14 is standard; shorter reacts faster but is noisier.",
    "rsi_band":
        "Classic thresholds are 30 (oversold → buy) and 70 (overbought → sell). Wider "
        "bands (20/80) trigger rarely, on stronger extremes.",
    "ls_quantile":
        "Fraction of names in each leg of the long-short book. 0.2 = long the top 20% "
        "by signal, short the bottom 20%.",
}


def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def banner(text):
    st.markdown(
        f"<div class='tk-banner'><span class='tk-dot' style='color:{GREEN}'></span>"
        f"{text}</div>", unsafe_allow_html=True)


def page_header(title, subtitle):
    """Shared page top: gradient hero title + status banner."""
    st.markdown(f"<div class='mv-hero'>{title}</div>", unsafe_allow_html=True)
    banner(subtitle)


# One chart config for the whole app: no plotly toolbar (less clutter, less DOM).
PLOTLY_CONFIG = {"displayModeBar": False}


def show_chart(fig, key):
    """Render a plotly figure with the app-wide config. Always pass a unique key."""
    st.plotly_chart(fig, width="stretch", key=key, config=PLOTLY_CONFIG)


def chips(items, colors=None):
    """Render a row of small pill chips. `colors` maps item -> accent color."""
    html = ""
    for item in items:
        style = ""
        if colors is not None and item in colors:
            c = colors[item]
            style = f" style='border-color:{c};color:{c}'"
        html += f"<span class='mv-chip'{style}>{item}</span>"
    st.markdown(html, unsafe_allow_html=True)


def style_chart(fig, height=420):
    """Apply the app's dark theme to any plotly figure (the part every chart shares)."""
    fig.update_layout(template="plotly_dark", height=height,
                      margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=MONO))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def hide_nontrading_days(fig, dates):
    """Collapse weekends and holidays so candles sit flush with no empty gaps."""
    dates = pd.to_datetime(pd.Series(dates))
    all_business_days = pd.date_range(dates.min(), dates.max(), freq="B")
    traded_days = set(dates.dt.normalize())

    holidays = []
    for day in all_business_days:
        if day not in traded_days:
            holidays.append(day)

    breaks = [dict(bounds=["sat", "mon"])]
    if holidays:
        breaks.append(dict(values=holidays))
    fig.update_xaxes(rangebreaks=breaks)
    return fig


# ------------------------------------------------------------------
# Cached compute
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(ckpt_path, _mtime):
    return engine.load_model(ckpt_path)


@st.cache_data(show_spinner=True, ttl=60 * 60)
def get_panel(tickers, end_iso):
    """Download + featurize a universe. Cached on the DATA only (tickers + date),
    NOT the model — so switching checkpoints reuses this instead of re-downloading
    and re-computing every feature. This is the expensive step."""
    return engine.build_panel(tickers, engine.START_DATE, end_iso)


@st.cache_data(show_spinner=True, ttl=60 * 60)
def get_predictions(tickers, ckpt_path, _mtime, end_iso):
    model, _ = get_model(ckpt_path, _mtime)
    panel = get_panel(tickers, end_iso)   # reused across every model
    return engine.predict(panel, model)


@st.cache_data(show_spinner=True, ttl=60 * 60)
def get_single_prediction(ticker, ckpt_path, _mtime, end_iso):
    """Predict one (possibly out-of-universe) ticker fetched live. Cross-sectional
    features are neutralized when ranked against itself — fine for a lookup."""
    model, _ = get_model(ckpt_path, _mtime)
    panel = get_panel((ticker,), end_iso)   # reused across every model
    return engine.predict(panel, model)


# ------------------------------------------------------------------
# Shared sidebar
# ------------------------------------------------------------------
def _checkpoint_label(meta):
    """Short dropdown label built from a parsed checkpoint's metrics."""
    if meta["ic"] is None:
        return meta["name"]   # couldn't parse metrics — show the raw filename
    version = meta["version"]
    version_text = f"v{version}" if version is not None else "v?"
    return (f"ic {meta['ic']:+.3f} · acc {meta['acc']:.2f} · "
            f"e{meta['epoch']:02d} · {version_text}")


def _filter_checkpoints(metas):
    """Sidebar filters (IC range / accuracy range / version). Returns the kept
    checkpoints. Checkpoints whose metrics didn't parse bypass the numeric filters
    so they're never hidden."""
    ic_values = sorted(m["ic"] for m in metas if m["ic"] is not None)
    acc_values = sorted(m["acc"] for m in metas if m["acc"] is not None)
    versions = engine.list_versions()   # every version_N folder in lightning_logs

    with st.sidebar.expander("🔎 Filter models"):
        ic_range = None
        if len(ic_values) >= 2 and ic_values[0] < ic_values[-1]:
            lo, hi = float(ic_values[0]), float(ic_values[-1])
            ic_range = st.slider("IC range", lo, hi, (lo, hi), step=0.001, format="%.3f")

        acc_range = None
        if len(acc_values) >= 2 and acc_values[0] < acc_values[-1]:
            lo, hi = float(acc_values[0]), float(acc_values[-1])
            acc_range = st.slider("Accuracy range", lo, hi, (lo, hi), step=0.01)

        # Dropdown of every version in lightning_logs, plus "All".
        version_choice = st.selectbox(
            "Version", ["All"] + versions,
            format_func=lambda v: "All versions" if v == "All" else f"v{v}")

    kept = []
    for m in metas:
        if ic_range is not None and m["ic"] is not None:
            if m["ic"] < ic_range[0] or m["ic"] > ic_range[1]:
                continue
        if acc_range is not None and m["acc"] is not None:
            if m["acc"] < acc_range[0] or m["acc"] > acc_range[1]:
                continue
        if version_choice != "All" and m["version"] != version_choice:
            continue
        kept.append(m)
    return kept


def pick_checkpoint():
    """Shared sidebar: model/checkpoint selector (with filters) + architecture readout.
    Used by every page so the 'change model' control is consistent."""
    inject_theme()
    st.sidebar.header("⚙️ MODEL")

    metas = engine.list_checkpoint_metas()
    if not metas:
        st.sidebar.error("No checkpoints found. Train first (`python lightning_train.py`).")
        st.stop()

    metas = _filter_checkpoints(metas)
    if not metas:
        st.sidebar.warning("No checkpoints match the filters.")
        st.stop()

    # Best IC first so the strongest model is the default pick.
    metas.sort(key=lambda m: m["ic"] if m["ic"] is not None else float("-inf"), reverse=True)
    labels = {m["path"]: _checkpoint_label(m) for m in metas}
    sorted_paths = [m["path"] for m in metas]

    ckpt = st.sidebar.selectbox("Checkpoint", sorted_paths, format_func=lambda p: labels[p])
    mtime = os.path.getmtime(ckpt)
    _, info = get_model(ckpt, mtime)
    with st.sidebar.expander("Architecture"):
        arch_type = info.get("type", "LSTM")
        hidden_label = "d" if arch_type == "Transformer" else "h"
        st.write(f"{arch_type} · {info['layers']}L · {hidden_label}{info['hidden']} · "
                 f"{info['input']}f · {info['classes']}c")
        if info["missing"]:
            st.warning(f"Missing weights: {info['missing']}")
    return ckpt, mtime, info


def controls():
    ckpt, mtime, info = pick_checkpoint()

    raw = st.sidebar.text_area(
        "Universe — any tickers (comma/space separated)",
        value="AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA", height=78,
        help="Type ANY symbols. These are the names the model ranks across; bigger "
             "= richer cross-sectional ranks but slower on first load.")
    # Split the text box on commas/spaces/newlines into a clean ticker list.
    text = raw.replace("\n", ",").replace(" ", ",")
    universe = []
    for part in text.split(","):
        symbol = part.strip().upper()
        if symbol and symbol not in universe:
            universe.append(symbol)
    universe = sorted(universe)
    if not universe:
        st.sidebar.warning("Enter at least one ticker.")
        st.stop()
    with st.sidebar.expander("Suggested tickers"):
        st.caption(", ".join(engine.UNIVERSE[:40]))

    today = dt.date.today()
    asof = st.sidebar.date_input("Collect data up to (as-of)", value=today,
                                 min_value=dt.date(2016, 1, 1), max_value=today)

    preds = get_predictions(tuple(sorted(universe)), ckpt, mtime, today.isoformat())
    asof_ts = pd.Timestamp(asof)
    view = preds[preds["date"] <= asof_ts]

    st.sidebar.caption(f"{len(view):,} preds · {view['ticker'].nunique()} tickers "
                       f"· thru {asof:%Y-%m-%d}")
    return {"preds": preds, "view": view, "asof": asof_ts, "asof_date": asof,
            "universe": universe, "ckpt": ckpt, "mtime": mtime, "info": info}


def caveats():
    st.caption(
        "⚠️ Research tool. Faint cross-sectional ranking edge (IC ≈ 0.05), not per-stock "
        "precision — judge by the Ranking page, not classification accuracy. Backtest "
        "figures are gross, overlapping-horizon, and exclude costs.")
