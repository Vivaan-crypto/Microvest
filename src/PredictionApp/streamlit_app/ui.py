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

:root{ --cyan:#00e5ff; --green:#2ecc71; --red:#ff3b6b; --amber:#f1c40f;
       --panel:#0d1320; --line:rgba(0,229,255,.18); --txt:#cdd6e4; }

.stApp{
  background:
    radial-gradient(1200px 600px at 80% -10%, rgba(0,229,255,.06), transparent),
    linear-gradient(180deg,#070b12 0%, #0a0f1a 100%);
  background-attachment: fixed;
  color:var(--txt);
  font-family:'JetBrains Mono', ui-monospace, "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", monospace;
}
/* faint grid */
.stApp::before{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(0,229,255,.035) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,229,255,.035) 1px,transparent 1px);
  background-size:34px 34px; }

h1,h2,h3{ font-family:'Orbitron', "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important; letter-spacing:.04em;
  color:#eaf6ff !important; text-shadow:0 0 18px rgba(0,229,255,.25); }
h1{ border-bottom:1px solid var(--line); padding-bottom:.35em; }

/* metric cards */
[data-testid="stMetric"]{
  background:linear-gradient(160deg,rgba(13,19,32,.9),rgba(9,13,22,.9));
  border:1px solid var(--line); border-radius:12px; padding:14px 16px;
  box-shadow:0 0 0 1px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,229,255,.35);
}
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace; color:var(--cyan);
  text-shadow:0 0 12px rgba(0,229,255,.35); }
[data-testid="stMetricLabel"]{ text-transform:uppercase; letter-spacing:.08em;
  font-size:.72rem; opacity:.75; }

/* sidebar — mono text comes from the .stApp cascade; do NOT use a `*` override,
   it clobbers Streamlit's Material icon font and prints ligatures as raw text
   (e.g. "arrow_right" over an expander). */
[data-testid="stSidebar"]{ background:#070b12; border-right:1px solid var(--line); }

/* keep Streamlit's Material icons intact everywhere */
[data-testid="stIconMaterial"], span.material-symbols-rounded,
span.material-symbols-outlined, [class*="material-symbols"]{
  font-family:'Material Symbols Rounded','Material Symbols Outlined' !important; }

/* inputs / buttons */
.stTextInput input, .stDateInput input, .stMultiSelect div[data-baseweb="select"]>div,
.stSelectbox div[data-baseweb="select"]>div{
  background:#0b111c !important; border:1px solid var(--line) !important; color:var(--txt) !important; }
.stButton>button{ background:transparent; border:1px solid var(--cyan); color:var(--cyan);
  border-radius:8px; font-family:'JetBrains Mono'; }
.stButton>button:hover{ background:rgba(0,229,255,.12); box-shadow:0 0 14px rgba(0,229,255,.35); }
[data-testid="stHeader"]{ background:transparent; }
hr{ border-color:var(--line) !important; }

/* the terminal-style banner used on each page */
.tk-banner{ font-family:'JetBrains Mono'; border:1px solid var(--line); border-radius:10px;
  background:linear-gradient(90deg,rgba(0,229,255,.08),transparent);
  padding:6px 14px; margin-bottom:10px; color:var(--cyan); font-size:.8rem; letter-spacing:.06em; }
.tk-dot{ height:9px;width:9px;border-radius:50%;display:inline-block;margin-right:6px;
  box-shadow:0 0 8px currentColor; }
</style>
"""


def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def banner(text):
    st.markdown(
        f"<div class='tk-banner'><span class='tk-dot' style='color:#2ecc71'></span>"
        f"{text}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Cached compute
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(ckpt_path, _mtime):
    return engine.load_model(ckpt_path)


@st.cache_data(show_spinner=True, ttl=60 * 60)
def get_predictions(tickers, ckpt_path, _mtime, end_iso):
    model, _ = get_model(ckpt_path, _mtime)
    panel = engine.build_panel(tickers, engine.START_DATE, end_iso)
    return engine.predict(panel, model)


@st.cache_data(show_spinner=True, ttl=60 * 60)
def get_single_prediction(ticker, ckpt_path, _mtime, end_iso):
    """Predict one (possibly out-of-universe) ticker fetched live. Cross-sectional
    features are neutralized when ranked against itself — fine for a lookup."""
    model, _ = get_model(ckpt_path, _mtime)
    panel = engine.build_panel((ticker,), engine.START_DATE, end_iso)
    return engine.predict(panel, model)


# ------------------------------------------------------------------
# Shared sidebar
# ------------------------------------------------------------------
def pick_checkpoint():
    """Shared sidebar: model/checkpoint selector + architecture readout.
    Used by every page so the 'change model' control is consistent."""
    inject_theme()
    st.sidebar.header("⚙️ MODEL")
    cks = engine.list_checkpoints()
    if not cks:
        st.sidebar.error("No checkpoints found. Train first (`python lightning_train.py`).")
        st.stop()
    ckpt = st.sidebar.selectbox("Checkpoint", cks, format_func=os.path.basename)
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
    universe = sorted({t.strip().upper() for t in raw.replace("\n", ",").replace(" ", ",").split(",") if t.strip()})
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
