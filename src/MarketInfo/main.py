# main.py  (Stooq-based, MACD, indicator toggles, custom tickers support)

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf  # still used for snapshot only
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# -----------------------------------------------------------------------------
# BASE CONFIGURATION (built-in tickers)
# -----------------------------------------------------------------------------

BASE_TICKERS = [
    # --- Technology ---
    ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
    ("NVDA", "Technology"), ("AVGO", "Technology"), ("AMD", "Technology"),
    ("INTC", "Technology"), ("ASML", "Technology"), ("CRM", "Technology"),
    ("ADBE", "Technology"),

    # --- Communication Services ---
    ("META", "Communication Services"), ("NFLX", "Communication Services"),
    ("DIS", "Communication Services"),

    # --- Consumer Discretionary ---
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"), ("TGT", "Consumer Discretionary"),

    # --- Financials ---
    ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    ("GS", "Financials"), ("MS", "Financials"), ("V", "Financials"),
    ("MA", "Financials"), ("BLK", "Financials"),

    # --- Energy ---
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    ("SLB", "Energy"),

    # --- Health Care ---
    ("UNH", "Health Care"), ("ABBV", "Health Care"),
    ("LLY", "Health Care"), ("JNJ", "Health Care"), ("PFE", "Health Care"),

    # --- Consumer Staples ---
    ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    ("COST", "Consumer Staples"), ("WMT", "Consumer Staples"),

    # --- Industrials ---
    ("CAT", "Industrials"), ("BA", "Industrials"), ("UPS", "Industrials"),

    # --- Indices ---
    ("^SPX", "Index"),
    ("^NDX", "Index"),
    ("^DJI", "Index"),

    # --- Watchlist ---
    ("APA", "Watchlist"), ("SOFI", "Watchlist"),
]

BASE_SYMBOLS = [t[0] for t in BASE_TICKERS]
BASE_SECTORS = {t: s for t, s in BASE_TICKERS}
WATCHLIST_TICKERS = [t[0] for t in BASE_TICKERS if t[1] == "Watchlist"]

SNAPSHOT_REFRESH_MS = 30000  # 30 seconds

# -----------------------------------------------------------------------------
# COLORS / THEME
# -----------------------------------------------------------------------------

GREEN, RED = "#10b981", "#f43f5e"
SLATE_50, SLATE_700, SLATE_800, SLATE_900, SLATE_950 = (
    "#f8fafc", "#334155", "#1e293b", "#020617", "#020617"
)
BLUE, ORANGE, PURPLE, YELLOW = "#3b82f6", "#f97316", "#8b5cf6", "#eab308"
WHITE, BLACK = "#ffffff", SLATE_950
CARD_BG, BORDER = SLATE_900, SLATE_800
TEXT_PRIMARY, TEXT_MUTED = SLATE_50, SLATE_700

# Base overlay style: full-screen, hidden by default
OVERLAY_BASE_STYLE = {
    "position": "fixed",
    "top": "0",
    "left": "0",
    "right": "0",
    "bottom": "0",
    "backgroundColor": BLACK,
    "zIndex": "1000",
    "padding": "40px",
    "borderRadius": "20px",
    "display": "none",          # important: not blocking clicks when closed
    "flexDirection": "column",
}

# -----------------------------------------------------------------------------
# DATA LAYER
# -----------------------------------------------------------------------------

def get_snapshot(ticker_tuples) -> pd.DataFrame:
    """
    Snapshot for the heatmap.
    Uses yfinance (batched) for all tickers in ticker_tuples = [(symbol, sector), ...]
    """
    if not ticker_tuples:
        return pd.DataFrame()

    symbols = [t[0] for t in ticker_tuples]
    sectors_map = {t: s for t, s in ticker_tuples}

    try:
        data = yf.download(
            " ".join(symbols),
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        logger.exception("get_snapshot: yf.download failed")
        return pd.DataFrame()

    rows = []
    for symbol in symbols:
        try:
            cols = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
            closes, vols = cols["Close"], cols["Volume"]

            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            change = round(((last - prev) / prev) * 100, 2)
            size = np.log10(max(last * float(vols.iloc[-1]), 1))

            rows.append(
                {
                    "Ticker": symbol,
                    "Sector": sectors_map.get(symbol, "Custom"),
                    "Last": round(last, 2),
                    "Change": change,
                    "Size": size,
                    "Open": cols["Open"],
                    "High": cols["High"],
                    "Low": cols["Low"],
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


# ---------- Free non-yfinance OHLCV source for the chart (Stooq) ----------

def _to_stooq_symbol(ticker: str) -> str:
    """
    Map your ticker symbols to Stooq symbols.

    - Plain US stocks: AAPL -> aapl.us
    - Indices starting with ^: ^SPX -> ^spx
    - If ticker already has a '.' (e.g. SHOP.TO), keep it as-is (lowercased).
    """
    t = ticker.strip()
    if t.startswith("^"):
        return t.lower()
    if "." in t:
        return t.lower()
    return t.lower() + ".us"


def get_hist_ohlcv(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq (free, no key) for the last `days` days.

    Returns DataFrame indexed by Date with columns:
    ['Open', 'High', 'Low', 'Close', 'Volume'] or empty df on failure.
    """
    try:
        symbol = _to_stooq_symbol(ticker)
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url)

        if df.empty:
            return df

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.rename(columns=str.title)
        df = df.set_index("Date").sort_index()

        if days is not None and len(df) > 0:
            cutoff = df.index.max() - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]

        return df
    except Exception:
        logger.exception("get_hist_ohlcv failed for %s", ticker)
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# VISUAL HELPERS
# -----------------------------------------------------------------------------

def build_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return px.scatter(title="No data")

    max_abs = max(1.0, df["Change"].abs().max())

    fig = px.treemap(
        df,
        path=["Sector", "Ticker"],
        values="Size",
        color="Change",
        range_color=[-max_abs, max_abs],
        color_continuous_scale=[(0, RED), (0.5, SLATE_800), (1, GREEN)],
    )

    customdata = np.stack([df["Change"].values, df["Last"].values], axis=-1)

    fig.update_traces(
        customdata=customdata,
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        textfont=dict(size=16, color=WHITE, family="Inter", weight=600),
        hovertemplate=[
            (
                "<b>%{label}</b><br>"
                "Change: %{customdata[0]:+.2f}%<br>"
                "Last: $%{customdata[1]:.2f}<extra></extra>"
            )
            for _ in range(len(df))
        ],
    )

    fig.update_layout(
        margin=dict(t=25, l=5, r=5, b=5),
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=TEXT_PRIMARY, family="Inter"),
        uirevision="keep",
    )
    return fig


def make_info_card(row: dict | None) -> html.Div:
    if not row:
        return html.Div(
            "Select a ticker",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%",
                "background": CARD_BG,
                "borderRadius": "20px",
                "border": f"1px solid {BORDER}",
                "color": TEXT_MUTED,
                "fontSize": "16px",
                "fontWeight": "600",
            },
        )

    color = GREEN if row["Change"] > 0 else (RED if row["Change"] < 0 else TEXT_MUTED)
    arrow = "↑" if row["Change"] > 0 else ("↓" if row["Change"] < 0 else "→")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        row["Ticker"],
                        style={
                            "fontWeight": "800",
                            "fontSize": "32px",
                            "color": TEXT_PRIMARY,
                            "lineHeight": "1",
                        },
                    ),
                    html.Div(
                        row["Sector"],
                        style={
                            "fontSize": "12px",
                            "color": TEXT_MUTED,
                            "marginTop": "6px",
                            "fontWeight": "600",
                            "textTransform": "uppercase",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        f"${row['Last']:.2f}",
                        style={
                            "fontSize": "28px",
                            "color": TEXT_PRIMARY,
                            "fontWeight": "700",
                        },
                    ),
                    html.Div(
                        "Current Price",
                        style={
                            "fontSize": "11px",
                            "color": TEXT_MUTED,
                            "marginTop": "4px",
                            "fontWeight": "600",
                            "textTransform": "uppercase",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    html.Span(
                        arrow,
                        style={"marginRight": "6px", "fontSize": "18px"},
                    ),
                    html.Span(f"{abs(row['Change']):.2f}%"),
                ],
                style={
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": WHITE,
                    "padding": "12px 20px",
                    "borderRadius": "14px",
                    "background": color,
                },
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "auto 1fr auto",
            "gap": "24px",
            "alignItems": "center",
            "background": CARD_BG,
            "borderRadius": "20px",
            "border": f"1px solid {BORDER}",
            "padding": "20px 28px",
        },
    )


def make_chart(row: dict | None, prefs: dict | None = None) -> go.Figure | None:
    """
    Chart using Stooq data.

    prefs keys:
      - sma20, sma50, bb, rsi, macd   (all booleans)
    """
    if not row:
        return None

    ticker = row["Ticker"]
    hist = get_hist_ohlcv(ticker, days=365)
    if hist.empty:
        logger.warning("make_chart: empty history for %s", ticker)
        return go.Figure()

    close = hist["Close"]

    # --- Indicators ---
    hist["SMA20"] = close.rolling(20).mean()
    hist["SMA50"] = close.rolling(50).mean()

    hist["BB_mid"] = close.rolling(20).mean()
    hist["BB_std"] = close.rolling(20).std()
    hist["BB_up"] = hist["BB_mid"] + 2 * hist["BB_std"]
    hist["BB_low"] = hist["BB_mid"] - 2 * hist["BB_std"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    hist["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    hist["MACD"] = ema12 - ema26
    hist["MACD_signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
    hist["MACD_hist"] = hist["MACD"] - hist["MACD_signal"]

    if prefs is None:
        prefs = {"sma20": True, "sma50": True, "bb": True, "rsi": True, "macd": True}

    # --- Subplots: Price, Volume, MACD, RSI ---
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.15, 0.10],
        vertical_spacing=0.04,
    )

    # Row 1: Candles + overlays
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if prefs.get("bb", True):
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["BB_up"],
                mode="lines",
                line=dict(width=1, color=PURPLE, dash="dash"),
                name="BB Upper",
                opacity=0.45,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["BB_low"],
                mode="lines",
                line=dict(width=1, color=PURPLE, dash="dash"),
                name="BB Lower",
                opacity=0.45,
                fill="tonexty",
                fillcolor="rgba(139,92,246,0.08)",
            ),
            row=1,
            col=1,
        )

    if prefs.get("sma20", True):
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["SMA20"],
                mode="lines",
                line=dict(width=2, color=BLUE),
                name="SMA 20",
            ),
            row=1,
            col=1,
        )
    if prefs.get("sma50", True):
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["SMA50"],
                mode="lines",
                line=dict(width=2, color=ORANGE),
                name="SMA 50",
            ),
            row=1,
            col=1,
        )

    # Row 2: Volume
    vol_colors = [
        GREEN if hist["Close"].iloc[i] >= hist["Open"].iloc[i] else RED
        for i in range(len(hist))
    ]
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist["Volume"],
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Row 3: MACD
    if prefs.get("macd", True):
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["MACD"],
                mode="lines",
                line=dict(width=1.5, color=BLUE),
                name="MACD",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["MACD_signal"],
                mode="lines",
                line=dict(width=1.5, color=ORANGE),
                name="Signal",
            ),
            row=3,
            col=1,
        )

        macd_hist_colors = [
            GREEN if v >= 0 else RED for v in hist["MACD_hist"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist["MACD_hist"],
                marker_color=macd_hist_colors,
                name="MACD Hist",
                showlegend=False,
                opacity=0.5,
            ),
            row=3,
            col=1,
        )

    # Row 4: RSI
    if prefs.get("rsi", True):
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["RSI"],
                mode="lines",
                line=dict(width=2, color=YELLOW),
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_hline(
            y=70, line_dash="dash", line_color=RED, opacity=0.5, row=4, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color=GREEN, opacity=0.5, row=4, col=1
        )

    # --- Axes ---
    x_min, x_max = hist.index.min(), (hist.index.max() + timedelta(days=1))
    low_min = float(hist["Low"].min())
    high_max = float(hist["High"].max())
    pad = (high_max - low_min) * 0.05

    fig.update_yaxes(
        title_text="Price",
        range=[low_min - pad, high_max + pad],
        fixedrange=True,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Volume",
        fixedrange=True,
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="MACD",
        fixedrange=True,
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        fixedrange=True,
        row=4,
        col=1,
    )

    # Disable range slider
    for r in range(1, 5):
        fig.update_xaxes(
            rangeslider_visible=False,
            row=r,
            col=1,
        )

    fig.update_xaxes(
        range=[x_min, x_max],
        rangeslider_visible=False,
        showgrid=True,
        gridcolor=BORDER,
        row=4,
        col=1,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
        ],
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title=dict(
            text=f"<b>{ticker}</b>",
            font=dict(size=20, color=TEXT_PRIMARY),
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=80, b=40),
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=TEXT_PRIMARY, size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            y=1.05,
            x=0.5,
            xanchor="center",
            bgcolor=CARD_BG,
            bordercolor=BORDER,
            borderwidth=1,
        ),
        dragmode="zoom",
        hovermode="x unified",
    )

    return fig


# -----------------------------------------------------------------------------
# DASH APP SETUP
# -----------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        # Stores
        dcc.Store(id="snapshot_data"),
        dcc.Store(id="selected_ticker"),
        dcc.Store(
            id="indicator_prefs",
            data={
                "sma20": True,
                "sma50": True,
                "bb": True,
                "rsi": True,
                "macd": True,
            },
        ),
        dcc.Store(id="custom_tickers", data=[]),  # list of {"symbol":..., "sector":...}

        # Main
        html.Div(
            [
                # Back button
                html.Button(
                    [html.Span("← "), html.Span("BACK")],
                    id="nav_back",
                    n_clicks=0,
                    style={
                        "display": "none",
                        "position": "fixed",
                        "top": "20px",
                        "left": "20px",
                        "zIndex": "2001",
                        "padding": "12px 24px",
                        "background": WHITE,
                        "border": "none",
                        "borderRadius": "12px",
                        "color": BLACK,
                        "fontSize": "13px",
                        "fontWeight": "700",
                        "cursor": "pointer",
                    },
                ),

                # Heatmap
                dcc.Graph(
                    id="heatmap",
                    style={
                        "height": "calc(100vh - 300px)",
                        "minHeight": "500px",
                        "borderRadius": "20px",
                    },
                ),

                # Full-screen overlay: chart + overlay info card + indicators
                html.Div(
                    [
                        html.Div(
                            id="chart_content",
                            style={"flex": "1 1 auto", "minHeight": "0"},
                        ),
                        html.Div(
                            id="overlay_info_card",
                            style={"marginTop": "16px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "40px",
                                        "height": "4px",
                                        "borderRadius": "999px",
                                        "backgroundColor": BORDER,
                                        "margin": "0 auto 8px auto",
                                    }
                                ),
                                html.Div(
                                    "INDICATORS",
                                    style={
                                        "color": TEXT_MUTED,
                                        "fontWeight": "700",
                                        "fontSize": "11px",
                                        "letterSpacing": "1px",
                                        "textAlign": "center",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="indicator_checklist",
                                    options=[
                                        {"label": "SMA 20", "value": "sma20"},
                                        {"label": "SMA 50", "value": "sma50"},
                                        {"label": "Bollinger Bands", "value": "bb"},
                                        {"label": "RSI", "value": "rsi"},
                                        {"label": "MACD", "value": "macd"},
                                    ],
                                    value=["sma20", "sma50", "bb", "rsi", "macd"],
                                    inputStyle={"marginRight": "8px"},
                                    labelStyle={
                                        "marginRight": "16px",
                                        "color": TEXT_PRIMARY,
                                        "fontWeight": "600",
                                        "fontSize": "12px",
                                    },
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "justifyContent": "center",
                                        "gap": "8px",
                                    },
                                ),
                            ],
                            style={
                                "marginTop": "16px",
                                "padding": "12px 16px 16px 16px",
                                "background": CARD_BG,
                                "borderRadius": "18px 18px 0 0",
                                "border": f"1px solid {BORDER}",
                                "boxShadow": "0 -10px 30px rgba(15,23,42,0.85)",
                            },
                        ),
                    ],
                    id="chart_overlay",
                    style=OVERLAY_BASE_STYLE.copy(),
                ),
            ],
            style={"flex": 1, "position": "relative", "minHeight": "500px"},
        ),

        # Snapshot refresh
        dcc.Interval(
            id="interval",
            interval=SNAPSHOT_REFRESH_MS,
            n_intervals=0,
        ),

        # Bottom: info card + stats + map tickers control
        html.Div(
            [
                html.Div(
                    [
                        html.Div(id="info_card"),
                        # Small spacer
                        html.Div(style={"height": "12px"}),
                        # Custom tickers control
                        html.Div(
                            [
                                html.Div(
                                    "MARKET MAP TICKERS",
                                    style={
                                        "color": TEXT_MUTED,
                                        "fontWeight": "700",
                                        "fontSize": "11px",
                                        "letterSpacing": "1px",
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="new_ticker_input",
                                            placeholder="Ticker (e.g. SHOP)",
                                            style={
                                                "backgroundColor": SLATE_900,
                                                "border": f"1px solid {BORDER}",
                                                "borderRadius": "10px",
                                                "padding": "6px 10px",
                                                "color": TEXT_PRIMARY,
                                                "width": "40%",
                                                "fontSize": "12px",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="new_ticker_sector",
                                            options=[
                                                {"label": "Technology", "value": "Technology"},
                                                {"label": "Communication Services", "value": "Communication Services"},
                                                {"label": "Consumer Discretionary", "value": "Consumer Discretionary"},
                                                {"label": "Financials", "value": "Financials"},
                                                {"label": "Energy", "value": "Energy"},
                                                {"label": "Health Care", "value": "Health Care"},
                                                {"label": "Consumer Staples", "value": "Consumer Staples"},
                                                {"label": "Industrials", "value": "Industrials"},
                                                {"label": "Index", "value": "Index"},
                                                {"label": "Watchlist", "value": "Watchlist"},
                                                {"label": "Custom", "value": "Custom"},
                                            ],
                                            value="Custom",
                                            clearable=False,
                                            style={
                                                "backgroundColor": SLATE_900,
                                                "borderRadius": "10px",
                                                "border": f"1px solid {BORDER}",
                                                "color": TEXT_PRIMARY,
                                                "width": "35%",
                                                "fontSize": "12px",
                                            },
                                        ),
                                        html.Button(
                                            "+ ADD",
                                            id="add_ticker_btn",
                                            n_clicks=0,
                                            style={
                                                "marginLeft": "8px",
                                                "padding": "6px 12px",
                                                "borderRadius": "10px",
                                                "border": "none",
                                                "background": GREEN,
                                                "color": WHITE,
                                                "fontSize": "12px",
                                                "fontWeight": "700",
                                                "cursor": "pointer",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "8px",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="custom_tickers_checklist",
                                    options=[],
                                    value=[],
                                    labelStyle={
                                        "marginRight": "10px",
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "11px",
                                    },
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "4px",
                                    },
                                ),
                            ],
                            style={
                                "background": CARD_BG,
                                "borderRadius": "14px",
                                "border": f"1px solid {BORDER}",
                                "padding": "12px 14px",
                                "marginTop": "8px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            "MARKET STATS",
                            style={
                                "color": TEXT_MUTED,
                                "fontWeight": "700",
                                "fontSize": "11px",
                                "letterSpacing": "1px",
                            },
                        ),
                        html.Div(id="stats"),
                    ],
                    style={
                        "background": CARD_BG,
                        "borderRadius": "20px",
                        "border": f"1px solid {BORDER}",
                        "padding": "20px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 1fr",
                "height": "300%",
                "gap": "16px",
            },
        ),
    ],
    style={
        "backgroundColor": BLACK,
        "minHeight": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "fontFamily": "Inter, sans-serif",
        "padding": "16px",
        "gap": "16px",
    },
)

# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------

@app.callback(
    Output("indicator_prefs", "data"),
    Input("indicator_checklist", "value"),
)
def persist_indicator_prefs(vals):
    try:
        vals = vals or []
        return {
            "sma20": "sma20" in vals,
            "sma50": "sma50" in vals,
            "bb": "bb" in vals,
            "rsi": "rsi" in vals,
            "macd": "macd" in vals,
        }
    except Exception:
        logger.exception("persist_indicator_prefs failed")
        return {
            "sma20": True,
            "sma50": True,
            "bb": True,
            "rsi": True,
            "macd": True,
        }


@app.callback(
    Output("custom_tickers", "data"),
    Output("custom_tickers_checklist", "options"),
    Output("custom_tickers_checklist", "value"),
    Input("add_ticker_btn", "n_clicks"),
    Input("custom_tickers_checklist", "value"),
    State("new_ticker_input", "value"),
    State("new_ticker_sector", "value"),
    State("custom_tickers", "data"),
    prevent_initial_call=True,
)
def manage_custom_tickers(add_clicks, selected_symbols, new_symbol, new_sector, current_list):
    """
    - When ADD is clicked: append new ticker (if valid) to custom list.
    - When checklist changes: treat un-checked symbols as removed.
    """
    try:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        current_list = current_list or []
        selected_symbols = selected_symbols or []

        # Helper to rebuild options + values
        def build_outputs(lst):
            opts = [
                {"label": f"{t['symbol']} ({t['sector']})", "value": t["symbol"]}
                for t in lst
            ]
            vals = [t["symbol"] for t in lst]
            return lst, opts, vals

        # Checklist changed -> removal
        if triggered == "custom_tickers_checklist":
            filtered = [t for t in current_list if t["symbol"] in selected_symbols]
            return build_outputs(filtered)

        # Add button clicked
        if triggered == "add_ticker_btn":
            if not new_symbol:
                # nothing to add; just reflect current state
                return build_outputs(current_list)

            symbol = new_symbol.strip().upper()
            if not symbol:
                return build_outputs(current_list)

            sector = new_sector or "Custom"

            # dedupe
            if symbol not in [t["symbol"] for t in current_list]:
                current_list.append({"symbol": symbol, "sector": sector})

            return build_outputs(current_list)

        # Fallback: no real trigger
        return build_outputs(current_list)

    except Exception:
        logger.exception("manage_custom_tickers failed")
        return current_list or [], [], []


@app.callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("interval", "n_intervals"),
    State("custom_tickers", "data"),
)
def update_snapshot_cb(n: int, custom_tickers):
    try:
        # combine base + custom
        ticker_tuples = BASE_TICKERS.copy()
        custom_tickers = custom_tickers or []
        existing_symbols = {t[0] for t in ticker_tuples}

        for ct in custom_tickers:
            sym = ct.get("symbol")
            sector = ct.get("sector", "Custom")
            if sym and sym not in existing_symbols:
                ticker_tuples.append((sym, sector))
                existing_symbols.add(sym)

        df = get_snapshot(ticker_tuples)
        fig = build_heatmap(df)
        data = df.to_dict("records")
        return fig, data
    except Exception:
        logger.exception("update_snapshot_cb failed")
        return px.scatter(title="No data"), []


@app.callback(
    Output("info_card", "children"),
    Output("stats", "children"),
    Input("snapshot_data", "data"),
    State("selected_ticker", "data"),
)
def update_info(data, ticker):
    try:
        if not data:
            return make_info_card(None), html.Div()

        df = pd.DataFrame(data)

        if ticker and ticker in df["Ticker"].values:
            row_dict = df[df["Ticker"] == ticker].iloc[0].to_dict()
        else:
            row_dict = None
        info = make_info_card(row_dict)

        gainers = (df["Change"] > 0).sum()
        losers = (df["Change"] < 0).sum()
        avg = df["Change"].mean()

        watchlist_mask = df["Ticker"].isin(WATCHLIST_TICKERS)
        watchlist_df = df[watchlist_mask]

        if not watchlist_df.empty:
            watchlist_gainers = (watchlist_df["Change"] > 0).sum()
            watchlist_losers = (watchlist_df["Change"] < 0).sum()
            watchlist_avg = watchlist_df["Change"].mean()
        else:
            watchlist_gainers = watchlist_losers = 0
            watchlist_avg = 0.0

        stats = html.Div(
            [
                html.Div(
                    "MARKET",
                    style={
                        "color": TEXT_MUTED,
                        "fontWeight": "700",
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Gainers",
                                    style={
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Span(
                                    str(gainers),
                                    style={
                                        "color": GREEN,
                                        "fontSize": "18px",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Losers",
                                    style={
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Span(
                                    str(losers),
                                    style={
                                        "color": RED,
                                        "fontSize": "18px",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Avg",
                                    style={
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Span(
                                    f"{avg:+.2f}%",
                                    style={
                                        "color": GREEN if avg > 0 else RED,
                                        "fontSize": "18px",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    style={
                        "height": "1px",
                        "backgroundColor": BORDER,
                        "margin": "10px 0",
                    }
                ),
                html.Div(
                    "WATCHLIST",
                    style={
                        "color": TEXT_MUTED,
                        "fontWeight": "700",
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Gainers",
                                    style={
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Span(
                                    str(watchlist_gainers),
                                    style={
                                        "color": GREEN,
                                        "fontSize": "16px",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Losers",
                                    style={
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Span(
                                    str(watchlist_losers),
                                    style={
                                        "color": RED,
                                        "fontSize": "16px",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Avg",
                                    style={
                                        "color": TEXT_PRIMARY,
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Span(
                                    f"{watchlist_avg:+.2f}%",
                                    style={
                                        "color": GREEN
                                        if watchlist_avg > 0
                                        else RED,
                                        "fontSize": "16px",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    style={
                        "height": "1px",
                        "backgroundColor": BORDER,
                        "margin": "10px 0 6px",
                    }
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "4px"},
        )

        return info, stats

    except Exception:
        logger.exception("update_info failed")
        return make_info_card(None), html.Div()


@app.callback(
    Output("selected_ticker", "data"),
    Output("chart_overlay", "style"),
    Output("nav_back", "style"),
    Input("heatmap", "clickData"),
    Input("nav_back", "n_clicks"),
    State("snapshot_data", "data"),
    State("selected_ticker", "data"),
    State("chart_overlay", "style"),
)
def handle_click(click, nav, data, ticker, current_style):
    """
    Open overlay when user clicks a heatmap cell.
    Close overlay when user clicks BACK.
    """
    try:
        hide = OVERLAY_BASE_STYLE.copy()
        show_nav_hidden = {
            **{
                "display": "none",
                "position": "fixed",
                "top": "20px",
                "left": "20px",
                "zIndex": "2001",
                "padding": "12px 24px",
                "background": WHITE,
                "border": "none",
                "borderRadius": "12px",
                "color": BLACK,
                "fontSize": "13px",
                "fontWeight": "700",
                "cursor": "pointer",
            }
        }
        show_nav_visible = show_nav_hidden.copy()
        show_nav_visible["display"] = "block"

        if not data:
            return None, hide, show_nav_hidden

        df = pd.DataFrame(data)
        triggered = (
            callback_context.triggered[0]["prop_id"]
            if callback_context.triggered
            else ""
        )

        # BACK -> close overlay
        if triggered == "nav_back.n_clicks":
            return ticker, hide, show_nav_hidden

        # No heatmap click => keep closed
        if not click or "points" not in click:
            return ticker, hide, show_nav_hidden

        label = click["points"][0].get("label")
        if label not in df["Ticker"].values:
            return ticker, hide, show_nav_hidden

        show = OVERLAY_BASE_STYLE.copy()
        show["display"] = "flex"

        return label, show, show_nav_visible

    except Exception:
        logger.exception("handle_click failed")
        return None, OVERLAY_BASE_STYLE.copy(), {
            "display": "none",
            "position": "fixed",
            "top": "20px",
            "left": "20px",
            "zIndex": "2001",
            "padding": "12px 24px",
            "background": WHITE,
            "border": "none",
            "borderRadius": "12px",
            "color": BLACK,
            "fontSize": "13px",
            "fontWeight": "700",
            "cursor": "pointer",
        }


@app.callback(
    Output("chart_content", "children"),
    Input("selected_ticker", "data"),
    Input("indicator_prefs", "data"),
    State("snapshot_data", "data"),
)
def update_chart_content(ticker, indicator_prefs, data):
    try:
        if not ticker or not data:
            return None

        df = pd.DataFrame(data)
        if ticker not in df["Ticker"].values:
            return None

        row = df[df["Ticker"] == ticker].iloc[0].to_dict()
        fig = make_chart(row, prefs=indicator_prefs)

        return dcc.Graph(
            figure=fig,
            config={"displayModeBar": True, "scrollZoom": True},
            style={"width": "100%", "height": "100%"},
        )
    except Exception:
        logger.exception("update_chart_content failed")
        return None


@app.callback(
    Output("overlay_info_card", "children"),
    Input("snapshot_data", "data"),
    Input("selected_ticker", "data"),
)
def update_overlay_info_card(data, ticker):
    """
    Info card shown INSIDE the overlay (so when chart is open, you still see the card).
    """
    try:
        if not data or not ticker:
            return html.Div()

        df = pd.DataFrame(data)
        if ticker not in df["Ticker"].values:
            return html.Div()

        row_dict = df[df["Ticker"] == ticker].iloc[0].to_dict()
        return make_info_card(row_dict)
    except Exception:
        logger.exception("update_overlay_info_card failed")
        return html.Div()


# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
