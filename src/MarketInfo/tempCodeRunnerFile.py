import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

# ---------------- CONFIG ----------------

TICKERS = [
    ("AAPL", "Technology"),
    ("MSFT", "Technology"),
    ("GOOGL", "Technology"),
    ("AMZN", "Consumer Discretionary"),
    ("META", "Communication Services"),
    ("NVDA", "Technology"),
    ("TSLA", "Consumer Discretionary"),
    # ("AVGO", "Technology"),
    # ("AMD", "Technology"),
    # ("INTC", "Technology"),
    # ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    # ("GS", "Financials"), ("MS", "Financials"), ("V", "Financials"),
    # ("MA", "Financials"), ("PYPL", "Financials"),
    # ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    # ("UNH", "Health Care"), ("PFE", "Health Care"),
    # ("ABBV", "Health Care"), ("LLY", "Health Care"),
    # ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    # ("WMT", "Consumer Staples"), ("COST", "Consumer Staples"),
    # ("HD", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    # ("NKE", "Consumer Discretionary"), ("DIS", "Communication Services"),
    # ("SPY", "ETF"), ("QQQ", "ETF"), ("IWM", "ETF"),
    # ("XLK", "ETF"), ("XLF", "ETF"), ("XLE", "ETF"),
    # ("XLV", "ETF"), ("XLU", "ETF"),
]

SYMBOLS = [t[0] for t in TICKERS]
SECTORS = {t: s for t, s in TICKERS}

SNAPSHOT_REFRESH_MS = 2000  # 5 seconds

# ---------------- COLORS ----------------

GREEN = "#00ff7f"
RED = "#ff4d4d"
GREY = "#9ba4b5"
WHITE = "#ffffff"
BLACK = "#090909"
DARK_GREY = "#222222"

PANEL_BG = "#060812"
PANEL_BORDER = "#262b3a"
DASHED_BORDER = "#2b3142"
MUTED_TEXT = "#666f82"
MUTED_TEXT_ALT = "#555e70"

HEADER_GRADIENT_START = "#15192b"
HEADER_GRADIENT_END = "#1b2838"
HOVER_NEUTRAL = "#0a0d15"  # unused now but handy if you style more later

IN_TILE = False


# ---------------- DATA: SNAPSHOT ONLY ----------------


def get_snapshot():
    """
    Pull ~1 week of hourly data for all tickers,
    compute last 1-hour % change and approximate size by dollar volume.
    Only function that hits yfinance -> keeps app fast.
    """
    data = yf.download(
        " ".join(SYMBOLS),
        period="2d",
        interval="1h",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    data.index = data.index.tz_convert("America/New_York")

    rows = []
    for s in SYMBOLS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                cols = data[s]
            else:
                cols = data

            closes = cols["Close"]
            vols = cols["Volume"]

            last_close = float(closes.iloc[-1])
            change_pct = closes.pct_change().iloc[-1] * 100.0

            last_vol = float(vols.iloc[-1])
            size_val = max(last_close * last_vol, 1.0)  # dollar volume-ish

            rows.append(
                {
                    "Ticker": s,
                    "Sector": SECTORS.get(s, "Unknown"),
                    "Last": last_close,
                    "Change": change_pct,
                    "Size": np.log10(size_val),
                    "Open": cols["Open"],
                    "High": cols["High"],
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)

def build_heatmap(df: pd.DataFrame):
    global IN_TILE
    if df.empty:
        return px.scatter(title="No data")

    # Prevent color washout if moves are tiny
    max_abs = max(1.0, df["Change"].abs().max())

    fig = px.treemap(
        df,
        path=["Sector", "Ticker"],
        values="Size",
        color="Change",
        range_color=[-max_abs, max_abs],
        color_continuous_scale=[
            (0.0, RED),  # hot red
            (0.5, DARK_GREY),  # dark neutral
            (1.0, GREEN),  # neon green
        ],
        hover_data={},  # keep hover minimal
    )
    # Required to show additional info (change + last) in text/hover
    customdata = np.stack([df["Change"].values, df["Last"].values], axis=-1)

    fig.update_traces(
        customdata=customdata,
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%",
        textfont=dict(size=13, color=WHITE),
        hovertemplate=[
            (
                ""
                if "/" not in p
                else "<b>%{label}</b><br>Change: %{customdata[0]:.2f}%<br>Last: $%{customdata[1]:.2f}<extra></extra>"
            )
            for p in df["Sector"] + "/" + df["Ticker"]
        ],
    )
    if IN_TILE:
        dicts = dict(t=25, l=2, r=2, b=10)
    else:
        dicts = dict(t=25, l=2, r=2, b=10)
    fig.update_layout(
        margin=dicts,
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=WHITE),
        uirevision="keep-treemap-state",
    )

    return fig

def make_info_card(row: dict | None):
    global IN_TILE
    """
    Build the bottom info card from a single row of the snapshot.
    No extra data calls.
    """
    if row is None:
        IN_TILE = False
        return html.Div(
            "Click a tile to see details.",
            style={"color": WHITE, "fontSize": "14px"},
        )
    else:
        IN_TILE = True

    color_change = GREEN if row["Change"] > 0 else (RED if row["Change"] < 0 else GREY)

    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "background": DARK_GREY,
            "borderRadius": "8px",
            "padding": "8px 12px",
        },
        children=[
            html.Div(
                children=[
                    html.Div(
                        row["Ticker"],
                        style={"fontWeight": "700", "fontSize": "18px", "color": WHITE},
                    ),
                    html.Div(
                        row["Sector"],
                        style={"fontSize": "12px", "color": GREY},
                    ),
                ]
            ),
            html.Div(
                children=[
                    html.Div(
                        f"${row['Last']:.2f}",
                        style={
                            "fontSize": "18px",
                            "textAlign": "right",
                            "color": WHITE,
                        },
                    ),
                    html.Div(
                        "Last price",
                        style={"fontSize": "12px", "color": GREY, "textAlign": "right"},
                    ),
                ]
            ),
            html.Div(
                f"{row['Change']:+.2f}%",
                style={
                    "fontSize": "14px",
                    "fontWeight": "600",
                    "color": color_change,
                    "padding": "4px 8px",
                    "borderRadius": "999px",
                    "border": f"2px solid {color_change}",
                },
            ),
        ],
    )

def make_ticker_graph(row: dict | None):
    if row is None:
        return html.Div(
            "Click a tile to see a price chart.",
            style={"color": MUTED_TEXT_ALT, "fontSize": "13px"},
        )

    symbol = row["Ticker"]

    # Fetch intraday data
    hist = yf.download(
        symbol,
        period="1mo",
        interval="1h",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    # Indicators
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()

    # Price (row 1) + Volume (row 2)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # SMAs
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["SMA20"],
            mode="lines",
            line=dict(width=1.3, color=GREY),
            name="SMA20",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["SMA50"],
            mode="lines",
            line=dict(width=1.2, color="#f5f5f5"),
            name="SMA50",
        ),
        row=1,
        col=1,
    )

    # Volume bars (clean, no candles)
    vol_colors = np.where(hist["Close"] >= hist["Open"], GREEN, RED)
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist["Volume"],
            marker_color=vol_colors,
            opacity=0.5,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    # Volume axis
    fig.update_yaxes(
        title_text="Volume",
        showgrid=False,
        row=2,
        col=1,
    )

    # X-axis formatting + range breaks (applied to both rows)
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # skip weekends
            dict(bounds=[16.5, 9.5], pattern="hour"),  # skip after-hours
            dict(bounds=[hist.index.min(), hist.index.max()])
        ],
        tickformat="%b %d<br>%H:%M",
        rangeslider_visible=False,
        showgrid=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        rangebreaks=[
            dict(bounds=[float(hist['Low'].min()) * 0.8, float(hist['High'].max()) * 1.2])
        ],
        showgrid=True,
        row=1,
        col=1)
    # Layout
    fig.update_layout(
        title=f"{symbol}",
        margin=dict(l=10, r=10, t=30, b=20),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=WHITE, size=11),
        showlegend=False,
        dragmode="pan",
    )

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "scrollZoom": True},
        style={"width": "100%", "height": "100%"},
    )


# ---------------- DASH APP ----------------

app = Dash(__name__)

# ---------------- CALLBACKS ----------------
app.layout = html.Div(
    style={
        "backgroundColor": BLACK,
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
    },
    children=[
        dcc.Store(id="snapshot_data"),
        dcc.Store(id="selected_ticker"),
        # Main area: heatmap
        html.Div(
            style={"flex": 1},
            children=[
                dcc.Graph(id="heatmap", style={"height": "90vh"}),
            ],
        ),
        dcc.Interval(
            id="snapshot_interval", interval=SNAPSHOT_REFRESH_MS, n_intervals=0
        ),
        # Bottom: left = info + empty chart GUI, right = empty news GUI
        html.Div(
            style={
                "display": "flex",
                "height": "30vh",
                "marginTop": "6px",
                "gap": "8px",
            },
            children=[
                # Left side
                html.Div(
                    style={
                        "flex": 2,
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                    children=[
                        html.Div(id="info_card"),
                        html.Div(
                            # chart GUI placeholder (no logic)
                            style={
                                "flex": 1,
                                "background": PANEL_BG,
                                "borderRadius": "8px",
                                "border": f"1px dashed {DASHED_BORDER}",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "color": WHITE,
                                "fontSize": "14px",
                                "padding": "8px 12px",
                            },
                            id="ticker_graph",
                        ),
                    ],
                ),
                # Right side
                html.Div(
                    style={
                        "flex": 1,
                        "background": PANEL_BG,
                        "borderRadius": "8px",
                        "border": f"1px solid {PANEL_BORDER}",
                        "padding": "6px 8px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                    children=[
                        html.Div(
                            "Market News (GUI only – logic disabled)",
                            style={
                                "color": WHITE,
                                "fontWeight": "600",
                                "fontSize": "13px",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            style={
                                "flex": 1,
                                "overflowY": "auto",
                                "color": MUTED_TEXT,
                                "fontSize": "12px",
                            },
                            children=[
                                html.Div(
                                    "This panel is ready for future news integration."
                                ),
                                html.Div("Right now it’s just a styled placeholder."),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# 1) Refresh snapshot + heatmap
@app.callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("snapshot_interval", "n_intervals"),
)
def update_snapshot(n):
    df = get_snapshot()
    fig = build_heatmap(df)
    return fig, df.to_dict("records")


# 2) Handle tile click -> update selected ticker + info card
@app.callback(
    Output("selected_ticker", "data"),
    Output("info_card", "children"),
    Output("ticker_graph", "children"),
    Input("heatmap", "clickData"),
    State("snapshot_data", "data"),
    State("selected_ticker", "data"),
)
def on_tile_click(click_data, snapshot_data, current_ticker):
    if not snapshot_data:
        return current_ticker, make_info_card(None), make_ticker_graph(None)

    df = pd.DataFrame(snapshot_data)

    if not click_data or "points" not in click_data:
        # No click: if already selected, show that; else blank
        if current_ticker and current_ticker in df["Ticker"].values:
            print(current_ticker)
            row = df[df["Ticker"] == current_ticker].iloc[0].to_dict()
            return current_ticker, make_info_card(row), make_ticker_graph(row)
        return None, make_info_card(None), make_ticker_graph(None)

    label = click_data["points"][0].get("label")
    if label not in df["Ticker"].values:
        return current_ticker, make_info_card(None), make_ticker_graph(None)

    row = df[df["Ticker"] == label].iloc[0].to_dict()
    return label, make_info_card(row), make_ticker_graph(row)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)