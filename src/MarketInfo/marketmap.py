import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from datetime import timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS = [
    ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
    ("NVDA", "Technology"), ("AVGO", "Watchlist"), ("AMD", "Technology"),
    ("INTC", "Technology"), ("ASML", "Technology"), ("CRM", "Technology"),
    ("ADBE", "Technology"),
    ("META", "Communication Services"), ("NFLX", "Communication Services"),
    ("DIS", "Communication Services"),
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"), ("TGT", "Consumer Discretionary"),
    ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    ("GS", "Financials"), ("MS", "Financials"), ("V", "Financials"),
    ("MA", "Financials"), ("BLK", "Financials"),
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    ("SLB", "Energy"),
    ("UNH", "Health Care"), ("ABBV", "Health Care"),
    ("LLY", "Health Care"), ("JNJ", "Health Care"), ("PFE", "Health Care"),
    ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    ("COST", "Consumer Staples"), ("WMT", "Consumer Staples"),
    ("CAT", "Industrials"), ("BA", "Industrials"), ("UPS", "Industrials"),
    ("^SPX", "Index"), ("^NDX", "Index"), ("^DJI",  "Index"),
    ("APA", "Energy"), ("SOFI", "Watchlist")
]

SYMBOLS = [t[0] for t in TICKERS]
SECTORS = {t: s for t, s in TICKERS}
WATCHLIST_TICKERS = [t[0] for t in TICKERS if t[1] == "Watchlist"]

SNAPSHOT_REFRESH_MS = 5000

# =============================================================================
# COLORS / THEME
# =============================================================================

GREEN, RED = "#10b981", "#f43f5e"
SLATE_50, SLATE_700, SLATE_800, SLATE_900, SLATE_950 = (
    "#f8fafc", "#334155", "#1e293b", "#0f172a", "#020617"
)
BLUE, ORANGE, PURPLE, YELLOW = "#3b82f6", "#f97316", "#8b5cf6", "#eab308"
WHITE, BLACK = "#ffffff", SLATE_950
CARD_BG, BORDER = SLATE_900, SLATE_800
TEXT_PRIMARY, TEXT_MUTED = SLATE_50, SLATE_700

# =============================================================================
# DATA LAYER (unchanged)
# =============================================================================

def get_snapshot() -> pd.DataFrame:
    data = yf.download(
        " ".join(SYMBOLS),
        period="2d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    rows = []
    for symbol in SYMBOLS:
        try:
            cols = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
            closes, vols = cols["Close"], cols["Volume"]

            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            change = round(((last - prev) / prev) * 100, 2)
            size = np.log10(max(last * float(vols.iloc[-1]), 1))

            rows.append({
                "Ticker": symbol,
                "Sector": SECTORS.get(symbol, "Unknown"),
                "Last": round(last, 2),
                "Change": change,
                "Size": size,
                "Open": cols["Open"],
                "High": cols["High"],
                "Low": cols["Low"],
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

# =============================================================================
# VISUALIZATION HELPERS (unchanged)
# =============================================================================

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
            html.Div([
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
            ]),

            html.Div([
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
            ]),

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


def make_chart(row: dict | None) -> go.Figure | None:
    if not row:
        return None

    ticker = row["Ticker"]

    hist = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if hist.empty:
        return go.Figure()

    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    hist["BB_mid"] = hist["Close"].rolling(20).mean()
    hist["BB_std"] = hist["Close"].rolling(20).std()
    hist["BB_up"] = hist["BB_mid"] + 2 * hist["BB_std"]
    hist["BB_low"] = hist["BB_mid"] - 2 * hist["BB_std"]

    delta = hist["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    hist["RSI"] = 100 - (100 / (1 + gain / loss))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.25, 0.15],
        vertical_spacing=0.06,
    )

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

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["RSI"],
            mode="lines",
            line=dict(width=2, color=YELLOW),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=70, line_dash="dash", line_color=RED, opacity=0.5, row=3, col=1
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color=GREEN, opacity=0.5, row=3, col=1
    )

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
        title_text="RSI",
        range=[0, 100],
        fixedrange=True,
        row=3,
        col=1,
    )

    fig.update_xaxes(
        range=[x_min, x_max],
        rangeslider_visible=False,
        showgrid=True,
        gridcolor=BORDER,
        row=3,
        col=1,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Skip weekends
        ],
    )

    fig.update_layout(
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

# =============================================================================
# DASH APP SETUP (single-page)
# =============================================================================

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Store(id="snapshot_data"),
        dcc.Store(id="selected_ticker"),

        # Main dashboard content
        html.Div(
            [
                html.Button([html.Span("← "), html.Span("BACK")], id="nav_back", n_clicks=0,
                    style={"display":"none","position":"absolute","top":"20px","left":"20px","zIndex":"2000","padding":"12px 24px",
                           "background": WHITE,"border": "none","borderRadius": "12px","color": BLACK,"fontSize": "13px","fontWeight": "700","cursor":"pointer"}),

                dcc.Graph(id="heatmap", style={"height":"calc(100vh - 300px)","minHeight":"500px","borderRadius":"20px"}),

                html.Div(html.Div(id="chart_content", style={"height":"100%"}, n_clicks=0),
                         id="chart_overlay", n_clicks=0,
                         style={"position":"absolute","top":"0","left":"0","right":"0","bottom":"0","opacity":"0","pointerEvents":"none",
                                "backgroundColor":BLACK,"zIndex":"1000","padding":"40px","borderRadius":"20px","transition":"opacity 0.3s ease"}),
            ],
            style={"flex":1,"position":"relative","minHeight":"500px"},
        ),

        dcc.Interval(id="interval", interval=SNAPSHOT_REFRESH_MS, n_intervals=0),

        html.Div([html.Div(id="info_card"),
                  html.Div([html.Div("MARKET STATS", style={"color": TEXT_MUTED, "fontWeight": "700", "fontSize": "11px", "letterSpacing": "1px"}),
                            html.Div(id="stats")],
                           style={"background": CARD_BG, "borderRadius": "20px", "border": f"1px solid {BORDER}", "padding": "20px", "display": "flex", "flexDirection": "column", "gap": "12px"})],
                 style={"display":"grid","gridTemplateColumns":"2fr 1fr","height":"300%","gap":"16px"}),
    ],
    style={"backgroundColor":BLACK,"minHeight":"100vh","display":"flex","flexDirection":"column","fontFamily":"Inter, sans-serif","padding":"16px","gap":"16px"},
)

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("interval", "n_intervals"),
)
def update_snapshot(n: int):
    df = get_snapshot()
    return build_heatmap(df), df.to_dict("records")


@app.callback(
    Output("info_card", "children"),
    Output("stats", "children"),
    Input("snapshot_data", "data"),
    State("selected_ticker", "data"),
)
def update_info(data, ticker):
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
                    html.Div([html.Span("Gainers", style={"color": TEXT_PRIMARY, "fontSize": "13px"}), html.Span(str(watchlist_gainers), style={"color": GREEN, "fontSize": "16px", "fontWeight": "700"})], style={"display":"flex","justifyContent":"space-between"}),
                    html.Div([html.Span("Losers", style={"color": TEXT_PRIMARY, "fontSize": "13px"}), html.Span(str(watchlist_losers), style={"color": RED, "fontSize":"16px","fontWeight":"700"})], style={"display":"flex","justifyContent":"space-between"}),
                    html.Div([html.Span("Avg", style={"color": TEXT_PRIMARY, "fontSize":"13px"}), html.Span(f"{watchlist_avg:+.2f}%", style={"color": GREEN if watchlist_avg > 0 else RED,"fontSize":"16px","fontWeight":"700"})], style={"display":"flex","justifyContent":"space-between"}),
                ],
                style={"display":"flex","flexDirection":"column","gap":"6px"},
            ),

            html.Div(
                style={
                    "height": "1px",
                    "backgroundColor": BORDER,
                    "margin": "10px 0 6px",
                }
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
        },
    )

    return info, stats


@app.callback(
    Output("selected_ticker", "data"),
    Output("chart_overlay", "style"),
    Output("chart_content", "children"),
    Output("nav_back", "style"),
    Input("heatmap", "clickData"),
    Input("nav_back", "n_clicks"),
    Input("chart_overlay", "n_clicks"),
    State("snapshot_data", "data"),
    State("selected_ticker", "data"),
    State("chart_overlay", "style"),
)
def handle_click(click, nav, overlay, data, ticker, style):
    hide = {
        "position": "absolute",
        "top": "0",
        "left": "0",
        "right": "0",
        "bottom": "0",
        "opacity": "0",
        "pointerEvents": "none",
        "backgroundColor": BLACK,
        "zIndex": "1000",
        "padding": "40px",
        "borderRadius": "20px",
        "transition": "opacity 0.3s ease",
    }

    show_nav = {
        "display": "block",
        "position": "absolute",
        "top": "20px",
        "left": "20px",
        "zIndex": "2000",
        "padding": "12px 24px",
        "background": WHITE,
        "border": "none",
        "borderRadius": "12px",
        "color": BLACK,
        "fontSize": "13px",
        "fontWeight": "700",
        "cursor": "pointer",
    }

    if not data:
        return ticker, hide, None, {"display": "none"}

    df = pd.DataFrame(data)

    if (
        callback_context.triggered
        and callback_context.triggered[0]["prop_id"] in ["nav_back.n_clicks"]
    ):
        if style and style.get("opacity") == "1":
            return ticker, hide, None, {"display": "none"}

    if not click or "points" not in click:
        return ticker, hide, None, {"display": "none"}

    label = click["points"][0].get("label")
    if label not in df["Ticker"].values:
        return ticker, hide, None, {"display": "none"}

    row = df[df["Ticker"] == label].iloc[0].to_dict()
    show = {**hide, "opacity": "1", "pointerEvents": "all"}

    chart_graph = dcc.Graph(
        figure=make_chart(row),
        config={"displayModeBar": True, "scrollZoom": True},
        style={"width": "100%", "height": "100%"},
    )

    return label, show, chart_graph, show_nav


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
