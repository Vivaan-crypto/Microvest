"""
Market dashboard workspace.

The layout is intentionally simpler now:
- the heatmap stays centered as the primary surface,
- the info card lives in a bottom strip,
- the detailed chart opens as a full overlay only after a heatmap click,
- indicators are grouped into a clearer schema.
"""

from __future__ import annotations

from dash import Input, Output, State, callback, callback_context, dcc, html
import dash
import pandas as pd
import plotly.graph_objects as go

from config import Colors, SNAPSHOT_REFRESH_MS, SYMBOLS, SECTORS
from src.MarketInfo.charts import create_heatmap, create_stock_chart, get_premium_layout
from src.MarketInfo.components import ChartContainer, InfoPanel
from src.MarketInfo.data import all_stock_data, get_stock_by_ticker, single_stock_data
from src.MarketInfo.indicators import (
    DEFAULT_INDICATORS,
    INDICATOR_GROUPS,
    INDICATOR_SPECS,
    build_indicator_options,
    group_selected_indicators,
    normalize_indicator_selection,
)
from styles import overlay

dash.register_page(__name__, path="/", name="Dashboard")

TICKER_OPTIONS = [
    {"label": f"{ticker}  |  {SECTORS.get(ticker, 'Unknown')}", "value": ticker}
    for ticker in SYMBOLS
]

TIMEFRAME_OPTIONS = [
    {"label": "1M", "value": "1M"},
    {"label": "3M", "value": "3M"},
    {"label": "6M", "value": "6M"},
    {"label": "1Y", "value": "1Y"},
    {"label": "2Y", "value": "2Y"},
    {"label": "5Y", "value": "5Y"},
]

DEFAULT_TICKER = TICKER_OPTIONS[0]["value"] if TICKER_OPTIONS else None


def _timeframe_to_days(value: str) -> int:
    mapping = {"1M": 31, "3M": 92, "6M": 183, "1Y": 365, "2Y": 730, "5Y": 1825}
    return mapping.get(value or "1Y", 365)


def _shell_style() -> dict:
    return {
        "minHeight": "100vh",
        "padding": "16px",
        "background": (
            "radial-gradient(circle at top left, rgba(0, 229, 255, 0.08), transparent 24%), "
            "radial-gradient(circle at top right, rgba(124, 58, 237, 0.08), transparent 26%), "
            "linear-gradient(180deg, #07070b, #0b0d14 58%, #08090d)"
        ),
        "color": Colors.TEXT_PRIMARY,
        "fontFamily": "'Outfit', sans-serif",
    }


def _panel_style() -> dict:
    return {
        "background": "rgba(16, 18, 28, 0.96)",
        "border": "1px solid rgba(148, 163, 184, 0.14)",
        "borderRadius": "18px",
        "boxShadow": "0 18px 60px rgba(0, 0, 0, 0.35)",
        "backdropFilter": "blur(18px)",
        "WebkitBackdropFilter": "blur(18px)",
        "overflow": "hidden",
    }


def _label_style() -> dict:
    return {
        "fontSize": "10px",
        "fontWeight": "800",
        "letterSpacing": "0.16em",
        "textTransform": "uppercase",
        "color": Colors.TEXT_MUTED,
        "fontFamily": "'JetBrains Mono', monospace",
    }


def _chip(label: str, color: str, fill: str) -> html.Div:
    return html.Div(
        label,
        style={
            "padding": "4px 10px",
            "borderRadius": "999px",
            "border": f"1px solid {color}55",
            "background": fill,
            "color": color,
            "fontSize": "10px",
            "fontWeight": "800",
            "letterSpacing": "0.04em",
            "whiteSpace": "nowrap",
        },
    )


def _grouped_indicator_rack(selected: list[str]) -> html.Div:
    grouped = group_selected_indicators(selected)
    if not grouped:
        return html.Div(
            "No indicators selected",
            style={
                **_panel_style(),
                "padding": "16px 18px",
                "color": Colors.TEXT_MUTED,
                "fontSize": "12px",
            },
        )

    group_cards = []
    for group in INDICATOR_GROUPS:
        items = [key for key in group["items"] if key in selected]
        if not items:
            continue

        group_cards.append(
            html.Div(
                [
                    html.Div(group["label"], style=_label_style()),
                    html.Div(
                        [
                            _chip(
                                INDICATOR_SPECS[key]["label"],
                                INDICATOR_SPECS[key]["color"],
                                f"{INDICATOR_SPECS[key]['color']}15",
                            )
                            for key in items
                        ],
                        style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginTop": "10px"},
                    ),
                ],
                style={
                    "padding": "14px 16px",
                    "border": "1px solid rgba(148, 163, 184, 0.12)",
                    "borderRadius": "14px",
                    "background": "rgba(10, 12, 20, 0.78)",
                },
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Indicator Rack", style=_label_style()),
                    html.Div(
                        f"{len(selected)} active in {len(grouped)} groups",
                        style={
                            "fontSize": "11px",
                            "color": Colors.TEXT_MUTED,
                            "fontFamily": "'JetBrains Mono', monospace",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            ),
            html.Div(
                group_cards,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
                    "gap": "10px",
                    "marginTop": "12px",
                },
            ),
        ],
        style={**_panel_style(), "padding": "16px 18px"},
    )


def _movers_panel(frame: pd.DataFrame) -> html.Div:
    if frame.empty or not {"Ticker", "Change"}.issubset(frame.columns):
        return html.Div(
            "Market movers will appear here once snapshot data loads.",
            style={**_panel_style(), "padding": "18px", "color": Colors.TEXT_MUTED, "fontSize": "12px"},
        )

    top_gainers = frame.sort_values("Change", ascending=False).head(4)
    top_losers = frame.sort_values("Change", ascending=True).head(4)

    def _list(title: str, rows, positive: bool) -> html.Div:
        accent = "#22c55e" if positive else "#ef4444"
        return html.Div(
            [
                html.Div(title, style=_label_style()),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(row.Ticker, style={"fontWeight": "800"}),
                                html.Span(
                                    f"{row.Change:+.2f}%",
                                    style={
                                        "color": accent,
                                        "fontFamily": "'JetBrains Mono', monospace",
                                        "fontWeight": "800",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "padding": "6px 0",
                                "borderBottom": "1px solid rgba(148, 163, 184, 0.08)",
                            },
                        )
                        for row in rows.itertuples(index=False)
                    ],
                    style={"marginTop": "10px"},
                ),
            ],
            style={
                "padding": "14px 16px",
                "border": f"1px solid {accent}22",
                "borderRadius": "14px",
                "background": f"{accent}08",
            },
        )

    return html.Div(
        [
            _list("Top Gainers", top_gainers, True),
            _list("Top Losers", top_losers, False),
        ],
        style={**_panel_style(), "padding": "16px", "display": "grid", "gap": "12px"},
    )


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=Colors.TEXT_MUTED, size=18, family="Outfit"),
    )
    fig.update_layout(**get_premium_layout(height=750))
    return fig


def _overlay_content(
    ticker: str,
    timeframe: str,
    selected_indicators: list[str],
    stock: dict | None,
    hist_df: pd.DataFrame,
) -> html.Div:
    if stock:
        title = f"${stock['Ticker']}"
        subtitle = f"{stock['Sector']} | {stock['Last']:.2f} | {stock['Change']:+.2f}% | {timeframe}"
    else:
        title = ticker or "Chart"
        subtitle = f"{timeframe} | No snapshot available"

    if hist_df.empty:
        chart = _empty_figure("No price history available")
    else:
        chart = create_stock_chart(ticker, hist_df, selected_indicators=selected_indicators)
        if chart is None:
            chart = _empty_figure("No price history available")

    indicator_line = [
        _chip(
            INDICATOR_SPECS[key]["label"],
            INDICATOR_SPECS[key]["color"],
            f"{INDICATOR_SPECS[key]['color']}15",
        )
        for key in selected_indicators
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Chart Overlay", style=_label_style()),
                            html.Div(
                                [
                                    html.Div(
                                        title,
                                        style={
                                            "fontSize": "28px",
                                            "fontWeight": "900",
                                            "letterSpacing": "-0.04em",
                                            "lineHeight": "1",
                                        },
                                    ),
                                    html.Div(
                                        subtitle,
                                        style={
                                            "fontSize": "11px",
                                            "color": Colors.TEXT_MUTED,
                                            "fontFamily": "'JetBrains Mono', monospace",
                                            "marginTop": "6px",
                                        },
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Button(
                        "Close",
                        id="close_chart_overlay",
                        n_clicks=0,
                        style={
                            "border": "1px solid rgba(148, 163, 184, 0.2)",
                            "background": "rgba(17, 24, 39, 0.9)",
                            "color": Colors.TEXT_PRIMARY,
                            "borderRadius": "999px",
                            "padding": "10px 14px",
                            "fontSize": "12px",
                            "fontWeight": "800",
                            "letterSpacing": "0.08em",
                            "textTransform": "uppercase",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "gap": "16px",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                indicator_line,
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "14px"},
            ),
            ChartContainer(
                figure=chart,
                id="stock_chart",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
                height="72vh",
            ),
        ],
        style={
            "width": "min(96vw, 1560px)",
            "maxHeight": "92vh",
            "padding": "18px",
            "borderRadius": "22px",
            "background": "rgba(9, 12, 20, 0.96)",
            "border": "1px solid rgba(148, 163, 184, 0.18)",
            "boxShadow": "0 32px 90px rgba(0, 0, 0, 0.55)",
            "overflow": "hidden",
        },
    )


layout = html.Div(
    [
        html.Link(
            href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap",
            rel="stylesheet",
        ),
        dcc.Store(id="snapshot_data"),
        dcc.Store(id="selected_ticker", data=DEFAULT_TICKER),
        dcc.Store(id="selected_timeframe", data="1Y"),
        dcc.Store(id="chart_overlay_visible", data=False),
        dcc.Interval(id="refresh_interval", interval=SNAPSHOT_REFRESH_MS, n_intervals=0),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            style={
                                                "width": "14px",
                                                "height": "14px",
                                                "borderRadius": "4px",
                                                "background": "linear-gradient(135deg, #00e5ff, #00ff88)",
                                                "boxShadow": "0 0 18px rgba(0, 245, 255, 0.45)",
                                            }
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Microvest Terminal",
                                                    style={
                                                        "fontSize": "14px",
                                                        "fontWeight": "800",
                                                        "letterSpacing": "0.08em",
                                                        "textTransform": "uppercase",
                                                    },
                                                ),
                                                html.Div(
                                                    "Heatmap-first trading workspace",
                                                    style={
                                                        "fontSize": "10px",
                                                        "color": Colors.TEXT_MUTED,
                                                        "fontFamily": "'JetBrains Mono', monospace",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                ),
                                html.Div(
                                    "Live",
                                    style={
                                        "padding": "4px 10px",
                                        "borderRadius": "999px",
                                        "background": "rgba(34, 197, 94, 0.12)",
                                        "border": "1px solid rgba(34, 197, 94, 0.25)",
                                        "color": "#22c55e",
                                        "fontSize": "10px",
                                        "fontWeight": "800",
                                        "letterSpacing": "0.12em",
                                        "textTransform": "uppercase",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "14px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Dropdown(
                                        id="ticker_search",
                                        options=TICKER_OPTIONS,
                                        value=DEFAULT_TICKER,
                                        placeholder="Search ticker",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                    style={"width": "300px"},
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id="timeframe_select",
                                        options=TIMEFRAME_OPTIONS,
                                        value="1Y",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                    style={"width": "110px"},
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id="indicator_select",
                                        options=build_indicator_options(grouped=True),
                                        value=DEFAULT_INDICATORS,
                                        multi=True,
                                        searchable=True,
                                        placeholder="Add indicators",
                                    ),
                                    style={"minWidth": "420px", "flex": "1"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "flex": "1",
                                "justifyContent": "flex-end",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "18px",
                        "padding": "16px 20px",
                        **_panel_style(),
                    },
                ),

                html.Div(id="indicator_rack"),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Market Map", style=_label_style()),
                                        html.Div(
                                            "Heatmap stays centered as the market overview",
                                            style={
                                                "fontSize": "12px",
                                                "color": Colors.TEXT_MUTED,
                                                "marginTop": "4px",
                                                "fontFamily": "'Outfit', sans-serif",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("Click a tile to open the chart overlay", style={"fontSize": "11px", "color": Colors.TEXT_MUTED}),
                                        html.Div("Search updates the selected ticker", style={"fontSize": "11px", "color": Colors.TEXT_MUTED}),
                                    ],
                                    style={"textAlign": "right"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "padding": "14px 16px",
                                "borderBottom": "1px solid rgba(148, 163, 184, 0.12)",
                            },
                        ),
                        ChartContainer(figure={}, id="heatmap", config={"displayModeBar": False}, height="66vh"),
                    ],
                    style={**_panel_style(), "overflow": "hidden"},
                ),

                html.Div(
                    [
                        html.Div(id="info_panel"),
                        html.Div(id="movers_panel"),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 1.2fr) minmax(0, 0.8fr)",
                        "gap": "16px",
                    },
                ),
            ],
            style={
                "display": "grid",
                "gap": "16px",
                "maxWidth": "1680px",
                "margin": "0 auto",
            },
        ),

        html.Div(
            id="chart_overlay",
            children=html.Div(
                id="chart_content",
                children=html.Div(id="chart_overlay_body"),
                style={
                    "width": "100%",
                    "height": "100%",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
            style=overlay(False),
        ),
    ],
    style=_shell_style(),
)


@callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("refresh_interval", "n_intervals"),
)
def update_snapshot(_n_intervals):
    df = all_stock_data()
    return create_heatmap(df), df.to_dict("records")


@callback(
    Output("selected_ticker", "data"),
    Output("ticker_search", "value"),
    Output("chart_overlay_visible", "data"),
    Input("ticker_search", "value"),
    Input("heatmap", "clickData"),
    Input("close_chart_overlay", "n_clicks"),
    State("selected_ticker", "data"),
    State("chart_overlay_visible", "data"),
    State("snapshot_data", "data"),
    prevent_initial_call=False,
)
def sync_selected_ticker(search_value, heatmap_click, close_click, current_ticker, current_visible, snapshot_data):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    current_ticker = current_ticker or DEFAULT_TICKER
    current_visible = bool(current_visible)

    if trigger == "close_chart_overlay":
        return current_ticker, current_ticker, False

    if trigger == "heatmap" and heatmap_click and "points" in heatmap_click:
        point = heatmap_click["points"][0]
        ticker = (point.get("customdata") or [point.get("label")])[0]
        if ticker:
            return ticker, ticker, True

    if trigger == "ticker_search" and search_value:
        return search_value, search_value, current_visible

    if not current_ticker and snapshot_data:
        frame = pd.DataFrame(snapshot_data)
        if not frame.empty and "Ticker" in frame.columns:
            default = frame.iloc[0]["Ticker"]
            return default, default, current_visible

    return current_ticker, current_ticker, current_visible


@callback(
    Output("selected_timeframe", "data"),
    Input("timeframe_select", "value"),
)
def sync_timeframe(value):
    return value or "1Y"


@callback(
    Output("chart_overlay", "style"),
    Input("chart_overlay_visible", "data"),
)
def sync_overlay_visibility(visible):
    return overlay(bool(visible))


@callback(
    Output("indicator_rack", "children"),
    Output("info_panel", "children"),
    Output("movers_panel", "children"),
    Output("chart_overlay_body", "children"),
    Input("selected_ticker", "data"),
    Input("selected_timeframe", "data"),
    Input("indicator_select", "value"),
    Input("chart_overlay_visible", "data"),
    State("snapshot_data", "data"),
)
def render_workspace(ticker, timeframe, indicator_values, overlay_visible, snapshot_data):
    selected = normalize_indicator_selection(indicator_values)
    symbol = ticker or DEFAULT_TICKER
    days = _timeframe_to_days(timeframe)

    frame = pd.DataFrame(snapshot_data or [])
    stock = get_stock_by_ticker(frame, symbol) if not frame.empty else None

    if stock:
        info = InfoPanel(
            ticker=stock["Ticker"],
            sector=stock["Sector"],
            price=stock["Last"],
            change=stock["Change"],
        )
    else:
        info = InfoPanel()

    movers = _movers_panel(frame)
    rack = _grouped_indicator_rack(selected)

    overlay_body = []
    if overlay_visible:
        hist_df = single_stock_data(symbol, period_days=days)
        overlay_body = _overlay_content(symbol, timeframe, selected, stock, hist_df)

    return rack, info, movers, overlay_body
