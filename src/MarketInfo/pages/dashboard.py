"""
pages/dashboard.py - Main Stock Heatmap Dashboard
Your existing app content, migrated to a Dash page
"""

import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import pandas as pd

from src.MarketInfo.config import SNAPSHOT_REFRESH_MS, Colors
from src.MarketInfo.components import PageContainer, Header, ChartContainer, ChartOverlay, BackButton, InfoPanel
from src.MarketInfo.data import all_stock_data, single_stock_data, get_stock_by_ticker
from src.MarketInfo.charts import create_heatmap, create_stock_chart
from src.MarketInfo.styles import overlay

dash.register_page(__name__, path="/", name="Dashboard")

# =============================================================================
# LAYOUT
# =============================================================================

layout = html.Div([
    # Font import
    html.Link(
        href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap",
        rel="stylesheet"
    ),

    dcc.Store(id="snapshot_data"),
    dcc.Store(id="selected_ticker"),
    
    dcc.Interval(
        id="refresh_interval",
        interval=SNAPSHOT_REFRESH_MS,
        n_intervals=0
    ),

    # Inner content (mirrors your original app.layout children, minus PageContainer wrapper)
    html.Div([
        Header(),
    ], style={"padding": "32px 32px 0 32px"}),

    html.Div([
        BackButton(id="nav_back"),

        ChartContainer(
            figure={},
            id="heatmap",
            config={"displayModeBar": False}
        ),

        ChartOverlay(
            id="chart_overlay",
            content_id="chart_content",
            visible=False
        )
    ], style={
        "flex": "1",
        "position": "relative",
        "minHeight": "600px",
        "animation": "fadeIn 0.6s ease-out",
        "padding": "24px 32px 0 32px",
    }),

    html.Div(
        id="info_panel",
        style={
            "animation": "fadeIn 0.6s ease-out 0.2s both",
            "padding": "0 32px 32px 32px"
        }
    )
], style={
    "minHeight": "calc(100vh - 52px)",
    "background": Colors.BG_PRIMARY,
    "display": "flex",
    "flexDirection": "column",
    "gap": "0",
    "fontFamily": "'Outfit', sans-serif",
})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("refresh_interval", "n_intervals")
)
def update_snapshot(n_intervals):
    df = all_stock_data()
    fig = create_heatmap(df)
    data = df.to_dict("records")
    return fig, data


@callback(
    Output("info_panel", "children"),
    Input("heatmap", "clickData"),
    Input("snapshot_data", "data")
)
def update_info_panel(click_data, snapshot_data):
    if not snapshot_data:
        return InfoPanel()

    df = pd.DataFrame(snapshot_data)

    if not click_data or "points" not in click_data:
        return InfoPanel()

    point = click_data["points"][0]
    if "customdata" in point and point["customdata"] and len(point["customdata"]) > 0:
        ticker = point["customdata"][0]
    else:
        ticker = point.get("label")

    stock = get_stock_by_ticker(df, ticker)

    if not stock:
        return InfoPanel()

    return InfoPanel(
        ticker=stock["Ticker"],
        sector=stock["Sector"],
        price=stock["Last"],
        change=stock["Change"]
    )


@callback(
    Output("selected_ticker", "data"),
    Output("chart_overlay", "style"),
    Output("chart_content", "children"),
    Output("nav_back", "className"),
    Input("heatmap", "clickData"),
    Input("nav_back", "n_clicks"),
    State("snapshot_data", "data"),
    State("selected_ticker", "data"),
    State("chart_overlay", "style"),
    prevent_initial_call=True
)
def handle_chart_overlay(
    heatmap_click,
    back_click,
    snapshot_data,
    current_ticker,
    current_overlay_style
):
    hidden_overlay = overlay(visible=False)
    hidden_button = "btn-hidden"
    visible_button = "btn-visible"

    if not snapshot_data:
        return current_ticker, hidden_overlay, None, hidden_button

    df = pd.DataFrame(snapshot_data)

    ctx = callback_context
    if not ctx.triggered:
        return current_ticker, hidden_overlay, None, hidden_button

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "nav_back":
        return current_ticker, hidden_overlay, None, hidden_button

    if trigger_id == "heatmap":
        if not heatmap_click or "points" not in heatmap_click:
            return current_ticker, hidden_overlay, None, hidden_button

        point = heatmap_click["points"][0]

        if "customdata" in point and point["customdata"] and len(point["customdata"]) > 0:
            ticker = point["customdata"][0]
        else:
            ticker = point.get("label")

        if not ticker or ticker not in df["Ticker"].values:
            return current_ticker, hidden_overlay, None, hidden_button

        hist_df = single_stock_data(ticker)

        if hist_df.empty:
            return current_ticker, hidden_overlay, None, hidden_button

        fig = create_stock_chart(ticker, hist_df)

        if not fig:
            return current_ticker, hidden_overlay, None, hidden_button

        chart_component = ChartContainer(
            figure=fig,
            id="detail_chart",
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True
            }
        )

        return ticker, overlay(visible=True), chart_component, visible_button

    return current_ticker, hidden_overlay, None, hidden_button