"""
Main Application - Premium Stock Dashboard
Beautiful, modern interface with smooth interactions
"""

from dash import Dash, dcc, html, Input, Output, State, callback_context

# Import modules
from config import SNAPSHOT_REFRESH_MS, APP_HOST, APP_PORT, DEBUG_MODE, Colors
from components import (
    PageContainer, Header, ChartContainer, ChartOverlay,
    BackButton, InfoPanel
)
from data import all_stock_data, single_stock_data, get_stock_by_ticker
from charts import create_heatmap, create_stock_chart
from styles import overlay

# =============================================================================
# INITIALIZE APP
# =============================================================================

app = Dash(__name__)
app.title = "Premium Stock Dashboard"

# Custom CSS for animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            * {
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            body {
                margin: 0;
                overflow-x: hidden;
            }
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: ''' + Colors.BG_SECONDARY + ''';
            }
            ::-webkit-scrollbar-thumb {
                background: ''' + Colors.BORDER_BRIGHT + ''';
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: ''' + Colors.ACCENT_PRIMARY + ''';
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = PageContainer([
    # =========================================================================
    # DATA STORES
    # =========================================================================
    dcc.Store(id="snapshot_data"),
    dcc.Store(id="selected_ticker"),

    # =========================================================================
    # AUTO-REFRESH TIMER
    # =========================================================================
    dcc.Interval(
        id="refresh_interval",
        interval=SNAPSHOT_REFRESH_MS,
        n_intervals=0
    ),

    # =========================================================================
    # HEADER
    # =========================================================================
    Header(),

    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    html.Div([
        # Back button (floating, hidden by default)
        BackButton(id="nav_back"),

        # Heatmap
        ChartContainer(
            figure={},
            id="heatmap",
            config={"displayModeBar": False}
        ),

        # Chart overlay
        ChartOverlay(
            id="chart_overlay",
            content_id="chart_content",
            visible=False
        )
    ], style={
        "flex": "1",
        "position": "relative",
        "minHeight": "600px",
        "animation": "fadeIn 0.6s ease-out"
    }),

    # =========================================================================
    # INFO PANEL
    # =========================================================================
    html.Div(
        id="info_panel",
        style={"animation": "fadeIn 0.6s ease-out 0.2s both"}
    )
])


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("refresh_interval", "n_intervals")
)
def update_snapshot(n_intervals):
    """
    Fetch and display latest stock data
    """
    df = all_stock_data()
    fig = create_heatmap(df)
    data = df.to_dict("records")
    return fig, data


@app.callback(
    Output("info_panel", "children"),
    Input("heatmap", "clickData"),
    Input("snapshot_data", "data")
)
def update_info_panel(click_data, snapshot_data):
    """
    Update info panel when stock is clicked
    """
    import pandas as pd

    if not snapshot_data:
        return InfoPanel()

    df = pd.DataFrame(snapshot_data)

    if not click_data or "points" not in click_data:
        return InfoPanel()

    # Get ticker from customdata or label
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


@app.callback(
    Output("selected_ticker", "data"),
    Output("chart_overlay", "style"),
    Output("chart_content", "children"),
    Output("nav_back", "style"),
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
    """
    Handle chart overlay with smooth transitions
    No caching - fetches fresh data each time
    """
    import pandas as pd

    # Default states
    hidden_overlay = overlay(visible=False)
    hidden_button = {"display": "none"}
    visible_button = {
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "position": "fixed",
        "top": "24px",
        "left": "24px",
        "zIndex": "2000"
    }

    if not snapshot_data:
        print("No snapshot data")
        return current_ticker, hidden_overlay, None, hidden_button

    df = pd.DataFrame(snapshot_data)

    # Determine trigger
    ctx = callback_context
    if not ctx.triggered:
        print("No trigger")
        return current_ticker, hidden_overlay, None, hidden_button

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(f"Trigger ID: {trigger_id}")

    # Handle back button
    if trigger_id == "nav_back":
        print("Back button clicked")
        return current_ticker, hidden_overlay, None, hidden_button

    # Handle heatmap click
    if trigger_id == "heatmap":
        if not heatmap_click:
            print("No heatmap click data")
            return current_ticker, hidden_overlay, None, hidden_button

        if "points" not in heatmap_click:
            print("No points in click data")
            return current_ticker, hidden_overlay, None, hidden_button

        # Get ticker from click - try customdata first, then label
        point = heatmap_click["points"][0]

        # Try to get ticker from customdata (index 0)
        if "customdata" in point and point["customdata"] and len(point["customdata"]) > 0:
            ticker = point["customdata"][0]
            print(f"Got ticker from customdata: {ticker}")
        else:
            # Fallback to label
            ticker = point.get("label")
            print(f"Got ticker from label: {ticker}")

        if not ticker:
            print("No ticker found in click data")
            print(f"Point data: {point}")
            return current_ticker, hidden_overlay, None, hidden_button

        # Validate ticker
        if ticker not in df["Ticker"].values:
            print(f"Ticker {ticker} not found in data")
            print(f"Available tickers: {df['Ticker'].tolist()[:5]}...")
            return current_ticker, hidden_overlay, None, hidden_button

        print(f"Fetching fresh data for {ticker}")

        # Fetch fresh data (no caching)
        hist_df = single_stock_data(ticker)

        if hist_df.empty:
            print(f"Failed to fetch history for {ticker}")
            return current_ticker, hidden_overlay, None, hidden_button

        print(f"Fetched {len(hist_df)} rows")

        # Create chart
        print("Creating chart")
        fig = create_stock_chart(ticker, hist_df)

        if not fig:
            print("Failed to create chart")
            return current_ticker, hidden_overlay, None, hidden_button

        print("Chart created successfully")

        # Build chart component
        chart_component = ChartContainer(
            figure=fig,
            id="detail_chart",
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True
            }
        )

        visible_overlay = overlay(visible=True)

        print("Returning visible overlay")
        return ticker, visible_overlay, chart_component, visible_button

    # Default
    print("Returning default state")
    return current_ticker, hidden_overlay, None, hidden_button


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=DEBUG_MODE
    )