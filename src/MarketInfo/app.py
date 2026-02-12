"""
Main Application
Ties together all components using the Dash framework

Industry pattern: "Application Layer" / "Composition Root"
- Minimal business logic (delegates to other modules)
- Focused on wiring components together
- Easy to understand the app structure at a glance
"""

from dash import Dash, dcc, html, Input, Output, State, callback_context

# Import our organized modules
from config import SNAPSHOT_REFRESH_MS, APP_HOST, APP_PORT, DEBUG_MODE, WATCHLIST_TICKERS
from components import (
    PageContainer, TwoColumnLayout, BackButton, ChartContainer,
    ChartOverlay, TickerInfoCard, StatsContainer,
    MarketStatsPanel, WatchlistStatsPanel
)
from data import (
    fetch_snapshot_data, fetch_stock_history,
    calculate_market_stats, calculate_watchlist_stats,
    get_stock_by_ticker
)
from charts import create_heatmap, create_stock_chart
from styles import overlay

# =============================================================================
# INITIALIZE APP
# =============================================================================

app = Dash(__name__)
app.title = "Stock Market Dashboard"

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = PageContainer([
    # =========================================================================
    # DATA STORES (invisible components that hold state)
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
    # MAIN CONTENT AREA
    # =========================================================================
    html.Div([
        # Back button (hidden by default, shown when chart overlay is open)
        BackButton(id="nav_back"),

        # Main heatmap
        ChartContainer(
            figure={},
            id="heatmap",
            config={"displayModeBar": False}
        ),

        # Chart overlay (slides over heatmap when ticker is clicked)
        ChartOverlay(
            id="chart_overlay",
            content_id="chart_content",
            visible=False
        )
    ], style={
        "flex": 1,
        "position": "relative",
        "minHeight": "500px"
    }),

    # =========================================================================
    # BOTTOM INFO PANEL (2-column layout)
    # =========================================================================
    TwoColumnLayout(
        # Left column: Selected stock info
        left_content=html.Div(id="info_card"),

        # Right column: Market statistics
        right_content=html.Div(id="stats_container"),

        left_ratio=2,
        right_ratio=1
    )
])


# =============================================================================
# CALLBACKS (Application Logic)
# =============================================================================

@app.callback(
    Output("heatmap", "figure"),
    Output("snapshot_data", "data"),
    Input("refresh_interval", "n_intervals")
)
def update_snapshot(n_intervals):
    """
    Callback 1: Fetch and display latest stock data

    Triggered by: Timer (every SNAPSHOT_REFRESH_MS milliseconds)
    Updates: Heatmap visualization and stored snapshot data

    Industry note: This is the "data refresh" callback
    It's the single source of truth for market data
    """
    # Fetch latest data
    df = fetch_snapshot_data()

    # Generate heatmap visualization
    fig = create_heatmap(df)

    # Store data for other callbacks (convert to dict for JSON serialization)
    data = df.to_dict("records")

    return fig, data


@app.callback(
    Output("info_card", "children"),
    Output("stats_container", "children"),
    Input("snapshot_data", "data"),
    State("selected_ticker", "data")
)
def update_info_and_stats(snapshot_data, selected_ticker):
    """
    Callback 2: Update info card and statistics panels

    Triggered by: New snapshot data
    Updates: Ticker info card and market/watchlist stats

    Industry pattern: "Derived State"
    - Takes raw data and computes what should be displayed
    - Separates data transformation from presentation
    """
    import pandas as pd

    if not snapshot_data:
        # No data available yet
        return TickerInfoCard(), StatsContainer(
            MarketStatsPanel(),
            WatchlistStatsPanel()
        )

    df = pd.DataFrame(snapshot_data)

    # Update ticker info card
    if selected_ticker:
        stock = get_stock_by_ticker(df, selected_ticker)
        if stock:
            info_card = TickerInfoCard(
                ticker=stock["Ticker"],
                sector=stock["Sector"],
                price=stock["Last"],
                change_pct=stock["Change"]
            )
        else:
            info_card = TickerInfoCard()
    else:
        info_card = TickerInfoCard()

    # Calculate market statistics
    market_stats = calculate_market_stats(df)
    watchlist_stats = calculate_watchlist_stats(df, WATCHLIST_TICKERS)

    # Build stats panels
    stats = StatsContainer(
        MarketStatsPanel(
            gainers=market_stats["gainers"],
            losers=market_stats["losers"],
            avg_change=market_stats["avg_change"]
        ),
        WatchlistStatsPanel(
            gainers=watchlist_stats["gainers"],
            losers=watchlist_stats["losers"],
            avg_change=watchlist_stats["avg_change"]
        )
    )

    return info_card, stats


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
    State("chart_overlay", "style")
)
def handle_chart_interactions(
        heatmap_click,
        back_click,
        overlay_click,
        snapshot_data,
        current_ticker,
        current_overlay_style
):
    """
    Callback 3: Handle user interactions with charts

    Triggered by:
    - Clicking on heatmap (show detail chart)
    - Clicking back button (hide detail chart)
    - Clicking overlay background (hide detail chart)

    Industry pattern: "Event Handler"
    - Determines which interaction occurred
    - Updates UI state accordingly
    - Uses callback_context to identify trigger
    """
    import pandas as pd

    # Default states
    hidden_overlay = overlay(visible=False)
    hidden_button = {"display": "none"}
    visible_button = {
        "display": "block",
        "position": "absolute",
        "top": "20px",
        "left": "20px",
        "zIndex": "2000"
    }

    # Check if we have data
    if not snapshot_data:
        return current_ticker, hidden_overlay, None, hidden_button

    df = pd.DataFrame(snapshot_data)

    # Determine which input triggered this callback
    ctx = callback_context
    if not ctx.triggered:
        return current_ticker, hidden_overlay, None, hidden_button

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # =========================================================================
    # INTERACTION 1: Back button clicked
    # =========================================================================
    if trigger_id == "nav_back":
        if current_overlay_style and current_overlay_style.get("opacity") == "1":
            # Close the overlay
            return current_ticker, hidden_overlay, None, hidden_button

    # =========================================================================
    # INTERACTION 2: Heatmap clicked (show detail chart)
    # =========================================================================
    if trigger_id == "heatmap" and heatmap_click:
        if "points" not in heatmap_click:
            return current_ticker, hidden_overlay, None, hidden_button

        # Get clicked ticker
        ticker = heatmap_click["points"][0].get("label")

        # Validate ticker exists in our data
        if ticker not in df["Ticker"].values:
            return current_ticker, hidden_overlay, None, hidden_button

        # Fetch historical data for this ticker
        hist_df = fetch_stock_history(ticker, period="1y")

        if hist_df.empty:
            return current_ticker, hidden_overlay, None, hidden_button

        # Create detailed chart
        fig = create_stock_chart(ticker, hist_df)

        if not fig:
            return current_ticker, hidden_overlay, None, hidden_button

        # Build chart component
        chart_component = ChartContainer(
            figure=fig,
            id="detail_chart",
            config={"displayModeBar": True, "scrollZoom": True}
        )

        # Show overlay with chart
        visible_overlay = overlay(visible=True)

        return ticker, visible_overlay, chart_component, visible_button

    # =========================================================================
    # DEFAULT: No changes
    # =========================================================================
    return current_ticker, hidden_overlay, None, hidden_button


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    """
    Application entry point

    Industry note: This pattern allows the app to be imported
    without running (useful for testing and deployment)
    """
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=DEBUG_MODE
    )