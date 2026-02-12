"""
Reusable UI Components
Separates presentation (UI) from business logic

Industry pattern: "Component Library"
- Each function returns a Dash component
- Pure functions (no side effects)
- Easy to test and reuse
"""

from dash import html, dcc
import plotly.graph_objects as go
from config import Colors, Typography, Spacing
from styles import (
    flexbox, card, text, button, stat_row, divider,
    badge, merge_styles, conditional_style
)


# =============================================================================
# BASIC COMPONENTS (Building blocks)
# =============================================================================

def Stat(label, value, value_color=Colors.TEXT_PRIMARY):
    """
    Display a label-value pair

    Args:
        label: Left side text (e.g., "Gainers")
        value: Right side value (e.g., "25")
        value_color: Color for the value

    Example:
        Stat("Gainers", "25", Colors.SUCCESS)
    """
    return html.Div([
        html.Span(label, style=text(size=Typography.SIZE_BASE, color=Colors.TEXT_PRIMARY)),
        html.Span(
            str(value),
            style=text(size=Typography.SIZE_XL, weight=Typography.WEIGHT_BOLD, color=value_color)
        )
    ], style=stat_row())


def SectionHeader(title, uppercase=True):
    """
    Section header with consistent styling

    Example:
        SectionHeader("MARKET OVERVIEW")
    """
    return html.Div(
        title,
        style={
            **text(
                size=Typography.SIZE_XS,
                weight=Typography.WEIGHT_BOLD,
                color=Colors.TEXT_MUTED
            ),
            "textTransform": "uppercase" if uppercase else "none",
            "letterSpacing": "1px"
        }
    )


def Divider(margin=f"{Spacing.LG} 0"):
    """
    Horizontal divider line

    Example:
        Divider(margin="20px 0")
    """
    return html.Div(style=divider(margin))


def Badge(content, is_positive=None):
    """
    Badge showing percentage change with color

    Args:
        content: Text to display (e.g., "+2.5%")
        is_positive: True for green, False for red, None for neutral

    Example:
        Badge("+2.5%", is_positive=True)
    """
    if is_positive is None:
        bg_color = Colors.TEXT_MUTED
        arrow = "→"
    elif is_positive:
        bg_color = Colors.SUCCESS
        arrow = "↑"
    else:
        bg_color = Colors.DANGER
        arrow = "↓"

    return html.Div([
        html.Span(arrow, style={"marginRight": Spacing.SM, "fontSize": Typography.SIZE_LG}),
        html.Span(content)
    ], style=badge(bg_color, Colors.WHITE))


def BackButton(id="nav_back"):
    """
    Back button component

    Example:
        BackButton(id="my-back-btn")
    """
    return html.Button(
        [html.Span("← "), html.Span("BACK")],
        id=id,
        n_clicks=0,
        style=merge_styles(
            button("primary"),
            {
                "display": "none",
                "position": "absolute",
                "top": Spacing.XL,
                "left": Spacing.XL,
                "zIndex": "2000"
            }
        )
    )


# =============================================================================
# CARD COMPONENTS (Composite components)
# =============================================================================

def StatsCard(title, stats_list):
    """
    Card displaying a list of statistics

    Args:
        title: Card title (e.g., "MARKET")
        stats_list: List of Stat components

    Example:
        StatsCard("MARKET", [
            Stat("Gainers", 25, Colors.SUCCESS),
            Stat("Losers", 15, Colors.DANGER)
        ])
    """
    return html.Div([
        SectionHeader(title),
        html.Div(stats_list, style=flexbox(direction="column", gap=Spacing.SM))
    ], style=flexbox(direction="column", gap=Spacing.XS))


def TickerInfoCard(ticker=None, sector=None, price=None, change_pct=None):
    """
    Card showing ticker details

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        sector: Sector name (e.g., "Technology")
        price: Current price (e.g., 175.50)
        change_pct: Percentage change (e.g., 2.5)

    Example:
        TickerInfoCard("AAPL", "Technology", 175.50, 2.5)
    """
    if not ticker:
        return html.Div(
            "Select a ticker",
            style=merge_styles(
                card(),
                flexbox(align="center", justify="center"),
                text(size=Typography.SIZE_LG, weight=Typography.WEIGHT_MEDIUM, color=Colors.TEXT_MUTED),
                {"height": "100%"}
            )
        )

    is_positive = change_pct > 0 if change_pct != 0 else None

    return html.Div([
        # Ticker name and sector
        html.Div([
            html.Div(
                ticker,
                style=text(
                    size=Typography.SIZE_DISPLAY,
                    weight=Typography.WEIGHT_BLACK,
                    color=Colors.TEXT_PRIMARY
                )
            ),
            html.Div(
                sector,
                style={
                    **text(size=Typography.SIZE_SM, weight=Typography.WEIGHT_MEDIUM, color=Colors.TEXT_MUTED),
                    "marginTop": Spacing.SM,
                    "textTransform": "uppercase"
                }
            )
        ]),

        # Price
        html.Div([
            html.Div(
                f"${price:.2f}",
                style=text(size=Typography.SIZE_XXXL, weight=Typography.WEIGHT_BOLD, color=Colors.TEXT_PRIMARY)
            ),
            html.Div(
                "Current Price",
                style={
                    **text(size=Typography.SIZE_XS, weight=Typography.WEIGHT_MEDIUM, color=Colors.TEXT_MUTED),
                    "marginTop": Spacing.XS,
                    "textTransform": "uppercase"
                }
            )
        ]),

        # Change badge
        Badge(f"{abs(change_pct):.2f}%", is_positive)

    ], style=merge_styles(
        card(padding=f"{Spacing.XL} {Spacing.XXXL}"),
        {
            "display": "grid",
            "gridTemplateColumns": "auto 1fr auto",
            "gap": Spacing.XXL,
            "alignItems": "center"
        }
    ))


def EmptyState(message="No data available", icon="📊"):
    """
    Empty state component

    Example:
        EmptyState("No stocks selected", "📈")
    """
    return html.Div([
        html.Div(icon, style={"fontSize": "48px", "marginBottom": Spacing.LG}),
        html.Div(
            message,
            style=text(size=Typography.SIZE_LG, weight=Typography.WEIGHT_MEDIUM, color=Colors.TEXT_MUTED)
        )
    ], style=merge_styles(
        flexbox(direction="column", align="center", justify="center"),
        {"height": "100%", "padding": Spacing.XXXL}
    ))


# =============================================================================
# CHART COMPONENTS
# =============================================================================

def ChartContainer(figure, id="chart", config=None):
    """
    Wrapper for Plotly chart with consistent styling

    Args:
        figure: Plotly figure object
        id: Element ID
        config: Chart config options

    Example:
        ChartContainer(my_figure, id="heatmap")
    """
    default_config = {
        "displayModeBar": False,
        "scrollZoom": False
    }

    return dcc.Graph(
        id=id,
        figure=figure,
        config=config or default_config,
        style={"height": "100%", "width": "100%"}
    )


def ChartOverlay(id="chart_overlay", content_id="chart_content", visible=False):
    """
    Full-screen overlay for detailed charts

    Args:
        id: Overlay container ID
        content_id: Content container ID
        visible: Whether overlay is initially visible

    Example:
        ChartOverlay(visible=True)
    """
    from styles import overlay

    return html.Div(
        html.Div(id=content_id, style={"height": "100%"}),
        id=id,
        n_clicks=0,
        style=overlay(visible)
    )


# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def PageContainer(children):
    """
    Main page container with consistent padding and styling

    Example:
        PageContainer([Header(), Content(), Footer()])
    """
    return html.Div(
        children,
        style={
            "backgroundColor": Colors.BLACK,
            "minHeight": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "fontFamily": Typography.FONT_FAMILY,
            "padding": Spacing.LG,
            "gap": Spacing.LG
        }
    )


def TwoColumnLayout(left_content, right_content, left_ratio=2, right_ratio=1):
    """
    Two-column grid layout

    Args:
        left_content: Content for left column
        right_content: Content for right column
        left_ratio: Relative width of left column
        right_ratio: Relative width of right column

    Example:
        TwoColumnLayout(main_chart, sidebar, left_ratio=3, right_ratio=1)
    """
    from styles import grid

    return html.Div(
        [left_content, right_content],
        style=grid(columns=f"{left_ratio}fr {right_ratio}fr", gap=Spacing.LG)
    )


# =============================================================================
# DATA DISPLAY COMPONENTS
# =============================================================================

def MarketStatsPanel(gainers=0, losers=0, avg_change=0.0):
    """
    Complete market statistics panel

    Example:
        MarketStatsPanel(gainers=25, losers=15, avg_change=1.2)
    """
    avg_color = Colors.SUCCESS if avg_change > 0 else Colors.DANGER if avg_change < 0 else Colors.TEXT_MUTED

    return StatsCard("MARKET", [
        Stat("Gainers", gainers, Colors.SUCCESS),
        Stat("Losers", losers, Colors.DANGER),
        Stat("Avg", f"{avg_change:+.2f}%", avg_color)
    ])


def WatchlistStatsPanel(gainers=0, losers=0, avg_change=0.0):
    """
    Watchlist statistics panel

    Example:
        WatchlistStatsPanel(gainers=2, losers=1, avg_change=-0.5)
    """
    avg_color = Colors.SUCCESS if avg_change > 0 else Colors.DANGER if avg_change < 0 else Colors.TEXT_MUTED

    return StatsCard("WATCHLIST", [
        Stat("Gainers", gainers, Colors.SUCCESS),
        Stat("Losers", losers, Colors.DANGER),
        Stat("Avg", f"{avg_change:+.2f}%", avg_color)
    ])


def StatsContainer(market_stats, watchlist_stats):
    """
    Container for all stats panels

    Example:
        StatsContainer(
            MarketStatsPanel(25, 15, 1.2),
            WatchlistStatsPanel(2, 1, -0.5)
        )
    """
    return html.Div([
        SectionHeader("MARKET STATS"),
        html.Div([
            market_stats,
            Divider(),
            watchlist_stats,
            Divider(margin=f"{Spacing.LG} 0 {Spacing.SM}")
        ], style=flexbox(direction="column", gap=Spacing.XS))
    ], style=card())