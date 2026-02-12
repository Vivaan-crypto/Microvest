"""
Chart Generation
Handles all Plotly chart creation logic

Industry pattern: "Chart Factory" / "Visualization Layer"
- Separates chart logic from business logic and UI
- Pure functions (input data → output figure)
- Easy to test and modify
- Consistent chart styling
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from typing import Optional

from config import Colors
from data import calculate_technical_indicators, calculate_volume_colors


# =============================================================================
# CHART CONFIGURATION
# =============================================================================

def get_base_layout(title: str = "") -> dict:
    """
    Get base layout config for all charts

    Industry best practice: Centralize common chart settings
    to maintain consistency across all visualizations
    """
    return {
        "paper_bgcolor": Colors.BLACK,
        "plot_bgcolor": Colors.BLACK,
        "font": {"color": Colors.TEXT_PRIMARY, "family": "Inter, sans-serif"},
        "title": {
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 20, "color": Colors.TEXT_PRIMARY},
            "x": 0.5
        },
        "hovermode": "x unified",
        "showlegend": True
    }


# =============================================================================
# HEATMAP / TREEMAP
# =============================================================================

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create stock performance heatmap (treemap visualization)

    Args:
        df: DataFrame with columns [Ticker, Sector, Change, Size]

    Returns:
        Plotly Figure object

    Visual hierarchy:
        - First level: Sectors (Technology, Financials, etc.)
        - Second level: Individual stocks
        - Size: Market weight (price × volume)
        - Color: Percentage change (red = down, green = up)
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=Colors.TEXT_MUTED)
        )
        fig.update_layout(get_base_layout())
        return fig

    # Calculate color range (symmetric around 0)
    max_abs_change = max(1.0, df["Change"].abs().max())

    # Create treemap
    fig = px.treemap(
        df,
        path=["Sector", "Ticker"],  # Hierarchy: Sector → Ticker
        values="Size",  # Box size based on market weight
        color="Change",  # Color based on % change
        color_continuous_scale=[
            (0, Colors.DANGER),  # Most negative = red
            (0.5, Colors.SLATE_800),  # Zero = dark gray
            (1, Colors.SUCCESS)  # Most positive = green
        ],
        range_color=[-max_abs_change, max_abs_change]
    )

    # Prepare custom data for hover and labels
    customdata = np.column_stack([df["Change"].values, df["Last"].values])

    # Update traces with custom formatting
    fig.update_traces(
        customdata=customdata,
        # Text shown on each box
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        textfont=dict(size=16, color=Colors.WHITE, family="Inter", weight=600),
        # Hover tooltip
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Change: %{customdata[0]:+.2f}%<br>"
            "Price: $%{customdata[1]:.2f}"
            "<extra></extra>"
        ),
    )

    # Update layout
    fig.update_layout(
        **get_base_layout(),
        margin=dict(t=25, l=5, r=5, b=5),
        uirevision="keep"  # Preserve zoom/pan on updates
    )

    return fig


# =============================================================================
# DETAILED STOCK CHART
# =============================================================================

def create_stock_chart(ticker: str, hist_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create comprehensive stock chart with technical indicators

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        hist_df: Historical OHLCV DataFrame

    Returns:
        Plotly Figure with 3 subplots:
        1. Price chart with candlesticks, SMA, Bollinger Bands
        2. Volume chart
        3. RSI indicator

    Industry note: This is a "factory function" pattern -
    it encapsulates complex chart creation logic
    """
    if hist_df.empty:
        return None

    # Calculate technical indicators
    hist_df = calculate_technical_indicators(hist_df)

    # Create 3-panel layout
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.25, 0.15],  # Price: 60%, Volume: 25%, RSI: 15%
        vertical_spacing=0.06,
        subplot_titles=("", "", "")  # No titles on subplots
    )

    # =========================================================================
    # PANEL 1: Price Chart with Technical Indicators
    # =========================================================================

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"],
            high=hist_df["High"],
            low=hist_df["Low"],
            close=hist_df["Close"],
            increasing_line_color=Colors.SUCCESS,
            decreasing_line_color=Colors.DANGER,
            name="Price",
            showlegend=False
        ),
        row=1, col=1
    )

    # Bollinger Bands (upper)
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_up"],
            mode="lines",
            line=dict(width=1, color=Colors.PURPLE, dash="dash"),
            name="BB Upper",
            opacity=0.45
        ),
        row=1, col=1
    )

    # Bollinger Bands (lower) with fill
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_low"],
            mode="lines",
            line=dict(width=1, color=Colors.PURPLE, dash="dash"),
            name="BB Lower",
            opacity=0.45,
            fill="tonexty",  # Fill to previous trace (BB upper)
            fillcolor="rgba(139,92,246,0.08)"
        ),
        row=1, col=1
    )

    # 20-day Simple Moving Average
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA20"],
            mode="lines",
            line=dict(width=2, color=Colors.BLUE),
            name="SMA 20"
        ),
        row=1, col=1
    )

    # 50-day Simple Moving Average
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA50"],
            mode="lines",
            line=dict(width=2, color=Colors.ORANGE),
            name="SMA 50"
        ),
        row=1, col=1
    )

    # =========================================================================
    # PANEL 2: Volume Chart
    # =========================================================================

    volume_colors = calculate_volume_colors(hist_df)

    fig.add_trace(
        go.Bar(
            x=hist_df.index,
            y=hist_df["Volume"],
            marker_color=volume_colors,
            name="Volume",
            showlegend=False
        ),
        row=2, col=1
    )

    # =========================================================================
    # PANEL 3: RSI Indicator
    # =========================================================================

    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["RSI"],
            mode="lines",
            line=dict(width=2, color=Colors.YELLOW),
            name="RSI",
            showlegend=False
        ),
        row=3, col=1
    )

    # RSI overbought/oversold lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=Colors.DANGER,
        opacity=0.5,
        row=3, col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=Colors.SUCCESS,
        opacity=0.5,
        row=3, col=1
    )

    # =========================================================================
    # Y-AXIS CONFIGURATION
    # =========================================================================

    # Calculate price range with padding
    low_min = float(hist_df["Low"].min())
    high_max = float(hist_df["High"].max())
    price_padding = (high_max - low_min) * 0.05

    fig.update_yaxes(
        title_text="Price",
        range=[low_min - price_padding, high_max + price_padding],
        fixedrange=True,
        row=1, col=1
    )

    fig.update_yaxes(
        title_text="Volume",
        fixedrange=True,
        row=2, col=1
    )

    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        fixedrange=True,
        row=3, col=1
    )

    # =========================================================================
    # X-AXIS CONFIGURATION
    # =========================================================================

    x_min = hist_df.index.min()
    x_max = hist_df.index.max() + timedelta(days=1)

    fig.update_xaxes(
        range=[x_min, x_max],
        rangeslider_visible=False,
        showgrid=True,
        gridcolor=Colors.BORDER,
        row=3, col=1,
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # Hide weekends
        ]
    )

    # =========================================================================
    # OVERALL LAYOUT
    # =========================================================================

    fig.update_layout(
        **get_base_layout(title=ticker),
        margin=dict(l=60, r=60, t=80, b=40),
        legend=dict(
            orientation="h",
            y=1.05,
            x=0.5,
            xanchor="center",
            bgcolor=Colors.CARD_BG,
            bordercolor=Colors.BORDER,
            borderwidth=1
        ),
        dragmode="zoom",
        height=600
    )

    return fig


# =============================================================================
# SIMPLIFIED CHART VARIATIONS
# =============================================================================

def create_simple_price_chart(ticker: str, hist_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a simplified price chart (just candlestick, no indicators)

    Args:
        ticker: Stock symbol
        hist_df: Historical OHLCV DataFrame

    Returns:
        Simplified Plotly Figure

    Use case: When you need a quick view without technical indicators
    """
    if hist_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"],
            high=hist_df["High"],
            low=hist_df["Low"],
            close=hist_df["Close"],
            increasing_line_color=Colors.SUCCESS,
            decreasing_line_color=Colors.DANGER
        )
    )

    fig.update_layout(
        **get_base_layout(title=ticker),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "") -> go.Figure:
    """
    Generic line chart creator

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title

    Returns:
        Line chart Figure

    Industry pattern: Generic chart builders can be reused across your app
    """
    fig = px.line(df, x=x_col, y=y_col)

    fig.update_traces(
        line=dict(color=Colors.BLUE, width=2)
    )

    fig.update_layout(
        **get_base_layout(title=title),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig