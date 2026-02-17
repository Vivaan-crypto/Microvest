"""
Chart Generation - Premium Visualizations
Stunning, modern charts for financial data
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from typing import Optional

from config import Colors, Typography, Effects
from data import calculate_technical_indicators, calculate_volume_colors


# =============================================================================
# CHART CONFIGURATION
# =============================================================================

def get_premium_layout(title: str = "", height: int = 600) -> dict:
    """
    Premium dark theme layout for charts
    """
    return {
        "paper_bgcolor": Colors.BG_PRIMARY,
        "plot_bgcolor": Colors.BG_SECONDARY,
        "font": {
            "color": Colors.TEXT_PRIMARY,
            "family": Typography.FONT_DISPLAY,
            "size": 13
        },
        "title": {
            "text": f"<b>{title}</b>",
            "font": {
                "size": 24,
                "color": Colors.TEXT_PRIMARY,
                "family": Typography.FONT_DISPLAY
            },
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top"
        },
        "hovermode": "x unified",
        "showlegend": True,
        "height": height,
    }


# =============================================================================
# HEATMAP / TREEMAP
# =============================================================================

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Premium stock performance heatmap with vibrant colors
    """

    # Calculate color range
    max_abs_change = max(1.0, df["Change"].abs().max())

    # Create treemap with vibrant colors
    fig = px.treemap(
        df,
        path=["Sector", "Ticker"],
        values="Size",
        color="Change",
        color_continuous_scale=[
            (0, Colors.DANGER),
            (0.45, Colors.DANGER_DIM),
            (0.5, Colors.BG_TERTIARY),
            (0.55, Colors.SUCCESS_DIM),
            (1, Colors.SUCCESS)
        ],
        range_color=[-max_abs_change, max_abs_change]
    )

    # Prepare custom data
    customdata = np.column_stack([df["Change"].values, df["Last"].values])

    # Update traces with premium styling
    fig.update_traces(
        customdata=customdata,
        texttemplate="<b>%{label}</b><br><span style='font-family: JetBrains Mono'>%{customdata[0]:+.2f}%</span>",
        textfont=dict(
            size=14,
            color=Colors.TEXT_PRIMARY,
            family=Typography.FONT_DISPLAY
        ),
        textposition="middle center",
        hovertemplate=(
            "<b style='font-size:16px'>%{label}</b><br>"
            "<span style='font-family: JetBrains Mono; font-size:18px; font-weight:700'>"
            "%{customdata[0]:+.2f}%</span><br>"
            "<span style='font-family: JetBrains Mono'>$%{customdata[1]:.2f}</span>"
            "<extra></extra>"
        ),
        marker=dict(
            line=dict(width=2, color=Colors.BG_PRIMARY),
            cornerradius=4
        )
    )

    # Update layout
    fig.update_layout(
        get_premium_layout(),
        margin=dict(t=20, l=5, r=5, b=5),
        uirevision="keep",
        coloraxis_showscale=False
    )

    return fig


# =============================================================================
# DETAILED STOCK CHART
# =============================================================================

def create_stock_chart(ticker: str, hist_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Premium stock chart with technical indicators

    Modern design with vibrant colors and smooth animations
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
        row_heights=[0.6, 0.25, 0.15],
        vertical_spacing=0.04,
        subplot_titles=("", "", "")
    )

    # =========================================================================
    # PANEL 1: CANDLESTICK CHART WITH INDICATORS
    # =========================================================================

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"],
            high=hist_df["High"],
            low=hist_df["Low"],
            close=hist_df["Close"],
            increasing_line_color=Colors.SUCCESS,
            increasing_fillcolor=Colors.SUCCESS,
            decreasing_line_color=Colors.DANGER,
            decreasing_fillcolor=Colors.DANGER,
            name="Price",
            showlegend=False,
            increasing=dict(line=dict(width=1)),
            decreasing=dict(line=dict(width=1))
        ),
        row=1, col=1
    )

    # Bollinger Bands - Upper
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_up"],
            mode="lines",
            line=dict(width=1.5, color=Colors.CHART_PURPLE, dash="dot"),
            name="BB Upper",
            opacity=0.6,
            showlegend=True
        ),
        row=1, col=1
    )

    # Bollinger Bands - Lower with fill
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_low"],
            mode="lines",
            line=dict(width=1.5, color=Colors.CHART_PURPLE, dash="dot"),
            name="BB Lower",
            opacity=0.6,
            fill="tonexty",
            fillcolor=f"{Colors.CHART_PURPLE}15",
            showlegend=True
        ),
        row=1, col=1
    )

    # SMA 20
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA20"],
            mode="lines",
            line=dict(width=2, color=Colors.ACCENT_PRIMARY),
            name="SMA 20",
            showlegend=True
        ),
        row=1, col=1
    )

    # SMA 50
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA50"],
            mode="lines",
            line=dict(width=2, color=Colors.CHART_ORANGE),
            name="SMA 50",
            showlegend=True
        ),
        row=1, col=1
    )

    # =========================================================================
    # PANEL 2: VOLUME
    # =========================================================================

    volume_colors = calculate_volume_colors(hist_df)

    fig.add_trace(
        go.Bar(
            x=hist_df.index,
            y=hist_df["Volume"],
            marker_color=volume_colors,
            marker_line_width=0,
            name="Volume",
            showlegend=False,
            opacity=0.7
        ),
        row=2, col=1
    )

    # =========================================================================
    # PANEL 3: RSI
    # =========================================================================

    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["RSI"],
            mode="lines",
            line=dict(width=2.5, color=Colors.CHART_YELLOW),
            name="RSI",
            showlegend=False,
            fill="tozeroy",
            fillcolor=f"{Colors.CHART_YELLOW}15"
        ),
        row=3, col=1
    )

    # RSI levels
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=Colors.DANGER,
        line_width=1,
        opacity=0.5,
        row=3, col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=Colors.SUCCESS,
        line_width=1,
        opacity=0.5,
        row=3, col=1
    )

    # =========================================================================
    # AXIS STYLING
    # =========================================================================

    # Price axis
    low_min = float(hist_df["Low"].min())
    high_max = float(hist_df["High"].max())
    price_padding = (high_max - low_min) * 0.05

    fig.update_yaxes(
        title_text="<b>PRICE</b>",
        title_font=dict(size=11, family=Typography.FONT_DISPLAY),
        range=[low_min - price_padding, high_max + price_padding],
        fixedrange=True,
        gridcolor=Colors.BORDER,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=Colors.BORDER_BRIGHT,
        row=1, col=1
    )

    # Volume axis
    fig.update_yaxes(
        title_text="<b>VOLUME</b>",
        title_font=dict(size=11, family=Typography.FONT_DISPLAY),
        fixedrange=True,
        gridcolor=Colors.BORDER,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=Colors.BORDER_BRIGHT,
        row=2, col=1
    )

    # RSI axis
    fig.update_yaxes(
        title_text="<b>RSI</b>",
        title_font=dict(size=11, family=Typography.FONT_DISPLAY),
        range=[0, 100],
        fixedrange=True,
        gridcolor=Colors.BORDER,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=Colors.BORDER_BRIGHT,
        row=3, col=1
    )

    # X-axis
    x_min = hist_df.index.min()
    x_max = hist_df.index.max() + timedelta(days=1)

    fig.update_xaxes(
        range=[x_min, x_max],
        rangeslider_visible=False,
        showgrid=True,
        gridcolor=Colors.BORDER,
        gridwidth=1,
        showline=True,
        linewidth=1,
        linecolor=Colors.BORDER_BRIGHT,
        row=3, col=1,
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.update_layout(
        **get_premium_layout(title=ticker, height=700),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0,
            bgcolor=Colors.BG_ELEVATED,
            bordercolor=Colors.BORDER_BRIGHT,
            borderwidth=1,
            font=dict(size=11, family=Typography.FONT_DISPLAY)
        ),
        dragmode="zoom",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=Colors.BG_ELEVATED,
            bordercolor=Colors.BORDER_BRIGHT,
            font=dict(
                family=Typography.FONT_MONO,
                size=12,
                color=Colors.TEXT_PRIMARY
            )
        )
    )

    return fig
