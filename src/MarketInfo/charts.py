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
    Premium dark theme layout for charts with glassmorphism
    """
    return {
        "paper_bgcolor": "rgba(10, 10, 15, 0.6)",  # Semi-transparent
        "plot_bgcolor": "rgba(19, 19, 26, 0.4)",  # Very transparent
        "font": {
            "color": Colors.TEXT_PRIMARY,
            "family": Typography.FONT_DISPLAY,
            "size": 13
        },
        "title": {
            "text": f"<b>{title}</b>" if title else "",
            "font": {
                "size": 32,
                "color": Colors.TEXT_PRIMARY,
                "family": Typography.FONT_DISPLAY,
                "weight": 700
            },
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top"
        },
        "showlegend": True,
        "height": height,
        "margin": dict(l=70, r=40, t=100, b=60),
        #"scrollZoom": True,
        "dragmode": "pan",
        "hovermode": "x unified",
        "hoverlabel": dict(
            bgcolor="rgba(32, 32, 46, 0.95)",  # Glassmorphic background
            bordercolor=Colors.SUCCESS,  # Green border
            font=dict(
                family=Typography.FONT_MONO,
                size=13,
                color=Colors.TEXT_PRIMARY
            ),
            align="left"
        )
    }


# =============================================================================
# HEATMAP / TREEMAP
# =============================================================================

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Premium stock performance heatmap with vibrant colors
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=Colors.TEXT_MUTED, family=Typography.FONT_DISPLAY)
        )
        fig.update_layout(get_premium_layout())
        return fig

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
    customdata = np.column_stack([
        df["Ticker"].values,
        df["Change"].values,
        df["Last"].values
    ])

    # Update traces with premium styling
    fig.update_traces(
        customdata=customdata,
        texttemplate="<b>%{label}</b><br><span style='font-family: JetBrains Mono'>%{customdata[1]:+.2f}%</span>",
        textfont=dict(
            size=14,
            color=Colors.TEXT_PRIMARY,
            family=Typography.FONT_DISPLAY
        ),
        textposition="middle center",
        hovertemplate=(
            "<b style='font-size:16px'>%{customdata[0]}</b><br>"
            "<span style='font-family: JetBrains Mono; font-size:18px; font-weight:700'>"
            "%{customdata[1]:+.2f}%</span><br>"
            "<span style='font-family: JetBrains Mono'>$%{customdata[2]:.2f}</span>"
            "<extra></extra>"
        ),
        marker=dict(
            line=dict(width=2, color=Colors.BG_PRIMARY),
            cornerradius=4
        )
    )

    # Update layout
    fig.update_layout(
        **get_premium_layout(),

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

    hist_df['Daily_Change'] = ((hist_df['Close'] - hist_df['Open']) / hist_df['Open'] * 100)
    print(hist_df['Daily_Change'].head())
    # Candlestick with semi-transparent colors
    fig.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"],
            high=hist_df["High"],
            low=hist_df["Low"],
            close=hist_df["Close"],
            customdata=hist_df['Daily_Change'],
            increasing_line_color=Colors.SUCCESS,
            increasing_fillcolor=f"{Colors.SUCCESS}",  # 80% opacity
            decreasing_line_color=Colors.DANGER,
            decreasing_fillcolor=f"{Colors.DANGER}",  # 80% opacity
            showlegend=False,
            increasing=dict(line=dict(width=1.5)),
            decreasing=dict(line=dict(width=1.5)),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Open: $%{open:.2f}<br>"
                "High: $%{high:.2f}<br>"
                "Low: $%{low:.2f}<br>"
                "Close: $%{close:.2f}<br>"
                "Change: $%{customdata:+.2f}%<extra></extra>"
                "<span style='font-family: JetBrains Mono; font-size:14px; font-weight:700'>"
            )
        ),
        row=1, col=1
    )

    # Bollinger Bands - Upper with glow effect
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_up"],
            mode="lines",
            line=dict(width=2, color=Colors.CHART_PURPLE_TRANSPARENT),
            name="BB Upper",
            opacity=0.8,
            showlegend=True
        ),
        row=1, col=1
    )

    # Bollinger Bands - Lower with transparent fill
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_low"],
            mode="lines",
            line=dict(width=2, color=Colors.CHART_PURPLE_TRANSPARENT, dash="dash"),
            name="BB Lower",
            opacity= 0.8,
            fill="tonexty",
            fillcolor=f"{Colors.CHART_PURPLE_TRANSPARENT}",  # 12% opacity - very glassy
            showlegend=True
        ),
        row=1, col=1
    )

    # SMA 20 - Electric cyan with glow
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA20"],
            mode="lines",
            line=dict(width=3, color=Colors.ACCENT_PRIMARY),
            name="SMA 20",
            showlegend=True,
            opacity=0.9
        ),
        row=1, col=1
    )

    # SMA 50 - Orange with glow
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA50"],
            mode="lines",
            line=dict(width=3, color=Colors.CHART_ORANGE),
            name="SMA 50",
            showlegend=True,
            opacity=0.9
        ),
        row=1, col=1
    )

    # =========================================================================
    # PANEL 2: VOLUME with gradient effect
    # =========================================================================

    volume_colors = calculate_volume_colors(hist_df)

    fig.add_trace(
        go.Bar(
            x=hist_df.index,
            y=hist_df["Volume"],
            marker=dict(
                color=volume_colors,
                line=dict(width=0),
                opacity=0.6  # Semi-transparent bars
            ),
            name="Volume",
            showlegend=False
        ),
        row=2, col=1
    )

    # =========================================================================
    # PANEL 3: RSI with glow effect
    # =========================================================================

    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["RSI_14"],
            mode="lines",
            line=dict(width=2, color=Colors.CHART_PURPLE),
            name="RSI",
            showlegend=False,
            opacity=1
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["RSI_29"],
            mode="lines",
            line=dict(width=2, color=Colors.CHART_YELLOW),
            name="RSI",
            showlegend=False,
            opacity=1
        ),
        row=3, col=1
    )

    # RSI levels with subtle lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=Colors.DANGER,
        line_width=1.5,
        opacity=0.4,
        row=3, col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=Colors.SUCCESS,
        line_width=1.5,
        opacity=0.4,
        row=3, col=1
    )

    # Add subtle middle line at 50
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color=Colors.TEXT_DIM,
        line_width=1,
        opacity=0.3,
        row=3, col=1
    )

    # =========================================================================
    # AXIS STYLING - Premium glassmorphism
    # =========================================================================

    # Price axis
    low_min = float(hist_df["Low"].min())
    high_max = float(hist_df["High"].max())
    price_padding = (high_max - low_min) * 0.05

    fig.update_yaxes(
        title_text="<b>PRICE ($)</b>",
        title_font=dict(
            size=18,
            family=Typography.FONT_DISPLAY,
            color=Colors.TEXT_SECONDARY
        ),
        range=[low_min - price_padding, high_max + price_padding],
        gridcolor="rgba(160, 160, 184, 0.16)",  # Very subtle grid
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor=Colors.BORDER_BRIGHT,
        nticks = 30,
        automargin = False,
        tickfont=dict(
            family=Typography.FONT_MONO,
            size=12,
            color=Colors.TEXT_SECONDARY
        ),
        row=1, col=1
    )

    # Volume axis
    fig.update_yaxes(
        title_text="<b>VOLUME</b>",
        title_font=dict(
            size=16,
            family=Typography.FONT_DISPLAY,
            color=Colors.TEXT_SECONDARY
        ),
        fixedrange=True,
        gridcolor="rgba(160, 160, 184, 0.16)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        nticks=13,
        automargin=False,
        linecolor=Colors.BORDER_BRIGHT,
        tickfont=dict(
            family=Typography.FONT_MONO,
            size=10,
            color=Colors.TEXT_SECONDARY
        ),
        row=2, col=1
    )

    # RSI axis
    fig.update_yaxes(
        title_text="<b>RSI</b>",
        title_font=dict(
            size=12,
            family=Typography.FONT_DISPLAY,
            color=Colors.TEXT_SECONDARY
        ),
        range=[0, 100],
        fixedrange=True,
        gridcolor="rgba(160, 160, 184, 0.16)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        nticks=10,
        automargin=False,
        linecolor=Colors.BORDER_BRIGHT,
        tickfont=dict(
            family=Typography.FONT_MONO,
            size=10,
            color=Colors.TEXT_SECONDARY
        ),
        row=3, col=1
    )

    # X-axis
    x_min = hist_df.index.min()
    x_max = hist_df.index.max() + timedelta(days=1)

    fig.update_xaxes(
        range=[x_min, x_max],
        rangeslider_visible=False,
        showgrid=True,
        gridcolor="rgba(160, 160, 184, 0.08)",
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor=Colors.BORDER_BRIGHT,
        tickfont=dict(
            family=Typography.FONT_MONO,
            size=11,
            color=Colors.TEXT_SECONDARY
        ),
        row=3, col=1,
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )

    # =========================================================================
    # LAYOUT - Premium glassmorphism
    # =========================================================================

    fig.update_layout(
        **get_premium_layout(title=ticker, height=750),
        xaxis_rangeslider_visible=False,
    )

    return fig
