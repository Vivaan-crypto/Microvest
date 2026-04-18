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

from config import Colors, Typography
from data import calculate_volume_colors
from indicators import compute_indicators, normalize_indicator_selection, INDICATOR_SPECS


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

def create_stock_chart(
    ticker: str,
    hist_df: pd.DataFrame,
    selected_indicators: Optional[list[str]] = None,
) -> Optional[go.Figure]:
    """
    Premium stock chart with technical indicators

    Modern design with vibrant colors and smooth animations
    """
    if hist_df.empty or len(hist_df) < 2:
        return None

    selected_indicators = normalize_indicator_selection(selected_indicators)
    hist_df = compute_indicators(hist_df, selected_indicators)

    # Create a TradingView-style 3 panel layout
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.64, 0.18, 0.18],
        vertical_spacing=0.04,
        subplot_titles=("", "", "")
    )

    # =========================================================================
    # PANEL 1: CANDLESTICK CHART WITH INDICATORS
    # =========================================================================

    hist_df["Daily_Change"] = ((hist_df["Close"] - hist_df["Open"]) / hist_df["Open"] * 100)
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
                "Change: %{customdata:+.2f}%<extra></extra>"
            )
        ),
        row=1, col=1
    )

    overlay_traces = [
        ("sma20", "SMA20"),
        ("sma50", "SMA50"),
        ("sma100", "SMA100"),
        ("sma200", "SMA200"),
        ("ema9", "EMA9"),
        ("ema21", "EMA21"),
        ("ema50", "EMA50"),
        ("vwap", "VWAP"),
        ("psar", "PSAR"),
        ("supertrend", "SUPERTREND"),
    ]

    for indicator_key, column_name in overlay_traces:
        if indicator_key in selected_indicators and column_name in hist_df.columns:
            spec = INDICATOR_SPECS[indicator_key]
            fig.add_trace(
                go.Scatter(
                    x=hist_df.index,
                    y=hist_df[column_name],
                    mode="lines",
                    line=dict(width=2 if "sma" in indicator_key or "ema" in indicator_key else 1.8, color=spec["color"]),
                    name=spec["label"],
                    opacity=0.95,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    if "bbands" in selected_indicators and {"BB_MID", "BB_UP", "BB_LOW"}.issubset(hist_df.columns):
        fig.add_trace(
            go.Scatter(
                x=hist_df.index,
                y=hist_df["BB_UP"],
                mode="lines",
                line=dict(width=1.4, color=Colors.CHART_PURPLE_TRANSPARENT),
                name="BB Upper",
                opacity=0.8,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df.index,
                y=hist_df["BB_LOW"],
                mode="lines",
                line=dict(width=1.4, color=Colors.CHART_PURPLE_TRANSPARENT, dash="dash"),
                name="BB Lower",
                opacity=0.8,
                fill="tonexty",
                fillcolor="rgba(168,85,247,0.08)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if "ichimoku" in selected_indicators and {"TENKAN", "KIJUN", "SENKOU_A", "SENKOU_B"}.issubset(hist_df.columns):
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["TENKAN"], mode="lines", line=dict(width=1.4, color="#60a5fa"), name="Tenkan", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["KIJUN"], mode="lines", line=dict(width=1.4, color="#22c55e"), name="Kijun", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["SENKOU_A"], mode="lines", line=dict(width=1, color="rgba(96,165,250,0.35)"), name="Senkou A", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["SENKOU_B"], mode="lines", line=dict(width=1, color="rgba(34,197,94,0.35)"), name="Senkou B", showlegend=True), row=1, col=1)

    if "donchian" in selected_indicators and {"DONCHIAN_HIGH", "DONCHIAN_LOW"}.issubset(hist_df.columns):
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["DONCHIAN_HIGH"], mode="lines", line=dict(width=1.2, color="#38bdf8", dash="dot"), name="Donchian High", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["DONCHIAN_LOW"], mode="lines", line=dict(width=1.2, color="#38bdf8", dash="dot"), name="Donchian Low", showlegend=True), row=1, col=1)

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

    if "volume_ma" in selected_indicators and "VOLUME_MA" in hist_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hist_df.index,
                y=hist_df["VOLUME_MA"],
                mode="lines",
                line=dict(width=1.6, color=INDICATOR_SPECS["volume_ma"]["color"]),
                name="Volume MA",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # =========================================================================
    # PANEL 3: RSI with glow effect
    # =========================================================================

    lower_added = False
    if "rsi14" in selected_indicators and "RSI14" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["RSI14"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["rsi14"]["color"]), name="RSI 14", showlegend=True), row=3, col=1)
        lower_added = True
    if "rsi29" in selected_indicators and "RSI29" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["RSI29"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["rsi29"]["color"]), name="RSI 29", showlegend=True), row=3, col=1)
        lower_added = True
    if "macd" in selected_indicators and {"MACD", "MACD_SIGNAL", "MACD_HIST"}.issubset(hist_df.columns):
        fig.add_trace(go.Bar(x=hist_df.index, y=hist_df["MACD_HIST"], name="MACD Hist", marker_color="rgba(0,229,255,0.25)", showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["MACD"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["macd"]["color"]), name="MACD", showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["MACD_SIGNAL"], mode="lines", line=dict(width=2, color="#f59e0b"), name="MACD Signal", showlegend=True), row=3, col=1)
        lower_added = True
    if "stochastic" in selected_indicators and {"STOCH_K", "STOCH_D"}.issubset(hist_df.columns):
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["STOCH_K"], mode="lines", line=dict(width=2, color="#f97316"), name="Stoch %K", showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["STOCH_D"], mode="lines", line=dict(width=2, color="#22c55e"), name="Stoch %D", showlegend=True), row=3, col=1)
        lower_added = True
    if "adx" in selected_indicators and "ADX" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["ADX"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["adx"]["color"]), name="ADX", showlegend=True), row=3, col=1)
        lower_added = True
    if "cci" in selected_indicators and "CCI" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["CCI"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["cci"]["color"]), name="CCI", showlegend=True), row=3, col=1)
        lower_added = True
    if "atr" in selected_indicators and "ATR" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["ATR"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["atr"]["color"]), name="ATR", showlegend=True), row=3, col=1)
        lower_added = True
    if "roc" in selected_indicators and "ROC" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["ROC"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["roc"]["color"]), name="ROC", showlegend=True), row=3, col=1)
        lower_added = True
    if "mfi" in selected_indicators and "MFI" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["MFI"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["mfi"]["color"]), name="MFI", showlegend=True), row=3, col=1)
        lower_added = True
    if "obv" in selected_indicators and "OBV" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["OBV"], mode="lines", line=dict(width=2, color=INDICATOR_SPECS["obv"]["color"]), name="OBV", showlegend=True), row=3, col=1)
        lower_added = True

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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(size=11, color=Colors.TEXT_SECONDARY),
        ),
    )

    return fig
