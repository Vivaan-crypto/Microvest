"""
============================================================
  AI Insights Page — FinanceAI Pro
  Powered by OpenAI GPT-4o
  Multi-source news intelligence, decision engine, n8n automation
============================================================
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import os
from datetime import datetime
import json
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from openai import OpenAI
from src.MarketInfo.news import build_news_context as build_news_context_from_news
from src.MarketInfo.data import all_stock_data, single_stock_data
from src.MarketInfo.indicators import compute_indicators

dash.register_page(__name__, path="/ai", name="AI Insights")

# =============================================================================
# OPENAI SETUP
# =============================================================================
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    config_path = Path(__file__).resolve().parents[1] / "ConfigurationFiles" / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            api_key = (config.get("api_keys", {}) or {}).get("gpt", "").strip()
        except Exception:
            api_key = ""

openai_client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-5-nano"

SYSTEM_PROMPT = """You are FinanceAI Pro — an elite institutional-grade financial analyst.

You synthesize real-time multi-source news streams, fundamental data, technical signals, and macroeconomic context to deliver precise, actionable investment intelligence.

Your responses are:
- Structured and decisive — always conclude with a clear BUY / HOLD / SELL signal and confidence level (High / Medium / Low)
- Backed by named sources when available (e.g., "Per Reuters...", "CNBC reports...")
- Calibrated for risk — always include key downside risks
- Professional but readable — no filler, no vague hedging

When given news context, explicitly reference which sources support your thesis.
Format decisions as:
  ▸ SIGNAL: [BUY/HOLD/SELL]  |  CONFIDENCE: [High/Medium/Low]  |  HORIZON: [Short/Medium/Long-term]
"""

# =============================================================================
# COLORS — dark terminal-grade theme
# =============================================================================

C = {
    "bg":       "#080810",
    "bg2":      "#0c0c18",
    "surface":  "#111120",
    "surface2": "#16162a",
    "surface3": "#1c1c30",
    "border":   "rgba(255,255,255,0.06)",
    "border2":  "rgba(255,255,255,0.10)",
    "accent":   "#00e5ff",   # electric cyan
    "accent2":  "#00ff88",   # neon green
    "accent3":  "#7c3aed",   # deep violet
    "accent4":  "#f59e0b",   # amber
    "green":    "#00ff88",
    "red":      "#ff3366",
    "yellow":   "#f59e0b",
    "text":     "#f0f0ff",
    "muted":    "#4a4a6a",
    "dim":      "#8080a0",
}

# =============================================================================
# QUICK PROMPTS
# =============================================================================

QUICK_PROMPTS = [
    ("⚡ Full Analysis",    "Give me a comprehensive technical and fundamental analysis of {ticker}, including recent news signals."),
    ("🎯 Decision",         "Based on current news, macro context, and technicals — give me a BUY, HOLD, or SELL decision on {ticker} with confidence level."),
    ("📰 News Impact",      "Analyze the latest news for {ticker} across all major sources. What's the dominant narrative and how does it affect the price?"),
    ("⚠️  Risk Assessment", "What are the top 5 risks (macro, sector, company-specific) that could negatively impact {ticker} right now?"),
    ("📊 vs Competitors",   "Compare {ticker} against its main competitors. Who has the stronger risk/reward profile and why?"),
    ("🌍 Macro Lens",       "How does current Fed policy, inflation data, and global macro trends impact the outlook for {ticker}?"),
]

MARKET_SUMMARY_PROMPT = """
You are analyzing live market conditions. Provide a structured market intelligence brief covering:

1. **Index Pulse** — S&P 500, NASDAQ, DOW trends today
2. **Sector Rotation** — Which sectors are seeing inflows/outflows
3. **Volatility Regime** — VIX context and what it means
4. **Top 3 Market-Moving Themes** — The narratives driving price action
5. **Decision Framework** — Risk-on or Risk-off environment? Recommended posture.

Be precise. Use institutional language. End with a clear market bias.
"""

# =============================================================================
# NEWS SOURCE CONFIG
# =============================================================================

NEWS_SOURCES = [
    {"id": "cnbc",       "name": "CNBC",           "icon": "📡", "color": "#00e5ff"},
    {"id": "reuters",    "name": "Reuters",         "icon": "🔵", "color": "#00ff88"},
    {"id": "bloomberg",  "name": "Bloomberg",       "icon": "🟠", "color": "#f59e0b"},
    {"id": "wsj",        "name": "WSJ",             "icon": "📋", "color": "#7c3aed"},
    {"id": "ft",         "name": "Financial Times", "icon": "🔴", "color": "#ff3366"},
]

N8N_AUTOMATIONS = [
    {"label": "📧 Email Alert on BUY Signal",      "desc": "Trigger email when AI confidence ≥ High"},
    {"label": "📲 Slack Notify on Earnings",        "desc": "Post to #trading channel on earnings beats"},
    {"label": "📊 Log Decisions to Google Sheets",  "desc": "Auto-log all AI signals with timestamp"},
    {"label": "🔔 Webhook on Sentiment Shift",      "desc": "Fire webhook when sentiment flips bearish"},
    {"label": "📁 Save Summaries to Notion",        "desc": "Archive daily market summaries to your Notion DB"},
]

ANALYSIS_MODES = [
    {"label": "Deep Dive", "value": "deep_dive"},
    {"label": "News Digest", "value": "news_digest"},
    {"label": "Risk Scan", "value": "risk_scan"},
    {"label": "Compare", "value": "compare"},
    {"label": "Macro", "value": "macro"},
]

# =============================================================================
# COMPONENT HELPERS
# =============================================================================

def status_dot(color: str, pulse: bool = False):
    return html.Div(style={
        "width": "7px", "height": "7px", "borderRadius": "50%",
        "background": color,
        "boxShadow": f"0 0 8px {color}",
        "animation": "glowPulse 2s ease infinite" if pulse else "none",
        "flexShrink": "0",
    })


def badge(text: str, color: str = None):
    color = color or C["accent"]
    return html.Span(text, style={
        "background": f"{color}22",
        "border": f"1px solid {color}55",
        "color": color,
        "fontSize": "9px", "fontWeight": "800",
        "padding": "2px 8px", "borderRadius": "20px",
        "letterSpacing": "0.1em", "textTransform": "uppercase",
        "fontFamily": "'JetBrains Mono', monospace",
    })


def section_label(text: str):
    return html.Div(text, style={
        "fontSize": "9px", "fontWeight": "700", "color": C["muted"],
        "letterSpacing": "2px", "textTransform": "uppercase",
        "marginBottom": "10px", "fontFamily": "'JetBrains Mono', monospace",
    })


def build_ticker_context(ticker: str) -> str:
    """Build a compact market context block from the live snapshot and history."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return ""

    lines: list[str] = []
    try:
        snapshot = all_stock_data()
        if not snapshot.empty:
            row = snapshot[snapshot["Ticker"] == ticker]
            if not row.empty:
                stock = row.iloc[0]
                lines.append(
                    f"Snapshot: last={stock['Last']:.2f}, change={stock['Change']:+.2f}%, "
                    f"sector={stock['Sector']}, volume={int(stock.get('Volume', 0))}"
                )
    except Exception:
        pass

    try:
        history = single_stock_data(ticker, period_days=220)
        if not history.empty and len(history) > 30:
            enriched = compute_indicators(history, ["sma20", "sma50", "ema21", "rsi14", "macd", "volume_ma", "bbands"])
            latest = enriched.iloc[-1]
            close = float(latest["Close"])
            trend = "above" if close > float(latest.get("SMA20", close)) else "below"
            lines.append(
                f"Technicals: close={close:.2f}, sma20={latest.get('SMA20', float('nan')):.2f}, "
                f"sma50={latest.get('SMA50', float('nan')):.2f}, rsi14={latest.get('RSI14', float('nan')):.1f}, "
                f"macd={latest.get('MACD', float('nan')):.2f}, trend={trend} sma20"
            )
    except Exception:
        pass

    return "\n".join(lines)


def message_bubble(role: str, content: str, timestamp: str = ""):
    is_user = role == "user"

    # Parse signal line if present in AI response
    signal_color = C["dim"]
    if "▸ SIGNAL: BUY" in content:
        signal_color = C["green"]
    elif "▸ SIGNAL: SELL" in content:
        signal_color = C["red"]
    elif "▸ SIGNAL: HOLD" in content:
        signal_color = C["yellow"]

    return html.Div([
        html.Div(
            "YOU" if is_user else "AI",
            style={
                "width": "32px", "height": "32px", "borderRadius": "50%",
                "background": f"linear-gradient(135deg, {C['accent']}, {C['accent2']})" if is_user
                else f"linear-gradient(135deg, {C['accent3']}, {C['accent']})",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "fontSize": "9px", "fontWeight": "900", "color": "#fff",
                "flexShrink": "0",
                "boxShadow": f"0 0 14px {'rgba(0,229,255,0.35)' if is_user else 'rgba(124,58,237,0.45)'}",
                "fontFamily": "'JetBrains Mono', monospace",
                "letterSpacing": "0.05em",
            }
        ),
        html.Div([
            html.Div(content, style={
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.8",
                "fontSize": "14px",
                "color": C["text"] if is_user else "#d0d0f0",
                "fontFamily": "'Inter', 'Outfit', sans-serif",
                "borderLeft": f"2px solid {signal_color}" if not is_user and signal_color != C["dim"] else "none",
                "paddingLeft": "12px" if not is_user and signal_color != C["dim"] else "0",
            }),
            html.Div(timestamp, style={
                "fontSize": "9px", "color": C["muted"],
                "marginTop": "8px", "textAlign": "right",
                "fontFamily": "'JetBrains Mono', monospace",
            }) if timestamp else None
        ], style={
            "background": C["surface2"] if is_user else C["surface3"],
            "border": f"1px solid {'rgba(0,229,255,0.15)' if is_user else C['border2']}",
            "borderRadius": "4px 14px 14px 14px" if not is_user else "14px 4px 14px 14px",
            "padding": "14px 18px",
            "maxWidth": "84%",
            "boxShadow": "0 6px 28px rgba(0,0,0,0.4)",
        })
    ], style={
        "display": "flex",
        "flexDirection": "row" if not is_user else "row-reverse",
        "alignItems": "flex-start",
        "gap": "10px",
        "marginBottom": "16px",
        "animation": "slideInUp 0.28s cubic-bezier(0.4,0,0.2,1) both",
    })


def render_sentiment_bar(score: float):
    if score > 0.6:
        color, label = C["green"], "BULLISH"
    elif score < 0.4:
        color, label = C["red"], "BEARISH"
    else:
        color, label = C["yellow"], "NEUTRAL"

    pct = f"{score * 100:.0f}%"
    return html.Div([
        html.Div([
            # Track
            html.Div(style={
                "height": "8px", "borderRadius": "4px",
                "background": "rgba(255,255,255,0.05)",
                "overflow": "hidden", "position": "relative",
            }, children=[
                html.Div(style={
                    "height": "100%", "borderRadius": "4px",
                    "width": pct,
                    "background": f"linear-gradient(90deg, {color}99, {color})",
                    "transition": "width 1s cubic-bezier(0.4,0,0.2,1)",
                    "boxShadow": f"0 0 10px {color}66",
                })
            ]),
        ]),
        html.Div([
            html.Span(label, style={
                "fontSize": "11px", "fontWeight": "800", "color": color,
                "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "1.5px",
            }),
            html.Span(f" {pct}", style={
                "fontSize": "10px", "color": C["muted"],
                "fontFamily": "'JetBrains Mono', monospace",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginTop": "8px"}),
    ])


def news_source_pill(source: dict, active: bool = True):
    return html.Div([
        html.Span(source["icon"], style={"fontSize": "11px"}),
        html.Span(source["name"], style={
            "fontSize": "10px", "fontWeight": "700",
            "fontFamily": "'JetBrains Mono', monospace",
            "letterSpacing": "0.05em",
        }),
        html.Div(style={
            "width": "5px", "height": "5px", "borderRadius": "50%",
            "background": source["color"] if active else C["muted"],
            "boxShadow": f"0 0 6px {source['color']}" if active else "none",
        })
    ], style={
        "display": "flex", "alignItems": "center", "gap": "5px",
        "background": f"{source['color']}11" if active else "rgba(255,255,255,0.03)",
        "border": f"1px solid {source['color']}33" if active else f"1px solid {C['border']}",
        "color": source["color"] if active else C["muted"],
        "padding": "4px 9px", "borderRadius": "20px",
        "cursor": "pointer",
        "transition": "all 0.18s ease",
    })


def n8n_automation_row(item: dict):
    return html.Div([
        html.Div([
            html.Div(item["label"], style={
                "fontSize": "12px", "fontWeight": "700",
                "color": C["text"], "fontFamily": "'Outfit', sans-serif",
            }),
            html.Div(item["desc"], style={
                "fontSize": "10px", "color": C["dim"],
                "fontFamily": "'Outfit', sans-serif", "marginTop": "2px",
            }),
        ], style={"flex": "1"}),
        html.Div([
            html.Div(style={
                "width": "32px", "height": "18px", "borderRadius": "9px",
                "background": "rgba(255,255,255,0.06)",
                "border": f"1px solid {C['border2']}",
                "position": "relative", "cursor": "pointer",
            }, children=[
                html.Div(style={
                    "width": "12px", "height": "12px", "borderRadius": "50%",
                    "background": C["muted"],
                    "position": "absolute", "top": "2px", "left": "2px",
                    "transition": "all 0.2s ease",
                })
            ])
        ])
    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "10px 0",
        "borderBottom": f"1px solid {C['border']}",
    })


# =============================================================================
# INJECTED STYLES
# =============================================================================

INJECTED_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;600;700;800;900&display=swap');

    * { box-sizing: border-box; }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 6px rgba(0,229,255,0.25); }
        50%       { box-shadow: 0 0 20px rgba(0,229,255,0.7); }
    }
    @keyframes typingDot {
        0%, 80%, 100% { transform: scale(0.55); opacity: 0.35; }
        40%            { transform: scale(1);    opacity: 1; }
    }
    @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
    }
    @keyframes scanline {
        0%   { background-position: 0 0; }
        100% { background-position: 0 100px; }
    }

    /* Chat input */
    .ai-text-input {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #f0f0ff !important;
        font-size: 14px !important;
        padding: 11px 14px !important;
        outline: none !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    .ai-text-input:focus {
        border-color: rgba(0,229,255,0.4) !important;
        box-shadow: 0 0 0 3px rgba(0,229,255,0.08) !important;
        background: rgba(255,255,255,0.05) !important;
    }
    .ai-text-input::placeholder { color: #4a4a6a !important; }

    /* Ticker input */
    .ai-ticker-input {
        background: rgba(0,229,255,0.04) !important;
        border: 1px solid rgba(0,229,255,0.2) !important;
        border-radius: 8px !important;
        color: #00e5ff !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        padding: 7px 12px !important;
        width: 100px !important;
        outline: none !important;
        transition: all 0.2s ease !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .ai-ticker-input:focus {
        border-color: rgba(0,229,255,0.55) !important;
        box-shadow: 0 0 0 3px rgba(0,229,255,0.1) !important;
    }

    .ai-mode-select .Select-control {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        min-height: 38px !important;
        box-shadow: none !important;
    }
    .ai-mode-select .Select-placeholder,
    .ai-mode-select .Select-value-label {
        color: #f0f0ff !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .ai-mode-select .Select-menu-outer {
        background: #0c0c18 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* Send button */
    .ai-send-btn {
        background: linear-gradient(135deg, #00e5ff, #00ff88);
        border: none;
        border-radius: 10px;
        color: #08080f;
        font-size: 16px;
        font-weight: 900;
        cursor: pointer;
        height: 44px;
        min-width: 54px;
        transition: all 0.18s ease;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Inter', sans-serif;
    }
    .ai-send-btn:hover { transform: scale(1.05); box-shadow: 0 0 22px rgba(0,229,255,0.55); }
    .ai-send-btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }

    /* Quick chips */
    .ai-quick-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        color: #8080a0;
        padding: 5px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.18s ease;
        white-space: nowrap;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.02em;
    }
    .ai-quick-chip:hover {
        background: rgba(0,229,255,0.10);
        border-color: rgba(0,229,255,0.4);
        color: #00e5ff;
        transform: translateY(-1px);
    }

    /* Summary button */
    .ai-summary-btn {
        background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(0,255,136,0.08));
        border: 1px solid rgba(0,229,255,0.25);
        color: #00e5ff;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.18s ease;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.04em;
    }
    .ai-summary-btn:hover {
        background: linear-gradient(135deg, rgba(0,229,255,0.18), rgba(0,255,136,0.18));
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,229,255,0.2);
    }

    /* Typing dots */
    .typing-dot {
        width: 6px; height: 6px;
        background: #00e5ff;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
    }
    .typing-dot:nth-child(1) { animation: typingDot 1.2s ease infinite; }
    .typing-dot:nth-child(2) { animation: typingDot 1.2s ease infinite 0.18s; }
    .typing-dot:nth-child(3) { animation: typingDot 1.2s ease infinite 0.36s; }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,229,255,0.3); }

    /* Stat card hover */
    .stat-card {
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        border-color: rgba(0,229,255,0.3) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
    }
"""

# Register style injection once at import time instead of embedding the callback
# inside the layout tree.
dash.clientside_callback(
    f"""
    function() {{
        if (document.getElementById('ai-injected-styles')) return '';
        var s = document.createElement('style');
        s.id = 'ai-injected-styles';
        s.textContent = `{INJECTED_CSS}`;
        document.head.appendChild(s);
        return '';
    }}
    """,
    Output("ai-page-styles", "children"),
    Input("ai-page-styles", "id"),
)

# =============================================================================
# PAGE LAYOUT
# =============================================================================

layout = html.Div([
    # Style injection
    html.Div(id="ai-page-styles", style={"display": "none"}),

    # Stores
    dcc.Store(id="ai-chat-history", data=[]),
    dcc.Store(id="ai-is-thinking", data=False),
    dcc.Store(id="ai-recent-text", data=""),
    dcc.Store(id="ai-mode", data="deep_dive"),

    # ── Top bar ─────────────────────────────────────────────────────────────
    html.Div([
        # Left: brand + model badge
        html.Div([
            html.Div([
                html.Div(style={
                    "width": "28px", "height": "28px", "borderRadius": "8px",
                    "background": f"linear-gradient(135deg, {C['accent3']}, {C['accent']})",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "14px",
                }),
                html.Div([
                    html.Span("FinanceAI ", style={
                        "fontWeight": "900", "fontSize": "15px", "color": C["text"],
                        "fontFamily": "'Outfit', sans-serif", "letterSpacing": "-0.02em",
                    }),
                    html.Span("Pro", style={
                        "fontWeight": "900", "fontSize": "15px",
                        "background": f"linear-gradient(90deg, {C['accent']}, {C['accent2']})",
                        "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                        "fontFamily": "'Outfit', sans-serif",
                    }),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
            badge("GPT-4o", C["accent"]),
            html.Div([
                status_dot(C["green"], pulse=True),
                html.Span("Live", style={
                    "fontSize": "10px", "color": C["green"],
                    "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "700",
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "5px",
                      "background": f"{C['green']}11",
                      "border": f"1px solid {C['green']}33",
                      "borderRadius": "20px", "padding": "3px 9px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),

        # Right: news source pills
        html.Div([
            html.Div("STREAMS:", style={
                "fontSize": "9px", "color": C["muted"], "fontWeight": "700",
                "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "1.5px",
                "alignSelf": "center",
            }),
            *[news_source_pill(s) for s in NEWS_SOURCES],
        ], style={"display": "flex", "alignItems": "center", "gap": "6px", "flexWrap": "wrap"}),

    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "14px 22px",
        "borderBottom": f"1px solid {C['border']}",
        "background": C["surface"],
        "marginBottom": "0",
    }),

    # ── Main content ─────────────────────────────────────────────────────────
    html.Div([

        # ── LEFT: Chat panel ────────────────────────────────────────────────
        html.Div([

            # Chat header
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(style={
                            "width": "8px", "height": "8px", "borderRadius": "50%",
                            "background": C["accent3"],
                            "boxShadow": f"0 0 10px {C['accent3']}",
                        }),
                        html.Span("Decision Engine", style={
                            "fontWeight": "700", "fontSize": "13px", "color": C["text"],
                            "fontFamily": "'Outfit', sans-serif",
                        }),
                        badge("Multi-Source", C["accent3"]),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                ]),
                # Ticker input
                html.Div([
                    html.Span("$", style={
                        "color": C["accent"], "fontWeight": "900", "fontSize": "14px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    dcc.Input(
                        id="ai-ticker-input",
                        type="text",
                        placeholder="AAPL",
                        maxLength=6,
                        debounce=False,
                        className="ai-ticker-input",
                    ),
                    dcc.Dropdown(
                        id="ai-mode-select",
                        options=ANALYSIS_MODES,
                        value="deep_dive",
                        clearable=False,
                        searchable=False,
                        className="ai-mode-select",
                        style={"width": "160px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "14px 18px",
                "borderBottom": f"1px solid {C['border']}",
                "background": C["surface"],
            }),

            # Quick prompt chips
            html.Div([
                html.Button(label, id={"type": "ai-quick-chip", "index": i},
                            className="ai-quick-chip", n_clicks=0)
                for i, (label, _) in enumerate(QUICK_PROMPTS)
            ], style={
                "display": "flex", "gap": "6px", "padding": "10px 14px",
                "overflowX": "auto", "borderBottom": f"1px solid {C['border']}",
                "background": C["bg2"],
                "scrollbarWidth": "none",
            }),

            # Messages
            html.Div(
                id="ai-chat-messages",
                children=[
                    message_bubble("assistant",
                        "Welcome to FinanceAI Pro — your institutional-grade decision engine.\n\n"
                        "I synthesize signals from CNBC, Reuters, Bloomberg, WSJ, and Financial Times "
                        "to deliver precise, multi-source investment intelligence.\n\n"
                        "→ Enter a ticker symbol above\n"
                        "→ Select a quick-action chip or type your own question\n"
                        "→ Every response ends with a clear BUY / HOLD / SELL signal\n\n"
                        "What are we analyzing today?", "")
                ],
                style={
                    "flex": "1", "overflowY": "auto",
                    "padding": "20px 16px",
                    "display": "flex", "flexDirection": "column",
                    "minHeight": "0",
                }
            ),

            # Typing indicator
            html.Div([
                html.Div(style={
                    "width": "32px", "height": "32px", "borderRadius": "50%",
                    "background": f"linear-gradient(135deg, {C['accent3']}, {C['accent']})",
                    "flexShrink": "0",
                }),
                html.Div([
                    html.Span(className="typing-dot"),
                    html.Span(className="typing-dot"),
                    html.Span(className="typing-dot"),
                ], style={
                    "background": C["surface3"],
                    "border": f"1px solid {C['border2']}",
                    "borderRadius": "4px 14px 14px 14px",
                    "padding": "11px 16px",
                    "display": "flex", "alignItems": "center", "gap": "3px",
                })
            ], id="ai-typing-indicator", style={
                "display": "none", "alignItems": "flex-start",
                "gap": "10px", "padding": "0 16px 12px",
            }),

            # Input bar
            html.Div([
                dcc.Input(
                    id="ai-user-input",
                    type="text",
                    placeholder="Ask anything — tickers, strategy, macro, risk...",
                    debounce=False,
                    className="ai-text-input",
                    style={"flex": "1", "height": "44px"},
                    n_submit=0,
                ),
                html.Button("➤", id="ai-send-btn", className="ai-send-btn", n_clicks=0),
            ], style={
                "display": "flex", "gap": "9px", "padding": "12px 16px",
                "borderTop": f"1px solid {C['border']}",
                "background": C["surface"],
            })

        ], style={
            "flex": "1", "display": "flex", "flexDirection": "column",
            "background": C["bg2"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "14px", "overflow": "hidden",
            "minHeight": "600px",
            "animation": "slideInUp 0.35s ease both",
        }),

        # ── RIGHT: Intelligence panel ────────────────────────────────────────
        html.Div([

            # ── Market Summary ─────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("🌐", style={"fontSize": "14px"}),
                        html.Span("Market Intelligence", style={
                            "fontWeight": "700", "fontSize": "13px", "color": C["text"],
                            "fontFamily": "'Outfit', sans-serif",
                        }),
                    ], style={"display": "flex", "alignItems": "center", "gap": "7px"}),
                    html.Button("Generate", id="ai-summary-btn",
                                className="ai-summary-btn", n_clicks=0),
                ], style={
                    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "padding": "13px 16px",
                    "borderBottom": f"1px solid {C['border']}",
                }),
                html.Div(
                    id="ai-market-summary",
                    children=html.Div([
                        html.Div("📊", style={"fontSize": "26px", "marginBottom": "8px"}),
                        html.Div(
                            "Generate an AI-powered snapshot of current market conditions, "
                            "sector rotation, and key themes.",
                            style={"color": C["muted"], "fontSize": "12px", "lineHeight": "1.7",
                                   "textAlign": "center", "fontFamily": "'Outfit', sans-serif",
                                   "maxWidth": "220px"}
                        )
                    ], style={"padding": "28px 16px", "display": "flex",
                              "flexDirection": "column", "alignItems": "center"}),
                    style={"overflowY": "auto", "maxHeight": "220px"}
                ),
            ], style={
                "borderBottom": f"1px solid {C['border']}",
            }),

            # ── Signal & Sentiment ──────────────────────────────────────────
            html.Div([
                section_label("AI Sentiment"),
                html.Div(id="ai-sentiment-bar", children=[
                    html.Div(style={
                        "height": "8px", "borderRadius": "4px",
                        "background": "rgba(255,255,255,0.05)",
                    })
                ]),
                html.Div(id="ai-sentiment-label",
                         children=html.Span("No analysis yet", style={
                             "color": C["muted"], "fontSize": "10px",
                             "fontFamily": "'JetBrains Mono', monospace",
                         }),
                         style={"marginTop": "2px"}),
            ], style={"padding": "16px", "borderBottom": f"1px solid {C['border']}"}),

            # ── Active Ticker + Stats ───────────────────────────────────────
            html.Div([
                section_label("Active Ticker"),
                html.Div(id="ai-active-ticker-display", children=html.Div(
                    "Enter a ticker →",
                    style={"color": C["muted"], "fontSize": "12px",
                           "fontFamily": "'Outfit', sans-serif"}
                )),
            ], style={"padding": "16px", "borderBottom": f"1px solid {C['border']}"}),

            # ── Signal History ─────────────────────────────────────────────
            html.Div([
                section_label("Signal Log"),
                html.Div(id="ai-signal-log", children=[
                    html.Div("No signals generated yet.",
                             style={"color": C["muted"], "fontSize": "11px",
                                    "fontFamily": "'Outfit', sans-serif"})
                ], style={"display": "flex", "flexDirection": "column", "gap": "6px"}),
            ], style={"padding": "16px", "borderBottom": f"1px solid {C['border']}",
                      "flex": "1", "overflowY": "auto"}),

            # ── n8n Automations ────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("⚡", style={"fontSize": "13px"}),
                        html.Span("n8n Automations", style={
                            "fontWeight": "700", "fontSize": "12px", "color": C["text"],
                            "fontFamily": "'Outfit', sans-serif",
                        }),
                        badge("Coming Soon", C["accent4"]),
                    ], style={"display": "flex", "alignItems": "center", "gap": "7px"}),
                    html.Div("Connect →", style={
                        "fontSize": "10px", "color": C["accent"],
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontWeight": "700", "cursor": "pointer",
                        "opacity": "0.5",
                    })
                ], style={
                    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "marginBottom": "10px",
                }),
                html.Div([
                    html.Div("Automate your trading workflow with n8n triggers. Connect signals, "
                             "alerts, and summaries to Slack, email, Notion, Google Sheets, and more.",
                             style={
                                 "fontSize": "11px", "color": C["dim"],
                                 "fontFamily": "'Outfit', sans-serif",
                                 "lineHeight": "1.6", "marginBottom": "12px",
                             }),
                    *[n8n_automation_row(item) for item in N8N_AUTOMATIONS],
                    html.Div(
                        [
                            html.Div(
                                "Integration setup",
                                style={
                                    "fontSize": "10px",
                                    "fontWeight": "800",
                                    "letterSpacing": "0.12em",
                                    "textTransform": "uppercase",
                                    "color": C["muted"],
                                    "marginBottom": "8px",
                                },
                            ),
                            dcc.Input(
                                id="n8n-webhook-url",
                                type="text",
                                placeholder="Webhook URL",
                                className="ai-text-input",
                                style={"width": "100%", "marginBottom": "8px"},
                            ),
                            dcc.Dropdown(
                                id="n8n-trigger-select",
                                options=[
                                    {"label": "BUY signal", "value": "buy"},
                                    {"label": "SELL signal", "value": "sell"},
                                    {"label": "Market summary", "value": "summary"},
                                    {"label": "Sentiment change", "value": "sentiment"},
                                ],
                                value="buy",
                                clearable=False,
                                searchable=False,
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="n8n-secret",
                                        type="password",
                                        placeholder="Secret token",
                                        className="ai-text-input",
                                        style={"flex": "1"},
                                    ),
                                    html.Button(
                                        "Save integration",
                                        className="ai-summary-btn",
                                        n_clicks=0,
                                        style={"minWidth": "140px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                            ),
                        ],
                        style={
                            "marginTop": "14px",
                            "padding": "12px",
                            "borderRadius": "12px",
                            "border": "1px solid rgba(245, 158, 11, 0.16)",
                            "background": "rgba(245, 158, 11, 0.04)",
                        },
                    ),
                    html.Div([
                        html.Span("🔗 ", style={"fontSize": "11px"}),
                        html.Span("Connect your n8n instance to enable",
                                  style={"fontSize": "10px", "color": C["muted"],
                                         "fontFamily": "'Outfit', sans-serif"}),
                    ], style={
                        "marginTop": "12px", "padding": "8px 12px",
                        "background": f"{C['accent4']}0a",
                        "border": f"1px dashed {C['accent4']}33",
                        "borderRadius": "8px", "display": "flex", "alignItems": "center", "gap": "5px",
                    })
                ]),
            ], style={
                "padding": "14px 16px",
                "background": f"{C['accent4']}05",
                "borderTop": f"1px solid {C['accent4']}22",
            }),

        ], style={
            "width": "300px", "flexShrink": "0",
            "display": "flex", "flexDirection": "column",
            "background": C["surface"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "14px", "overflow": "hidden",
            "animation": "slideInUp 0.35s ease 0.08s both",
        }),

    ], style={
        "display": "flex", "gap": "16px",
        "padding": "16px 22px 22px",
        "height": "calc(100vh - 110px)",
        "boxSizing": "border-box",
        "maxWidth": "1800px",
        "margin": "0 auto",
    })

], style={"background": C["bg"], "minHeight": "100vh"})


# =============================================================================
# OPENAI HELPER
# =============================================================================

def call_openai(
    messages: list[dict],
    news_context: str = "",
    analysis_mode: str = "deep_dive",
    ticker_context: str = "",
) -> str:
    """Call GPT-4o with optional live news context injected."""
    try:
        if openai_client is None:
            return (
                "OpenAI is not configured yet. Set the `OPENAI_API_KEY` "
                "environment variable to enable AI responses."
            )

        mode_guidance = {
            "deep_dive": "Provide a full investment memo with thesis, setup, catalysts, risks, and action plan.",
            "news_digest": "Summarize the news flow, identify what is actually market moving, and separate signal from noise.",
            "risk_scan": "Focus on downside risk, invalidation levels, balance sheet pressure, and scenario failure points.",
            "compare": "Compare the ticker against peers on moat, growth, valuation, margins, and risk/reward.",
            "macro": "Tie the ticker to rates, inflation, liquidity, sector rotation, and macro regime.",
        }.get(analysis_mode, "Provide a structured market memo.")

        system = (
            SYSTEM_PROMPT
            + "\n\nYou are operating inside a professional research workspace."
            + "\nUse this format: Thesis | What matters now | Technicals | Risks | Catalysts | Action."
            + "\nCite news sources by name when you rely on them."
            + "\nIf the context is thin, say what is missing instead of inventing details."
            + f"\nMode focus: {mode_guidance}"
        )
        if ticker_context:
            system += f"\n\n--- TICKER CONTEXT ---\n{ticker_context}\n--- END TICKER CONTEXT ---"
        if news_context:
            system += f"\n\n--- LIVE NEWS CONTEXT ---\n{news_context}\n--- END NEWS CONTEXT ---"

        formatted = [{"role": "system", "content": system}]
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            formatted.append({"role": role, "content": m["content"]})

        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted,
            max_completion_tokens=1200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GPT-4o error: {str(e)}"


def extract_sentiment(text: str) -> float:
    t = text.lower()
    bull = sum(t.count(w) for w in [
        "bullish", "buy", "strong", "growth", "upside", "positive",
        "outperform", "opportunity", "rally", "breakout", "upgrade"
    ])
    bear = sum(t.count(w) for w in [
        "bearish", "sell", "weak", "decline", "downside", "negative",
        "underperform", "risk", "caution", "fall", "downgrade", "miss"
    ])
    total = bull + bear
    return bull / total if total > 0 else 0.5


def extract_signal(text: str):
    """Extract BUY/SELL/HOLD from response text."""
    t = text.upper()
    if "SIGNAL: BUY" in t or "▸ BUY" in t:
        return "BUY", C["green"]
    elif "SIGNAL: SELL" in t or "▸ SELL" in t:
        return "SELL", C["red"]
    elif "SIGNAL: HOLD" in t or "▸ HOLD" in t:
        return "HOLD", C["yellow"]
    return None, C["muted"]


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("ai-mode", "data"),
    Input("ai-mode-select", "value"),
)
def sync_ai_mode(value):
    return value or "deep_dive"


@callback(
    Output("ai-chat-messages", "children"),
    Output("ai-user-input", "value"),
    Output("ai-chat-history", "data"),
    Output("ai-is-thinking", "data"),
    Output("ai-recent-text", "data"),
    Input("ai-send-btn", "n_clicks"),
    Input("ai-user-input", "n_submit"),
    State("ai-user-input", "value"),
    State("ai-ticker-input", "value"),
    State("ai-mode", "data"),
    State("ai-chat-history", "data"),
    State("ai-chat-messages", "children"),
    prevent_initial_call=True
)
def show_user_message(send_clicks, n_submit, user_text, ticker, mode, history, current_messages):
    if not user_text or not user_text.strip():
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    ticker_str = (ticker or "").upper().strip()
    mode_str = (mode or "deep_dive").replace("_", " ").title()
    context_prefix = f"[Analyzing ticker: ${ticker_str}] " if ticker_str else ""
    full_prompt = f"[Mode: {mode_str}] {context_prefix}{user_text.strip()}"

    history = history or []
    history.append({"role": "user", "content": full_prompt})

    ts = datetime.now().strftime("%H:%M")
    new_messages = list(current_messages) + [message_bubble("user", user_text.strip(), ts)]

    return new_messages, "", history, True, user_text


@callback(
    Output("ai-typing-indicator", "style"),
    Input("ai-is-thinking", "data"),
)
def toggle_typing(is_thinking):
    base = {"alignItems": "flex-start", "gap": "10px", "padding": "0 16px 12px"}
    return {**base, "display": "flex"} if is_thinking else {"display": "none"}


@callback(
    Output("ai-chat-messages", "children", allow_duplicate=True),
    Output("ai-chat-history", "data", allow_duplicate=True),
    Output("ai-is-thinking", "data", allow_duplicate=True),
    Output("ai-sentiment-bar", "children"),
    Output("ai-sentiment-label", "children"),
    Output("ai-signal-log", "children"),
    Input("ai-chat-history", "data"),
    State("ai-chat-messages", "children"),
    State("ai-recent-text", "data"),
    State("ai-ticker-input", "value"),
    State("ai-mode", "data"),
    State("ai-signal-log", "children"),
    prevent_initial_call=True
)
def get_ai_response(history, current_messages, user_text, ticker, mode, signal_log):
    if not history or history[-1]["role"] != "user":
        return (dash.no_update,) * 6

    ticker_str = (ticker or "").upper().strip()
    mode_str = mode or "deep_dive"
    news_ctx = build_news_context_from_news(ticker=ticker_str, limit=10)
    ticker_ctx = build_ticker_context(ticker_str)
    ai_response = call_openai(history, news_context=news_ctx, analysis_mode=mode_str, ticker_context=ticker_ctx)
    history.append({"role": "assistant", "content": ai_response})

    # Clean markdown
    html_text = markdown.markdown(ai_response)
    clean_text = BeautifulSoup(html_text, "html.parser").get_text()

    ts = datetime.now().strftime("%H:%M")
    new_messages = list(current_messages) + [message_bubble("assistant", clean_text, ts)]

    # Sentiment
    score = extract_sentiment(ai_response)
    sentiment_ui = render_sentiment_bar(score)
    label_text = "BULLISH" if score > 0.6 else "BEARISH" if score < 0.4 else "NEUTRAL"
    label_color = C["green"] if score > 0.6 else C["red"] if score < 0.4 else C["yellow"]

    # Signal extraction
    signal, sig_color = extract_signal(ai_response)
    new_log = list(signal_log) if signal_log else []
    if signal:
        new_log.insert(0, html.Div([
            html.Div([
                html.Span(signal, style={
                    "fontSize": "10px", "fontWeight": "800",
                    "color": sig_color, "fontFamily": "'JetBrains Mono', monospace",
                    "background": f"{sig_color}18",
                    "border": f"1px solid {sig_color}44",
                    "padding": "2px 8px", "borderRadius": "4px",
                }),
                html.Span(f" ${ticker_str}" if ticker_str else "",
                          style={"fontSize": "11px", "color": C["dim"],
                                 "fontFamily": "'JetBrains Mono', monospace",
                                 "fontWeight": "700"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
            html.Div(ts, style={"fontSize": "9px", "color": C["muted"],
                                "fontFamily": "'JetBrains Mono', monospace"}),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "7px 10px",
            "background": f"{sig_color}08",
            "border": f"1px solid {sig_color}22",
            "borderRadius": "7px",
        }))
        new_log = new_log[:8]  # Keep last 8

    return (
        new_messages,
        history,
        False,
        sentiment_ui,
        html.Span(label_text, style={"color": label_color, "fontWeight": "800",
                                     "fontSize": "11px",
                                     "fontFamily": "'JetBrains Mono', monospace",
                                     "letterSpacing": "1.5px"}),
        new_log,
    )


@callback(
    Output("ai-user-input", "value", allow_duplicate=True),
    Input({"type": "ai-quick-chip", "index": dash.ALL}, "n_clicks"),
    State("ai-ticker-input", "value"),
    prevent_initial_call=True
)
def fill_quick_prompt(n_clicks_list, ticker):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return dash.no_update
    triggered = ctx.triggered[0]["prop_id"]
    idx = json.loads(triggered.split(".")[0])["index"]
    _, prompt_template = QUICK_PROMPTS[idx]
    ticker_str = (ticker or "TICKER").upper().strip()
    return prompt_template.replace("{ticker}", ticker_str)


@callback(
    Output("ai-market-summary", "children"),
    Input("ai-summary-btn", "n_clicks"),
    prevent_initial_call=True
)
def generate_market_summary(n_clicks):
    if not n_clicks:
        return dash.no_update

    news_ctx = build_news_context_from_news(limit=10)
    response = call_openai(
        [{"role": "user", "content": MARKET_SUMMARY_PROMPT}],
        news_context=news_ctx,
        analysis_mode="macro",
    )

    html_text = markdown.markdown(response)
    clean = BeautifulSoup(html_text, "html.parser").get_text()

    return html.Div([
        html.Div([
            html.Div([
                html.Span("🌐", style={"fontSize": "13px"}),
                html.Span("Market Brief", style={
                    "fontWeight": "800", "fontSize": "12px", "color": C["accent"],
                    "letterSpacing": "0.08em", "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Span(datetime.now().strftime("%H:%M"), style={
                    "fontSize": "9px", "color": C["muted"],
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginLeft": "auto",
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "7px",
                      "marginBottom": "10px"}),
            html.Div(clean, style={
                "whiteSpace": "pre-wrap", "fontSize": "12px",
                "lineHeight": "1.8", "color": C["dim"],
                "fontFamily": "'Inter', sans-serif",
            }),
        ], style={"padding": "14px 16px"})
    ], style={"animation": "slideInUp 0.35s ease both"})


@callback(
    Output("ai-active-ticker-display", "children"),
    Input("ai-ticker-input", "value"),
)
def update_ticker_display(ticker):
    if not ticker or not ticker.strip():
        return html.Div("Enter a ticker →",
                        style={"color": C["muted"], "fontSize": "11px",
                               "fontFamily": "'Outfit', sans-serif"})
    t = ticker.upper().strip()
    return html.Div([
        html.Div([
            html.Span(f"${t}", style={
                "fontSize": "20px", "fontWeight": "900", "color": C["accent"],
                "letterSpacing": "3px", "fontFamily": "'JetBrains Mono', monospace",
            }),
            html.Div(style={
                "width": "6px", "height": "6px", "borderRadius": "50%",
                "background": C["green"], "boxShadow": f"0 0 8px {C['green']}",
                "animation": "glowPulse 2s infinite", "alignSelf": "center",
            }),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
        html.Div("Use chips or type a question ↗", style={
            "fontSize": "10px", "color": C["muted"], "marginTop": "3px",
            "fontFamily": "'Outfit', sans-serif",
        })
    ], style={"animation": "slideInUp 0.25s ease both"})
