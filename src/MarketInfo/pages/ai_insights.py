"""Dash AI insights page — institutional terminal redesign."""

from __future__ import annotations

import os
from datetime import datetime

import dash
from dash import Input, Output, State, callback, dcc, html
from openai import OpenAI

from MarketInfo.data import all_stock_data, get_stock_by_ticker, single_stock_data
from MarketInfo.indicators import compute_indicators
from MarketInfo.news import build_news_context


dash.register_page(__name__, path="/ai", name="AI Insights")

MODEL_NAME = "gpt-5-nano"
ANALYSIS_MODES = [
    {"label": "Deep Dive", "value": "deep_dive"},
    {"label": "News Digest", "value": "news_digest"},
    {"label": "Risk Scan", "value": "risk_scan"},
    {"label": "Compare", "value": "compare"},
    {"label": "Macro Lens", "value": "macro"},
]

SYSTEM_PROMPT = (
    "You are a professional market research copilot. "
    "Respond clearly and with structure. Use the sections: Thesis, What matters now, "
    "Technicals, Risks, Catalysts, Action. Cite source names when using the news context. "
    "If context is missing, say what is missing."
)

api_key = os.getenv("OPENAI_API_KEY", "").strip()
openai_client = OpenAI(api_key=api_key) if api_key else None

# ── colour tokens ──────────────────────────────────────────────────────────────
C = {
    "bg": "#080c14",
    "surface": "#0d1321",
    "surface2": "#111827",
    "border": "rgba(255,255,255,0.07)",
    "border2": "rgba(255,255,255,0.12)",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "dim": "#374151",
    "accent": "#0ea5e9",
    "accent2": "#38bdf8",
    "green": "#10b981",
    "red": "#f43f5e",
    "amber": "#f59e0b",
    "purple": "#8b5cf6",
    "tag_bg": "rgba(14,165,233,0.10)",
}

FONT_IMPORTS = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500;600&"
    "family=IBM+Plex+Sans:wght@300;400;500;600;700&"
    "display=swap"
)

# ── shared inline styles ───────────────────────────────────────────────────────
PANEL = {
    "background": C["surface"],
    "border": f"1px solid {C['border']}",
    "borderRadius": "4px",
    "padding": "20px",
}

INPUT_BASE = {
    "background": C["bg"],
    "color": C["text"],
    "border": f"1px solid {C['border2']}",
    "borderRadius": "3px",
    "fontFamily": "'IBM Plex Sans', sans-serif",
    "fontSize": "13px",
    "padding": "8px 10px",
    "outline": "none",
    "width": "100%",
    "boxSizing": "border-box",
}

BTN_PRIMARY = {
    "background": C["accent"],
    "color": "#000",
    "border": "none",
    "borderRadius": "3px",
    "fontFamily": "'IBM Plex Sans', sans-serif",
    "fontWeight": "600",
    "fontSize": "12px",
    "letterSpacing": "0.06em",
    "padding": "9px 20px",
    "cursor": "pointer",
    "textTransform": "uppercase",
}

BTN_GHOST = {
    **BTN_PRIMARY,
    "background": "transparent",
    "color": C["accent2"],
    "border": f"1px solid {C['accent']}",
}

LABEL_STYLE = {
    "fontSize": "10px",
    "fontWeight": "600",
    "letterSpacing": "0.12em",
    "color": C["muted"],
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "display": "block",
    "fontFamily": "'IBM Plex Mono', monospace",
}

SECTION_TITLE = {
    "fontSize": "10px",
    "fontWeight": "600",
    "letterSpacing": "0.14em",
    "color": C["accent"],
    "textTransform": "uppercase",
    "fontFamily": "'IBM Plex Mono', monospace",
    "marginBottom": "14px",
    "paddingBottom": "8px",
    "borderBottom": f"1px solid {C['border']}",
    "display": "flex",
    "alignItems": "center",
    "gap": "8px",
}

PRE_PANEL = {
    "whiteSpace": "pre-wrap",
    "fontSize": "11px",
    "lineHeight": "1.7",
    "fontFamily": "'IBM Plex Mono', monospace",
    "color": C["muted"],
    "background": C["bg"],
    "padding": "12px 14px",
    "borderRadius": "3px",
    "border": f"1px solid {C['border']}",
    "maxHeight": "180px",
    "overflowY": "auto",
}


# ── helper: pill badge ─────────────────────────────────────────────────────────
def pill(label: str, color: str = C["accent"]) -> html.Span:
    return html.Span(
        label,
        style={
            "fontSize": "10px",
            "fontWeight": "600",
            "letterSpacing": "0.08em",
            "color": color,
            "background": f"{color}18",
            "border": f"1px solid {color}40",
            "borderRadius": "2px",
            "padding": "2px 7px",
            "fontFamily": "'IBM Plex Mono', monospace",
            "textTransform": "uppercase",
        },
    )


# ── context helpers (unchanged logic) ─────────────────────────────────────────
def build_ticker_context(ticker: str) -> str:
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return ""
    lines: list[str] = []
    try:
        snapshot = all_stock_data()
        stock = get_stock_by_ticker(snapshot, ticker) if not snapshot.empty else None
        if stock:
            lines.append(
                f"Snapshot: ticker={stock['Ticker']}, sector={stock['Sector']}, "
                f"last={stock['Last']:.2f}, change={stock['Change']:+.2f}%, "
                f"volume={int(stock.get('Volume', 0))}"
            )
    except Exception:
        pass
    try:
        history = single_stock_data(ticker, period_days=220)
        if not history.empty and len(history) > 40:
            enriched = compute_indicators(
                history,
                ["sma20", "sma50", "ema21", "rsi14", "macd", "bbands", "volume_ma"],
            )
            latest = enriched.iloc[-1]
            close = float(latest["Close"])
            trend = "above" if close > float(latest.get("SMA20", close)) else "below"
            lines.append(
                f"Technicals: close={close:.2f}, sma20={latest.get('SMA20', float('nan')):.2f}, "
                f"sma50={latest.get('SMA50', float('nan')):.2f}, "
                f"rsi14={latest.get('RSI14', float('nan')):.1f}, "
                f"macd={latest.get('MACD', float('nan')):.2f}, trend={trend} sma20"
            )
    except Exception:
        pass
    return "\n".join(lines)


def call_openai(user_text: str, ticker: str, mode: str) -> str:
    if openai_client is None:
        return "OpenAI is not configured. Set OPENAI_API_KEY to enable AI responses."
    mode_guidance = {
        "deep_dive": "Provide a full investment memo with balanced bullish and bearish arguments.",
        "news_digest": "Summarize the recent news and separate signal from noise.",
        "risk_scan": "Focus on downside risk, invalidation levels, and scenario failure points.",
        "compare": "Compare the ticker against likely peers on setup quality, moat, and risk/reward.",
        "macro": "Connect the ticker to rates, liquidity, sector rotation, and macro regime.",
    }.get(mode, "Provide a structured market memo.")
    ticker_context = build_ticker_context(ticker)
    news_context = build_news_context(ticker=ticker, limit=8)
    messages = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\nMode focus: {mode_guidance}\n\n"
                f"Ticker context:\n{ticker_context or 'No ticker context available.'}\n\n"
                f"News context:\n{news_context or 'No recent news context available.'}"
            ),
        },
        {"role": "user", "content": user_text},
    ]
    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_completion_tokens=1200
        )
        return resp.choices[0].message.content or "No response returned."
    except Exception as exc:
        return f"AI request failed: {exc}"


# ── chat bubble ────────────────────────────────────────────────────────────────
def chat_bubble(role: str, content: str, timestamp: str) -> html.Div:
    is_user = role == "user"
    tag_color = C["accent"] if is_user else C["purple"]
    tag_label = "YOU" if is_user else "COPILOT"
    border_col = C["accent"] if is_user else C["purple"]
    bg = f"{border_col}08"
    css_class = "chat-bubble" if is_user else "chat-bubble-ai"
    return html.Div(
        [
            html.Div(
                [
                    pill(tag_label, tag_color),
                    html.Span(
                        timestamp,
                        style={
                            "fontSize": "10px",
                            "color": C["muted"],
                            "fontFamily": "'IBM Plex Mono', monospace",
                            "marginLeft": "auto",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "10px"},
            ),
            html.Div(
                content,
                style={
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.7",
                    "fontSize": "13px",
                    "color": C["text"],
                    "fontFamily": "'IBM Plex Sans', sans-serif",
                },
            ),
        ],
        className=css_class,
        style={
            "padding": "14px 16px",
            "borderRadius": "3px",
            "border": f"1px solid {border_col}28",
            "borderLeft": f"3px solid {border_col}",
            "background": bg,
            "marginBottom": "10px",
        },
    )


def thinking_bubble() -> html.Div:
    return html.Div(
        [
            html.Span("COPILOT", className="thinking-label"),
            html.Div(
                [html.Span(), html.Span(), html.Span()],
                className="thinking-dots",
            ),
            html.Span(
                "Analyzing...",
                style={
                    "fontSize": "11px",
                    "color": C["muted"],
                    "fontFamily": "'IBM Plex Mono', monospace",
                    "marginLeft": "4px",
                },
            ),
        ],
        className="thinking-bubble",
        id="thinking-indicator",
    )


# ── divider ────────────────────────────────────────────────────────────────────
def divider(label: str = "", color: str = C["muted"]) -> html.Div:
    return html.Div(
        [html.Span(label, style={"color": color, "fontSize": "10px", "letterSpacing": "0.10em",
                                 "fontFamily": "'IBM Plex Mono', monospace", "padding": "0 8px"})],
        style={"display": "flex", "alignItems": "center", "gap": "6px",
               "margin": "16px 0", "borderTop": f"1px solid {C['border']}",
               "paddingTop": "16px"},
    )


# ── layout ─────────────────────────────────────────────────────────────────────


layout = html.Div(
    [
        dcc.Store(id="ai-chat-history", data=[]),
        dcc.Store(id="ai-thinking", data=False),
        html.Link(href=FONT_IMPORTS, rel="stylesheet"),

        # ── topbar ─────────────────────────────────────────────────────────────
        html.Div(
            [
                # left: branding
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("AI", style={"color": C["accent"], "fontWeight": "700"}),
                                html.Span(" INSIGHTS", style={"color": C["text"]}),
                            ],
                            style={
                                "fontSize": "15px",
                                "fontWeight": "700",
                                "letterSpacing": "0.14em",
                                "fontFamily": "'IBM Plex Mono', monospace",
                            },
                        ),
                        html.Div(
                            "Market research copilot  ·  Powered by OpenAI GPT-4o",
                            style={"fontSize": "11px", "color": C["muted"], "fontFamily": "'IBM Plex Mono', monospace",
                                   "marginTop": "2px"},
                        ),
                    ]
                ),
                # right: status chips
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(className="live-dot"),
                                html.Span("LIVE", style={"fontSize": "10px", "fontWeight": "700",
                                                         "fontFamily": "'IBM Plex Mono', monospace",
                                                         "color": C["green"], "letterSpacing": "0.1em"}),
                            ],
                            style={"display": "flex", "alignItems": "center", "background": f"{C['green']}12",
                                   "border": f"1px solid {C['green']}30", "borderRadius": "3px",
                                   "padding": "5px 10px"},
                        ),
                        pill("GPT-4o", C["purple"]),
                        pill("RSS + TA", C["amber"]),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "padding": "14px 20px",
                "background": C["surface"],
                "border": f"1px solid {C['border']}",
                "borderRadius": "4px",
                "marginBottom": "12px",
            },
        ),

        # ── main grid ──────────────────────────────────────────────────────────
        html.Div(
            [
                # ── LEFT: chat panel ──────────────────────────────────────────
                html.Div(
                    [
                        # section header
                        html.Div(
                            [
                                html.Span("⬡", style={"fontSize": "12px"}),
                                "RESEARCH TERMINAL",
                            ],
                            style=SECTION_TITLE,
                        ),

                        # ticker + mode row
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("TICKER", style=LABEL_STYLE),
                                        dcc.Input(
                                            id="ai-ticker-input",
                                            value="AAPL",
                                            type="text",
                                            style={
                                                **INPUT_BASE,
                                                "fontFamily": "'IBM Plex Mono', monospace",
                                                "fontWeight": "600",
                                                "fontSize": "15px",
                                                "color": C["accent2"],
                                                "letterSpacing": "0.08em",
                                                "textTransform": "uppercase",
                                            },
                                        ),
                                    ],
                                    style={"flex": "0 0 110px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("ANALYSIS MODE", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="ai-mode-select",
                                            options=ANALYSIS_MODES,
                                            value="deep_dive",
                                            clearable=False,
                                            style={
                                                **INPUT_BASE,
                                                "fontFamily": "'IBM Plex Mono', monospace",
                                                "fontWeight": "600",
                                                "fontSize": "15px",
                                                "color": C["accent2"],
                                                "letterSpacing": "0.08em",
                                                "textTransform": "uppercase",
                                            },
                                        ),
                                    ],
                                    style={"flex": "1"},
                                ),
                            ],
                            style={"display": "flex", "gap": "12px", "marginBottom": "16px", "alignItems": "flex-end"},
                        ),

                        # prompt input
                        html.Label("PROMPT", style=LABEL_STYLE),
                        dcc.Textarea(
                            id="ai-user-input",
                            value="",
                            placeholder="Enter a question, thesis challenge, or request a structured memo...",
                            style={
                                **INPUT_BASE,
                                "minHeight": "110px",
                                "resize": "vertical",
                                "lineHeight": "1.6",
                                "padding": "10px 12px",
                            },
                        ),

                        # action buttons
                        html.Div(
                            [
                                html.Button(
                                    [html.Span("▶  "), "SEND"],
                                    id="ai-send-btn",
                                    n_clicks=0,
                                    style=BTN_PRIMARY,
                                ),
                                html.Button(
                                    [html.Span("⊞  "), "GENERATE BRIEF"],
                                    id="ai-summary-btn",
                                    n_clicks=0,
                                    style=BTN_GHOST,
                                ),
                            ],
                            style={"display": "flex", "gap": "10px", "marginTop": "14px"},
                        ),

                        html.Div(
                            style={"height": "1px", "background": C["border"], "margin": "20px 0"},
                        ),

                        # chat messages
                        html.Div("CONVERSATION", style={**LABEL_STYLE, "marginBottom": "12px"}),
                        html.Div(
                            id="ai-chat-messages",
                            style={"maxHeight": "480px", "overflowY": "auto", "paddingRight": "4px"},
                        ),
                    ],
                    style={**PANEL, "flex": "1.6"},
                ),

                # ── RIGHT: context + integrations ─────────────────────────────
                html.Div(
                    [
                        # live context
                        html.Div(
                            [html.Span("◈", className="section-icon-spin", style={"fontSize": "12px"}), "LIVE CONTEXT"],
                            style=SECTION_TITLE,
                        ),
                        html.Div(
                            [
                                html.Div("TECHNICAL SNAPSHOT", style=LABEL_STYLE),
                                html.Pre(id="ai-live-context", className="context-panel", style=PRE_PANEL),
                            ],
                            style={"marginBottom": "20px"},
                        ),

                        # news feed
                        html.Div(
                            [
                                html.Div("NEWS FEED CONTEXT", style=LABEL_STYLE),
                                html.Pre(id="ai-news-context", className="context-panel", style=PRE_PANEL),
                            ],
                            style={"marginBottom": "20px"},
                        ),

                        # divider
                        html.Div(
                            style={"height": "1px", "background": C["border"], "margin": "4px 0 20px"},
                        ),

                        # n8n integrations
                        html.Div(
                            [html.Span("⇄", style={"fontSize": "13px"}), "N8N INTEGRATIONS"],
                            style=SECTION_TITLE,
                        ),

                        html.Div(
                            [
                                html.Label("WEBHOOK URL", style=LABEL_STYLE),
                                dcc.Input(
                                    id="n8n-webhook",
                                    type="text",
                                    placeholder="https://your-n8n.cloud/webhook/...",
                                    style={**INPUT_BASE, "marginBottom": "12px"},
                                ),

                                html.Label("TRIGGER EVENT", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="n8n-trigger",
                                    options=[
                                        {"label": "Signal Generated", "value": "signal"},
                                        {"label": "Market Brief Ready", "value": "brief"},
                                        {"label": "Risk Alert", "value": "risk"},
                                    ],
                                    value="signal",
                                    clearable=False,
                                    style={
                                        **INPUT_BASE,
                                        "fontFamily": "'IBM Plex Mono', monospace",
                                        "fontWeight": "600",
                                        "fontSize": "15px",
                                        "color": ["accent2"],
                                        "letterSpacing": "0.08em",
                                        "textTransform": "uppercase",
                                    },

                                ),

                                html.Label("SHARED SECRET", style=LABEL_STYLE),
                                dcc.Input(
                                    id="n8n-secret",
                                    type="password",
                                    placeholder="••••••••••••",
                                    style={**INPUT_BASE, "marginBottom": "12px"},
                                ),

                                html.Div(
                                    [
                                        html.Span("○",
                                                  style={"color": C["amber"], "fontSize": "9px", "marginRight": "5px"}),
                                        "Persistence not wired — placeholder UI only.",
                                    ],
                                    style={
                                        "fontSize": "11px",
                                        "color": C["muted"],
                                        "fontFamily": "'IBM Plex Mono', monospace",
                                        "background": f"{C['amber']}08",
                                        "border": f"1px solid {C['amber']}20",
                                        "borderRadius": "3px",
                                        "padding": "8px 10px",
                                        "display": "flex",
                                        "alignItems": "center",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={**PANEL, "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
        ),
    ],
    style={
        "minHeight": "100vh",
        "background": C["bg"],
        "color": C["text"],
        "fontFamily": "'IBM Plex Sans', sans-serif",
        "padding": "16px 20px",
    },
)


# ── callbacks (unchanged logic) ────────────────────────────────────────────────
@callback(
    Output("ai-live-context", "children"),
    Output("ai-news-context", "children"),
    Input("ai-ticker-input", "value"),
)
def update_context_panels(ticker):
    ticker = (ticker or "").upper().strip()
    return (
        build_ticker_context(ticker) or "Awaiting valid ticker symbol...",
        build_news_context(ticker=ticker, limit=6) or "News context unavailable.",
    )


@callback(
    Output("ai-thinking", "data"),
    Output("ai-chat-history", "data", allow_duplicate=True),
    Input("ai-send-btn", "n_clicks"),
    Input("ai-summary-btn", "n_clicks"),
    prevent_initial_call=True,
)
def set_thinking(_send, _brief):
    return True, dash.no_update


@callback(
    Output("ai-chat-history", "data"),
    Output("ai-user-input", "value"),
    Output("ai-thinking", "data", allow_duplicate=True),
    Input("ai-send-btn", "n_clicks"),
    State("ai-user-input", "value"),
    State("ai-ticker-input", "value"),
    State("ai-mode-select", "value"),
    State("ai-chat-history", "data"),
    prevent_initial_call=True,
)
def handle_send(_clicks, user_text, ticker, mode, history):
    if not user_text or not user_text.strip():
        return history or [], user_text, False
    history = history or []
    history.append({"role": "user", "content": user_text.strip(),
                    "timestamp": datetime.now().strftime("%H:%M:%S")})
    answer = call_openai(user_text.strip(), ticker or "", mode or "deep_dive")
    history.append({"role": "assistant", "content": answer,
                    "timestamp": datetime.now().strftime("%H:%M:%S")})
    return history, "", False


@callback(
    Output("ai-chat-history", "data", allow_duplicate=True),
    Output("ai-thinking", "data", allow_duplicate=True),
    Input("ai-summary-btn", "n_clicks"),
    State("ai-ticker-input", "value"),
    State("ai-chat-history", "data"),
    prevent_initial_call=True,
)
def handle_summary(_clicks, ticker, history):
    ticker = (ticker or "").upper().strip()
    history = history or []
    prompt = (
        f"Summarize the live market brief for {ticker or 'the selected market'}. "
        "Keep it concise and actionable."
    )
    answer = call_openai(prompt, ticker, "news_digest")
    history.append({"role": "assistant", "content": answer,
                    "timestamp": datetime.now().strftime("%H:%M:%S")})
    return history, False


@callback(
    Output("ai-chat-messages", "children"),
    Input("ai-chat-history", "data"),
    Input("ai-thinking", "data"),
)
def render_chat(history, is_thinking):
    history = history or []
    children = []

    if not history and not is_thinking:
        return html.Div(
            "No messages yet — enter a prompt above or generate a brief.",
            className="chat-empty",
        )

    children = [chat_bubble(item["role"], item["content"], item["timestamp"]) for item in history]

    if is_thinking:
        children.append(thinking_bubble())

    return children
