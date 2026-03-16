"""
pages/ai_insights.py — AI Insights Page powered by Google Gemini
Chat, stock analysis, and market summaries.

Setup:
    pip install google-generativeai
    export GEMINI_API_KEY="your-key-here"

Note: Using gemini-2.5-pro-preview-03-25 — the latest available Gemini Pro.
      Update MODEL_NAME below when newer versions release.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
from google import genai
import yaml
from datetime import datetime
import json
import markdown
from bs4 import BeautifulSoup
from src.MarketInfo.response_agent import get_response
import langchain
dash.register_page(__name__, path="/ai", name="AI Insights")

# =============================================================================
# GEMINI SETUP
# =============================================================================
with open("../ConfigurationFiles/config.yaml", "r") as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config["api_keys"]["gemini"])
MODEL_NAME = "gemini-3-flash-preview"

# =============================================================================
# COLORS (matches your existing dark theme)
# =============================================================================

C = {
    "bg": "#0a0a0f",
    "bg2": "#0f0f1a",
    "surface": "#13131a",
    "surface2": "#1a1a24",
    "border": "rgba(255,255,255,0.07)",
    "accent": "#00f5ff",  # your existing electric cyan
    "accent2": "#00ff88",  # your existing neon green
    "accent3": "#ff00e5",  # your existing hot magenta
    "green": "#00ff88",
    "red": "#ff0055",
    "text": "#ffffff",
    "muted": "#606078",
    "dim": "#a0a0b8",
}

# =============================================================================
# QUICK PROMPT CHIPS
# =============================================================================

QUICK_PROMPTS = [
    ("📊 Analyze", "Give me a detailed technical and fundamental analysis of {ticker}."),
    ("🔮 Outlook", "What is the 3–6 month price outlook for {ticker}? Include key risks and catalysts."),
    ("📰 News Impact", "Summarize recent news for {ticker} and how it might affect the stock price."),
    ("💡 Buy/Sell/Hold", "Would you recommend buying, holding, or selling {ticker} right now? Explain your reasoning."),
    ("📈 Competitors", "Who are {ticker}'s main competitors and how does it compare to them?"),
    ("🌍 Macro Risks", "What macroeconomic risks could negatively impact {ticker} in the near term?"),
]

MARKET_SUMMARY_PROMPT = (
    "Give me a concise but insightful summary of current stock market conditions. "
    "Cover: major index trends, sector rotation, volatility, and 2-3 key themes traders should watch. "
    "Use clear sections."
)


# =============================================================================
# COMPONENT HELPERS
# =============================================================================

def message_bubble(role: str, content: str, timestamp: str = ""):
    is_user = role == "user"
    return html.Div([
        html.Div(
            "You" if is_user else "AI",
            style={
                "width": "30px", "height": "30px",
                "borderRadius": "50%",
                "background": f"linear-gradient(135deg, {C['accent']}, {C['accent2']})" if is_user
                else f"linear-gradient(135deg, {C['accent3']}, {C['accent']})",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "fontSize": "11px", "fontWeight": "800", "color": "#000",
                "flexShrink": "0",
                "boxShadow": f"0 0 12px {'rgba(0,245,255,0.4)' if is_user else 'rgba(255,0,229,0.4)'}",
                "fontFamily": "'Outfit', sans-serif",
            }
        ),
        html.Div([
            html.Div(content, style={
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.75",
                "fontSize": "18px",
                "color": C["text"],
                "fontFamily": "'Outfit', sans-serif",
            }),
            html.Div(timestamp, style={
                "fontSize": "8px", "color": C["muted"],
                "marginTop": "6px", "textAlign": "right",
                "fontFamily": "'JetBrains Mono', monospace",
            }) if timestamp else None
        ], style={
            "background": C["surface2"] if is_user else C["surface"],
            "border": f"1px solid {'rgba(0,245,255,0.2)' if is_user else C['border']}",
            "borderRadius": "4px 14px 14px 14px" if not is_user else "14px 4px 14px 14px",
            "padding": "12px 16px",
            "maxWidth": "82%",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.35)",
        })
    ], style={
        "display": "flex",
        "flexDirection": "row" if not is_user else "row-reverse",
        "alignItems": "flex-start",
        "gap": "10px",
        "marginBottom": "14px",
        "animation": "slideInUp 0.3s ease both",
    })


def render_sentiment_bar(score: float):
    color = C["green"] if score > 0.55 else C["red"] if score < 0.45 else C["accent"]
    label = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
    return [
        html.Div([
            html.Div(style={
                "height": "6px", "borderRadius": "3px",
                "width": f"{score * 100:.0f}%",
                "background": f"linear-gradient(90deg, {color}, {color}88)",
                "transition": "width 0.8s cubic-bezier(0.4,0,0.2,1)",
                "boxShadow": f"0 0 8px {color}66",
            })
        ], style={
            "height": "6px", "borderRadius": "3px",
            "background": "rgba(255,255,255,0.06)", "overflow": "hidden",
        }),
        html.Div(label, style={
            "fontSize": "11px", "marginTop": "6px", "textAlign": "center",
            "color": color, "fontWeight": "700",
            "fontFamily": "'JetBrains Mono', monospace",
        })
    ]


# =============================================================================
# PAGE LAYOUT
# =============================================================================

layout = html.Div([
    # CSS injected via a hidden iframe trick — html.Style doesn't exist in Dash
    html.Div(id="ai-page-styles", **{"data-dash-is-loading": "true"}, style={"display": "none"}),
    dash.clientside_callback(
        """
        function() {
            if (document.getElementById('ai-injected-styles')) return '';
            var style = document.createElement('style');
            style.id = 'ai-injected-styles';
            style.textContent = `
                @keyframes slideInUp {
                    from { opacity: 0; transform: translateY(14px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
                @keyframes glowPulse {
                    0%, 100% { box-shadow: 0 0 8px rgba(0,245,255,0.2); }
                    50%       { box-shadow: 0 0 22px rgba(0,245,255,0.6); }
                }
                @keyframes typingDot {
                    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
                    40%            { transform: scale(1);   opacity: 1; }
                }
                .ai-quick-chip {
                    background: rgba(0,245,255,0.06);
                    border: 1px solid rgba(0,245,255,0.2);
                    color: #00f5ff;
                    padding: 5px 13px;
                    border-radius: 20px;
                    font-size: 13px;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    white-space: nowrap;
                    font-family: 'Outfit', sans-serif;
                    letter-spacing: 0.02em;
                }
                .ai-quick-chip:hover {
                    background: rgba(0,245,255,0.18);
                    border-color: rgba(0,245,255,0.55);
                    color: #fff;
                    transform: translateY(-1px);
                }
                .ai-send-btn {
                    background: linear-gradient(135deg, #00f5ff, #00ff88);
                    border: none;
                    border-radius: 9px;
                    color: #0a0a0f;
                    font-size: 20px;
                    font-weight: 900;
                    cursor: pointer;
                    height: 42px;
                    min-width: 50px;
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .ai-send-btn:hover { transform: scale(1.06); box-shadow: 0 0 18px rgba(0,245,255,0.5); }
                .ai-send-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }
                .ai-text-input {
                    background: rgba(255,255,255,0.03) !important;
                    border: 1px solid rgba(255,255,255,0.09) !important;
                    border-radius: 9px !important;
                    color: #ffffff !important;
                    font-size: 18px !important;
                    padding: 10px 13px !important;
                    outline: none !important;
                    transition: border-color 0.2s ease !important;
                    font-family: 'Outfit', sans-serif !important;
                }
                .ai-text-input:focus {
                    border-color: rgba(0,245,255,0.45) !important;
                    box-shadow: 0 0 0 3px rgba(0,245,255,0.1) !important;
                }
                .ai-ticker-input {
                    background: rgba(255,255,255,0.04) !important;
                    border: 1px solid rgba(255,255,255,0.09) !important;
                    border-radius: 7px !important;
                    color: #00f5ff !important;
                    font-size: 14px !important;
                    font-weight: 800 !important;
                    letter-spacing: 3px !important;
                    text-transform: uppercase !important;
                    padding: 7px 12px !important;
                    width: 110px !important;
                    transition: border-color 0.2s ease !important;
                    font-family: 'JetBrains Mono', monospace !important;
                }
                .ai-ticker-input:focus {
                    border-color: rgba(0,245,255,0.45) !important;
                    box-shadow: 0 0 0 3px rgba(0,245,255,0.1) !important;
                    outline: none !important;
                }
                .ai-summary-btn {
                    background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(0,255,136,0.12));
                    border: 1px solid rgba(0,245,255,0.3);
                    color: #00f5ff;
                    padding: 4px 4px;
                    border-radius: 9px;
                    font-size: 16px;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    font-family: 'Outfit', sans-serif;
                    letter-spacing: 0.05em;
                }
                .ai-summary-btn:hover {
                    background: linear-gradient(135deg, rgba(0,245,255,0.22), rgba(0,255,136,0.22));
                    transform: translateY(-1px);
                    box-shadow: 0 4px 18px rgba(0,245,255,0.2);
                }
                .typing-dot {
                    width: 6px; height: 6px;
                    background: #00f5ff;
                    border-radius: 50%;
                    display: inline-block;
                    margin: 0 2px;
                }
                .typing-dot:nth-child(1) { animation: typingDot 1.2s ease infinite; }
                .typing-dot:nth-child(2) { animation: typingDot 1.2s ease infinite 0.2s; }
                .typing-dot:nth-child(3) { animation: typingDot 1.2s ease infinite 0.4s; }
            `;
            document.head.appendChild(style);
            return '';
        }
        """,
        Output("ai-page-styles", "children"),
        Input("ai-page-styles", "id"),
    ),

    # Stores
    dcc.Store(id="ai-chat-history", data=[]),
    # Add this store to track if AI is thinking
    dcc.Store(id="ai-is-thinking", data=[]),
    dcc.Store(id="recent-text", data=""),
    # Page wrapper
    html.Div([

        # ── LEFT: Chat panel ────────────────────────────────────────────────
        html.Div([

            # Header
            html.Div([
                html.Div([
                    html.Div(style={
                        "width": "7px", "height": "7px", "borderRadius": "50%",
                        "background": C["green"],
                        "boxShadow": f"0 0 8px {C['green']}",
                        "animation": "glowPulse 2s ease infinite"
                    }),
                    html.Span("Gemini", style={
                        "color": C["text"], "fontWeight": "800", "fontSize": "14px",
                        "letterSpacing": "0.05em", "fontFamily": "'Outfit', sans-serif"
                    }),
                    html.Span("2.5 Pro", style={
                        "background": f"linear-gradient(135deg, {C['accent']}, {C['accent2']})",
                        "color": "#000", "fontSize": "9px", "fontWeight": "800",
                        "padding": "2px 8px", "borderRadius": "20px", "letterSpacing": "0.08em",
                        "fontFamily": "'JetBrains Mono', monospace",
                    })
                ], style={"display": "flex", "alignItems": "center", "gap": "9px"}),

                # Ticker input
                html.Div([
                    html.Span("$", style={"color": C["accent"], "fontWeight": "900", "fontSize": "15px",
                                          "fontFamily": "'JetBrains Mono', monospace"}),
                    dcc.Input(
                        id="ai-ticker-input",
                        type="text",
                        placeholder="AAPL",
                        maxLength=6,
                        debounce=False,
                        className="ai-ticker-input",
                    )
                ], style={"display": "flex", "alignItems": "center", "gap": "5px"})
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "14px 18px",
                "borderBottom": f"1px solid {C['border']}",
                "background": C["surface"],
                "borderRadius": "14px 14px 0 0",
            }),

            # Quick prompt chips
            html.Div([
                html.Button(label, id={"type": "ai-quick-chip", "index": i},
                            className="ai-quick-chip", n_clicks=0)
                for i, (label, _) in enumerate(QUICK_PROMPTS)
            ], style={
                "display": "flex", "gap": "7px", "padding": "10px 14px",
                "overflowX": "auto", "borderBottom": f"1px solid {C['border']}",
                "background": C["bg2"],
            }),

            # Messages area
            html.Div(
                id="ai-chat-messages",
                children=[
                    message_bubble("assistant",
                                   "👋 Hey! I'm your AI market analyst, powered by Gemini.\n\n"
                                   "Enter a ticker above, then:\n"
                                   "• Tap a quick chip for instant analysis\n"
                                   "• Ask me anything about stocks or strategy\n"
                                   "• Hit 'Market Summary' on the right for a broad overview\n\n"
                                   "What would you like to explore?", "")
                ],
                style={
                    "flex": "1", "overflowY": "auto",
                    "padding": "18px 14px",
                    "display": "flex", "flexDirection": "column",
                    "minHeight": "0",
                }
            ),

            # Typing indicator
            html.Div([
                html.Div(style={
                    "width": "30px", "height": "30px", "borderRadius": "50%", "flexShrink": "0",
                    "background": f"linear-gradient(135deg, {C['accent3']}, {C['accent']})",
                }),
                html.Div([
                    html.Span(className="typing-dot"),
                    html.Span(className="typing-dot"),
                    html.Span(className="typing-dot"),
                ], style={
                    "background": C["surface"], "border": f"1px solid {C['border']}",
                    "borderRadius": "4px 14px 14px 14px",
                    "padding": "10px 16px", "display": "flex", "alignItems": "center", "gap": "3px"
                })
            ], id="ai-typing-indicator", style={
                "display": "none", "alignItems": "flex-start",
                "gap": "10px", "padding": "0 14px 10px",
            }),

            # Input bar
            html.Div([
                dcc.Input(
                    id="ai-user-input",
                    type="text",
                    placeholder="Ask anything about stocks, markets, or strategy...",
                    debounce=False,
                    className="ai-text-input",
                    style={"flex": "1", "height": "42px"},
                    n_submit=0,
                ),
                html.Button("➤", id="ai-send-btn", className="ai-send-btn", n_clicks=0),
            ], style={
                "display": "flex", "gap": "9px", "padding": "12px 14px",
                "borderTop": f"1px solid {C['border']}",
                "background": C["surface"],
                "borderRadius": "0 0 14px 14px",
            })

        ], style={
            "flex": "1", "display": "flex", "flexDirection": "column",
            "background": C["bg2"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "14px", "overflow": "hidden",
            "minHeight": "600px",
            "animation": "slideInUp 0.4s ease both",
        }),

        # ── RIGHT: Market intelligence panel ───────────────────────────────
        html.Div([

            # Panel header
            html.Div([
                html.Div([
                    html.Span("🌍", style={"fontSize": "16px"}),
                    html.Span("Market Intelligence", style={
                        "fontWeight": "700", "fontSize": "15px", "color": C["text"],
                        "fontFamily": "'Outfit', sans-serif", "letterSpacing": "0.05em"
                    })
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                html.Button("Generate Summary", id="ai-summary-btn",
                            className="ai-summary-btn", n_clicks=0)
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "14px 18px", "borderBottom": f"1px solid {C['border']}",
            }),

            # Summary output
            html.Div(
                id="ai-market-summary",
                children=html.Div([
                    html.Div("🔍", style={"fontSize": "28px", "marginBottom": "10px"}),
                    html.Div(
                        "Click 'Generate Summary' for an AI-powered snapshot of current market conditions.",
                        style={"color": C["muted"], "fontSize": "15px", "lineHeight": "1.7",
                               "textAlign": "center", "fontFamily": "'Outfit', sans-serif"}
                    )
                ], style={"padding": "36px 18px", "display": "flex",
                          "flexDirection": "column", "alignItems": "center"}),
                style={"overflowY": "auto", "flex": "1"}
            ),

            html.Hr(style={"borderColor": C["border"], "margin": "0"}),

            # Sentiment meter
            html.Div([
                html.Div("AI Sentiment", style={
                    "fontSize": "10px", "fontWeight": "700", "color": C["muted"],
                    "letterSpacing": "1.5px", "textTransform": "uppercase",
                    "marginBottom": "10px", "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Div(id="ai-sentiment-bar", children=[
                    html.Div(style={
                        "height": "6px", "borderRadius": "3px",
                        "background": "rgba(255,255,255,0.06)",
                    })
                ]),
                html.Div(id="ai-sentiment-label",
                         children="No analysis yet",
                         style={"fontSize": "11px", "color": C["muted"], "marginTop": "6px",
                                "textAlign": "center", "fontFamily": "'JetBrains Mono', monospace"})
            ], style={"padding": "16px 18px"}),

            # Active ticker display
            html.Div([
                html.Div("Active Ticker", style={
                    "fontSize": "10px", "fontWeight": "700", "color": C["muted"],
                    "letterSpacing": "1.5px", "textTransform": "uppercase",
                    "marginBottom": "10px", "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Div(id="ai-active-ticker-display", children=html.Div(
                    "None selected",
                    style={"color": C["muted"], "fontSize": "12px",
                           "fontFamily": "'Outfit', sans-serif"}
                ))
            ], style={"padding": "0 18px 18px"}),

        ], style={
            "width": "320px", "flexShrink": "0",
            "display": "flex", "flexDirection": "column",
            "background": C["surface"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "14px", "overflow": "hidden",
            "animation": "slideInUp 0.4s ease 0.1s both",
        }),

    ], style={
        "display": "flex", "gap": "18px",
        "padding": "22px",
        "height": "calc(100vh - 52px)",
        "boxSizing": "border-box",
        "maxWidth": "1800px",
        "margin": "0 auto",
    })
], style={"background": C["bg"], "minHeight": "calc(100vh - 52px)"})


# =============================================================================
# GEMINI HELPER
# =============================================================================

def call_gemini(messages: list[dict]) -> str:  # ← Only takes messages, not user_input
    try:
        # Get the last user message
        user_message = messages[-1]["content"]

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"

def extract_sentiment(text: str) -> float:
    t = text.lower()
    bull = sum(t.count(w) for w in
               ["bullish", "buy", "strong", "growth", "upside", "positive", "outperform", "opportunity", "rally"])
    bear = sum(t.count(w) for w in
               ["bearish", "sell", "weak", "decline", "downside", "negative", "underperform", "risk", "caution",
                "fall"])
    total = bull + bear
    return bull / total if total > 0 else 0.5


# =============================================================================
# CALLBACKS
# =============================================================================


# Callback 1: Show user message and mark as "thinking"
@callback(
    Output("ai-chat-messages", "children"),
    Output("ai-user-input", "value"),
    Output("ai-chat-history", "data"),
    Output("ai-is-thinking", "data"),  # Add this
    Output("recent-text", "data"),
    Input("ai-send-btn", "n_clicks"),
    Input("ai-user-input", "n_submit"),
    State("ai-user-input", "value"),
    State("ai-ticker-input", "value"),
    State("ai-chat-history", "data"),
    State("ai-chat-messages", "children"),
    prevent_initial_call=True
)
def show_user_message_immediately(send_clicks, n_submit, user_text, ticker, history, current_messages):
    if not user_text or not user_text.strip():
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    ticker_str = (ticker or "").upper().strip()
    context = f"[Ticker context: {ticker_str}] " if ticker_str else ""
    full_prompt = context + user_text.strip()

    history = history or []
    history.append({"role": "user", "content": full_prompt})

    ts = datetime.now().strftime("%H:%M")
    new_messages = list(current_messages) + [
        message_bubble("user", user_text.strip(), ts),
    ]

    return new_messages, "", history, True, user_text


# Callback 2: Show/hide typing indicator based on thinking state
@callback(
    Output("ai-typing-indicator", "style"),
    Input("ai-is-thinking", "data"),
)
def toggle_typing_indicator(is_thinking):
    if is_thinking:
        return {
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "10px",
            "padding": "0 14px 10px",
        }
    return {"display": "none"}


@callback(
    Output("ai-chat-messages", "children", allow_duplicate=True),
    Output("ai-chat-history", "data", allow_duplicate=True),
    Output("ai-is-thinking", "data", allow_duplicate=True),
    Output("ai-sentiment-bar", "children"),
    Output("ai-sentiment-label", "children"),
    Input("ai-chat-history", "data"),
    State("ai-chat-messages", "children"),
    State("recent-text", "data"),  # ← Changed to State
    prevent_initial_call=True
)
def get_ai_response(history, current_messages, user_text):  # ← Fixed parameter order
    if not history or history[-1]["role"] != "user":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Get AI response (slow)
    #ai_response = call_gemini(history)  # ← Only pass history
    message = history[-1]
    ai_response = get_response(message)
    history.append({"role": "model", "content": ai_response})

    # Convert markdown to plain text
    message_html = markdown.markdown(ai_response)
    message = BeautifulSoup(message_html, "html.parser").get_text()

    ts = datetime.now().strftime("%H:%M")
    new_messages = list(current_messages) + [
        message_bubble("assistant", message, ts),
    ]

    score = extract_sentiment(ai_response)
    sentiment_bar = render_sentiment_bar(score)
    label_text = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
    label_color = C["green"] if score > 0.55 else C["red"] if score < 0.45 else C["dim"]

    return (
        new_messages,  # ← Was returning user_text, should return new_messages
        history,
        False,  # Set thinking to False
        sentiment_bar,
        html.Span(label_text, style={"color": label_color, "fontWeight": "700"})
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
    response = call_gemini([{"role": "user", "content": MARKET_SUMMARY_PROMPT}])
    return html.Div([
        html.Div([
            html.Div("🌐  Market Summary", style={
                "fontWeight": "800", "fontSize": "15px", "color": C["accent"],
                "marginBottom": "10px", "letterSpacing": "0.08em",
                "fontFamily": "'JetBrains Mono', monospace",
            }),
            html.Div(response, style={
                "whiteSpace": "pre-wrap", "fontSize": "15px",
                "lineHeight": "1.8", "color": C["dim"],
                "fontFamily": "'Outfit', sans-serif",
            }),
            html.Div(
                f"Generated {datetime.now().strftime('%H:%M:%S')}",
                style={"fontSize": "10px", "color": C["muted"], "marginTop": "10px",
                       "textAlign": "right", "fontFamily": "'JetBrains Mono', monospace"}
            )
        ], style={"padding": "14px 18px"})
    ], style={"animation": "slideInUp 0.4s ease both"})


@callback(
    Output("ai-active-ticker-display", "children"),
    Input("ai-ticker-input", "value"),
)
def update_ticker_display(ticker):
    if not ticker or not ticker.strip():
        return html.Div("None selected",
                        style={"color": C["muted"], "fontSize": "12px",
                               "fontFamily": "'Outfit', sans-serif"})
    t = ticker.upper().strip()
    return html.Div([
        html.Span(f"${t}", style={
            "fontSize": "22px", "fontWeight": "800", "color": C["accent"],
            "letterSpacing": "3px", "fontFamily": "'JetBrains Mono', monospace",
        }),
        html.Div("Use chips or type a question →", style={
            "fontSize": "10px", "color": C["muted"], "marginTop": "4px",
            "fontFamily": "'Outfit', sans-serif",
        })
    ], style={"animation": "slideInUp 0.3s ease both"})
