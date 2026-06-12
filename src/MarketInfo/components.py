"""
UI Components - Premium Design
Beautiful, modern components for financial dashboard
"""

from dash import html, dcc
from config import Colors, Typography, Spacing, Effects
from datetime import time, datetime
from styles import (
    flexbox, grid, text, card, button, badge,
    overlay, merge_styles, glass_card
)


# =============================================================================
# FONTS
# =============================================================================

def get_font_imports():
    """
    Google Fonts import for premium typography
    """
    return html.Link(
        href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap",
        rel="stylesheet"
    )


# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def PageContainer(children):
    """
    Premium page container - clean dark background
    """
    return html.Div([
        get_font_imports(),

        # Main content
        html.Div(
            children,
            style={
                "minHeight": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "fontFamily": Typography.FONT_BODY,
                "padding": Spacing.XXL,
                "gap": Spacing.XL,
                "position": "relative",
                "background": Colors.BG_PRIMARY  # Solid dark background
            }
        )
    ])


def Header(title="STOCK MARKET"):
    """
    Premium header with gradient text
    """
    return html.Div([
        html.Div([
            html.H1(
                title,
                style={
                    "fontFamily": Typography.FONT_DISPLAY,
                    "fontSize": Typography.SIZE_HERO,
                    "fontWeight": Typography.WEIGHT_BLACK,
                    "lineHeight": Typography.LINE_TIGHT,
                    "background": Colors.GRADIENT_PRIMARY,
                    "WebkitBackgroundClip": "text",
                    "WebkitTextFillColor": "transparent",
                    "backgroundClip": "text",
                    "margin": "0",
                    "letterSpacing": "-0.03em"
                }
            ),
            html.Div(
                "Real-time market visualization",
                style={
                    **text(
                        size=Typography.SIZE_SM,
                        color=Colors.TEXT_SECONDARY,
                        font=Typography.FONT_DISPLAY
                    ),
                    "marginTop": Spacing.SM,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.1em"
                }
            )
        ]),

        # Live indicator

        html.Div([
            html.Div(style={
                "width": "8px",
                "height": "8px",
                "borderRadius": "50%",
                "background": isMarketLive(Colors.SUCCESS, Colors.DANGER),
                "boxShadow": Effects.GLOW_SUCCESS,
                "animation": "pulse 1s ease-in-out infinite"
            }),
            html.Span(
                isMarketLive("Live", "Closed"),
                style={
                    **text(
                        size=Typography.SIZE_XS,
                        weight=Typography.WEIGHT_BOLD,
                        color=isMarketLive(Colors.SUCCESS, Colors.DANGER),
                        font=Typography.FONT_MONO
                    ),
                    "letterSpacing": "0.1em"
                }
            )
        ], style={
            **flexbox(gap=Spacing.SM),
            "padding": f"{Spacing.SM} {Spacing.MD}",
            "background": f"{isMarketLive(Colors.SUCCESS, Colors.DANGER)}10",
            "border": f"1px solid {isMarketLive(Colors.SUCCESS, Colors.DANGER)}20",
            "borderRadius": Effects.RADIUS_FULL
        })
    ], style=flexbox(justify="space-between", align="flex-start"))


def BackButton(id="nav_back"):
    """
    Sleek floating back button with glassmorphic style
    """
    return html.Button(
        html.Span("←", style={
            "fontSize": "24px",
            "fontWeight": "300",
            "lineHeight": "1"
        }),
        id=id,
        className="btn-hidden",
        n_clicks=0,
        style={
            "position": "fixed",
            "top": Spacing.XS,
            "left": "0px",
            "zIndex": "2000",
            "width": "40px",
            "height": "40px",
            "padding": "0",
            "background": "rgba(205, 0, 0, 0.8)",
            "backdropFilter": "blur(12px)",
            "WebkitBackdropFilter": "blur(12px)",
            "border": f"3px solid {Colors.BORDER_BRIGHT}",
            "borderRadius": "50%",
            "color": Colors.TEXT_PRIMARY,
            "cursor": "pointer",
            "transition": "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
            "boxShadow": "0 4px 16px rgba(0, 0, 0, 0.6)",
            "alignItems": "center",
            "justifyContent": "center"
        },
    )


# =============================================================================
# CHART COMPONENTS
# =============================================================================

def ChartContainer(figure, id="chart", config=None):
    """
    Premium chart container with enhanced glass effect
    """
    default_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "scrollZoom": True
    }

    return html.Div([
        dcc.Graph(
            id=id,
            figure=figure,
            config=config or default_config,
            style={"height": "100%", "width": "100%"}
        )
    ], style={
        "background": "rgba(32, 32, 46, 0.3)",  # More transparent
        "backdropFilter": "blur(20px)",
        "WebkitBackdropFilter": "blur(20px)",
        "border": "1px solid rgba(160, 160, 184, 0.2)",
        "borderRadius": Effects.RADIUS_LG,
        "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.4)",
        "padding": "0",
        "overflow": "hidden",
        "minHeight": "100%",
        "height": "100%"
    })


def ChartOverlay(id="chart_overlay", content_id="chart_content", visible=False):
    """
    Premium full-screen overlay
    """
    return html.Div(
        html.Div(
            id=content_id,
            style={
                "width": "100vw",
                "height": "100vh",
            }
        ),
        id=id,
        style=overlay(visible)
    )


# =============================================================================
# DATA DISPLAY COMPONENTS
# =============================================================================

def StockBadge(ticker, price, change):
    """
    Premium stock badge with live data
    """
    is_positive = change > 0
    color = Colors.SUCCESS if is_positive else Colors.DANGER
    arrow = "↑" if is_positive else "↓"

    return html.Div([
        # Ticker
        html.Div(
            ticker,
            style={
                **text(
                    size=Typography.SIZE_XXL,
                    weight=Typography.WEIGHT_BLACK,
                    font=Typography.FONT_DISPLAY
                ),
                "letterSpacing": "-0.02em"
            }
        ),

        # Price
        html.Div(
            f"${price:,.2f}",
            style={
                **text(
                    size=Typography.SIZE_HERO,
                    weight=Typography.WEIGHT_BOLD,
                    font=Typography.FONT_MONO
                ),
                "lineHeight": Typography.LINE_TIGHT,
                "marginTop": Spacing.SM
            }
        ),

        # Change badge
        html.Div([
            html.Span(arrow, style={"fontSize": Typography.SIZE_LG}),
            html.Span(f"{abs(change):.2f}%")
        ], style={
            **badge(color),
            "marginTop": Spacing.MD,
            "fontSize": Typography.SIZE_MD
        })
    ], style={
        **card(elevated=True),
        "background": f"linear-gradient(135deg, {Colors.BG_ELEVATED} 0%, {Colors.BG_SECONDARY} 100%)",
        "minWidth": "280px"
    })


def SectorTag(sector):
    """
    Sector category tag
    """
    return html.Div(
        sector,
        style={
            "padding": f"{Spacing.XS} {Spacing.MD}",
            "background": f"{Colors.ACCENT_PRIMARY}15",
            "border": f"1px solid {Colors.ACCENT_PRIMARY}40",
            "borderRadius": Effects.RADIUS_SM,
            "color": Colors.ACCENT_PRIMARY,
            "fontSize": Typography.SIZE_TINY,
            "fontWeight": Typography.WEIGHT_BOLD,
            "fontFamily": Typography.FONT_MONO,
            "textTransform": "uppercase",
            "letterSpacing": "0.1em"
        }
    )


def InfoPanel(ticker=None, sector=None, price=None, change=None):
    """
    Premium information panel
    """
    if not ticker:
        return html.Div(
            [
                html.Div("📊", style={"fontSize": "48px", "marginBottom": Spacing.LG}),
                html.Div(
                    "Select a stock from the heatmap",
                    style={
                        **text(
                            size=Typography.SIZE_LG,
                            color=Colors.TEXT_MUTED,
                            font=Typography.FONT_DISPLAY
                        ),
                        "textAlign": "center"
                    }
                )
            ],
            style={
                **card(elevated=True),
                **flexbox(direction="column", align="center", justify="center"),
                "minHeight": "200px",
                "background": f"linear-gradient(135deg, {Colors.BG_ELEVATED} 0%, {Colors.BG_SECONDARY} 100%)"
            }
        )

    is_positive = change > 0
    color = Colors.SUCCESS if is_positive else Colors.DANGER
    arrow = "↑" if is_positive else "↓"

    return html.Div([
        # Top row - Ticker and sector
        html.Div([
            html.Div([
                html.H2(
                    ticker,
                    style={
                        **text(
                            size=Typography.SIZE_XXXL,
                            weight=Typography.WEIGHT_BLACK,
                            font=Typography.FONT_DISPLAY
                        ),
                        "margin": "0",
                        "letterSpacing": "-0.02em"
                    }
                ),
                SectorTag(sector)
            ], style=flexbox(align="center", gap=Spacing.MD)),

            # Change badge
            html.Div([
                html.Span(arrow, style={"fontSize": Typography.SIZE_XL, "marginRight": Spacing.SM}),
                html.Span(f"{abs(change):.2f}%")
            ], style={
                **badge(color),
                "fontSize": Typography.SIZE_LG,
                "padding": f"{Spacing.MD} {Spacing.XL}"
            })
        ], style=flexbox(justify="space-between", align="center")),

        # Divider
        html.Div(style={
            "height": "1px",
            "background": f"linear-gradient(90deg, transparent 0%, {Colors.BORDER_BRIGHT} 50%, transparent 100%)",
            "margin": f"{Spacing.XL} 0"
        }),

        # Price display
        html.Div([
            html.Div(
                "CURRENT PRICE",
                style={
                    **text(
                        size=Typography.SIZE_TINY,
                        weight=Typography.WEIGHT_BOLD,
                        color=Colors.TEXT_MUTED,
                        font=Typography.FONT_DISPLAY
                    ),
                    "letterSpacing": "0.15em",
                    "marginBottom": Spacing.SM
                }
            ),
            html.Div(
                f"${price:,.2f}",
                style={
                    "fontFamily": Typography.FONT_MONO,
                    "fontSize": Typography.SIZE_HERO,
                    "fontWeight": Typography.WEIGHT_BOLD,
                    "lineHeight": Typography.LINE_TIGHT,
                    "background": Colors.GRADIENT_PRIMARY,
                    "WebkitBackgroundClip": "text",
                    "WebkitTextFillColor": "transparent",
                    "backgroundClip": "text"
                }
            )
        ])
    ], style={
        **card(elevated=True),
        "background": f"linear-gradient(135deg, {Colors.BG_ELEVATED} 0%, {Colors.BG_SECONDARY} 100%)",
        "position": "relative",
        "overflow": "hidden"
    })


# =============================================================================
# LOADING STATES
# =============================================================================

def LoadingSpinner():
    """
    Premium loading spinner
    """
    return html.Div([
        html.Div(style={
            "width": "40px",
            "height": "40px",
            "border": f"3px solid {Colors.BORDER}",
            "borderTop": f"3px solid {Colors.ACCENT_PRIMARY}",
            "borderRadius": "50%",
            "animation": "spin 1s linear infinite"
        }),
        html.Div(
            "Loading...",
            style={
                **text(
                    size=Typography.SIZE_SM,
                    color=Colors.TEXT_MUTED,
                    font=Typography.FONT_DISPLAY
                ),
                "marginTop": Spacing.MD
            }
        )
    ], style={
        **flexbox(direction="column", align="center", justify="center"),
        "padding": Spacing.XXXL
    })


def isMarketLive(a, b):
    start_time = time(9, 30)
    end_time = time(16, 0)
    now = datetime.now().time()
    if start_time <= now <= end_time:
        return a
    else:
        return b
