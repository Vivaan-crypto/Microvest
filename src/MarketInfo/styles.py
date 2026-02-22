"""
UI Style Helpers - Premium Design System
Beautiful, modern styles for financial dashboard
"""

from config import Colors, Spacing, Typography, Effects


# =============================================================================
# GLASS MORPHISM & MODERN EFFECTS
# =============================================================================

def glass_card(blur="16px", opacity=0.1):
    """
    Modern glassmorphism card effect
    """
    return {
        "background": f"rgba(32, 32, 46, {opacity})",
        "backdropFilter": f"blur({blur})",
        "WebkitBackdropFilter": f"blur({blur})",
        "border": f"1px solid {Colors.BORDER_BRIGHT}",
        "borderRadius": Effects.RADIUS_LG,
        "boxShadow": Effects.SHADOW_MD
    }


def neon_border(color=Colors.ACCENT_PRIMARY):
    """
    Glowing neon border effect
    """
    return {
        "border": f"1px solid {color}",
        "boxShadow": f"0 0 10px {color}40, inset 0 0 10px {color}20"
    }


# =============================================================================
# BASE STYLES
# =============================================================================

def flexbox(
        direction="row",
        align="center",
        justify="flex-start",
        gap=Spacing.MD,
        wrap="nowrap"
):
    """Flexbox layout helper"""
    return {
        "display": "flex",
        "flexDirection": direction,
        "alignItems": align,
        "justifyContent": justify,
        "gap": gap,
        "flexWrap": wrap
    }


def grid(columns="1fr", rows="auto", gap=Spacing.LG):
    """CSS Grid helper"""
    return {
        "display": "grid",
        "gridTemplateColumns": columns,
        "gridTemplateRows": rows,
        "gap": gap
    }


def text(
        size=Typography.SIZE_BASE,
        weight=Typography.WEIGHT_NORMAL,
        color=Colors.TEXT_PRIMARY,
        font=Typography.FONT_BODY
):
    """Text styling helper"""
    return {
        "fontSize": size,
        "fontWeight": weight,
        "color": color,
        "fontFamily": font,
        "letterSpacing": "-0.01em"
    }


# =============================================================================
# COMPONENT STYLES
# =============================================================================

def card(elevated=False):
    """
    Premium card component
    """
    base = {
        "background": Colors.BG_ELEVATED if elevated else Colors.BG_SECONDARY,
        "border": f"1px solid {Colors.BORDER}",
        "borderRadius": Effects.RADIUS_LG,
        "padding": Spacing.XXL,
        "boxShadow": Effects.SHADOW_MD if elevated else Effects.SHADOW_SM,
        "transition": Effects.TRANSITION_BASE
    }
    return base


def button(variant="primary"):
    """
    Modern button styles
    """
    base = {
        "padding": f"{Spacing.MD} {Spacing.XL}",
        "borderRadius": Effects.RADIUS_MD,
        "fontWeight": Typography.WEIGHT_SEMIBOLD,
        "fontSize": Typography.SIZE_SM,
        "fontFamily": Typography.FONT_DISPLAY,
        "cursor": "pointer",
        "transition": Effects.TRANSITION_FAST,
        "border": "none",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em"
    }

    variants = {
        "primary": {
            "background": Colors.GRADIENT_PRIMARY,
            "color": Colors.BG_PRIMARY,
            "boxShadow": Effects.GLOW_ACCENT
        },
        "danger": {
            "background": Colors.GRADIENT_DANGER,
            "color": Colors.TEXT_PRIMARY,
            "boxShadow": Effects.GLOW_DANGER
        },
        "ghost": {
            "background": "rgba(32, 32, 46, 0.6)",
            "backdropFilter": "blur(12px)",
            "WebkitBackdropFilter": "blur(12px)",
            "color": Colors.TEXT_PRIMARY,
            "border": f"1px solid {Colors.BORDER_BRIGHT}"
        }
    }

    return {**base, **variants.get(variant, variants["primary"])}

def badge(color=Colors.SUCCESS):
    """
    Modern badge with glow effect
    """
    glow_map = {
        Colors.SUCCESS: Effects.GLOW_SUCCESS,
        Colors.DANGER: Effects.GLOW_DANGER,
        Colors.ACCENT_PRIMARY: Effects.GLOW_ACCENT
    }

    return {
        "display": "inline-flex",
        "alignItems": "center",
        "gap": Spacing.XS,
        "padding": f"{Spacing.SM} {Spacing.MD}",
        "borderRadius": Effects.RADIUS_FULL,
        "background": f"{color}20",
        "border": f"1px solid {color}",
        "color": color,
        "fontSize": Typography.SIZE_SM,
        "fontWeight": Typography.WEIGHT_BOLD,
        "fontFamily": Typography.FONT_MONO,
        "boxShadow": glow_map.get(color, Effects.GLOW_ACCENT),
        "letterSpacing": "0.02em"
    }


def stat_display():
    """
    Large stat number display
    """
    return {
        "fontFamily": Typography.FONT_MONO,
        "fontSize": Typography.SIZE_HERO,
        "fontWeight": Typography.WEIGHT_BOLD,
        "lineHeight": Typography.LINE_TIGHT,
        "background": Colors.OVERLAY,
        "WebkitBackgroundClip": "text",
        "WebkitTextFillColor": "transparent",
        "backgroundClip": "text"
    }


def overlay(visible=False):
    """
    Full-screen overlay with blur effect
    """
    return {
        "position": "fixed",
        "top": "0",
        "left": "0",
        "right": "0",
        "bottom": "0",
        "background": Colors.OVERLAY,
        "backdropFilter": Effects.BLUR_LG,
        "WebkitBackdropFilter": Effects.BLUR_LG,
        "opacity": "1" if visible else "0",
        "pointerEvents": "all" if visible else "none",
        "zIndex": "1000",
        "transition": "opacity 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
        "padding": f"{Spacing.XXXL}",  # top right bottom left,
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }


# =============================================================================
# ANIMATED ELEMENTS
# =============================================================================

def pulse_animation():
    """
    Subtle pulse animation for live data
    """
    return {
        "animation": "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "@keyframes pulse": {
            "0%, 100%": {"opacity": 1},
            "50%": {"opacity": 0.7}
        }
    }


def slide_up():
    """
    Slide up entrance animation
    """
    return {
        "animation": "slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
        "@keyframes slideUp": {
            "from": {
                "opacity": 0,
                "transform": "translateY(20px)"
            },
            "to": {
                "opacity": 1,
                "transform": "translateY(0)"
            }
        }
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def merge_styles(*style_dicts):
    """
    Merge multiple style dictionaries
    """
    merged = {}
    for style_dict in style_dicts:
        if style_dict:
            merged.update(style_dict)
    return merged


def hover_lift():
    """
    Hover effect that lifts element
    """
    return {
        ":hover": {
            "transform": "translateY(-2px)",
            "boxShadow": Effects.SHADOW_LG
        }
    }
