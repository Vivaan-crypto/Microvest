"""
UI Style Helpers
Reusable style generators to eliminate repetitive inline styles

Industry pattern: "Style as Functions"
- Keeps components clean
- Easy to maintain consistency
- DRY (Don't Repeat Yourself) principle
"""

from config import Colors, Spacing, Typography, Layout


# =============================================================================
# BASE STYLES (Most commonly reused)
# =============================================================================

def flexbox(
        direction="row",
        align="center",
        justify="flex-start",
        gap=Spacing.MD,
        wrap="nowrap"
):
    """
    Flexbox layout helper

    Example:
        style = flexbox(direction="column", align="stretch", gap="20px")
    """
    return {
        "display": "flex",
        "flexDirection": direction,
        "alignItems": align,
        "justifyContent": justify,
        "gap": gap,
        "flexWrap": wrap
    }


def card(padding=Spacing.XL, extra_styles=None):
    """
    Standard card container

    Example:
        style = card(padding="30px")
    """
    base = {
        "background": Colors.CARD_BG,
        "borderRadius": Layout.BORDER_RADIUS,
        "border": f"1px solid {Colors.BORDER}",
        "padding": padding
    }
    return {**base, **(extra_styles or {})}


def text(size=Typography.SIZE_BASE, weight=Typography.WEIGHT_NORMAL, color=Colors.TEXT_PRIMARY):
    """
    Text styling helper

    Example:
        style = text(size="20px", weight="700", color=Colors.SUCCESS)
    """
    return {
        "fontSize": size,
        "fontWeight": weight,
        "color": color,
        "fontFamily": Typography.FONT_FAMILY
    }


def button(variant="primary"):
    """
    Button styles with variants

    Variants: primary, secondary, ghost
    """
    base = {
        "padding": f"{Spacing.MD} {Spacing.XXL}",
        "border": "none",
        "borderRadius": Layout.BORDER_RADIUS_SM,
        "cursor": "pointer",
        "fontWeight": Typography.WEIGHT_BOLD,
        "fontSize": Typography.SIZE_BASE,
        "transition": "all 0.2s ease"
    }

    variants = {
        "primary": {
            "background": Colors.WHITE,
            "color": Colors.BLACK
        },
        "secondary": {
            "background": Colors.CARD_BG,
            "color": Colors.TEXT_PRIMARY,
            "border": f"1px solid {Colors.BORDER}"
        },
        "ghost": {
            "background": "transparent",
            "color": Colors.TEXT_PRIMARY
        }
    }

    return {**base, **variants.get(variant, variants["primary"])}


# =============================================================================
# COMPONENT-SPECIFIC STYLES
# =============================================================================

def stat_row():
    """Style for stat display rows (label + value)"""
    return flexbox(direction="row", justify="space-between", gap=Spacing.SM)


def divider(margin=f"{Spacing.LG} 0"):
    """Horizontal divider line"""
    return {
        "height": "1px",
        "backgroundColor": Colors.BORDER,
        "margin": margin
    }


def badge(bg_color=Colors.SUCCESS, text_color=Colors.WHITE):
    """
    Badge/pill style

    Example:
        style = badge(bg_color=Colors.DANGER)
    """
    return {
        "display": "inline-flex",
        "alignItems": "center",
        "padding": f"{Spacing.SM} {Spacing.LG}",
        "borderRadius": Layout.BORDER_RADIUS_MD,
        "background": bg_color,
        "color": text_color,
        "fontSize": Typography.SIZE_XL,
        "fontWeight": Typography.WEIGHT_BOLD
    }


def overlay(visible=False):
    """
    Full-screen overlay for modals/charts

    Example:
        style = overlay(visible=True)
    """
    return {
        "position": "absolute",
        "top": "0",
        "left": "0",
        "right": "0",
        "bottom": "0",
        "opacity": "1" if visible else "0",
        "pointerEvents": "all" if visible else "none",
        "backgroundColor": Colors.BLACK,
        "zIndex": "1000",
        "padding": Spacing.XXXL,
        "borderRadius": Layout.BORDER_RADIUS,
        "transition": "opacity 0.3s ease"
    }


# =============================================================================
# LAYOUT HELPERS
# =============================================================================

def grid(columns="1fr", rows="auto", gap=Spacing.LG):
    """
    CSS Grid helper

    Example:
        style = grid(columns="2fr 1fr", gap="20px")
    """
    return {
        "display": "grid",
        "gridTemplateColumns": columns,
        "gridTemplateRows": rows,
        "gap": gap
    }


def absolute_position(top=None, right=None, bottom=None, left=None, z_index=None):
    """
    Absolute positioning helper

    Example:
        style = absolute_position(top="20px", left="20px", z_index="2000")
    """
    style = {"position": "absolute"}
    if top is not None:
        style["top"] = top
    if right is not None:
        style["right"] = right
    if bottom is not None:
        style["bottom"] = bottom
    if left is not None:
        style["left"] = left
    if z_index is not None:
        style["zIndex"] = str(z_index)
    return style


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def merge_styles(*style_dicts):
    """
    Merge multiple style dictionaries
    Later styles override earlier ones

    Example:
        style = merge_styles(
            card(),
            flexbox(direction="column"),
            {"backgroundColor": "red"}
        )
    """
    merged = {}
    for style_dict in style_dicts:
        if style_dict:
            merged.update(style_dict)
    return merged


def conditional_style(condition, true_style, false_style=None):
    """
    Apply styles based on condition

    Example:
        style = conditional_style(
            is_positive,
            {"color": Colors.SUCCESS},
            {"color": Colors.DANGER}
        )
    """
    return true_style if condition else (false_style or {})