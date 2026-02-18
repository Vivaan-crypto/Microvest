"""
Configuration and Constants
Centralizes all app configuration with enhanced design tokens
"""

# =============================================================================
# STOCK CONFIGURATION
# =============================================================================

TICKERS = [
    ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
    # ("NVDA", "Technology"), ("AVGO", "Technology"), ("AMD", "Technology"),
    # ("INTC", "Technology"), ("ASML", "Technology"), ("CRM", "Technology"),
    # ("ADBE", "Technology"),
    # ("META", "Communication Services"), ("NFLX", "Communication Services"),
    # ("DIS", "Communication Services"),
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    # ("HD", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    # ("NKE", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"),
    # ("MCD", "Consumer Discretionary"), ("TGT", "Consumer Discretionary"),
    # ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    # ("GS", "Financials"), ("MS", "Financials"), ("V", "Financials"),
    # ("MA", "Financials"), ("BLK", "Financials"),
    # ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    # ("SLB", "Energy"),
    # ("UNH", "Health Care"), ("ABBV", "Health Care"),
    # ("LLY", "Health Care"), ("JNJ", "Health Care"), ("PFE", "Health Care"),
    # ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    # ("COST", "Consumer Staples"), ("WMT", "Consumer Staples"),
    # ("CAT", "Industrials"), ("BA", "Industrials"), ("UPS", "Industrials"),
    # ("^SPX", "Index"), ("^NDX", "Index"), ("^DJI", "Index")
]

# Derived configurations
SYMBOLS = [t[0] for t in TICKERS]
SECTORS = {t: s for t, s in TICKERS}

# =============================================================================
# APP SETTINGS
# =============================================================================

SNAPSHOT_REFRESH_MS = 5000  # Milliseconds between data refreshes
APP_HOST = "127.0.0.1"
APP_PORT = 8050
DEBUG_MODE = False


# =============================================================================
# ENHANCED DESIGN SYSTEM - Modern Financial Dashboard
# =============================================================================

class Colors:
    """
    Premium dark theme with vibrant accents
    Inspired by modern financial terminals
    """
    # Base colors - Deep dark theme
    BG_PRIMARY = "#0a0a0f"
    BG_SECONDARY = "#13131a"
    BG_TERTIARY = "#1a1a24"
    BG_ELEVATED = "#20202e"

    # Vibrant accent colors
    ACCENT_PRIMARY = "#00f5ff"  # Electric cyan
    ACCENT_SECONDARY = "#ff00e5"  # Hot magenta
    ACCENT_TERTIARY = "#00ff88"  # Neon green

    # Market colors
    SUCCESS = "#00ff88"  # Bright green
    SUCCESS_DIM = "#00cc6a"
    SUCCESS_GLOW = "rgba(0, 255, 136, 0.2)"

    DANGER = "#ff0055"  # Hot pink/red
    DANGER_DIM = "#cc0044"
    DANGER_GLOW = "rgba(255, 0, 85, 0.2)"

    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a0a0b8"
    TEXT_MUTED = "#606078"
    TEXT_DIM = "#404050"

    # Chart colors
    CHART_BLUE = "#3b82f6"
    CHART_PURPLE = "rgba(67, 34, 98, 0.6)"
    CHART_ORANGE = "#fb923c"
    CHART_YELLOW = "#fbbf24"

    # UI elements
    BORDER = "#2a2a38"
    BORDER_BRIGHT = "#3a3a48"
    OVERLAY = "rgba(10, 10, 15, 0.95)"

    # Gradients
    GRADIENT_PRIMARY = "linear-gradient(135deg, #00f5ff 0%, #00ff88 100%)"
    GRADIENT_DANGER = "linear-gradient(135deg, #ff0055 0%, #ff00e5 100%)"
    GRADIENT_NEUTRAL = "linear-gradient(135deg, #3a3a48 0%, #2a2a38 100%)"


class Typography:
    """
    Modern, distinctive typography system
    Using JetBrains Mono for data/numbers and Outfit for UI
    """
    # Font families
    FONT_MONO = "'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', monospace"
    FONT_DISPLAY = "'Outfit', -apple-system, BlinkMacSystemFont, sans-serif"
    FONT_BODY = "'Outfit', -apple-system, BlinkMacSystemFont, sans-serif"

    # Font sizes
    SIZE_TINY = "10px"
    SIZE_XS = "11px"
    SIZE_SM = "13px"
    SIZE_BASE = "15px"
    SIZE_MD = "16px"
    SIZE_LG = "18px"
    SIZE_XL = "22px"
    SIZE_XXL = "28px"
    SIZE_XXXL = "36px"
    SIZE_DISPLAY = "48px"
    SIZE_HERO = "64px"

    # Font weights
    WEIGHT_LIGHT = "300"
    WEIGHT_NORMAL = "400"
    WEIGHT_MEDIUM = "500"
    WEIGHT_SEMIBOLD = "600"
    WEIGHT_BOLD = "700"
    WEIGHT_BLACK = "800"

    # Line heights
    LINE_TIGHT = "1.2"
    LINE_NORMAL = "1.5"
    LINE_RELAXED = "1.75"


class Spacing:
    """
    Consistent spacing scale - 4px base unit
    """
    XXS = "2px"
    XS = "4px"
    SM = "8px"
    MD = "12px"
    LG = "16px"
    XL = "24px"
    XXL = "32px"
    XXXL = "48px"
    HUGE = "64px"


class Effects:
    """
    Modern visual effects
    """
    # Shadows
    SHADOW_SM = "0 2px 8px rgba(0, 0, 0, 0.4)"
    SHADOW_MD = "0 4px 16px rgba(0, 0, 0, 0.5)"
    SHADOW_LG = "0 8px 32px rgba(0, 0, 0, 0.6)"
    SHADOW_XL = "0 16px 48px rgba(0, 0, 0, 0.7)"

    # Glows
    GLOW_SUCCESS = "0 0 20px rgba(0, 255, 136, 0.3), 0 0 40px rgba(0, 255, 136, 0.1)"
    GLOW_DANGER = "0 0 20px rgba(255, 0, 85, 0.3), 0 0 40px rgba(255, 0, 85, 0.1)"
    GLOW_ACCENT = "0 0 20px rgba(0, 245, 255, 0.3), 0 0 40px rgba(0, 245, 255, 0.1)"

    # Border radius
    RADIUS_SM = "6px"
    RADIUS_MD = "10px"
    RADIUS_LG = "16px"
    RADIUS_XL = "24px"
    RADIUS_FULL = "9999px"

    # Blur
    BLUR_SM = "blur(8px)"
    BLUR_MD = "blur(16px)"
    BLUR_LG = "blur(24px)"

    # Transitions
    TRANSITION_FAST = "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)"
    TRANSITION_BASE = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
    TRANSITION_SLOW = "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)"


class Layout:
    """
    Layout constants
    """
    CONTAINER_MAX_WIDTH = "1800px"
    HEATMAP_MIN_HEIGHT = "600px"
    CHART_HEIGHT = "700px"