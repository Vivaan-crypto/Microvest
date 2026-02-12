"""
Configuration and Constants
Centralizes all app configuration in one place
"""

# =============================================================================
# STOCK CONFIGURATION
# =============================================================================

TICKERS = [
    ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
    ("NVDA", "Technology"), ("AVGO", "Watchlist"), ("AMD", "Technology"),
    ("INTC", "Technology"), ("ASML", "Technology"), ("CRM", "Technology"),
    ("ADBE", "Technology"),
    ("META", "Communication Services"), ("NFLX", "Communication Services"),
    ("DIS", "Communication Services"),
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"), ("TGT", "Consumer Discretionary"),
    ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    ("GS", "Financials"), ("MS", "Financials"), ("V", "Financials"),
    ("MA", "Financials"), ("BLK", "Financials"),
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    ("SLB", "Energy"),
    ("UNH", "Health Care"), ("ABBV", "Health Care"),
    ("LLY", "Health Care"), ("JNJ", "Health Care"), ("PFE", "Health Care"),
    ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    ("COST", "Consumer Staples"), ("WMT", "Consumer Staples"),
    ("CAT", "Industrials"), ("BA", "Industrials"), ("UPS", "Industrials"),
    ("^SPX", "Index"), ("^NDX", "Index"), ("^DJI", "Index"),
    ("APA", "Energy"), ("SOFI", "Watchlist")
]

# Derived configurations
SYMBOLS = [t[0] for t in TICKERS]
SECTORS = {t: s for t, s in TICKERS}
WATCHLIST_TICKERS = [t[0] for t in TICKERS if t[1] == "Watchlist"]

# =============================================================================
# APP SETTINGS
# =============================================================================

SNAPSHOT_REFRESH_MS = 5000  # Milliseconds between data refreshes
APP_HOST = "127.0.0.1"
APP_PORT = 8050
DEBUG_MODE = False


# =============================================================================
# THEME / DESIGN TOKENS
# =============================================================================

class Colors:
    """
    Centralized color palette
    Industry best practice: Use a class to group related constants
    """
    # Semantic colors
    SUCCESS = "#10b981"
    DANGER = "#f43f5e"

    # Grayscale
    SLATE_50 = "#f8fafc"
    SLATE_700 = "#334155"
    SLATE_800 = "#1e293b"
    SLATE_900 = "#0f172a"
    SLATE_950 = "#020617"

    # Accent colors
    BLUE = "#3b82f6"
    ORANGE = "#f97316"
    PURPLE = "#8b5cf6"
    YELLOW = "#eab308"

    # Aliases for easier use
    WHITE = "#ffffff"
    BLACK = SLATE_950
    CARD_BG = SLATE_900
    BORDER = SLATE_800
    TEXT_PRIMARY = SLATE_50
    TEXT_MUTED = SLATE_700


class Spacing:
    """
    Consistent spacing scale
    Industry standard: 4px base unit (4, 8, 12, 16, 20, 24...)
    """
    XS = "4px"
    SM = "8px"
    MD = "12px"
    LG = "16px"
    XL = "20px"
    XXL = "24px"
    XXXL = "32px"


class Typography:
    """
    Typography system
    """
    FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

    # Font sizes
    SIZE_XS = "11px"
    SIZE_SM = "12px"
    SIZE_BASE = "13px"
    SIZE_MD = "14px"
    SIZE_LG = "16px"
    SIZE_XL = "18px"
    SIZE_XXL = "20px"
    SIZE_XXXL = "28px"
    SIZE_DISPLAY = "32px"

    # Font weights
    WEIGHT_NORMAL = "400"
    WEIGHT_MEDIUM = "600"
    WEIGHT_BOLD = "700"
    WEIGHT_BLACK = "800"


class Layout:
    """
    Layout constants
    """
    BORDER_RADIUS = "20px"
    BORDER_RADIUS_SM = "12px"
    BORDER_RADIUS_MD = "14px"

    CONTAINER_MAX_WIDTH = "1600px"
    HEATMAP_MIN_HEIGHT = "500px"