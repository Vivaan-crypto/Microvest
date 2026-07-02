"""
Single source of truth for every research knob.

Every cached artifact (raw panel, feature panel, walk-forward predictions) is
stamped with CONFIG_VERSION. Bump the version whenever you change anything that
alters the *meaning* of the data — label definition, feature list, horizon,
universe — so stale caches can never silently feed a new experiment.

v1  (implicit): fixed +/-5% absolute-return labels, 26 features, 100 tickers,
    history from 2014, single train/test split at TRAIN_END.
v2: beta-adjusted vol-normalized z-score labels, factor features added
    (short-term reversal, 12-1 momentum, idiosyncratic vol), S&P 500 universe,
    history from 2005, walk-forward evaluation.
"""

import os

CONFIG_VERSION = 2

# ------------------------------------------------------------------
# Horizon / windowing
# ------------------------------------------------------------------
HORIZON = 5          # forward-return horizon (trading days)
WINDOW = 20          # sequence length fed to the model

# ------------------------------------------------------------------
# Label definition (v2): beta-adjusted, vol-normalized z-score
#
#   fwd_excess = fwd_ret - beta_60d * mkt_fwd_ret
#   fwd_z      = fwd_excess / (vol_20d * sqrt(HORIZON))
#   label      = Long if fwd_z >= +Z_THRESHOLD, Short if <= -Z_THRESHOLD,
#                else NoTrade
#
# "Significant move" now means: moved more than Z_THRESHOLD x its own normal
# weekly range, beyond what its market exposure explains. Comparable across
# calm/wild stocks and calm/wild years — unlike the old flat +/-5%.
# ------------------------------------------------------------------
Z_THRESHOLD = 1.0
VOL_WINDOW = 20      # daily-return vol window used to normalize the label
BETA_WINDOW = 60     # rolling beta window (already a feature: beta_60d)

# ------------------------------------------------------------------
# History / universe
# ------------------------------------------------------------------
START_DATE = "2004-06-01"   # extra pre-2005 history so 252d features warm up
END_DATE = "2026-01-01"
TRAIN_END = "2021-08-16"    # legacy single-split boundary (kept for the app)

# Universe: "sp500" -> current S&P 500 constituents (fetched + cached), or
# "legacy100" -> the original hand-picked 100 mega-caps.
# NOTE — survivorship bias: both lists are *today's* members, so training data
# only contains companies that survived. Backtest numbers are inflated by this;
# a true fix needs point-in-time constituent data (paid). Treat all results as
# relative comparisons between models, not absolute expected returns.
UNIVERSE_NAME = "sp500"

# ------------------------------------------------------------------
# Walk-forward folds: expanding window, one test year each.
# Train on everything <= train_end, purge HORIZON rows at the boundary
# (their labels peek across it), test on the following calendar year.
# ------------------------------------------------------------------
WALK_FORWARD_FOLDS = [
    # (name,   train_end,     test_start,    test_end)
    ("2019", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2020", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2021", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2025", "2024-12-31", "2025-01-01", "2025-12-31"),
]

# ------------------------------------------------------------------
# Training defaults for walk-forward (per fold). Kept modest: 7 folds retrain
# from scratch, so per-fold cost matters. EarlyStopping on val IC cuts it more.
# ------------------------------------------------------------------
WF_MAX_EPOCHS = 40
WF_PATIENCE = 8
WF_BATCH_SIZE = 256
WF_LR = 1e-3

# ------------------------------------------------------------------
# Paths (relative to src/PredictionApp/, where scripts run from)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")          # parquet caches
WF_DIR = os.path.join(DATA_DIR, "walkforward")       # per-fold predictions
