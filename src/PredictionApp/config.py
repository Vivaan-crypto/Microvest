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
v3: label-tail hygiene -- floor the vol denominator (VOL_FLOOR) and winsorize
    fwd_z (Z_CLIP) so tiny-vol names / earnings gaps stop defining the class
    edges and the regression target. Concentrated (top-K) grading added.
"""

import os

CONFIG_VERSION = 3

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

# Label-tail hygiene (v3). Without these, tiny vol_20d denominators (quiet names
# / stale prices) explode fwd_z to |z| ~ 200+ on micro-moves, and earnings gaps
# add a few |z| ~ 20 rows -- both corrupt the class edges AND dominate the
# regression target / the extreme picks a concentrated book depends on.
VOL_FLOOR = 0.005    # floor on daily vol_20d in the label denominator (~0.5%/day)
Z_CLIP = 5.0         # winsorize fwd_z to +/- this many sigmas (p99 is ~2.8)

# ------------------------------------------------------------------
# Portfolio construction: this is a CONCENTRATED swing-trading strategy
# (< 5 positions at a time), not a diversified cross-sectional book. Breadth
# is fixed at 2*TOP_K per day regardless of universe size, so the pooled/
# cross-sectional IC (which implicitly assumes a broad, diversified book) is
# NOT the primary target -- scorecard.py's topk_* columns (edge/Sharpe/hit
# rate on just the K highest-conviction longs and K lowest-conviction shorts
# per day) are. See metrics.topk_return.
# ------------------------------------------------------------------
TOP_K = 3

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
# Model size. Deliberately tiny: the target's edge is ~0.03 IC, so capacity
# mostly buys memorization -- and CPU step time scales with it. Measured on
# this box: 256x3 = 379 ms/step (~35 h/fold on sp500); 64-hidden = 13 ms/step.
# ------------------------------------------------------------------
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.2

# ------------------------------------------------------------------
# Training defaults for walk-forward (per fold). Kept modest: 7 folds retrain
# from scratch, so per-fold cost matters. EarlyStopping on val IC cuts it more.
# ------------------------------------------------------------------
SEED = 42            # per-run determinism; fold-to-fold comparisons need it
WF_MAX_EPOCHS = 40
WF_PATIENCE = 8
WF_BATCH_SIZE = 256
WF_LR = 1e-3
WEIGHT_DECAY = 1e-3  # applied to weight matrices only (not biases/LayerNorm)
# Windows: DataLoader workers spawn full process copies of the fold's feature
# arrays (hundreds of MB each on sp500) and __getitem__ is a trivial slice, so
# workers cost more than they save. 0 = load in the main process.
WF_NUM_WORKERS = 0

# ------------------------------------------------------------------
# Paths (relative to src/PredictionApp/, where scripts run from)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")          # parquet caches
WF_DIR = os.path.join(DATA_DIR, "walkforward")       # per-fold predictions
