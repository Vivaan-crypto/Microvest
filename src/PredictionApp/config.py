"""Single source of truth for every research knob.

Cached artifacts are stamped with CONFIG_VERSION; bump it whenever a change
alters the *meaning* of the data (labels, features, horizon, universe) so stale
caches can't feed a new experiment.

v1: fixed +/-5% labels, 26 features, 100 tickers, single split.
v2: beta-adjusted vol-normalized z-score labels, factor features, S&P 500,
    history from 2005, walk-forward eval.
v3: label-tail hygiene (VOL_FLOOR, Z_CLIP) + concentrated top-K grading.
"""

import os

import torch

CONFIG_VERSION = 3

# Horizon / windowing
HORIZON = 5  # forward-return horizon (trading days)
WINDOW = 20  # sequence length fed to the model

# Label (v2): beta-adjusted, vol-normalized z-score.
#   fwd_z = (fwd_ret - beta_60d*mkt_fwd_ret) / (vol_20d * sqrt(HORIZON))
#   label = Long if fwd_z >= +Z_THRESHOLD, Short if <= -Z_THRESHOLD, else NoTrade
Z_THRESHOLD = 1.0
VOL_WINDOW = 20
BETA_WINDOW = 60

# Label-tail hygiene (v3): floor the vol denominator and winsorize fwd_z so
# tiny-vol names and earnings gaps stop defining the class edges / regression target.
VOL_FLOOR = 0.005  # floor on daily vol_20d (~0.5%/day)
Z_CLIP = 5.0  # winsorize fwd_z to +/- this many sigmas

# Concentrated swing strategy (< 5 positions): breadth is fixed at 2*TOP_K/day,
# so scorecard.py's topk_* columns are the target, not pooled cross-sectional IC.
TOP_K = 3

# History / universe
START_DATE = "2004-06-01"  # pre-2005 history so 252d features warm up
END_DATE = "2026-01-01"
TRAIN_END = "2021-08-16"  # legacy single-split boundary (kept for the app)

# "sp500" -> current S&P 500 constituents (cached), "legacy100" -> hand-picked 100.
# Survivorship bias: both are *today's* members, so backtest numbers are inflated;
# treat all results as relative model comparisons, not absolute expected returns.
UNIVERSE_NAME = "sp500"

# Walk-forward folds: expanding window, train <= train_end (purge HORIZON rows at
# the boundary), test the following calendar year.
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

# Model size. Tiny on purpose: the target's edge is ~0.03 IC, so capacity mostly
# buys memorization, and CPU step time scales with it (256x3 = 379ms/step vs 13ms).
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.2

# Walk-forward training defaults (per fold). Kept modest: 7 folds retrain from scratch.
SEED = 42  # per-run determinism; fold-to-fold comparisons need it
WF_MAX_EPOCHS = 40
WF_PATIENCE = 8
WF_BATCH_SIZE = 256
WF_LR = 1e-3
WEIGHT_DECAY = 1e-3  # weight matrices only (not biases/LayerNorm)

# DataLoader workers default to 0 on Windows: spawn pickles a full copy of the
# fold's tensors into every worker (a multi-minute stall that looks like a hang),
# and __getitem__ is a trivial in-memory slice, so workers cost more than they save.
WF_NUM_WORKERS = 0
DATALOADER_WORKERS = 0 if os.name == "nt" else 15

# Compute device: auto-detect CUDA, else CPU. Force CPU with MICROVEST_FORCE_CPU=1.
FORCE_CPU = os.environ.get("MICROVEST_FORCE_CPU", "0") == "1"
CUDA_AVAILABLE = torch.cuda.is_available() and not FORCE_CPU

ACCELERATOR = "gpu" if CUDA_AVAILABLE else "cpu"
DEVICES = 1  # single-GPU; "auto" would trigger DDP on multi-GPU

# fp32 by default (bit-comparable across devices). MICROVEST_MIXED=1 trades a little
# numerical noise for ~1.5-2x on Ampere+ via bf16 (no loss-scaling headaches).
PRECISION = (
    "bf16-mixed"
    if os.environ.get("MICROVEST_MIXED", "0") == "1" and CUDA_AVAILABLE
    else "32-true"
)

PIN_MEMORY = CUDA_AVAILABLE  # overlaps host->GPU copies; pure overhead on CPU

# Single-split trainer batch. Small batches leave the GPU launch-overhead bound
# (bs=32 -> 81s/epoch, bs=1024 -> 3.9s on the RTX 5060 Ti); keep 32 on CPU.
TRAIN_BATCH_SIZE = 1024 if CUDA_AVAILABLE else 32

if CUDA_AVAILABLE:
    torch.set_float32_matmul_precision("high")  # TF32 matmuls: free Ampere+ speedup


def torch_device() -> torch.device:
    """Device for manual (non-Lightning) paths: walk-forward scoring, app inference."""
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")


# Paths (relative to src/PredictionApp/, where scripts run from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")  # parquet caches
WF_DIR = os.path.join(DATA_DIR, "walkforward")  # per-fold predictions
