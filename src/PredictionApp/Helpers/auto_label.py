import pandas as pd
import numpy as np

INPUT = "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/test.csv"
OUTPUT = "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/test.csv"

# ============================================================
# LABELING STRATEGY CONFIGURATION
# ============================================================
HORIZON = 5          # Days ahead to predict

# --- Choose ONE labeling mode ---
MODE = "binary_threshold"   # Options: "binary_threshold", "ternary_threshold", "quantile"

# For "binary_threshold" and "ternary_threshold":
# The threshold below determines Up/Down cutoff
# 7% was too strict → 91% Flat. Use 1-2% for balanced classes.
THRESHOLD = 0.01     # 1% move over HORIZON days → Up or Down

# For "quantile" mode: top/bottom X% of returns become Up/Down, rest Flat
QUANTILE_PCT = 0.33  # Top 33% = Up, Bottom 33% = Down, Middle 34% = Flat


def label_ticker_binary(group, horizon, threshold):
    """
    Binary: Up (1) vs Down (0). Flat samples excluded at filtering step.
    Use confidence of predicted probability as Flat signal at inference.
    """
    group = group.copy()
    future_close = group["Close"].shift(-horizon)
    ret = (future_close - group["Close"]) / group["Close"]

    group["Label"] = np.where(ret >= threshold, 1,
                     np.where(ret <= -threshold, 0, -2))  # -2 = exclude (flat)
    group.loc[group.index[-horizon:], "Label"] = -2  # Insufficient future data
    return group


def label_ticker_ternary(group, horizon, threshold):
    """
    3-class: Down (0), Flat (1), Up (2).
    Uses a sensible threshold (1-2%) rather than 7%.
    """
    group = group.copy()
    future_close = group["Close"].shift(-horizon)
    ret = (future_close - group["Close"]) / group["Close"]

    group["Label"] = np.where(ret >= threshold, 2,
                     np.where(ret <= -threshold, 0, 1))
    group.loc[group.index[-horizon:], "Label"] = -2
    return group


def label_ticker_quantile(group, horizon, quantile_pct):
    """
    Quantile-based: always produces balanced classes regardless of market conditions.
    Top quantile_pct = Up (2), Bottom quantile_pct = Down (0), Middle = Flat (1).
    Forces equal representation of Up and Down → no class imbalance.
    """
    group = group.copy()
    future_close = group["Close"].shift(-horizon)
    ret = (future_close - group["Close"]) / group["Close"]

    up_threshold = ret.quantile(1 - quantile_pct)
    down_threshold = ret.quantile(quantile_pct)

    group["Label"] = np.where(ret >= up_threshold, 2,
                     np.where(ret <= down_threshold, 0, 1))
    group.loc[group.index[-horizon:], "Label"] = -2
    return group


# ============================================================
# MAIN
# ============================================================
df = pd.read_csv(INPUT)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

if MODE == "binary_threshold":
    df = df.groupby("ticker", group_keys=False).apply(
        lambda g: label_ticker_binary(g, HORIZON, THRESHOLD)
    )
elif MODE == "ternary_threshold":
    df = df.groupby("ticker", group_keys=False).apply(
        lambda g: label_ticker_ternary(g, HORIZON, THRESHOLD)
    )
elif MODE == "quantile":
    df = df.groupby("ticker", group_keys=False).apply(
        lambda g: label_ticker_quantile(g, HORIZON, QUANTILE_PCT)
    )
else:
    raise ValueError(f"Unknown MODE: {MODE}")

# Remove samples with insufficient future data or flat (binary mode)
df = df[df["Label"] != -2].copy()
df["Label"] = df["Label"].astype(int)

df.to_csv(OUTPUT, index=False)
print(f"Mode: {MODE}")
print(f"Done. Saved to {OUTPUT}")
print(f"Label distribution:\n{df['Label'].value_counts()}")
print(f"Class balance: {dict(df['Label'].value_counts(normalize=True).round(3))}")
