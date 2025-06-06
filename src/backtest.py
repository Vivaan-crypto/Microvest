import numpy as np
import pandas as pd


def backtest(df, model, scaler, confidence_threshold=0.7):
    """
    Enhanced backtesting function with multiple performance metrics
    """
    # Get features from the model
    _, _, features = model

    # Prepare data
    X = df[features]
    y = (df["close"].shift(-3) / df["close"] - 1 > 0.005).astype(
        int
    )  # Same target as training

    # Scale features and get predictions
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs > confidence_threshold).astype(int)

    # Only consider times when we actually took a position
    positions = preds == 1
    n_trades = positions.sum()

    if n_trades == 0:
        return 0.0, 1.0  # No trades made

    # Calculate returns for our positions
    entry_prices = df["close"][positions]
    exit_prices = df["close"].shift(-3)[positions]
    trade_returns = (exit_prices / entry_prices - 1).values

    # Performance metrics
    accuracy = (preds[positions] == y[positions]).mean()
    cumulative_return = (1 + trade_returns).prod()
    win_rate = (trade_returns > 0).mean()
    avg_win = trade_returns[trade_returns > 0].mean()
    avg_loss = trade_returns[trade_returns <= 0].mean()
    profit_factor = (
        -avg_win * (win_rate) / (avg_loss * (1 - win_rate)) if avg_loss != 0 else np.inf
    )

    # Additional metrics we might want to display
    max_drawdown = 1 - (1 + trade_returns).cum
