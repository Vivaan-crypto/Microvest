# Microvest — Model Viewer (Streamlit)

A small dashboard over the trained LSTM signal.

## Run

From `src/PredictionApp/`:

```bash
streamlit run streamlit_app/Home.py
```

(Uses the same venv as training — `streamlit`, `plotly`, `yfinance` are already installed.)

## Pages

- **Home** — headline ranking metrics + predicted-direction mix for the selected window.
- **🎯 Classification** — pick a ticker, see its price chart colored by the model's
  predicted direction, the latest call + probabilities, and a **manual backtest**
  (accuracy, per-class accuracy, confusion matrix, predictions-vs-outcomes) over the
  window up to your **as-of** date.
- **🏆 Ranking** — the cross-sectional view that actually matters: as-of leaderboard
  (top = longs, bottom = shorts), IC / ICIR / decile spread / long-short Sharpe,
  decile-return bars, a long-short equity curve, and a rolling-IC stability chart.

## How it works

`engine.py` reuses the training feature pipeline (`preprocess.py`) so the model sees
exactly what it was trained on, then loads any checkpoint (architecture is derived
from the weights). Data is fetched once to *today* and cached; the **as-of** date is a
pure, leak-free view filter (all features are causal, scaler fit ≤ TRAIN_END), so
backtesting different cutoffs is instant after the first load.

## Notes

- First load downloads + featurizes the whole universe (cached for an hour). Start
  with a small universe in the sidebar for speed, then widen it.
- Read the model as a **ranking** signal (IC ≈ 0.05), not a per-stock oracle.
- Backtest numbers are gross, overlapping-horizon, and exclude costs.
