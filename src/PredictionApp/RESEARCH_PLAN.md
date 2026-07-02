# Microvest Research Plan & Decisions Log

This file is the "why" behind the pipeline. `config.py` is the "what".
Bump `CONFIG_VERSION` in config.py whenever labels/features/universe change.

## Goal

Predict whether a stock will make a **significant move of its own** (not just
riding the market) over the next 5 trading days — and evaluate that prediction
honestly enough that improvements are real, not luck.

## The v2 target (the core change)

Old (v1): `label = fwd 5d return >= +5% / <= -5% / else NoTrade`.
Problems: a flat 5% means different things for TSLA vs a utility; label mix
swings wildly between calm and volatile years; most "big moves" were really
the whole market moving.

New (v2):

```
fwd_excess = fwd_ret - beta_60d * mkt_fwd_ret     # the stock's OWN move
fwd_z      = fwd_excess / (vol_20d * sqrt(5))     # in units of its normal weekly wiggle
label      = Long  if fwd_z >= +1.0
             Short if fwd_z <= -1.0
             else NoTrade
```

- **beta-adjusted**: subtracts `beta x market move`, not market move 1-for-1,
  because high-beta names naturally amplify the market (TSLA ~2x, KO ~0.6x).
- **vol-normalized**: "significant" = moved > 1 sigma of its own weekly range.
  Same meaning for every stock, in every year.
- The continuous `fwd_z` is kept alongside the class label; the IC metric
  grades the model against it (train and grade on the same thing).
- Because the label only needs ONE market series (not a live cross-section),
  a single ticker can still be scored in isolation.

## Evaluation rules (Phase 0)

1. **Walk-forward**: expanding-window folds, one test year each (2019–2025),
   defined in `config.WALK_FORWARD_FOLDS`. Early stopping uses the last year
   of *train* as validation — the test year influences nothing.
2. **Purging**: any row whose 5-day label window crosses a train/val/test
   boundary is dropped.
3. **The referee is `scorecard.py`.** A change is good only if it improves the
   average fold IC AND helps in most folds.
4. Per-fold predictions are saved (parquet) so runs can be re-graded/compared
   without retraining.

Deliberately skipped for now (user choice): transaction-cost modeling (step 2),
LightGBM baseline (step 4). Revisit both — costs before trusting any Sharpe,
LightGBM as the likely-stronger backbone.

## Features (v2 = 30)

Added in v2 (documented, decades-robust factors):
- `ret_2d` — short-term reversal input
- `mom_12_1` — past-year return skipping the last month (classic momentum)
- `idio_vol_60d` — volatility of the stock's market-residual returns
  (documented negative predictor)
- `xs_rank_mom_12_1` — momentum rank vs. peers

Not yet added: sector-relative ranks (needs a sector mapping), earnings-date
proximity (needs an earnings calendar).

## Data

- Universe: current S&P 500 constituents (Wikipedia, cached). Fallback:
  the original 100 mega-caps (`legacy100`).
- History: 2004-06 onward (features need ~1yr warm-up, so usable rows start
  ~2005-06). Covers 2008, 2015, 2020, 2022 regimes.
- **Survivorship bias**: both universes are TODAY'S members — training data
  contains only companies that survived. All results are inflated by this;
  treat numbers as model-vs-model comparisons, not absolute expectations.
  Real fix needs point-in-time constituents (paid data).
- Everything cached to parquet under `Data/cache/` (`data_store.py`); keys
  include CONFIG_VERSION, so bumping the version invalidates stale caches.

## How to run

```bash
# from src/PredictionApp/
python walkforward.py                      # full walk-forward (slow, honest)
python walkforward.py --universe legacy100 --folds 2022 --epochs 10   # quick pass
python scorecard.py Data/walkforward/run_<timestamp>                  # grade a run
python scorecard.py --ckpt checkpoints/<run>/best-....ckpt            # grade old ckpt
python preprocess.py                       # legacy single-split tensors (app/training)
python lightning_train.py                  # legacy single-split training
```

## Compatibility notes

- Old checkpoints (26 features) can't score v2 panels (30 features).
  `scorecard.py --ckpt` still works for them ONLY against a v1-shaped panel;
  practically: retrain under v2 and compare scorecards.
- The Streamlit app follows `preprocess.FEATURE_COLS` automatically, so it
  works with v2 checkpoints once one is trained; `true_class` in the app now
  means the z-based label.

## What "good" looks like

- Avg fold IC (vs fwd_z) of +0.02 is real; +0.04 is solid; +0.06+ is strong.
- IC positive in >= 5 of 7 folds.
- Decile spread positive and roughly monotone.
- Don't chase F1/accuracy — class balance makes them misleading here.
