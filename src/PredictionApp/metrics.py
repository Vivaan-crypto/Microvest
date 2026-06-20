"""
Signal-quality metrics for a directional classifier.

The models here do NOT predict a % price change -- they predict a class
(Short / NoTrade / Long). Classic quant metrics like the Information Coefficient
need a *continuous* prediction to correlate against the realized forward return,
so we derive one from how confident the model is:

    signal = P(Long) - P(Short)      in [-1, 1]

A confidently-long row scores ~ +1, a confidently-short row ~ -1, and an
uncertain / NoTrade row ~ 0. Everything below operates on that signal, so it is
model-agnostic: pass the probabilities (or a precomputed signal) from the LSTM
or the Transformer -- they share the class order [Short, NoTrade, Long].

Definitions
-----------
IC   (Information Coefficient): correlation between signal and realized forward
     return. Rank IC (Spearman) is the quant standard; Pearson is also reported.
ICIR (IC Information Ratio): mean(IC_t) / std(IC_t) over a daily cross-sectional
     IC series, optionally annualized by sqrt(periods per year).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

TRADING_DAYS = 252


# ------------------------------------------------------------------
# Probabilities -> continuous signal
# ------------------------------------------------------------------
def signal_from_proba(proba: np.ndarray) -> np.ndarray:
    """Map class probabilities [P(Short), P(NoTrade), P(Long)] to P(Long)-P(Short)."""
    proba = np.asarray(proba)
    return proba[:, 2] - proba[:, 0]


# ------------------------------------------------------------------
# Pooled Information Coefficient
# ------------------------------------------------------------------
def information_coefficient(
    signal: np.ndarray, fwd_ret: np.ndarray, method: str = "spearman"
) -> float:
    """Pooled IC over all samples (one number).

    method: "spearman" (rank IC, robust, the default) or "pearson" (linear).
    """
    signal = np.asarray(signal, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    ok = np.isfinite(signal) & np.isfinite(fwd_ret)
    if ok.sum() < 3 or np.std(signal[ok]) == 0 or np.std(fwd_ret[ok]) == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(signal[ok], fwd_ret[ok])[0])
    return float(spearmanr(signal[ok], fwd_ret[ok])[0])


# ------------------------------------------------------------------
# Cross-sectional (daily) IC series + ICIR
# ------------------------------------------------------------------
def cross_sectional_ic(
    dates: np.ndarray,
    signal: np.ndarray,
    fwd_ret: np.ndarray,
    method: str = "spearman",
    min_names: int = 5,
) -> pd.Series:
    """Per-date IC across the ticker universe -> a time series indexed by date.

    On each date we correlate the signal against the forward return across all
    names trading that day. Needs a real cross-section (>= min_names tickers);
    dates with fewer names are skipped.
    """
    df = pd.DataFrame({"date": dates, "signal": signal, "fwd_ret": fwd_ret})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    out = {}
    for d, grp in df.groupby("date"):
        if len(grp) < min_names or grp["signal"].std() == 0 or grp["fwd_ret"].std() == 0:
            continue
        if method == "pearson":
            ic = pearsonr(grp["signal"], grp["fwd_ret"])[0]
        else:
            ic = spearmanr(grp["signal"], grp["fwd_ret"])[0]
        out[pd.Timestamp(d)] = ic
    return pd.Series(out, name=f"ic_{method}").sort_index()


def icir(daily_ic: pd.Series, annualize: bool = True,
         periods_per_year: int = TRADING_DAYS) -> Dict[str, float]:
    """Summarize a daily IC series: mean, std, hit rate, and ICIR.

    ICIR = mean(IC) / std(IC). Annualized by sqrt(periods_per_year) -- note this
    is optimistic for an overlapping-horizon daily series, so read it as relative.
    """
    ic = daily_ic.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ic) < 2 or ic.std() == 0:
        return {"ic_mean": float(ic.mean()) if len(ic) else float("nan"),
                "ic_std": float("nan"), "icir": float("nan"),
                "ic_hit_rate": float("nan"), "n_days": int(len(ic))}
    raw = ic.mean() / ic.std()
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir": float(raw * np.sqrt(periods_per_year)) if annualize else float(raw),
        "ic_hit_rate": float((ic > 0).mean()),  # fraction of days IC is positive
        "n_days": int(len(ic)),
    }


# ------------------------------------------------------------------
# Trading-flavored signal metrics
# ------------------------------------------------------------------
def directional_hit_rate(
    signal: np.ndarray, fwd_ret: np.ndarray, threshold: float = 0.0
) -> Dict[str, float]:
    """Of confidently-directional calls (|signal| > threshold), how often is the
    sign of the signal the same as the sign of the realized return?"""
    signal = np.asarray(signal, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    mask = np.isfinite(signal) & np.isfinite(fwd_ret) & (np.abs(signal) > threshold)
    n = int(mask.sum())
    if n == 0:
        return {"hit_rate": float("nan"), "coverage": 0.0, "n": 0}
    hits = np.sign(signal[mask]) == np.sign(fwd_ret[mask])
    return {
        "hit_rate": float(hits.mean()),
        "coverage": float(n / len(signal)),  # share of rows that cleared threshold
        "n": n,
    }


def decile_spread(
    signal: np.ndarray, fwd_ret: np.ndarray, n_buckets: int = 10
) -> Dict[str, float]:
    """Mean forward return of the top signal bucket minus the bottom bucket.

    A monotone, positive spread is the clearest sign the signal sorts winners
    from losers (the bread-and-butter quant sanity check)."""
    signal = np.asarray(signal, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    ok = np.isfinite(signal) & np.isfinite(fwd_ret)
    signal, fwd_ret = signal[ok], fwd_ret[ok]
    if len(signal) < n_buckets * 2 or np.std(signal) == 0:
        return {"top": float("nan"), "bottom": float("nan"), "spread": float("nan")}
    try:
        buckets = pd.qcut(signal, n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return {"top": float("nan"), "bottom": float("nan"), "spread": float("nan")}
    top = fwd_ret[buckets == np.nanmax(buckets)].mean()
    bottom = fwd_ret[buckets == np.nanmin(buckets)].mean()
    return {"top": float(top), "bottom": float(bottom), "spread": float(top - bottom)}


def long_short_sharpe(
    dates: np.ndarray, signal: np.ndarray, fwd_ret: np.ndarray,
    quantile: float = 0.2, periods_per_year: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Annualized Sharpe of a daily, dollar-neutral long-short book.

    Each date: long the top-`quantile` of names by signal, short the bottom,
    equal-weighted; the day's PnL is (mean long fwd_ret - mean short fwd_ret).
    Overlapping HORIZON-day returns make this optimistic -- treat as relative.
    """
    df = pd.DataFrame({"date": dates, "signal": signal, "fwd_ret": fwd_ret})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    pnl = {}
    for d, grp in df.groupby("date"):
        if len(grp) < 5 or grp["signal"].std() == 0:
            continue
        hi = grp["signal"].quantile(1 - quantile)
        lo = grp["signal"].quantile(quantile)
        longs = grp.loc[grp["signal"] >= hi, "fwd_ret"]
        shorts = grp.loc[grp["signal"] <= lo, "fwd_ret"]
        if len(longs) == 0 or len(shorts) == 0:
            continue
        pnl[pd.Timestamp(d)] = longs.mean() - shorts.mean()

    pnl = pd.Series(pnl).sort_index()
    if len(pnl) < 2 or pnl.std() == 0:
        return {"sharpe": float("nan"), "mean_pnl": float(pnl.mean()) if len(pnl) else float("nan"),
                "n_days": int(len(pnl))}
    return {
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)),
        "mean_pnl": float(pnl.mean()),
        "n_days": int(len(pnl)),
    }


# ------------------------------------------------------------------
# One-call report
# ------------------------------------------------------------------
def evaluate_signal(
    fwd_ret: np.ndarray,
    proba: Optional[np.ndarray] = None,
    signal: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    hit_threshold: float = 0.1,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute the full signal report from either probabilities or a raw signal.

    Pass `proba` ([N,3]) OR `signal` ([N]); pass `dates` ([N]) to unlock the
    cross-sectional ICIR and long-short Sharpe (they need a daily cross-section).
    """
    if signal is None:
        if proba is None:
            raise ValueError("Provide either `proba` or `signal`.")
        signal = signal_from_proba(proba)
    signal = np.asarray(signal, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)

    report: Dict[str, float] = {
        "ic_pearson": information_coefficient(signal, fwd_ret, "pearson"),
        "ic_spearman": information_coefficient(signal, fwd_ret, "spearman"),
    }
    report.update({f"hit_{k}": v for k, v in
                   directional_hit_rate(signal, fwd_ret, hit_threshold).items()})
    report.update({f"decile_{k}": v for k, v in
                   decile_spread(signal, fwd_ret).items()})

    if dates is not None:
        daily_ic = cross_sectional_ic(dates, signal, fwd_ret, "spearman")
        report.update(icir(daily_ic))
        report.update({f"ls_{k}": v for k, v in
                       long_short_sharpe(dates, signal, fwd_ret).items()})

    if verbose:
        _print_report(report, hit_threshold)
    return report


def _print_report(r: Dict[str, float], hit_threshold: float) -> None:
    def g(k):  # safe getter
        return r.get(k, float("nan"))

    print("\n=== Signal metrics (P(Long) - P(Short) vs. forward return) ===")
    print(f"  IC  (Pearson)        : {g('ic_pearson'):+.4f}")
    print(f"  IC  (Spearman/rank)  : {g('ic_spearman'):+.4f}")
    if "icir" in r:
        print(f"  Cross-sectional IC   : {g('ic_mean'):+.4f} "
              f"(std {g('ic_std'):.4f}, {int(g('n_days'))} days)")
        print(f"  ICIR (annualized)    : {g('icir'):+.3f}  "
              f"(IC>0 on {g('ic_hit_rate'):.1%} of days)")
    print(f"  Hit rate (|sig|>{hit_threshold:g}) : {g('hit_hit_rate'):.1%} "
          f"(coverage {g('hit_coverage'):.1%}, n={int(g('hit_n'))})")
    print(f"  Decile spread        : {g('decile_spread'):+.4f} "
          f"(top {g('decile_top'):+.4f} / bottom {g('decile_bottom'):+.4f})")
    if "ls_sharpe" in r:
        print(f"  Long-short Sharpe    : {g('ls_sharpe'):+.2f} "
              f"(mean daily PnL {g('ls_mean_pnl'):+.4f})")
    print("=" * 62)
