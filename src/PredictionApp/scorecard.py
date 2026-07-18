"""
The one referee for every experiment.

Grades a set of predictions per fold and overall, always on the same metrics,
always out-of-sample. A change to the model/features/labels "worked" only if
it improves the AVERAGE across folds — not just one lucky year.

Usage:
    # Grade a walk-forward run (the normal path)
    python scorecard.py Data/walkforward/run_20260701_120000

    # Grade a single legacy checkpoint over the post-TRAIN_END period
    python scorecard.py --ckpt checkpoints/<run>/best-model-....ckpt

Metric definitions (see metrics.py for the math):
    ic_z        pooled Spearman IC of signal vs fwd_z  — the trained target
    ic_ret      pooled Spearman IC of signal vs raw forward return
    xs_ic       mean daily cross-sectional IC (signal vs fwd_z across names)
    icir        annualized mean/std of the daily IC series (stability)
    decile      top-decile minus bottom-decile mean forward return
    hit         directional hit rate of confident calls (|signal| > 0.1)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import config
import metrics as M


def grade_frame(df: pd.DataFrame) -> dict:
    """All headline metrics for one prediction frame (one fold, or pooled)."""
    d = df.dropna(subset=["signal", "fwd_z", "fwd_ret"])
    if d.empty:
        return {}
    sig = d["signal"].values
    row = {
        "n": len(d),
        "ic_z": M.information_coefficient(sig, d["fwd_z"].values),
        "ic_ret": M.information_coefficient(sig, d["fwd_ret"].values),
        "decile": M.decile_spread(sig, d["fwd_ret"].values)["spread"],
        "hit": M.directional_hit_rate(sig, d["fwd_ret"].values, 0.1)["hit_rate"],
    }
    daily = M.cross_sectional_ic(d["date"].values, sig, d["fwd_z"].values)
    ic_stats = M.icir(daily)
    row["xs_ic"] = ic_stats["ic_mean"]
    row["icir"] = ic_stats["icir"]
    row["ic_days_pos"] = ic_stats["ic_hit_rate"]
    return row


def grade_frame_topk(df: pd.DataFrame, k: int = config.TOP_K) -> dict:
    """Concentrated-book view: edge/Sharpe/hit-rate on just the K highest-
    conviction longs and K lowest-conviction shorts per day (metrics.topk_return).

    This is the PRIMARY target for a <5-position swing strategy -- the pooled/
    cross-sectional numbers in grade_frame implicitly assume a diversified book
    (breadth = the full universe) and understate the bar a concentrated trader
    actually needs to clear (breadth = 2*k). See config.TOP_K.
    """
    d = df.dropna(subset=["signal", "fwd_ret"])
    if d.empty:
        return {}
    stats = M.topk_return(d["date"].values, d["signal"].values, d["fwd_ret"].values, k=k)
    return {"n": len(d), "topk_edge": stats["edge_mean"],
            "topk_sharpe": stats["edge_sharpe"], "topk_hit": stats["hit_rate"],
            "topk_days": stats["n_days"]}


def grade_frame_gated(df: pd.DataFrame, target_tpd: float = config.TOP_K) -> dict:
    """Concentrated CONFIDENCE-GATED view (the primary target): per fold, pick a
    |signal| threshold that averages ~target_tpd trades/day, then report the
    signed per-trade edge / Sharpe / hit rate on just those high-conviction
    positions (metrics.gated_return). This beats forced top-K because it trades
    nothing on low-conviction days instead of jamming noise into the book.
    """
    d = df.dropna(subset=["signal", "fwd_ret"])
    if d.empty:
        return {}
    n_days = pd.Series(d["date"]).nunique()
    frac = min(1.0, target_tpd * n_days / len(d))     # share of rows to trade
    thr = float(np.quantile(np.abs(d["signal"].values), 1.0 - frac))
    s = M.gated_return(d["date"].values, d["signal"].values, d["fwd_ret"].values, threshold=thr)
    return {"n": len(d), "thr": thr, "trades_day": s["trades_per_day"],
            "edge": s["edge_mean"], "sharpe": s["edge_sharpe"], "hit": s["hit_rate"]}


def print_table(rows: dict, fmt: dict, cols: list):
    table = pd.DataFrame(rows).T
    table.index.name = "fold"
    cols = [c for c in cols if c in table.columns]
    table = table[cols]
    out = table.copy()
    for c, f in fmt.items():
        if c in out.columns:
            out[c] = out[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "—")
    print(out.to_string())
    return table


def grade_run(run_dir: str):
    paths = sorted(glob.glob(os.path.join(run_dir, "fold_*.parquet")))
    if not paths:
        raise SystemExit(f"No fold_*.parquet files in {run_dir}")

    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Run: {run_dir}")
        print(f"  config v{meta.get('config_version')} · universe {meta.get('universe')} "
              f"({meta.get('n_tickers')} names) · z_thr {meta.get('z_threshold')} · "
              f"H {meta.get('horizon')}d\n")

    frames = [pd.read_parquet(p) for p in paths]
    rows, gated_rows = {}, {}
    for df in frames:
        name = df["fold"].iloc[0]
        rows[name] = grade_frame(df)
        gated_rows[name] = grade_frame_gated(df)

    pooled = pd.concat(frames, ignore_index=True)
    rows["ALL"] = grade_frame(pooled)
    gated_rows["ALL"] = grade_frame_gated(pooled)

    ic_fmt = {"n": "{:,.0f}", "ic_z": "{:+.4f}", "ic_ret": "{:+.4f}",
             "xs_ic": "{:+.4f}", "icir": "{:+.2f}", "ic_days_pos": "{:.1%}",
             "decile": "{:+.4f}", "hit": "{:.1%}"}
    ic_cols = ["n", "ic_z", "ic_ret", "xs_ic", "icir", "ic_days_pos", "decile", "hit"]
    gated_fmt = {"n": "{:,.0f}", "thr": "{:.3f}", "trades_day": "{:.1f}",
                "edge": "{:+.4f}", "sharpe": "{:+.2f}", "hit": "{:.1%}"}
    gated_cols = ["n", "thr", "trades_day", "edge", "sharpe", "hit"]

    print(f"=== Concentrated confidence-gated scorecard — the real target "
          f"(~{config.TOP_K} trades/day, all out-of-sample) ===")
    gated_table = print_table(gated_rows, gated_fmt, gated_cols)

    print("\n=== Diversified/cross-sectional scorecard — diagnostic only "
          "(assumes trading the whole ranked universe) ===")
    table = print_table(rows, ic_fmt, ic_cols)

    per_fold = gated_table.drop(index="ALL", errors="ignore")
    if len(per_fold) >= 1:
        mean_sharpe = per_fold["sharpe"].mean()
        mean_edge = per_fold["edge"].mean()
        mean_hit = per_fold["hit"].mean()
        n_pos = int((per_fold["edge"] > 0).sum())
        print(f"\nAvg fold gated edge/trade: {mean_edge:+.4f} · Sharpe: {mean_sharpe:+.2f} "
              f"· hit: {mean_hit:.1%} · edge positive in {n_pos}/{len(per_fold)} folds")
        print("This is the bar for a concentrated swing book: only high-conviction "
              "days trade, so it reflects what you'd actually hold.")


def grade_checkpoint(ckpt_path: str, universe: str):
    """Score one checkpoint over the honest OOS window (date > TRAIN_END)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "streamlit_app"))
    import engine  # noqa: E402
    import data_store  # noqa: E402
    import preprocess as pp  # noqa: E402

    model, info = engine.load_model(ckpt_path)
    print(f"Checkpoint: {ckpt_path}")
    print(f"  arch: {info['layers']}L h{info['hidden']} {info['input']}f\n")

    tickers = data_store.resolve_universe(universe)
    panel = pp.build_panel(tickers)
    preds = engine.predict(panel, model)
    oos = preds[preds["date"] > pd.Timestamp(pp.TRAIN_END)].copy()

    ic_fmt = {"n": "{:,.0f}", "ic_z": "{:+.4f}", "ic_ret": "{:+.4f}",
             "xs_ic": "{:+.4f}", "icir": "{:+.2f}", "ic_days_pos": "{:.1%}",
             "decile": "{:+.4f}", "hit": "{:.1%}"}
    ic_cols = ["n", "ic_z", "ic_ret", "xs_ic", "icir", "ic_days_pos", "decile", "hit"]
    gated_fmt = {"n": "{:,.0f}", "thr": "{:.3f}", "trades_day": "{:.1f}",
                "edge": "{:+.4f}", "sharpe": "{:+.2f}", "hit": "{:.1%}"}
    gated_cols = ["n", "thr", "trades_day", "edge", "sharpe", "hit"]

    print(f"=== Concentrated confidence-gated OOS scorecard (date > {pp.TRAIN_END}) ===")
    print_table({"OOS": grade_frame_gated(oos)}, gated_fmt, gated_cols)
    print(f"\n=== Diversified/cross-sectional OOS scorecard — diagnostic only ===")
    print_table({"OOS": grade_frame(oos)}, ic_fmt, ic_cols)


def main():
    ap = argparse.ArgumentParser(description="Grade predictions (walk-forward run or checkpoint)")
    ap.add_argument("run_dir", nargs="?", help="a Data/walkforward/run_* directory")
    ap.add_argument("--ckpt", help="grade a single checkpoint instead")
    ap.add_argument("--universe", default="legacy100", choices=["sp500", "legacy100"],
                    help="universe for --ckpt mode")
    args = ap.parse_args()

    if args.ckpt:
        grade_checkpoint(args.ckpt, args.universe)
    elif args.run_dir:
        grade_run(args.run_dir)
    else:
        ap.error("Give a run directory or --ckpt <path>")


if __name__ == "__main__":
    main()
