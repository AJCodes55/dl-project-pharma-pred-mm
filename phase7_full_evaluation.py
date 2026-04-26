#!/usr/bin/env python3
"""
Phase 7 full evaluation and analysis across ablations and baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METRIC_COLS = [
    "cumulative_return_pct",
    "ann_sharpe",
    "ann_sortino",
    "max_drawdown_pct",
    "win_rate",
    "alpha_vs_xph_ann_pct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7 full evaluation and analysis")
    p.add_argument("--project-root", default="", help="Optional project root")
    p.add_argument("--results-root", default="results")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--fda-events-path", default="All data DL project/fda_event_calendar.csv")
    p.add_argument("--output-dir", default="results/phase7_analysis")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--case-study-count", type=int, default=5)
    return p.parse_args()


def resolve_path(project_root: Path | None, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute() or project_root is None:
        return p
    return project_root / p


def annualized_sharpe(daily_ret: pd.Series) -> float:
    s = float(daily_ret.std(ddof=1))
    if s <= 1e-12:
        return 0.0
    return float((daily_ret.mean() / s) * np.sqrt(252.0))


def bootstrap_sharpe_ci(
    daily_ret: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    arr = np.asarray(daily_ret, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return {"boot_mean_sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    n = arr.size
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        s = float(sample.std(ddof=1))
        if s <= 1e-12:
            stats[i] = 0.0
        else:
            stats[i] = float((sample.mean() / s) * np.sqrt(252.0))
    return {
        "boot_mean_sharpe": float(np.mean(stats)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
    }


def load_metrics_rows(results_root: Path) -> pd.DataFrame:
    entries = [
        ("phase3_price_only/phase3_price_only_metrics.csv", "PPO_PriceOnly"),
        ("phase4_price_sentiment/phase4_price_sentiment_metrics.csv", "PPO_PriceSentiment"),
        ("phase5_basic_fda/phase5_fda_metrics.csv", "PPO_Phase5_BasicFDA"),
        ("phase5_rich_fda/phase5_fda_metrics.csv", "PPO_Phase5_RichFDA"),
        ("phase5_rich_fda_ct/phase5_fda_metrics.csv", "PPO_Phase5_RichFDA_CT"),
        ("phase6_sec/phase6_sec_metrics.csv", "PPO_Phase6_SEC"),
    ]
    rows = []
    baseline_rows = None
    for rel, strat in entries:
        p = results_root / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        row = df[df["strategy"] == strat].head(1)
        if not row.empty:
            rows.append(row.iloc[0].to_dict())
        if baseline_rows is None:
            b = df[df["strategy"].isin(["BuyHold_Equal", "EqualWeight_Monthly", "Momentum_20D", "XPH"])]
            if not b.empty:
                baseline_rows = b.copy()
    out = pd.DataFrame(rows)
    if baseline_rows is not None:
        out = pd.concat([out, baseline_rows], ignore_index=True)
    if out.empty:
        raise RuntimeError("No metrics rows loaded for Phase 7.")
    keep = ["strategy"] + METRIC_COLS
    out = out[keep].drop_duplicates(subset=["strategy"], keep="first").reset_index(drop=True)
    return out


def _read_curve(path: Path, ppo_col: str, strategy_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if ppo_col not in df.columns:
        raise ValueError(f"Expected column `{ppo_col}` in {path}")
    out = df[["date", ppo_col, "buy_hold_equal", "equal_weight_monthly", "momentum_20d", "xph"]].copy()
    out = out.rename(
        columns={
            ppo_col: strategy_col,
            "buy_hold_equal": "BuyHold_Equal",
            "equal_weight_monthly": "EqualWeight_Monthly",
            "momentum_20d": "Momentum_20D",
            "xph": "XPH",
        }
    )
    return out


def load_master_equity_curve(results_root: Path) -> pd.DataFrame:
    parts = []
    p3 = results_root / "phase3_price_only/phase3_price_only_equity_curve.csv"
    if p3.exists():
        parts.append(_read_curve(p3, "ppo_price_only", "PPO_PriceOnly"))
    p4 = results_root / "phase4_price_sentiment/phase4_price_sentiment_equity_curve.csv"
    if p4.exists():
        parts.append(_read_curve(p4, "ppo_price_sentiment", "PPO_PriceSentiment"))
    p5b = results_root / "phase5_basic_fda/phase5_fda_equity_curve.csv"
    if p5b.exists():
        parts.append(_read_curve(p5b, "ppo_phase5_fda", "PPO_Phase5_BasicFDA"))
    p5r = results_root / "phase5_rich_fda/phase5_fda_equity_curve.csv"
    if p5r.exists():
        parts.append(_read_curve(p5r, "ppo_phase5_fda", "PPO_Phase5_RichFDA"))
    p5c = results_root / "phase5_rich_fda_ct/phase5_fda_equity_curve.csv"
    if p5c.exists():
        parts.append(_read_curve(p5c, "ppo_phase5_fda", "PPO_Phase5_RichFDA_CT"))
    p6 = results_root / "phase6_sec/phase6_sec_equity_curve.csv"
    if p6.exists():
        parts.append(_read_curve(p6, "ppo_phase6_sec", "PPO_Phase6_SEC"))
    if not parts:
        raise RuntimeError("No equity curve files found for Phase 7.")

    merged = parts[0]
    for df in parts[1:]:
        merged = merged.merge(df, on=["date", "BuyHold_Equal", "EqualWeight_Monthly", "Momentum_20D", "XPH"], how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def compute_daily_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    out = equity_df.copy()
    val_cols = [c for c in out.columns if c != "date"]
    for c in val_cols:
        out[c] = out[c].astype(float).pct_change().fillna(0.0)
    return out


def build_bootstrap_table(daily_ret: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, c in enumerate([x for x in daily_ret.columns if x != "date"]):
        ci = bootstrap_sharpe_ci(daily_ret[c], n_bootstrap=n_bootstrap, seed=seed + i)
        rows.append({"strategy": c, "ann_sharpe_point": annualized_sharpe(daily_ret[c]), **ci})
    ci_df = pd.DataFrame(rows).sort_values("ann_sharpe_point", ascending=False).reset_index(drop=True)

    # Pairwise Sharpe deltas using aligned bootstrap draws.
    pairs = [("PPO_Phase6_SEC", "PPO_Phase5_BasicFDA"), ("PPO_Phase6_SEC", "PPO_PriceSentiment"), ("PPO_Phase6_SEC", "XPH")]
    rng = np.random.default_rng(seed + 999)
    deltas = []
    for a, b in pairs:
        if a not in daily_ret.columns or b not in daily_ret.columns:
            continue
        xa = np.asarray(daily_ret[a], dtype=np.float64)
        xb = np.asarray(daily_ret[b], dtype=np.float64)
        n = min(len(xa), len(xb))
        if n < 10:
            continue
        xa = xa[:n]
        xb = xb[:n]
        vals = np.empty(n_bootstrap, dtype=np.float64)
        for k in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            sa = annualized_sharpe(pd.Series(xa[idx]))
            sb = annualized_sharpe(pd.Series(xb[idx]))
            vals[k] = sa - sb
        deltas.append(
            {
                "strategy_a": a,
                "strategy_b": b,
                "delta_ann_sharpe_point": annualized_sharpe(daily_ret[a]) - annualized_sharpe(daily_ret[b]),
                "delta_ci_low": float(np.quantile(vals, 0.025)),
                "delta_ci_high": float(np.quantile(vals, 0.975)),
            }
        )
    delta_df = pd.DataFrame(deltas)
    return ci_df, delta_df


def build_interpretability_summary(results_root: Path) -> pd.DataFrame:
    rows = []
    phase5_files = [
        ("phase5_basic_fda/phase5_fda_interpretability.csv", "PPO_Phase5_BasicFDA"),
        ("phase5_rich_fda/phase5_fda_interpretability.csv", "PPO_Phase5_RichFDA"),
        ("phase5_rich_fda_ct/phase5_fda_interpretability.csv", "PPO_Phase5_RichFDA_CT"),
    ]
    for rel, strat in phase5_files:
        p = results_root / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "days_bucket" in df.columns:
            df = df.rename(columns={"days_bucket": "bucket"})
        df["source_strategy"] = strat
        df["signal_type"] = "fda_proximity"
        rows.append(df)

    p6 = results_root / "phase6_sec/phase6_sec_interpretability.csv"
    if p6.exists():
        df = pd.read_csv(p6)
        if "filing_recency_bucket" in df.columns:
            df = df.rename(columns={"filing_recency_bucket": "bucket"})
        df["source_strategy"] = "PPO_Phase6_SEC"
        df["signal_type"] = "sec_filing_recency"
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    wanted = ["source_strategy", "signal_type", "bucket", "mean_action", "buy_ratio", "hold_ratio", "sell_ratio", "n"]
    return out[[c for c in wanted if c in out.columns]]


def build_case_studies(
    fda_events_path: Path,
    feature_cfg_path: Path,
    equity_df: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    if not fda_events_path.exists():
        return pd.DataFrame()
    cfg = json.loads(feature_cfg_path.read_text())
    test_start = pd.Timestamp(cfg["test_start"]).normalize()
    test_end = pd.Timestamp(cfg["test_end"]).normalize()

    ev = pd.read_csv(fda_events_path, parse_dates=["date"])
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev = ev[(ev["date"] >= test_start) & (ev["date"] <= test_end)].copy()
    if ev.empty:
        return pd.DataFrame()
    score_col = "px_5d_ret_pct" if "px_5d_ret_pct" in ev.columns else None
    if score_col is not None:
        ev["_abs_score"] = ev[score_col].abs()
        ev = ev.sort_values("_abs_score", ascending=False)
    else:
        ev = ev.sort_values("date")
    ev = ev.head(count).reset_index(drop=True)

    eq = equity_df.sort_values("date").reset_index(drop=True).copy()
    eq_cols = [c for c in eq.columns if c != "date"]
    rows = []
    for _, r in ev.iterrows():
        d = pd.Timestamp(r["date"])
        idx_candidates = np.where(eq["date"].values >= np.datetime64(d))[0]
        if len(idx_candidates) == 0:
            continue
        i0 = int(idx_candidates[0])
        i_start = max(0, i0 - 3)
        i_end = min(len(eq) - 1, i0 + 5)
        for s in eq_cols:
            v0 = float(eq.loc[i_start, s])
            v1 = float(eq.loc[i_end, s])
            if v0 <= 0:
                ret = np.nan
            else:
                ret = (v1 / v0 - 1.0) * 100.0
            rows.append(
                {
                    "event_date": d.date().isoformat(),
                    "ticker": r.get("ticker", ""),
                    "event_type": r.get("event_type", ""),
                    "outcome": r.get("outcome", ""),
                    "strategy": s,
                    "window_return_pct_t-3_to_t+5": ret,
                }
            )
    return pd.DataFrame(rows)


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_return"] = out["cumulative_return_pct"].rank(ascending=False, method="min")
    out["rank_sharpe"] = out["ann_sharpe"].rank(ascending=False, method="min")
    out["rank_sortino"] = out["ann_sortino"].rank(ascending=False, method="min")
    out["rank_drawdown"] = out["max_drawdown_pct"].rank(ascending=False, method="min")
    out["rank_alpha"] = out["alpha_vs_xph_ann_pct"].rank(ascending=False, method="min")
    out["rank_mean"] = out[["rank_return", "rank_sharpe", "rank_sortino", "rank_drawdown", "rank_alpha"]].mean(axis=1)
    out = out.sort_values(["rank_mean", "ann_sharpe"], ascending=[True, False]).reset_index(drop=True)
    return out


def write_summary_md(output_dir: Path, master_metrics: pd.DataFrame) -> None:
    ppo = master_metrics[master_metrics["strategy"].str.startswith("PPO_", na=False)].copy()
    if ppo.empty:
        top_line = "No PPO strategies found in master metrics."
    else:
        top = ppo.sort_values("ann_sharpe", ascending=False).iloc[0]
        top_line = (
            f"Top PPO by Sharpe: {top['strategy']} "
            f"(Sharpe {top['ann_sharpe']:.3f}, Return {top['cumulative_return_pct']:.2f}%, "
            f"MaxDD {top['max_drawdown_pct']:.2f}%)."
        )
    text = (
        "# Phase 7 Final Summary\n\n"
        "This folder contains consolidated evaluation artifacts across Phases 3-6 and classical baselines.\n\n"
        f"- {top_line}\n"
        "- See `phase7_master_metrics.csv` for cross-phase ranking.\n"
        "- See `phase7_bootstrap_sharpe_ci.csv` and `phase7_bootstrap_sharpe_delta_ci.csv` for statistical confidence.\n"
        "- See `phase7_interpretability_summary.csv` and `phase7_case_studies.csv` for behavior diagnostics.\n"
    )
    (output_dir / "phase7_final_summary.md").write_text(text)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else None
    results_root = resolve_path(project_root, args.results_root)
    feature_cfg_path = resolve_path(project_root, args.feature_config_path)
    fda_events_path = resolve_path(project_root, args.fda_events_path)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_metrics = load_metrics_rows(results_root)
    master_metrics = add_rank_columns(master_metrics)
    master_metrics.to_csv(output_dir / "phase7_master_metrics.csv", index=False)

    equity = load_master_equity_curve(results_root)
    equity.to_csv(output_dir / "phase7_master_equity_curve.csv", index=False)
    daily_ret = compute_daily_returns(equity)
    daily_ret.to_csv(output_dir / "phase7_master_daily_returns.csv", index=False)

    sharpe_ci, sharpe_delta_ci = build_bootstrap_table(
        daily_ret=daily_ret,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )
    sharpe_ci.to_csv(output_dir / "phase7_bootstrap_sharpe_ci.csv", index=False)
    sharpe_delta_ci.to_csv(output_dir / "phase7_bootstrap_sharpe_delta_ci.csv", index=False)

    interp = build_interpretability_summary(results_root)
    interp.to_csv(output_dir / "phase7_interpretability_summary.csv", index=False)

    case_df = build_case_studies(
        fda_events_path=fda_events_path,
        feature_cfg_path=feature_cfg_path,
        equity_df=equity,
        count=args.case_study_count,
    )
    case_df.to_csv(output_dir / "phase7_case_studies.csv", index=False)

    write_summary_md(output_dir, master_metrics=master_metrics)

    print("Saved Phase 7 artifacts:")
    for name in [
        "phase7_master_metrics.csv",
        "phase7_master_equity_curve.csv",
        "phase7_master_daily_returns.csv",
        "phase7_bootstrap_sharpe_ci.csv",
        "phase7_bootstrap_sharpe_delta_ci.csv",
        "phase7_interpretability_summary.csv",
        "phase7_case_studies.csv",
        "phase7_final_summary.md",
    ]:
        print(" -", output_dir / name)


if __name__ == "__main__":
    main()
