#!/usr/bin/env python3
"""
Phase 8 live demo runner for in-class presentation.

Loads the already-trained Phase 6 SEC PPO model and runs a deterministic
backtest replay on the prepared test split. Produces compact artifacts
for quick storytelling during a live demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from phase2_trading_env import DEFAULT_TICKERS
from phase5_fda_ppo import (
    action_distribution,
    annualized_sharpe,
    max_drawdown,
    run_ppo_backtest,
)
from phase6_multimodal_env import make_phase6_sequence_env_from_processed

# Keep explicit import so custom extractor class is available for PPO.load.
from phase6_multimodal_policy import Phase6MultimodalExtractor  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run class-friendly live demo replay for Phase 6 model")
    p.add_argument("--project-root", default=".", help="Project root path")
    p.add_argument("--model-path", default="results/phase6_sec/ppo_phase6_sec.zip")
    p.add_argument("--dataset-path", default="results/phase6_sec/_test_scaled.csv")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--sec-feature-config-path", default="results/phase6_sec/phase6_sec_feature_config.json")
    p.add_argument("--run-meta-path", default="results/phase6_sec/phase6_run_metadata.json")
    p.add_argument("--phase6-metrics-path", default="results/phase6_sec/phase6_sec_metrics.csv")
    p.add_argument("--out-dir", default="results/live_demo")
    p.add_argument("--seed", type=int, default=-1, help="If < 0, uses selected_seed from metadata")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip PNG equity chart creation (useful if matplotlib is unavailable)",
    )
    return p.parse_args()


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative).expanduser()
    return p if p.is_absolute() else project_root / p


def compute_demo_metrics(equity: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
    cumulative_return_pct = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
    return {
        "cumulative_return_pct": cumulative_return_pct,
        "ann_sharpe": annualized_sharpe(daily_returns),
        "max_drawdown_pct": float(max_drawdown(equity) * 100.0),
        "daily_volatility_pct": float(daily_returns.std(ddof=1) * np.sqrt(252.0) * 100.0),
    }


def load_baseline_phase6_row(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    row = df[df["strategy"] == "PPO_Phase6_SEC"].head(1)
    if row.empty:
        return {}
    cols = ["cumulative_return_pct", "ann_sharpe", "max_drawdown_pct"]
    return {f"baseline_{c}": float(row.iloc[0][c]) for c in cols}


def build_action_summary(actions_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    flat_summary = action_distribution(actions_df)
    rows = []
    for ticker in DEFAULT_TICKERS:
        col = f"act_{ticker}"
        if col not in actions_df.columns:
            continue
        vals = actions_df[col].to_numpy()
        rows.append(
            {
                "ticker": ticker,
                "sell_ratio": float((vals == 0).mean()),
                "hold_ratio": float((vals == 1).mean()),
                "buy_ratio": float((vals == 2).mean()),
            }
        )
    ticker_df = pd.DataFrame(rows)
    return ticker_df, flat_summary


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = resolve_path(project_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(project_root, args.model_path)
    dataset_path = resolve_path(project_root, args.dataset_path)
    feature_config_path = resolve_path(project_root, args.feature_config_path)
    sec_feature_config_path = resolve_path(project_root, args.sec_feature_config_path)
    run_meta_path = resolve_path(project_root, args.run_meta_path)
    phase6_metrics_path = resolve_path(project_root, args.phase6_metrics_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

    selected_seed = 42
    selected_window = 20
    if run_meta_path.exists():
        meta = json.loads(run_meta_path.read_text())
        selected_seed = int(meta.get("selected_seed", selected_seed))
        selected_window = int(meta.get("window_size", selected_window))
    seed = selected_seed if args.seed < 0 else int(args.seed)

    env = make_phase6_sequence_env_from_processed(
        dataset_path=dataset_path,
        feature_config_path=feature_config_path,
        sec_feature_config_path=sec_feature_config_path,
        window_size=selected_window,
        tickers=DEFAULT_TICKERS,
        initial_cash=1_000_000.0,
        transaction_cost=0.001,
        trade_fraction=0.10,
    )

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is required for live demo. Install with: pip install stable-baselines3"
        ) from exc

    model = PPO.load(str(model_path), device=args.device)
    perf_df, actions_df = run_ppo_backtest(env, model, seed=seed)

    perf_df["date"] = pd.to_datetime(perf_df["date"])
    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    equity = perf_df.set_index("date")["portfolio_value"]
    daily = equity.pct_change().fillna(0.0)

    demo_metrics = compute_demo_metrics(equity, daily)
    baseline_metrics = load_baseline_phase6_row(phase6_metrics_path)
    metrics_row = {"strategy": "PPO_Phase6_SEC_LiveDemoReplay", "seed": seed, **demo_metrics, **baseline_metrics}
    metrics_df = pd.DataFrame([metrics_row])

    ticker_action_df, flat_action_summary = build_action_summary(actions_df)
    action_totals_df = pd.DataFrame([{"scope": "all_assets", **flat_action_summary}])

    equity_plot_path = out_dir / "phase8_live_demo_equity.png"
    plot_written = False
    if not args.skip_plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 5))
            plt.plot(equity.index, equity.values, linewidth=2.0, label="PPO Phase 6 Replay")
            plt.title("Phase 6 Live Demo: Portfolio Value Replay")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(alpha=0.25)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(equity_plot_path, dpi=150)
            plt.close()
            plot_written = True
        except ModuleNotFoundError:
            print("WARNING: matplotlib not installed; skipping equity PNG.")

    perf_path = out_dir / "phase8_live_demo_perf.csv"
    actions_path = out_dir / "phase8_live_demo_actions.csv"
    metrics_path = out_dir / "phase8_live_demo_metrics.csv"
    action_by_ticker_path = out_dir / "phase8_live_demo_action_by_ticker.csv"
    action_totals_path = out_dir / "phase8_live_demo_action_totals.csv"

    perf_df.to_csv(perf_path, index=False)
    actions_df.to_csv(actions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    ticker_action_df.to_csv(action_by_ticker_path, index=False)
    action_totals_df.to_csv(action_totals_path, index=False)

    print("Live demo replay complete.")
    print(f"Seed used: {seed}")
    print(f"Window size: {selected_window}")
    print(f"Final portfolio value: ${equity.iloc[-1]:,.2f}")
    print(f"Cumulative return: {demo_metrics['cumulative_return_pct']:.2f}%")
    print(f"Annualized Sharpe: {demo_metrics['ann_sharpe']:.3f}")
    print(f"Max drawdown: {demo_metrics['max_drawdown_pct']:.2f}%")
    print("\nSaved artifacts:")
    print(f" - {metrics_path}")
    print(f" - {perf_path}")
    print(f" - {actions_path}")
    print(f" - {action_by_ticker_path}")
    print(f" - {action_totals_path}")
    if plot_written:
        print(f" - {equity_plot_path}")


if __name__ == "__main__":
    main()
