#!/usr/bin/env python3
"""
PharmaTrade-MM Phase 3: PPO Price-Only baseline (script version).

This script mirrors `phase3_price_only_ppo.ipynb`:
- trains PPO on price-only features
- backtests on test split
- computes baseline strategies
- calculates metrics
- saves outputs under results/phase3_price_only/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from phase2_trading_env import DEFAULT_TICKERS, make_env_from_processed


def annualized_sharpe(daily_ret: pd.Series) -> float:
    s = daily_ret.std(ddof=1)
    if s <= 1e-12:
        return 0.0
    return float((daily_ret.mean() / s) * np.sqrt(252))


def annualized_sortino(daily_ret: pd.Series) -> float:
    downside = daily_ret[daily_ret < 0]
    ds = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if ds <= 1e-12:
        return 0.0
    return float((daily_ret.mean() / ds) * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def win_rate(daily_ret: pd.Series) -> float:
    return float((daily_ret > 0).mean())


def simulate_buy_hold_equal(prices: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    shares = (1_000_000.0 / len(prices.columns)) / prices.iloc[0]
    equity = (prices * shares).sum(axis=1)
    daily = equity.pct_change().fillna(0.0)
    return equity, daily


def simulate_equal_weight_monthly(prices: pd.DataFrame, tc: float = 0.001) -> Tuple[pd.Series, pd.Series]:
    dates = prices.index
    n = prices.shape[1]
    target_w = np.repeat(1.0 / n, n)
    weights = target_w.copy()
    equity = [1_000_000.0]
    prev_month = dates[0].month

    for i in range(1, len(dates)):
        cur_month = dates[i].month
        r = prices.iloc[i].values / prices.iloc[i - 1].values - 1.0
        gross_ret = float(np.dot(weights, r))
        value = equity[-1] * (1.0 + gross_ret)

        numer = weights * (1.0 + r)
        denom = numer.sum()
        weights = numer / denom if denom > 0 else target_w.copy()

        if cur_month != prev_month:
            turnover = np.abs(target_w - weights).sum()
            value = value * (1.0 - tc * turnover)
            weights = target_w.copy()
        prev_month = cur_month
        equity.append(value)

    equity_s = pd.Series(equity, index=dates)
    daily = equity_s.pct_change().fillna(0.0)
    return equity_s, daily


def simulate_momentum_20d(prices: pd.DataFrame, top_k: int = 3, tc: float = 0.001) -> Tuple[pd.Series, pd.Series]:
    dates = prices.index
    n = prices.shape[1]
    momentum = prices.pct_change(20).shift(1)
    weights = np.zeros(n)
    equity = [1_000_000.0]

    for i in range(1, len(dates)):
        m = momentum.iloc[i].values
        valid = np.where(np.isfinite(m) & (m > 0))[0]
        target = np.zeros(n)
        if len(valid) > 0:
            order = valid[np.argsort(m[valid])[::-1]]
            pick = order[:top_k]
            target[pick] = 1.0 / len(pick)

        turnover = np.abs(target - weights).sum()
        value = equity[-1] * (1.0 - tc * turnover)

        r = prices.iloc[i].values / prices.iloc[i - 1].values - 1.0
        gross_ret = float(np.dot(target, r))
        value = value * (1.0 + gross_ret)
        weights = target
        equity.append(value)

    equity_s = pd.Series(equity, index=dates)
    daily = equity_s.pct_change().fillna(0.0)
    return equity_s, daily


def normalize_to_1m(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] * 1_000_000.0


def compute_metrics(
    name: str,
    equity: pd.Series,
    daily_ret: pd.Series,
    xph_daily_ret: pd.Series,
) -> Dict[str, float]:
    common_idx = daily_ret.index.intersection(xph_daily_ret.index)
    aligned_ret = daily_ret.loc[common_idx]
    aligned_xph = xph_daily_ret.loc[common_idx]
    excess = aligned_ret - aligned_xph

    return {
        "strategy": name,
        "cumulative_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0),
        "ann_sharpe": annualized_sharpe(aligned_ret),
        "ann_sortino": annualized_sortino(aligned_ret),
        "max_drawdown_pct": float(max_drawdown(equity) * 100.0),
        "win_rate": win_rate(aligned_ret),
        "alpha_vs_xph_ann_pct": float(excess.mean() * 252 * 100.0),
    }


def run_ppo_backtest(env, model, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs, info = env.reset(seed=seed)
    done = False
    rows = []
    action_rows = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, step_info = env.step(action)
        rows.append(
            {
                "date": step_info["date"],
                "portfolio_value": step_info["portfolio_value"],
                "daily_return": step_info["daily_return"],
                "reward": reward,
                "drawdown": step_info["drawdown"],
                "sharpe": step_info["sharpe"],
                "cvar_penalty": step_info["cvar_penalty"],
            }
        )
        a = np.asarray(action).reshape(-1)
        action_rows.append(
            {"date": step_info["date"], **{f"act_{t}": int(a[i]) for i, t in enumerate(DEFAULT_TICKERS)}}
        )

    perf = pd.DataFrame(rows)
    perf["date"] = pd.to_datetime(perf["date"])
    perf = perf.sort_values("date").reset_index(drop=True)

    actions = pd.DataFrame(action_rows)
    actions["date"] = pd.to_datetime(actions["date"])
    actions = actions.sort_values("date").reset_index(drop=True)
    return perf, actions


def load_ppo_params(best_param_json: Path) -> Dict[str, float]:
    default = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "clip_range": 0.2,
    }
    if best_param_json.exists():
        payload = json.loads(best_param_json.read_text())
        default.update(payload.get("best_params", {}))

    default["n_steps"] = int(default["n_steps"])
    default["batch_size"] = int(default["batch_size"])
    default["n_epochs"] = int(default["n_epochs"])
    return default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 PPO price-only baseline")
    p.add_argument("--train-path", default="processed/train_dataset.csv")
    p.add_argument("--test-path", default="processed/test_dataset.csv")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--xph-path", default="processed/xph_processed.csv")
    p.add_argument("--results-dir", default="results/phase3_price_only")
    p.add_argument("--best-param-json", default="results/phase3_tuning/best_params_price_only.json")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "ppo_price_only.zip"

    cfg = json.loads(Path(args.feature_config_path).read_text())
    price_features = cfg["price_features"]
    ppo_params = load_ppo_params(Path(args.best_param_json))
    print("PPO params:", ppo_params)

    def build_price_only_env(dataset_path: str):
        return make_env_from_processed(
            dataset_path=dataset_path,
            feature_config_path=args.feature_config_path,
            tickers=DEFAULT_TICKERS,
            feature_columns_override=price_features,
            initial_cash=1_000_000.0,
            transaction_cost=0.001,
            trade_fraction=0.10,
            use_event_scaling=False,
        )

    vec_train_env = DummyVecEnv([lambda: build_price_only_env(args.train_path)])
    model = PPO("MlpPolicy", vec_train_env, verbose=1, seed=args.seed, **ppo_params)
    model.learn(total_timesteps=args.timesteps)
    model.save(str(model_path))
    print("Saved model:", model_path)

    test_env = build_price_only_env(args.test_path)
    ppo_perf, ppo_actions = run_ppo_backtest(test_env, model, seed=args.seed)

    test_df = pd.read_csv(args.test_path, parse_dates=["date"])
    prices = test_df.pivot(index="date", columns="ticker", values="close").sort_index()[DEFAULT_TICKERS]
    bh_equity, bh_daily = simulate_buy_hold_equal(prices)
    ew_equity, ew_daily = simulate_equal_weight_monthly(prices)
    mo_equity, mo_daily = simulate_momentum_20d(prices)

    xph = pd.read_csv(args.xph_path, parse_dates=["date"]).sort_values("date")
    xph = xph[(xph["date"] >= prices.index.min()) & (xph["date"] <= prices.index.max())].copy()
    xph_equity = normalize_to_1m(xph.set_index("date")["close"])
    xph_daily = xph_equity.pct_change().fillna(0.0)

    ppo_equity = ppo_perf.set_index("date")["portfolio_value"]
    ppo_daily = ppo_equity.pct_change().fillna(0.0)

    metric_rows = [
        compute_metrics("PPO_PriceOnly", ppo_equity, ppo_daily, xph_daily),
        compute_metrics("BuyHold_Equal", bh_equity, bh_daily, xph_daily),
        compute_metrics("EqualWeight_Monthly", ew_equity, ew_daily, xph_daily),
        compute_metrics("Momentum_20D", mo_equity, mo_daily, xph_daily),
        compute_metrics("XPH", xph_equity, xph_daily, xph_daily),
    ]
    metrics_df = pd.DataFrame(metric_rows).sort_values("ann_sharpe", ascending=False).reset_index(drop=True)

    metrics_path = results_dir / "phase3_price_only_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    curve_df = pd.DataFrame(
        {
            "date": ppo_equity.index,
            "ppo_price_only": ppo_equity.values,
            "buy_hold_equal": bh_equity.reindex(ppo_equity.index).values,
            "equal_weight_monthly": ew_equity.reindex(ppo_equity.index).values,
            "momentum_20d": mo_equity.reindex(ppo_equity.index).values,
            "xph": xph_equity.reindex(ppo_equity.index).values,
        }
    )
    curve_path = results_dir / "phase3_price_only_equity_curve.csv"
    curve_df.to_csv(curve_path, index=False)

    actions_path = results_dir / "phase3_price_only_actions.csv"
    ppo_actions.to_csv(actions_path, index=False)

    if curve_df.isnull().any().any():
        raise ValueError("NaN found in equity curve output")
    if not (curve_df[[c for c in curve_df.columns if c != "date"]] > 0).all().all():
        raise ValueError("Non-positive portfolio value found")
    if len(ppo_actions) != len(ppo_perf):
        raise ValueError("Action log length mismatch")

    print("\nSaved artifacts:")
    print(" -", metrics_path)
    print(" -", curve_path)
    print(" -", actions_path)
    print(" -", model_path)
    print("\nTop metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
