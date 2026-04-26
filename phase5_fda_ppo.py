#!/usr/bin/env python3
"""
PharmaTrade-MM Phase 5: PPO with FDA modality.

Ablations:
- basic_fda: price + sentiment + basic FDA timing/event-type
- rich_fda:  price + sentiment + full FDA feature set (incl. rich context)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from phase2_trading_env import DEFAULT_TICKERS
from phase5_multimodal_env import make_phase5_sequence_env_from_processed
from phase5_multimodal_policy import build_phase5_policy_kwargs
from phase5_sequence_utils import load_phase5_sequence_contract, scale_sentiment


def to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    return str(value)


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
    obs, _ = env.reset(seed=seed)
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
        action_rows.append({"date": step_info["date"], **{f"act_{t}": int(a[i]) for i, t in enumerate(DEFAULT_TICKERS)}})

    perf = pd.DataFrame(rows)
    perf["date"] = pd.to_datetime(perf["date"])
    perf = perf.sort_values("date").reset_index(drop=True)

    actions = pd.DataFrame(action_rows)
    actions["date"] = pd.to_datetime(actions["date"])
    actions = actions.sort_values("date").reset_index(drop=True)
    return perf, actions


def parse_seeds(seed_text: str) -> List[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def load_ppo_params(primary_tuned_json: Path, warm_start_json: Path) -> Dict[str, float]:
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
    source = "default"
    if primary_tuned_json.exists():
        payload = json.loads(primary_tuned_json.read_text())
        default.update(payload.get("best_params", {}))
        source = str(primary_tuned_json)
    elif warm_start_json.exists():
        payload = json.loads(warm_start_json.read_text())
        default.update(payload.get("best_params", {}))
        source = str(warm_start_json)
    default["n_steps"] = int(default["n_steps"])
    default["batch_size"] = int(default["batch_size"])
    default["n_epochs"] = int(default["n_epochs"])
    print("Using PPO params from:", source)
    return default


def train_two_stage(
    PPO,
    vec_env,
    params: Dict,
    seed: int,
    total_timesteps: int,
    device: str,
    stage_frac: float = 0.35,
):
    p = params.copy()
    ent_end = float(p.pop("ent_coef_final", 0.0001))
    model = PPO("MultiInputPolicy", vec_env, verbose=0, seed=seed, device=device, **p)
    first = max(1, int(total_timesteps * stage_frac))
    second = max(0, total_timesteps - first)
    model.learn(total_timesteps=first)
    if second > 0:
        model.ent_coef = ent_end
        model.learn(total_timesteps=second, reset_num_timesteps=False)
    return model


def action_distribution(actions_df: pd.DataFrame) -> Dict[str, float]:
    action_cols = [c for c in actions_df.columns if c.startswith("act_")]
    flat = actions_df[action_cols].to_numpy().reshape(-1)
    total = len(flat)
    if total == 0:
        return {"sell_ratio": 0.0, "hold_ratio": 0.0, "buy_ratio": 0.0, "max_ratio": 0.0}
    sell = float((flat == 0).sum()) / total
    hold = float((flat == 1).sum()) / total
    buy = float((flat == 2).sum()) / total
    return {"sell_ratio": sell, "hold_ratio": hold, "buy_ratio": buy, "max_ratio": max(sell, hold, buy)}


def build_param_candidates(base: Dict) -> List[Dict]:
    base = base.copy()
    base.setdefault("policy_kwargs", build_phase5_policy_kwargs())
    base.setdefault("ent_coef_final", 0.0001)
    base["n_steps"] = int(base["n_steps"])
    base["batch_size"] = int(base["batch_size"])
    base["n_epochs"] = int(base["n_epochs"])

    cands = []
    c0 = base.copy()
    c0["name"] = "base_tuned"
    cands.append(c0)

    c1 = base.copy()
    c1.update(
        {
            "name": "lower_lr_small_net",
            "learning_rate": float(base["learning_rate"]) * 0.5,
            "ent_coef": min(float(base.get("ent_coef", 0.001)), 0.001),
            "ent_coef_final": 0.0,
        }
    )
    cands.append(c1)

    c2 = base.copy()
    c2.update(
        {
            "name": "low_lr_stable_steps",
            "learning_rate": float(base["learning_rate"]) * 0.3,
            "n_steps": 1024,
            "batch_size": 128,
            "ent_coef": 0.001,
            "ent_coef_final": 0.0,
        }
    )
    cands.append(c2)

    c3 = base.copy()
    c3.update(
        {
            "name": "no_exploration_late",
            "ent_coef": float(base.get("ent_coef", 0.001)),
            "ent_coef_final": 0.0,
        }
    )
    cands.append(c3)

    for c in cands:
        c["n_steps"] = int(c["n_steps"])
        c["batch_size"] = int(min(c["batch_size"], c["n_steps"]))
        c["n_epochs"] = int(c["n_epochs"])
    return cands


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 PPO FDA ablations")
    p.add_argument(
        "--project-root",
        default="",
        help="Optional project root directory. Relative paths resolve from here.",
    )
    p.add_argument("--ablation", choices=["basic_fda", "rich_fda", "rich_fda_ct"], default="basic_fda")
    p.add_argument("--train-path", default="processed/train_dataset.csv")
    p.add_argument("--test-path", default="processed/test_dataset.csv")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--xph-path", default="processed/xph_processed.csv")
    p.add_argument("--results-root", default="results")
    p.add_argument("--phase5-best-param-json", default="")
    p.add_argument("--phase4-best-param-json", default="results/phase4_tuning/best_params_price_sentiment.json")
    p.add_argument("--phase3-best-param-json", default="results/phase3_tuning/best_params_price_only.json")
    p.add_argument("--phase3-metrics-path", default="results/phase3_price_only/phase3_price_only_metrics.csv")
    p.add_argument("--phase4-metrics-path", default="results/phase4_price_sentiment/phase4_price_sentiment_metrics.csv")
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--search-timesteps", type=int, default=60_000)
    p.add_argument("--seeds", default="7,42,123")
    p.add_argument("--val-split-date", default="2021-01-01")
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--sent-clip", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device used by SB3 PPO.",
    )
    return p.parse_args()


def resolve_path(project_root: Path | None, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute() or project_root is None:
        return p
    return project_root / p


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    args = parse_args()
    project_root = Path(args.project_root).expanduser() if args.project_root else None
    if project_root is not None:
        project_root = project_root.resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"--project-root does not exist: {project_root}")
        os.chdir(project_root)

    ablation = args.ablation
    results_root = resolve_path(project_root, args.results_root)
    train_path = resolve_path(project_root, args.train_path)
    test_path = resolve_path(project_root, args.test_path)
    feature_config_path = resolve_path(project_root, args.feature_config_path)
    xph_path = resolve_path(project_root, args.xph_path)
    phase4_best_param_json = resolve_path(project_root, args.phase4_best_param_json)
    phase3_best_param_json = resolve_path(project_root, args.phase3_best_param_json)
    phase3_metrics_path = resolve_path(project_root, args.phase3_metrics_path)
    phase4_metrics_path = resolve_path(project_root, args.phase4_metrics_path)
    phase5_best_param_json = resolve_path(project_root, args.phase5_best_param_json) if args.phase5_best_param_json else Path("_missing.json")

    if ablation == "basic_fda":
        subdir = "phase5_basic_fda"
    elif ablation == "rich_fda":
        subdir = "phase5_rich_fda"
    else:
        subdir = "phase5_rich_fda_ct"
    results_dir = results_root / subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "ppo_phase5_fda.zip"

    cfg = json.loads(feature_config_path.read_text())
    seeds = parse_seeds(args.seeds)
    contract = load_phase5_sequence_contract(
        feature_config_path=feature_config_path,
        window_size=args.window_size,
        ablation=ablation,
    )
    sent_cols = cfg["sentiment_features"]

    ppo_params = load_ppo_params(
        primary_tuned_json=phase5_best_param_json if phase5_best_param_json.exists() else phase4_best_param_json,
        warm_start_json=phase3_best_param_json,
    )
    candidate_params = build_param_candidates(ppo_params)

    raw_train = pd.read_csv(train_path, parse_dates=["date"])
    raw_test = pd.read_csv(test_path, parse_dates=["date"])
    if ablation == "rich_fda_ct":
        # Confound-aware flag from trailing clinical-trial activity (no lookahead).
        ct_cols = [c for c in raw_train.columns if c.startswith("ct_")]
        if ct_cols:
            raw_train["ct_confound_flag_5d"] = (raw_train[ct_cols].sum(axis=1) > 0).astype(float)
            raw_test["ct_confound_flag_5d"] = (raw_test[ct_cols].sum(axis=1) > 0).astype(float)
        else:
            raw_train["ct_confound_flag_5d"] = 0.0
            raw_test["ct_confound_flag_5d"] = 0.0
    scaled_train, scaled_test, scaler_stats = scale_sentiment(raw_train, raw_test, sent_cols, clip=args.sent_clip)
    scaled_train_path = results_dir / "_train_scaled.csv"
    scaled_test_path = results_dir / "_test_scaled.csv"
    scaled_train.to_csv(scaled_train_path, index=False)
    scaled_test.to_csv(scaled_test_path, index=False)
    (results_dir / "sentiment_scaler_stats.json").write_text(json.dumps(scaler_stats, indent=2))

    val_split = pd.Timestamp(args.val_split_date)
    sub_train = scaled_train[scaled_train["date"] < val_split].copy()
    sub_val = scaled_train[scaled_train["date"] >= val_split].copy()
    if sub_train.empty or sub_val.empty:
        raise ValueError("Validation split produced empty subset. Adjust --val-split-date.")
    sub_train_path = results_dir / "_sub_train_scaled.csv"
    sub_val_path = results_dir / "_sub_val_scaled.csv"
    sub_train.to_csv(sub_train_path, index=False)
    sub_val.to_csv(sub_val_path, index=False)

    print(f"Ablation: {ablation}")
    print(
        "Num features (price/sent/fda):",
        len(contract.price_features),
        len(contract.sentiment_features),
        len(contract.event_features),
    )
    print("Sequence window:", args.window_size)
    print("Robust seeds:", seeds)
    print("Timesteps (final):", args.timesteps, "| search:", args.search_timesteps)
    print("Device:", args.device)

    def build_env(dataset_path: str):
        return make_phase5_sequence_env_from_processed(
            dataset_path=dataset_path,
            feature_config_path=feature_config_path,
            ablation=ablation,
            window_size=args.window_size,
            tickers=DEFAULT_TICKERS,
            initial_cash=1_000_000.0,
            transaction_cost=0.001,
            trade_fraction=0.10,
            use_event_scaling=(ablation in {"rich_fda", "rich_fda_ct"}),
        )

    trial_rows = []
    best_score = -1e18
    best_cfg = None
    best_seed = args.seed
    for cand in candidate_params:
        seed_rows = []
        for s in seeds:
            vec_env = DummyVecEnv([lambda p=str(sub_train_path): build_env(p)])
            train_params = {k: v for k, v in cand.items() if k != "name"}
            model_tmp = train_two_stage(
                PPO,
                vec_env,
                train_params,
                seed=s,
                total_timesteps=args.search_timesteps,
                device=args.device,
            )
            val_env = build_env(str(sub_val_path))
            perf_tmp, actions_tmp = run_ppo_backtest(val_env, model_tmp, seed=s)
            daily = perf_tmp["daily_return"]
            eq = perf_tmp["portfolio_value"]
            sharpe_v = annualized_sharpe(daily)
            ret_v = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
            mdd_v = max_drawdown(eq)
            ad = action_distribution(actions_tmp)
            seed_rows.append({"candidate": cand["name"], "seed": s, "sharpe": sharpe_v, "cumret": ret_v, "mdd": mdd_v, **ad})

        seed_df = pd.DataFrame(seed_rows)
        mean_sh = float(seed_df["sharpe"].mean())
        mean_ret = float(seed_df["cumret"].mean())
        mean_mdd = float(seed_df["mdd"].mean())
        mean_sell = float(seed_df["sell_ratio"].mean())
        mean_hold = float(seed_df["hold_ratio"].mean())
        mean_buy = float(seed_df["buy_ratio"].mean())
        mean_max_action = float(seed_df["max_ratio"].mean())
        collapse_penalty = 0.25 if mean_max_action >= 0.90 else 0.0
        buy_penalty = 1.0 if mean_buy < 0.05 else 0.0
        hold_penalty = 1.0 if mean_hold > 0.85 else 0.0
        sharpe_gate_penalty = 0.75 if mean_sh <= 0 else 0.0
        score = mean_sh + 1.5 * mean_ret - 0.2 * abs(mean_mdd) - collapse_penalty - buy_penalty - hold_penalty - sharpe_gate_penalty
        trial_rows.append(
            {
                "candidate": cand["name"],
                "score": score,
                "mean_sharpe": mean_sh,
                "mean_cumret": mean_ret,
                "mean_mdd": mean_mdd,
                "mean_sell_ratio": mean_sell,
                "mean_hold_ratio": mean_hold,
                "mean_buy_ratio": mean_buy,
                "mean_max_action_ratio": mean_max_action,
                "collapse_penalty": collapse_penalty,
                "buy_penalty": buy_penalty,
                "hold_penalty": hold_penalty,
                "sharpe_gate_penalty": sharpe_gate_penalty,
                "params": json.dumps(to_jsonable({k: v for k, v in cand.items() if k != "name"})),
            }
        )
        print(
            f"[candidate={cand['name']}] score={score:.4f} sharpe={mean_sh:.4f} "
            f"ret={mean_ret:.4f} mdd={mean_mdd:.4f} "
            f"sell/hold/buy={mean_sell:.3f}/{mean_hold:.3f}/{mean_buy:.3f}"
        )
        if score > best_score:
            best_score = score
            best_cfg = cand
            best_seed = int(seed_df.sort_values("sharpe", ascending=False).iloc[0]["seed"])

    trials_df = pd.DataFrame(trial_rows).sort_values("score", ascending=False).reset_index(drop=True)
    trials_path = results_dir / "phase5_robust_search_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    if best_cfg is None:
        raise RuntimeError("No candidate config selected.")
    selected_cfg = {k: v for k, v in best_cfg.items() if k != "name"}
    (results_dir / "selected_phase5_params.json").write_text(
        json.dumps({"selected_name": best_cfg["name"], "params": to_jsonable(selected_cfg), "score": best_score}, indent=2)
    )
    run_meta_path = results_dir / "phase5_run_metadata.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "ablation": ablation,
                "architecture": "price_lstm_2x128 + sentiment_lstm_1x64 + fda_mlp_64 + cross_attention + fusion_256",
                "policy_type": "MultiInputPolicy",
                "window_size": args.window_size,
                "event_feature_count": len(contract.event_features),
                "event_features": contract.event_features,
                "use_event_scaling": (ablation in {"rich_fda", "rich_fda_ct"}),
                "seeds": seeds,
                "selected_seed": best_seed,
                "selected_candidate": best_cfg["name"],
                "selected_params": to_jsonable(selected_cfg),
                "search_timesteps": args.search_timesteps,
                "timesteps": args.timesteps,
            },
            indent=2,
        )
    )

    vec_train_env = DummyVecEnv([lambda p=str(scaled_train_path): build_env(p)])
    model = train_two_stage(
        PPO,
        vec_train_env,
        selected_cfg,
        seed=best_seed,
        total_timesteps=args.timesteps,
        device=args.device,
    )
    model.save(str(model_path))
    print("Saved model:", model_path)
    print("Selected candidate:", best_cfg["name"])
    print("Selected seed:", best_seed)

    test_env = build_env(str(scaled_test_path))
    ppo_perf, ppo_actions = run_ppo_backtest(test_env, model, seed=best_seed)
    ad_final = action_distribution(ppo_actions)
    if ad_final["buy_ratio"] <= 0.001:
        print("Detected buy_ratio ~ 0. Retrying with forced exploration config.")
        forced_cfg = selected_cfg.copy()
        forced_cfg.update(
            {
                "learning_rate": min(float(selected_cfg.get("learning_rate", 3e-4)), 1e-4),
                "n_steps": max(int(selected_cfg.get("n_steps", 2048)), 2048),
                "batch_size": 128,
                "ent_coef": max(float(selected_cfg.get("ent_coef", 0.001)), 0.01),
                "ent_coef_final": max(float(selected_cfg.get("ent_coef_final", 0.0001)), 0.001),
            }
        )
        vec_train_env_fb = DummyVecEnv([lambda p=str(scaled_train_path): build_env(p)])
        model = train_two_stage(
            PPO,
            vec_train_env_fb,
            forced_cfg,
            seed=123,
            total_timesteps=args.timesteps,
            device=args.device,
        )
        model.save(str(model_path))
        test_env_fb = build_env(str(scaled_test_path))
        ppo_perf, ppo_actions = run_ppo_backtest(test_env_fb, model, seed=123)
        ad_final = action_distribution(ppo_actions)
        print("Fallback action distribution:", ad_final)

    test_df = raw_test.copy()
    prices = test_df.pivot(index="date", columns="ticker", values="close").sort_index()[DEFAULT_TICKERS]
    bh_equity, bh_daily = simulate_buy_hold_equal(prices)
    ew_equity, ew_daily = simulate_equal_weight_monthly(prices)
    mo_equity, mo_daily = simulate_momentum_20d(prices)

    xph = pd.read_csv(xph_path, parse_dates=["date"]).sort_values("date")
    xph = xph[(xph["date"] >= prices.index.min()) & (xph["date"] <= prices.index.max())].copy()
    xph_equity = normalize_to_1m(xph.set_index("date")["close"])
    xph_daily = xph_equity.pct_change().fillna(0.0)

    ppo_equity = ppo_perf.set_index("date")["portfolio_value"]
    ppo_daily = ppo_equity.pct_change().fillna(0.0)
    strategy_name_map = {
        "basic_fda": "PPO_Phase5_BasicFDA",
        "rich_fda": "PPO_Phase5_RichFDA",
        "rich_fda_ct": "PPO_Phase5_RichFDA_CT",
    }
    strategy_name = strategy_name_map[ablation]
    metric_rows = [
        compute_metrics(strategy_name, ppo_equity, ppo_daily, xph_daily),
        compute_metrics("BuyHold_Equal", bh_equity, bh_daily, xph_daily),
        compute_metrics("EqualWeight_Monthly", ew_equity, ew_daily, xph_daily),
        compute_metrics("Momentum_20D", mo_equity, mo_daily, xph_daily),
        compute_metrics("XPH", xph_equity, xph_daily, xph_daily),
    ]
    metrics_df = pd.DataFrame(metric_rows).sort_values("ann_sharpe", ascending=False).reset_index(drop=True)
    metrics_path = results_dir / "phase5_fda_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    curve_df = pd.DataFrame(
        {
            "date": ppo_equity.index,
            "ppo_phase5_fda": ppo_equity.values,
            "buy_hold_equal": bh_equity.reindex(ppo_equity.index).values,
            "equal_weight_monthly": ew_equity.reindex(ppo_equity.index).values,
            "momentum_20d": mo_equity.reindex(ppo_equity.index).values,
            "xph": xph_equity.reindex(ppo_equity.index).values,
        }
    )
    curve_path = results_dir / "phase5_fda_equity_curve.csv"
    curve_df.to_csv(curve_path, index=False)

    actions_path = results_dir / "phase5_fda_actions.csv"
    ppo_actions.to_csv(actions_path, index=False)
    action_dist_path = results_dir / "phase5_action_distribution.csv"
    pd.DataFrame([ad_final]).to_csv(action_dist_path, index=False)

    # Interpretability output: exposure vs FDA days-to-event bucket.
    events = raw_test[["date", "ticker", "days_to_event"]].copy()
    events["date"] = pd.to_datetime(events["date"])
    actions_long = ppo_actions.copy()
    actions_long["date"] = pd.to_datetime(actions_long["date"])
    melted = actions_long.melt(id_vars=["date"], var_name="act_col", value_name="action")
    melted["ticker"] = melted["act_col"].str.replace("act_", "", regex=False)
    interp = melted.merge(events, on=["date", "ticker"], how="left")
    interp["days_bucket"] = pd.cut(
        interp["days_to_event"],
        bins=[-np.inf, 5, 20, 90, np.inf],
        labels=["event_0_5d", "near_6_20d", "far_21_90d", "no_event_90p"],
    )
    interp_summary = (
        interp.groupby("days_bucket", observed=False)["action"]
        .agg(
            mean_action="mean",
            buy_ratio=lambda s: float((s == 2).mean()),
            hold_ratio=lambda s: float((s == 1).mean()),
            sell_ratio=lambda s: float((s == 0).mean()),
            n="count",
        )
        .reset_index()
    )
    interp_path = results_dir / "phase5_fda_interpretability.csv"
    interp_summary.to_csv(interp_path, index=False)

    if curve_df.isnull().any().any():
        raise ValueError("NaN found in phase5 equity curve output")
    if not (curve_df[[c for c in curve_df.columns if c != "date"]] > 0).all().all():
        raise ValueError("Non-positive portfolio value found in phase5")
    if len(ppo_actions) != len(ppo_perf):
        raise ValueError("Phase5 action/perf length mismatch")

    comp_rows = []
    p3_path = phase3_metrics_path
    p4_path = phase4_metrics_path
    if p3_path.exists():
        p3 = pd.read_csv(p3_path)
        row = p3[p3["strategy"] == "PPO_PriceOnly"].head(1)
        if not row.empty:
            comp_rows.append(("phase3_price_only", row.iloc[0]))
    if p4_path.exists():
        p4 = pd.read_csv(p4_path)
        row = p4[p4["strategy"] == "PPO_PriceSentiment"].head(1)
        if not row.empty:
            comp_rows.append(("phase4_price_sentiment", row.iloc[0]))
    p5_row = metrics_df[metrics_df["strategy"] == strategy_name].head(1)
    if not p5_row.empty:
        comp_rows.append((f"phase5_{ablation}", p5_row.iloc[0]))

    if len(comp_rows) >= 2:
        keep = ["cumulative_return_pct", "ann_sharpe", "ann_sortino", "max_drawdown_pct", "alpha_vs_xph_ann_pct"]
        comp = pd.DataFrame({"metric": keep})
        for name, row in comp_rows:
            comp[name] = [float(row[m]) for m in keep]
        if "phase4_price_sentiment" in comp.columns and f"phase5_{ablation}" in comp.columns:
            comp["delta_p5_minus_p4"] = comp[f"phase5_{ablation}"] - comp["phase4_price_sentiment"]
        comp_path = results_dir / "phase3_phase4_phase5_comparison.csv"
        comp.to_csv(comp_path, index=False)
        print("Saved:", comp_path)

    print("\nSaved artifacts:")
    print(" -", metrics_path)
    print(" -", curve_path)
    print(" -", actions_path)
    print(" -", action_dist_path)
    print(" -", trials_path)
    print(" -", run_meta_path)
    print(" -", interp_path)
    print(" -", model_path)
    print("\nTop metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
