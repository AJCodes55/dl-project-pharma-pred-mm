#!/usr/bin/env python3
"""
PharmaTrade-MM Phase 6: PPO with SEC filing modality (Ablation 3).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from phase2_trading_env import DEFAULT_TICKERS
from phase5_fda_ppo import (
    action_distribution,
    annualized_sharpe,
    build_param_candidates,
    compute_metrics,
    load_ppo_params,
    max_drawdown,
    normalize_to_1m,
    parse_seeds,
    run_ppo_backtest,
    simulate_buy_hold_equal,
    simulate_equal_weight_monthly,
    simulate_momentum_20d,
    to_jsonable,
    train_two_stage,
)
from phase6_multimodal_env import make_phase6_sequence_env_from_processed
from phase6_multimodal_policy import build_phase6_policy_kwargs
from phase6_sec_pipeline import build_phase6_sec_features, save_sec_feature_config
from phase6_sequence_utils import (
    load_phase6_sequence_contract,
    scale_sec_features,
    scale_sentiment,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 PPO SEC ablation")
    p.add_argument("--project-root", default="", help="Optional project root directory.")
    p.add_argument("--train-path", default="processed/train_dataset.csv")
    p.add_argument("--test-path", default="processed/test_dataset.csv")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--xph-path", default="processed/xph_processed.csv")
    p.add_argument("--results-root", default="results")
    p.add_argument("--phase6-best-param-json", default="")
    p.add_argument("--phase5-best-param-json", default="results/phase5_basic_fda/selected_phase5_params.json")
    p.add_argument("--phase4-best-param-json", default="results/phase4_tuning/best_params_price_sentiment.json")
    p.add_argument("--phase3-best-param-json", default="results/phase3_tuning/best_params_price_only.json")
    p.add_argument("--phase3-metrics-path", default="results/phase3_price_only/phase3_price_only_metrics.csv")
    p.add_argument("--phase4-metrics-path", default="results/phase4_price_sentiment/phase4_price_sentiment_metrics.csv")
    p.add_argument("--phase5-metrics-path", default="results/phase5_basic_fda/phase5_fda_metrics.csv")
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--search-timesteps", type=int, default=60_000)
    p.add_argument("--seeds", default="7,42,123")
    p.add_argument("--val-split-date", default="2021-01-01")
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--sent-clip", type=float, default=3.0)
    p.add_argument("--sec-clip", type=float, default=5.0)
    p.add_argument(
        "--sec-feature-mode",
        default="finbert",
        choices=["finbert", "proxy"],
        help="Use strict FinBERT SEC embeddings or proxy SEC features.",
    )
    p.add_argument("--sec-half-life-days", type=float, default=14.0)
    p.add_argument("--sec-sparse-dir", default="All data DL project")
    p.add_argument("--sec-daily-emb-path", default="processed/sec_daily_embeddings.csv")
    p.add_argument("--sec-feature-config-path", default="processed/sec_embedding_feature_config.json")
    p.add_argument(
        "--allow-proxy-fallback",
        action="store_true",
        help="If finbert artifacts are missing, fallback to proxy features instead of failing.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def resolve_path(project_root: Path | None, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute() or project_root is None:
        return p
    return project_root / p


def _extract_best_params_from_selected(path: Path) -> Dict:
    payload = json.loads(path.read_text())
    params = payload.get("params")
    return params if isinstance(params, dict) else {}


def _merge_daily_sec_embeddings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    daily_emb_path: Path,
    sec_cfg_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if not daily_emb_path.exists():
        raise FileNotFoundError(f"Missing SEC daily embeddings: {daily_emb_path}")
    if not sec_cfg_path.exists():
        raise FileNotFoundError(f"Missing SEC feature config: {sec_cfg_path}")

    sec_cfg = json.loads(sec_cfg_path.read_text())
    sec_features = list(sec_cfg.get("sec_features", []))
    if not sec_features:
        raise ValueError(f"SEC feature config has no sec_features: {sec_cfg_path}")

    sec_daily = pd.read_csv(daily_emb_path, parse_dates=["date"])
    sec_daily["date"] = pd.to_datetime(sec_daily["date"])
    sec_daily["ticker"] = sec_daily["ticker"].astype(str).str.upper()
    missing = [c for c in sec_features if c not in sec_daily.columns]
    if missing:
        raise ValueError(f"SEC daily embedding file missing required columns: {missing}")

    train = train_df.copy()
    test = test_df.copy()
    for df in (train, test):
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()

    keep_cols = ["date", "ticker"] + sec_features
    train_aug = train.merge(sec_daily[keep_cols], on=["date", "ticker"], how="left")
    test_aug = test.merge(sec_daily[keep_cols], on=["date", "ticker"], how="left")
    for c in sec_features:
        train_aug[c] = train_aug[c].fillna(0.0)
        test_aug[c] = test_aug[c].fillna(0.0)

    meta = dict(sec_cfg)
    meta["sec_features"] = sec_features
    meta["source"] = str(daily_emb_path)
    return train_aug, test_aug, meta


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

    results_root = resolve_path(project_root, args.results_root)
    results_dir = results_root / "phase6_sec"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(project_root, args.train_path)
    test_path = resolve_path(project_root, args.test_path)
    feature_config_path = resolve_path(project_root, args.feature_config_path)
    xph_path = resolve_path(project_root, args.xph_path)
    sec_sparse_dir = resolve_path(project_root, args.sec_sparse_dir)
    sec_daily_emb_path = resolve_path(project_root, args.sec_daily_emb_path)
    sec_feature_config_path = resolve_path(project_root, args.sec_feature_config_path)
    phase4_best_param_json = resolve_path(project_root, args.phase4_best_param_json)
    phase3_best_param_json = resolve_path(project_root, args.phase3_best_param_json)
    phase3_metrics_path = resolve_path(project_root, args.phase3_metrics_path)
    phase4_metrics_path = resolve_path(project_root, args.phase4_metrics_path)
    phase5_metrics_path = resolve_path(project_root, args.phase5_metrics_path)
    phase6_best_param_json = resolve_path(project_root, args.phase6_best_param_json) if args.phase6_best_param_json else Path("_missing.json")
    phase5_best_param_json = resolve_path(project_root, args.phase5_best_param_json)

    cfg = json.loads(feature_config_path.read_text())
    sent_cols = cfg["sentiment_features"]
    seeds = parse_seeds(args.seeds)

    ppo_params = load_ppo_params(
        primary_tuned_json=phase6_best_param_json if phase6_best_param_json.exists() else phase4_best_param_json,
        warm_start_json=phase3_best_param_json,
    )
    if phase5_best_param_json.exists():
        ppo_params.update(_extract_best_params_from_selected(phase5_best_param_json))
    ppo_params["policy_kwargs"] = build_phase6_policy_kwargs()
    candidate_params = build_param_candidates(ppo_params)

    raw_train = pd.read_csv(train_path, parse_dates=["date"])
    raw_test = pd.read_csv(test_path, parse_dates=["date"])

    sec_mode_used = args.sec_feature_mode
    if args.sec_feature_mode == "finbert":
        try:
            sec_train, sec_test, sec_meta = _merge_daily_sec_embeddings(
                raw_train,
                raw_test,
                daily_emb_path=sec_daily_emb_path,
                sec_cfg_path=sec_feature_config_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            if not args.allow_proxy_fallback:
                raise RuntimeError(
                    "Strict Phase 6 requires prebuilt FinBERT SEC artifacts.\n"
                    "Run `phase6_sec_finbert_pipeline.py` first, or pass --allow-proxy-fallback."
                ) from exc
            print("WARNING: FinBERT SEC artifacts missing/invalid, falling back to proxy SEC features.")
            sec_mode_used = "proxy"
            sec_train, sec_test, sec_meta = build_phase6_sec_features(
                raw_train,
                raw_test,
                sparse_data_dir=sec_sparse_dir,
                half_life_days=args.sec_half_life_days,
            )
    else:
        sec_train, sec_test, sec_meta = build_phase6_sec_features(
            raw_train,
            raw_test,
            sparse_data_dir=sec_sparse_dir,
            half_life_days=args.sec_half_life_days,
        )
    sec_cfg_path = results_dir / "phase6_sec_feature_config.json"
    save_sec_feature_config(sec_cfg_path, sec_meta)

    sent_train, sent_test, sent_scaler = scale_sentiment(sec_train, sec_test, sent_cols, clip=args.sent_clip)
    scaled_train, scaled_test, sec_scaler = scale_sec_features(
        sent_train,
        sent_test,
        sec_cols=sec_meta["sec_features"],
        clip=args.sec_clip,
    )

    scaled_train_path = results_dir / "_train_scaled.csv"
    scaled_test_path = results_dir / "_test_scaled.csv"
    scaled_train.to_csv(scaled_train_path, index=False)
    scaled_test.to_csv(scaled_test_path, index=False)
    (results_dir / "sentiment_scaler_stats.json").write_text(json.dumps(sent_scaler, indent=2))
    (results_dir / "sec_scaler_stats.json").write_text(json.dumps(sec_scaler, indent=2))

    contract = load_phase6_sequence_contract(
        feature_config_path=feature_config_path,
        sec_feature_config_path=sec_cfg_path,
        window_size=args.window_size,
    )
    val_split = pd.Timestamp(args.val_split_date)
    sub_train = scaled_train[scaled_train["date"] < val_split].copy()
    sub_val = scaled_train[scaled_train["date"] >= val_split].copy()
    if sub_train.empty or sub_val.empty:
        raise ValueError("Validation split produced empty subset. Adjust --val-split-date.")
    sub_train_path = results_dir / "_sub_train_scaled.csv"
    sub_val_path = results_dir / "_sub_val_scaled.csv"
    sub_train.to_csv(sub_train_path, index=False)
    sub_val.to_csv(sub_val_path, index=False)

    print("Phase 6 SEC ablation")
    print("Num features (price/sent/sec):", len(contract.price_features), len(contract.sentiment_features), len(contract.sec_features))
    print("Sequence window:", args.window_size)
    print("Robust seeds:", seeds)
    print("Timesteps (final):", args.timesteps, "| search:", args.search_timesteps)
    print("Device:", args.device)
    print("SEC feature mode:", sec_mode_used)

    def build_env(dataset_path: str):
        return make_phase6_sequence_env_from_processed(
            dataset_path=dataset_path,
            feature_config_path=feature_config_path,
            sec_feature_config_path=sec_cfg_path,
            window_size=args.window_size,
            tickers=DEFAULT_TICKERS,
            initial_cash=1_000_000.0,
            transaction_cost=0.001,
            trade_fraction=0.10,
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
    trials_path = results_dir / "phase6_robust_search_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    if best_cfg is None:
        raise RuntimeError("No candidate config selected.")
    selected_cfg = {k: v for k, v in best_cfg.items() if k != "name"}
    (results_dir / "selected_phase6_params.json").write_text(
        json.dumps({"selected_name": best_cfg["name"], "params": to_jsonable(selected_cfg), "score": best_score}, indent=2)
    )

    run_meta_path = results_dir / "phase6_run_metadata.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "ablation": "phase6_sec",
                "architecture": "price_lstm_2x128 + sentiment_lstm_1x64 + sec_mlp_64 + cross_attention + fusion_256",
                "policy_type": "MultiInputPolicy",
                "window_size": args.window_size,
                "sec_feature_count": len(contract.sec_features),
                "sec_features": contract.sec_features,
                "seeds": seeds,
                "selected_seed": best_seed,
                "selected_candidate": best_cfg["name"],
                "selected_params": to_jsonable(selected_cfg),
                "search_timesteps": args.search_timesteps,
                "timesteps": args.timesteps,
                "sec_half_life_days": args.sec_half_life_days,
                "sec_feature_source": str(sec_meta.get("source", sec_sparse_dir)),
                "sec_feature_mode": sec_mode_used,
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
    model_path = results_dir / "ppo_phase6_sec.zip"
    model.save(str(model_path))
    print("Saved model:", model_path)
    print("Selected candidate:", best_cfg["name"])
    print("Selected seed:", best_seed)

    test_env = build_env(str(scaled_test_path))
    ppo_perf, ppo_actions = run_ppo_backtest(test_env, model, seed=best_seed)
    ad_final = action_distribution(ppo_actions)

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
    metric_rows = [
        compute_metrics("PPO_Phase6_SEC", ppo_equity, ppo_daily, xph_daily),
        compute_metrics("BuyHold_Equal", bh_equity, bh_daily, xph_daily),
        compute_metrics("EqualWeight_Monthly", ew_equity, ew_daily, xph_daily),
        compute_metrics("Momentum_20D", mo_equity, mo_daily, xph_daily),
        compute_metrics("XPH", xph_equity, xph_daily, xph_daily),
    ]
    metrics_df = pd.DataFrame(metric_rows).sort_values("ann_sharpe", ascending=False).reset_index(drop=True)
    metrics_path = results_dir / "phase6_sec_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    curve_df = pd.DataFrame(
        {
            "date": ppo_equity.index,
            "ppo_phase6_sec": ppo_equity.values,
            "buy_hold_equal": bh_equity.reindex(ppo_equity.index).values,
            "equal_weight_monthly": ew_equity.reindex(ppo_equity.index).values,
            "momentum_20d": mo_equity.reindex(ppo_equity.index).values,
            "xph": xph_equity.reindex(ppo_equity.index).values,
        }
    )
    curve_path = results_dir / "phase6_sec_equity_curve.csv"
    curve_df.to_csv(curve_path, index=False)

    actions_path = results_dir / "phase6_sec_actions.csv"
    ppo_actions.to_csv(actions_path, index=False)
    action_dist_path = results_dir / "phase6_action_distribution.csv"
    pd.DataFrame([ad_final]).to_csv(action_dist_path, index=False)

    # Interpretability: action behavior by filing recency bucket.
    if "sec_days_since_filing" in scaled_test.columns:
        sec_cols = ["date", "ticker", "sec_days_since_filing"]
        sec_days = scaled_test[sec_cols].copy()
        sec_days["date"] = pd.to_datetime(sec_days["date"])
        actions_long = ppo_actions.copy()
        actions_long["date"] = pd.to_datetime(actions_long["date"])
        melted = actions_long.melt(id_vars=["date"], var_name="act_col", value_name="action")
        melted["ticker"] = melted["act_col"].str.replace("act_", "", regex=False)
        interp = melted.merge(sec_days, on=["date", "ticker"], how="left")
        interp["filing_recency_bucket"] = pd.cut(
            interp["sec_days_since_filing"],
            bins=[-np.inf, 5, 20, 90, np.inf],
            labels=["filing_0_5d", "filing_6_20d", "filing_21_90d", "filing_90p"],
        )
        interp_summary = (
            interp.groupby("filing_recency_bucket", observed=False)["action"]
            .agg(
                mean_action="mean",
                buy_ratio=lambda s: float((s == 2).mean()),
                hold_ratio=lambda s: float((s == 1).mean()),
                sell_ratio=lambda s: float((s == 0).mean()),
                n="count",
            )
            .reset_index()
        )
    else:
        interp_summary = pd.DataFrame(
            [{"filing_recency_bucket": "not_available", "mean_action": np.nan, "buy_ratio": np.nan, "hold_ratio": np.nan, "sell_ratio": np.nan, "n": 0}]
        )
    interp_path = results_dir / "phase6_sec_interpretability.csv"
    interp_summary.to_csv(interp_path, index=False)

    comp_rows = []
    if phase3_metrics_path.exists():
        p3 = pd.read_csv(phase3_metrics_path)
        row = p3[p3["strategy"] == "PPO_PriceOnly"].head(1)
        if not row.empty:
            comp_rows.append(("phase3_price_only", row.iloc[0]))
    if phase4_metrics_path.exists():
        p4 = pd.read_csv(phase4_metrics_path)
        row = p4[p4["strategy"] == "PPO_PriceSentiment"].head(1)
        if not row.empty:
            comp_rows.append(("phase4_price_sentiment", row.iloc[0]))
    if phase5_metrics_path.exists():
        p5 = pd.read_csv(phase5_metrics_path)
        row = p5[p5["strategy"] == "PPO_Phase5_BasicFDA"].head(1)
        if row.empty:
            row = p5[p5["strategy"].str.startswith("PPO_Phase5", na=False)].head(1)
        if not row.empty:
            comp_rows.append(("phase5_frozen", row.iloc[0]))
    p6_row = metrics_df[metrics_df["strategy"] == "PPO_Phase6_SEC"].head(1)
    if not p6_row.empty:
        comp_rows.append(("phase6_sec", p6_row.iloc[0]))

    if len(comp_rows) >= 2:
        keep = ["cumulative_return_pct", "ann_sharpe", "ann_sortino", "max_drawdown_pct", "alpha_vs_xph_ann_pct"]
        comp = pd.DataFrame({"metric": keep})
        for name, row in comp_rows:
            comp[name] = [float(row[m]) for m in keep]
        if "phase6_sec" in comp.columns and "phase4_price_sentiment" in comp.columns:
            comp["delta_p6_minus_p4"] = comp["phase6_sec"] - comp["phase4_price_sentiment"]
        if "phase6_sec" in comp.columns and "phase5_frozen" in comp.columns:
            comp["delta_p6_minus_p5"] = comp["phase6_sec"] - comp["phase5_frozen"]
        comp_path = results_dir / "phase3_phase4_phase5_phase6_comparison.csv"
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
