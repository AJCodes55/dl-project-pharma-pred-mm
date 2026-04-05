#!/usr/bin/env python3
"""
PharmaTrade-MM Phase 2 trading environment.

This module provides a Gym-compatible environment that:
- consumes processed Phase 1 datasets (train/test)
- simulates portfolio execution over daily bars
- supports Buy/Hold/Sell actions per ticker
- applies transaction costs and FDA event-window position scaling
- computes a risk-aware reward with drawdown and CVaR penalties
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HAS_GYM = True
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ModuleNotFoundError as exc:
        HAS_GYM = False

        class _BaseEnv:
            """Fallback Env shim when gym is unavailable."""

            def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
                if seed is not None:
                    np.random.seed(seed)
                return None

        class _MultiDiscrete:
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec, dtype=np.int64)

            def sample(self):
                return np.array([np.random.randint(0, n) for n in self.nvec], dtype=np.int64)

            def __repr__(self):
                return f"MultiDiscrete({self.nvec.tolist()})"

        class _Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

            def __repr__(self):
                return f"Box(shape={self.shape}, dtype={self.dtype})"

        class _Spaces:
            MultiDiscrete = _MultiDiscrete
            Box = _Box

        class _GymShim:
            Env = _BaseEnv

        gym = _GymShim()  # type: ignore
        spaces = _Spaces()  # type: ignore
        print(
            "Warning: gymnasium/gym not installed. Using local fallback spaces for sanity checks. "
            "Install gymnasium for PPO training."
        )


DEFAULT_TICKERS = ["PFE", "JNJ", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB"]


@dataclass
class RewardConfig:
    sharpe_delta_weight: float = 0.05
    drawdown_weight: float = 0.10
    cvar_weight: float = 0.10
    cvar_alpha: float = 0.10
    cvar_min_obs: int = 20


class PharmaTradingEnv(gym.Env):
    """
    A discrete-action multi-asset trading environment.

    Action encoding per ticker:
      - 0: Sell (partial)
      - 1: Hold
      - 2: Buy (partial)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        tickers: Optional[Sequence[str]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        trade_fraction: float = 0.10,
        event_window_scale: float = 0.50,
        use_event_scaling: bool = True,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        self.df = data.copy()
        if "date" not in self.df.columns or "ticker" not in self.df.columns:
            raise ValueError("Input data must contain 'date' and 'ticker' columns.")

        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["ticker"] = self.df["ticker"].astype(str).str.upper()

        self.tickers = list(tickers) if tickers is not None else list(DEFAULT_TICKERS)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.trade_fraction = float(trade_fraction)
        self.event_window_scale = float(event_window_scale)
        self.use_event_scaling = bool(use_event_scaling)
        self.reward_cfg = reward_config or RewardConfig()

        if feature_columns is None:
            excluded = {"date", "ticker"}
            feature_columns = [c for c in self.df.columns if c not in excluded]
        self.feature_columns = list(feature_columns)

        if "close" not in self.feature_columns:
            raise ValueError("feature_columns must include 'close' for pricing.")
        if self.use_event_scaling and "is_event_window" not in self.df.columns:
            raise ValueError(
                "Input data must contain 'is_event_window' when use_event_scaling=True."
            )

        self._prepare_arrays()

        self.num_assets = len(self.tickers)
        self.num_features = len(self.feature_columns)

        # Portfolio context: positions (N) + normalized cash (1)
        obs_dim = self.num_assets * self.num_features + self.num_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([3] * self.num_assets)

        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets, dtype=np.float64)
        self.returns_history: List[float] = []
        self.prev_sharpe = 0.0
        self.peak_value = self.initial_cash

    def _prepare_arrays(self) -> None:
        df = self.df[self.df["ticker"].isin(self.tickers)].copy()
        if df.empty:
            raise ValueError("No rows left after filtering to requested tickers.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.dates = sorted(df["date"].unique())
        self.num_steps = len(self.dates)
        n = len(self.tickers)
        f = len(self.feature_columns)

        feat_arr = np.zeros((self.num_steps, n, f), dtype=np.float32)
        close_arr = np.zeros((self.num_steps, n), dtype=np.float64)
        event_arr = np.zeros((self.num_steps, n), dtype=np.int8)

        for t_idx, date in enumerate(self.dates):
            day = (
                df[df["date"] == date]
                .set_index("ticker")
                .reindex(self.tickers)
            )
            if day.isnull().any().any():
                missing = day[day["close"].isna()].index.tolist()
                raise ValueError(
                    f"Missing ticker rows on date {pd.Timestamp(date).date()}: {missing}"
                )

            feat_arr[t_idx] = day[self.feature_columns].to_numpy(dtype=np.float32)
            close_arr[t_idx] = day["close"].to_numpy(dtype=np.float64)
            if "is_event_window" in day.columns:
                event_arr[t_idx] = day["is_event_window"].to_numpy(dtype=np.int8)
            else:
                event_arr[t_idx] = np.zeros(n, dtype=np.int8)

        self.features_arr = feat_arr
        self.close_arr = close_arr
        self.event_arr = event_arr

    def _portfolio_value(self, step_idx: Optional[int] = None) -> float:
        idx = self.current_step if step_idx is None else step_idx
        prices = self.close_arr[idx]
        return float(self.cash + np.dot(self.positions, prices))

    def _compute_sharpe(self) -> float:
        if len(self.returns_history) < 2:
            return 0.0
        rets = np.asarray(self.returns_history, dtype=np.float64)
        std = rets.std(ddof=1)
        if std <= 1e-12:
            return 0.0
        return float((rets.mean() / std) * np.sqrt(252.0))

    def _compute_cvar_penalty(self) -> float:
        if len(self.returns_history) < self.reward_cfg.cvar_min_obs:
            return 0.0
        rets = np.asarray(self.returns_history, dtype=np.float64)
        q = np.quantile(rets, self.reward_cfg.cvar_alpha)
        tail = rets[rets <= q]
        if tail.size == 0:
            return 0.0
        cvar = float(tail.mean())
        return max(0.0, -cvar)

    def _get_observation(self) -> np.ndarray:
        feats = self.features_arr[self.current_step].reshape(-1)
        prices = self.close_arr[self.current_step]
        position_values = self.positions * prices
        portfolio_value = max(self._portfolio_value(), 1e-9)
        pos_weights = position_values / portfolio_value
        cash_ratio = np.array([self.cash / portfolio_value], dtype=np.float64)
        obs = np.concatenate([feats, pos_weights, cash_ratio], axis=0)
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets, dtype=np.float64)
        self.returns_history = []
        self.prev_sharpe = 0.0
        self.peak_value = self.initial_cash

        obs = self._get_observation()
        info = {
            "date": str(pd.Timestamp(self.dates[self.current_step]).date()),
            "portfolio_value": self._portfolio_value(),
            "cash": self.cash,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.num_assets:
            raise ValueError(f"Action length {action.shape[0]} != num_assets {self.num_assets}")
        if np.any((action < 0) | (action > 2)):
            raise ValueError("Actions must be in {0=sell, 1=hold, 2=buy}")

        prices = self.close_arr[self.current_step]
        is_event = self.event_arr[self.current_step]
        value_before = self._portfolio_value()

        for i in range(self.num_assets):
            a = int(action[i])
            price = float(prices[i])
            if price <= 0:
                continue

            if a == 0:
                # Sell a fraction of existing position.
                qty = int(np.floor(self.positions[i] * self.trade_fraction))
                if qty <= 0 and self.positions[i] > 0:
                    qty = 1
                qty = min(qty, int(self.positions[i]))
                if qty > 0:
                    gross = qty * price
                    fee = gross * self.transaction_cost
                    self.positions[i] -= qty
                    self.cash += gross - fee

            elif a == 2:
                # Buy with a fraction of current cash.
                buy_scale = self.trade_fraction
                if self.use_event_scaling and is_event[i] == 1:
                    buy_scale *= self.event_window_scale
                budget = self.cash * buy_scale
                qty = int(np.floor(budget / price))
                if qty > 0:
                    gross = qty * price
                    fee = gross * self.transaction_cost
                    total_cost = gross + fee
                    if total_cost <= self.cash:
                        self.positions[i] += qty
                        self.cash -= total_cost

        value_after = self._portfolio_value()
        daily_return = 0.0
        if value_before > 0:
            daily_return = (value_after - value_before) / value_before
        self.returns_history.append(float(daily_return))

        self.peak_value = max(self.peak_value, value_after)
        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = (self.peak_value - value_after) / self.peak_value

        sharpe = self._compute_sharpe()
        sharpe_delta = sharpe - self.prev_sharpe
        self.prev_sharpe = sharpe
        cvar_penalty = self._compute_cvar_penalty()

        reward = (
            float(daily_return)
            + self.reward_cfg.sharpe_delta_weight * float(sharpe_delta)
            - self.reward_cfg.drawdown_weight * float(drawdown)
            - self.reward_cfg.cvar_weight * float(cvar_penalty)
        )

        self.current_step += 1
        terminated = self.current_step >= self.num_steps
        truncated = False
        if terminated:
            self.current_step = self.num_steps - 1

        obs = self._get_observation()
        info = {
            "date": str(pd.Timestamp(self.dates[self.current_step]).date()),
            "portfolio_value": value_after,
            "cash": self.cash,
            "daily_return": float(daily_return),
            "drawdown": float(drawdown),
            "sharpe": float(sharpe),
            "cvar_penalty": float(cvar_penalty),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pv = self._portfolio_value()
        print(
            f"Step={self.current_step} Date={pd.Timestamp(self.dates[self.current_step]).date()} "
            f"Portfolio={pv:,.2f} Cash={self.cash:,.2f}"
        )


def load_feature_columns(feature_config_path: str | Path) -> List[str]:
    cfg = json.loads(Path(feature_config_path).read_text())
    ct = cfg.get("clinical_trial_features", [])
    return cfg["price_features"] + cfg["sentiment_features"] + cfg["fda_features"] + ct


def make_env_from_processed(
    dataset_path: str | Path,
    feature_config_path: str | Path,
    tickers: Optional[Sequence[str]] = None,
    feature_columns_override: Optional[Sequence[str]] = None,
    **env_kwargs,
) -> PharmaTradingEnv:
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    feature_cols = list(feature_columns_override) if feature_columns_override is not None else load_feature_columns(feature_config_path)
    return PharmaTradingEnv(
        data=df,
        tickers=tickers or DEFAULT_TICKERS,
        feature_columns=feature_cols,
        **env_kwargs,
    )


if __name__ == "__main__":
    env = make_env_from_processed(
        dataset_path="processed/train_dataset.csv",
        feature_config_path="processed/feature_config.json",
    )
    obs, info = env.reset(seed=42)
    print("Obs shape:", obs.shape)
    print("Reset info:", info)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, step_info = env.step(action)
        print(
            f"date={step_info['date']} reward={reward:.6f} "
            f"value={step_info['portfolio_value']:.2f}"
        )
        if done:
            break
