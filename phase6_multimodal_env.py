#!/usr/bin/env python3
"""
Phase 6 sequence-aware multimodal trading environment with SEC stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from phase2_trading_env import DEFAULT_TICKERS, RewardConfig, gym, spaces
from phase6_sequence_utils import (
    Phase6SequenceContract,
    load_phase6_sequence_contract,
    validate_dataframe_columns,
)


@dataclass
class SeqEnvConfig:
    initial_cash: float = 1_000_000.0
    transaction_cost: float = 0.001
    trade_fraction: float = 0.10
    reward_config: RewardConfig = field(default_factory=RewardConfig)


class PharmaSequenceSECEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        contract: Phase6SequenceContract,
        tickers: Optional[Sequence[str]] = None,
        config: Optional[SeqEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.df = data.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["ticker"] = self.df["ticker"].astype(str).str.upper()
        self.contract = contract
        self.tickers = list(tickers) if tickers is not None else list(contract.tickers)
        self.cfg = config or SeqEnvConfig()

        required = (
            ["date", "ticker", "close"]
            + contract.price_features
            + contract.sentiment_features
            + contract.sec_features
        )
        validate_dataframe_columns(self.df, required)
        self._prepare_arrays()
        self.num_assets = len(self.tickers)
        self.window = contract.window_size

        self.observation_space = spaces.Dict(
            {
                "price_seq": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window, self.num_assets, len(contract.price_features)),
                    dtype=np.float32,
                ),
                "sent_seq": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window, self.num_assets, len(contract.sentiment_features)),
                    dtype=np.float32,
                ),
                "sec_seq": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window, self.num_assets, len(contract.sec_features)),
                    dtype=np.float32,
                ),
                "portfolio_ctx": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_assets + 1,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete([3] * self.num_assets)

        self.current_step = 0
        self.cash = float(self.cfg.initial_cash)
        self.positions = np.zeros(self.num_assets, dtype=np.float64)
        self.returns_history: List[float] = []
        self.prev_sharpe = 0.0
        self.peak_value = float(self.cfg.initial_cash)

    def _prepare_arrays(self) -> None:
        df = self.df[self.df["ticker"].isin(self.tickers)].copy()
        if df.empty:
            raise ValueError("No rows left after filtering to requested tickers.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.dates = sorted(df["date"].unique())
        self.num_steps = len(self.dates)
        n = len(self.tickers)
        p = len(self.contract.price_features)
        s = len(self.contract.sentiment_features)
        u = len(self.contract.sec_features)

        price_arr = np.zeros((self.num_steps, n, p), dtype=np.float32)
        sent_arr = np.zeros((self.num_steps, n, s), dtype=np.float32)
        sec_arr = np.zeros((self.num_steps, n, u), dtype=np.float32)
        close_arr = np.zeros((self.num_steps, n), dtype=np.float64)

        for t_idx, date in enumerate(self.dates):
            day = df[df["date"] == date].set_index("ticker").reindex(self.tickers)
            if day.isnull().any().any():
                missing = day[day["close"].isna()].index.tolist()
                raise ValueError(f"Missing ticker rows on date {pd.Timestamp(date).date()}: {missing}")
            price_arr[t_idx] = day[self.contract.price_features].to_numpy(dtype=np.float32)
            sent_arr[t_idx] = day[self.contract.sentiment_features].to_numpy(dtype=np.float32)
            sec_arr[t_idx] = day[self.contract.sec_features].to_numpy(dtype=np.float32)
            close_arr[t_idx] = day["close"].to_numpy(dtype=np.float64)

        self.price_arr = price_arr
        self.sent_arr = sent_arr
        self.sec_arr = sec_arr
        self.close_arr = close_arr

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
        if len(self.returns_history) < self.cfg.reward_config.cvar_min_obs:
            return 0.0
        rets = np.asarray(self.returns_history, dtype=np.float64)
        q = np.quantile(rets, self.cfg.reward_config.cvar_alpha)
        tail = rets[rets <= q]
        if tail.size == 0:
            return 0.0
        cvar = float(tail.mean())
        return max(0.0, -cvar)

    def _get_seq_window(self, arr: np.ndarray) -> np.ndarray:
        start = max(0, self.current_step - self.window + 1)
        chunk = arr[start : self.current_step + 1]
        if chunk.shape[0] < self.window:
            pad = np.repeat(chunk[:1], repeats=self.window - chunk.shape[0], axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        return chunk.astype(np.float32)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        price_seq = self._get_seq_window(self.price_arr)
        sent_seq = self._get_seq_window(self.sent_arr)
        sec_seq = self._get_seq_window(self.sec_arr)
        prices = self.close_arr[self.current_step]
        position_values = self.positions * prices
        portfolio_value = max(self._portfolio_value(), 1e-9)
        pos_weights = position_values / portfolio_value
        cash_ratio = np.array([self.cash / portfolio_value], dtype=np.float64)
        portfolio_ctx = np.concatenate([pos_weights, cash_ratio], axis=0).astype(np.float32)
        return {
            "price_seq": price_seq,
            "sent_seq": sent_seq,
            "sec_seq": sec_seq,
            "portfolio_ctx": portfolio_ctx,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = float(self.cfg.initial_cash)
        self.positions = np.zeros(self.num_assets, dtype=np.float64)
        self.returns_history = []
        self.prev_sharpe = 0.0
        self.peak_value = float(self.cfg.initial_cash)
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
        value_before = self._portfolio_value()
        for i in range(self.num_assets):
            a = int(action[i])
            price = float(prices[i])
            if price <= 0:
                continue
            if a == 0:
                qty = int(np.floor(self.positions[i] * self.cfg.trade_fraction))
                if qty <= 0 and self.positions[i] > 0:
                    qty = 1
                qty = min(qty, int(self.positions[i]))
                if qty > 0:
                    gross = qty * price
                    fee = gross * self.cfg.transaction_cost
                    self.positions[i] -= qty
                    self.cash += gross - fee
            elif a == 2:
                budget = self.cash * self.cfg.trade_fraction
                qty = int(np.floor(budget / price))
                if qty > 0:
                    gross = qty * price
                    fee = gross * self.cfg.transaction_cost
                    total_cost = gross + fee
                    if total_cost <= self.cash:
                        self.positions[i] += qty
                        self.cash -= total_cost

        value_after = self._portfolio_value()
        daily_return = (value_after - value_before) / value_before if value_before > 0 else 0.0
        self.returns_history.append(float(daily_return))
        self.peak_value = max(self.peak_value, value_after)
        drawdown = (self.peak_value - value_after) / self.peak_value if self.peak_value > 0 else 0.0
        sharpe = self._compute_sharpe()
        sharpe_delta = sharpe - self.prev_sharpe
        self.prev_sharpe = sharpe
        cvar_penalty = self._compute_cvar_penalty()
        reward = (
            float(daily_return)
            + self.cfg.reward_config.sharpe_delta_weight * float(sharpe_delta)
            - self.cfg.reward_config.drawdown_weight * float(drawdown)
            - self.cfg.reward_config.cvar_weight * float(cvar_penalty)
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


def make_phase6_sequence_env_from_processed(
    dataset_path: str | Path,
    feature_config_path: str | Path,
    sec_feature_config_path: str | Path,
    window_size: int = 20,
    tickers: Optional[Sequence[str]] = None,
    **env_kwargs,
) -> PharmaSequenceSECEnv:
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    contract = load_phase6_sequence_contract(
        feature_config_path=feature_config_path,
        sec_feature_config_path=sec_feature_config_path,
        window_size=window_size,
    )
    env_cfg = SeqEnvConfig(
        initial_cash=float(env_kwargs.pop("initial_cash", 1_000_000.0)),
        transaction_cost=float(env_kwargs.pop("transaction_cost", 0.001)),
        trade_fraction=float(env_kwargs.pop("trade_fraction", 0.10)),
        reward_config=env_kwargs.pop("reward_config", RewardConfig()),
    )
    return PharmaSequenceSECEnv(
        data=df,
        contract=contract,
        tickers=tickers or DEFAULT_TICKERS,
        config=env_cfg,
    )
