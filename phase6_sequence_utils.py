#!/usr/bin/env python3
"""
Phase 6 sequence contract and preprocessing utilities (SEC ablation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Phase6SequenceContract:
    tickers: List[str]
    price_features: List[str]
    sentiment_features: List[str]
    sec_features: List[str]
    window_size: int

    @property
    def portfolio_context_dim(self) -> int:
        return len(self.tickers) + 1

    @property
    def price_shape(self) -> Tuple[int, int, int]:
        return (self.window_size, len(self.tickers), len(self.price_features))

    @property
    def sentiment_shape(self) -> Tuple[int, int, int]:
        return (self.window_size, len(self.tickers), len(self.sentiment_features))

    @property
    def sec_shape(self) -> Tuple[int, int, int]:
        return (self.window_size, len(self.tickers), len(self.sec_features))


def load_phase6_sequence_contract(
    feature_config_path: str | Path,
    sec_feature_config_path: str | Path,
    window_size: int,
) -> Phase6SequenceContract:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    cfg = json.loads(Path(feature_config_path).read_text())
    sec_cfg = json.loads(Path(sec_feature_config_path).read_text())
    sec_features = list(sec_cfg.get("sec_features", []))
    if not sec_features:
        raise ValueError("SEC feature config does not contain sec_features.")
    return Phase6SequenceContract(
        tickers=list(cfg["tickers"]),
        price_features=list(cfg["price_features"]),
        sentiment_features=list(cfg["sentiment_features"]),
        sec_features=sec_features,
        window_size=int(window_size),
    )


def validate_dataframe_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def scale_sentiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sent_cols: List[str],
    clip: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    tr = train_df.copy()
    te = test_df.copy()
    stats: Dict[str, Dict[str, float]] = {}
    for c in sent_cols:
        mu = float(tr[c].mean())
        sd = float(tr[c].std(ddof=0))
        if sd <= 1e-12:
            sd = 1.0
        tr[c] = ((tr[c] - mu) / sd).clip(-clip, clip)
        te[c] = ((te[c] - mu) / sd).clip(-clip, clip)
        stats[c] = {"mean": mu, "std": sd}
    return tr, te, stats


def scale_sec_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sec_cols: List[str],
    clip: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    tr = train_df.copy()
    te = test_df.copy()
    stats: Dict[str, Dict[str, float]] = {}
    for c in sec_cols:
        mu = float(tr[c].mean())
        sd = float(tr[c].std(ddof=0))
        if sd <= 1e-12:
            sd = 1.0
        tr[c] = ((tr[c] - mu) / sd).clip(-clip, clip)
        te[c] = ((te[c] - mu) / sd).clip(-clip, clip)
        stats[c] = {"mean": mu, "std": sd}
    return tr, te, stats
