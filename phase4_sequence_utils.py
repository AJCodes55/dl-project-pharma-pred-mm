#!/usr/bin/env python3
"""
Phase 4 sequence contract and preprocessing utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class SequenceContract:
    tickers: List[str]
    price_features: List[str]
    sentiment_features: List[str]
    window_size: int

    @property
    def portfolio_context_dim(self) -> int:
        # position weights per asset + normalized cash
        return len(self.tickers) + 1

    @property
    def price_shape(self) -> Tuple[int, int, int]:
        return (self.window_size, len(self.tickers), len(self.price_features))

    @property
    def sentiment_shape(self) -> Tuple[int, int, int]:
        return (self.window_size, len(self.tickers), len(self.sentiment_features))


def load_sequence_contract(feature_config_path: str | Path, window_size: int) -> SequenceContract:
    cfg = json.loads(Path(feature_config_path).read_text())
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    return SequenceContract(
        tickers=list(cfg["tickers"]),
        price_features=list(cfg["price_features"]),
        sentiment_features=list(cfg["sentiment_features"]),
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
