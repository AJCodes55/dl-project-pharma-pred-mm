#!/usr/bin/env python3
"""
Phase 5 sequence contract and preprocessing utilities (FDA ablations).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Phase5SequenceContract:
    tickers: List[str]
    price_features: List[str]
    sentiment_features: List[str]
    event_features: List[str]
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
    def event_shape(self) -> Tuple[int, int, int]:
        return (self.window_size, len(self.tickers), len(self.event_features))


def _select_event_features(cfg: Dict, ablation: str) -> List[str]:
    all_fda = list(cfg["fda_features"])
    if ablation == "rich_fda":
        return all_fda
    if ablation == "rich_fda_ct":
        ct = list(cfg.get("clinical_trial_features", []))
        # Confound context feature is derived in runtime from trailing ct_* signals.
        return list(dict.fromkeys(all_fda + ct + ["ct_confound_flag_5d"]))

    # Ablation 4 (basic FDA): days_to_event + FDA event type one-hots.
    basic = []
    if "days_to_event" in all_fda:
        basic.append("days_to_event")
    basic.extend([c for c in all_fda if c.startswith("evt_")])
    if "is_event_window" in all_fda:
        basic.append("is_event_window")
    return list(dict.fromkeys(basic))


def load_phase5_sequence_contract(
    feature_config_path: str | Path,
    window_size: int,
    ablation: str,
) -> Phase5SequenceContract:
    if ablation not in {"basic_fda", "rich_fda", "rich_fda_ct"}:
        raise ValueError("ablation must be one of: basic_fda, rich_fda, rich_fda_ct")
    if window_size < 2:
        raise ValueError("window_size must be >= 2")

    cfg = json.loads(Path(feature_config_path).read_text())
    return Phase5SequenceContract(
        tickers=list(cfg["tickers"]),
        price_features=list(cfg["price_features"]),
        sentiment_features=list(cfg["sentiment_features"]),
        event_features=_select_event_features(cfg, ablation=ablation),
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
