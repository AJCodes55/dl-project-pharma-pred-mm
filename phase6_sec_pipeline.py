#!/usr/bin/env python3
"""
Phase 6 SEC feature engineering utilities.

Builds daily SEC-related features from sparse filing-day sentiment files.
This is a no-lookahead pipeline: all daily features at t use filings with date <= t.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SEC_FEATURES = [
    "sec_sent_pos_decay",
    "sec_sent_neg_decay",
    "sec_sent_neu_decay",
    "sec_net_sent_decay",
    "sec_filing_count_today",
    "sec_filing_count_20d",
    "sec_days_since_filing",
    "sec_recent_filing_flag_5d",
]


def _safe_read_sparse_filing_files(sparse_data_dir: Path, tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        p = sparse_data_dir / f"sentiment_{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        if df.empty:
            continue
        df["ticker"] = t
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "sentiment_pos", "sentiment_neg", "sentiment_neu", "n_filings", "net_sentiment"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def _build_single_ticker_features(panel: pd.DataFrame, sparse: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    panel = panel.sort_values("date").copy()
    sparse = sparse.sort_values("date").copy()
    sparse = sparse.drop_duplicates(subset=["date"], keep="last")

    filing_map = sparse.set_index("date")
    decay = np.power(0.5, 1.0 / max(half_life_days, 1e-6))

    sec_pos = 0.0
    sec_neg = 0.0
    sec_neu = 1.0
    sec_net = 0.0
    last_filing_date = None
    rolling_counts: List[Tuple[pd.Timestamp, float]] = []
    out_rows = []

    for _, row in panel.iterrows():
        d = pd.Timestamp(row["date"])

        sec_pos *= decay
        sec_neg *= decay
        sec_net *= decay
        sec_neu = max(0.0, min(1.0, 1.0 - sec_pos - sec_neg))

        filing_count_today = 0.0
        if d in filing_map.index:
            hit = filing_map.loc[d]
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[-1]
            filing_count_today = float(hit.get("n_filings", 1.0))
            sec_pos = float(hit.get("sentiment_pos", sec_pos))
            sec_neg = float(hit.get("sentiment_neg", sec_neg))
            sec_neu = float(hit.get("sentiment_neu", sec_neu))
            sec_net = float(hit.get("net_sentiment", sec_net))
            last_filing_date = d

        rolling_counts.append((d, filing_count_today))
        cutoff = d - pd.Timedelta(days=20)
        rolling_counts = [(dd, v) for dd, v in rolling_counts if dd > cutoff]
        filing_count_20d = float(sum(v for _, v in rolling_counts))

        if last_filing_date is None:
            days_since = 999.0
        else:
            days_since = float((d - last_filing_date).days)
        recent_flag_5d = float(days_since <= 5.0)

        out_rows.append(
            {
                "date": d,
                "ticker": row["ticker"],
                "sec_sent_pos_decay": sec_pos,
                "sec_sent_neg_decay": sec_neg,
                "sec_sent_neu_decay": sec_neu,
                "sec_net_sent_decay": sec_net,
                "sec_filing_count_today": filing_count_today,
                "sec_filing_count_20d": filing_count_20d,
                "sec_days_since_filing": days_since,
                "sec_recent_filing_flag_5d": recent_flag_5d,
            }
        )
    return pd.DataFrame(out_rows)


def build_phase6_sec_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sparse_data_dir: str | Path,
    half_life_days: float = 14.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    sparse_dir = Path(sparse_data_dir)
    train = train_df.copy()
    test = test_df.copy()
    for df in (train, test):
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()

    tickers = sorted(train["ticker"].unique().tolist())
    sparse = _safe_read_sparse_filing_files(sparse_dir, tickers=tickers)

    full_panel = pd.concat([train[["date", "ticker"]], test[["date", "ticker"]]], ignore_index=True).drop_duplicates()
    sec_rows = []
    for t in tickers:
        pnl_t = full_panel[full_panel["ticker"] == t].copy()
        sp_t = sparse[sparse["ticker"] == t].copy() if not sparse.empty else sparse.copy()
        sec_rows.append(_build_single_ticker_features(pnl_t, sp_t, half_life_days=half_life_days))
    sec_daily = pd.concat(sec_rows, ignore_index=True)

    train_aug = train.merge(sec_daily, on=["date", "ticker"], how="left")
    test_aug = test.merge(sec_daily, on=["date", "ticker"], how="left")
    for c in DEFAULT_SEC_FEATURES:
        train_aug[c] = train_aug[c].fillna(0.0)
        test_aug[c] = test_aug[c].fillna(0.0)

    meta = {
        "sec_features": DEFAULT_SEC_FEATURES,
        "source": str(sparse_dir),
        "half_life_days": float(half_life_days),
        "num_sparse_rows": int(len(sparse)),
    }
    return train_aug, test_aug, meta


def save_sec_feature_config(path: str | Path, meta: Dict) -> None:
    p = Path(path)
    p.write_text(json.dumps(meta, indent=2))
