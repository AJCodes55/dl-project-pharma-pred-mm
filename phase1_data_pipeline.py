#!/usr/bin/env python3
"""
Phase 1 data pipeline for PharmaTrade-MM.

Builds:
1) Enhanced sentiment with exponential decay forward-fill
2) Daily FDA timing + event context features
3) Unified dataset merged on (date, ticker)
4) Train/test splits
5) XPH benchmark in standardized schema
6) feature_config.json for downstream model code
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Local-first defaults for this project workspace.
ZIP_PATH = "All data DL project.zip"
DATA_DIR = "All data DL project"
OUT_DIR = "processed"
EXTRACT_ZIP = False

TRAIN_START = "2018-02-20"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-03"
TEST_END = "2023-12-29"

MODEL_TICKERS = ["PFE", "JNJ", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB"]
EVENT_TYPES = ["ADCOM", "APPROVAL", "CRL"]
THERAPEUTIC_AREAS = [
    "Bone Disease",
    "Cardiology",
    "Dermatology",
    "Immunology",
    "Infectious Disease",
    "Neurology",
    "Oncology",
    "Psychiatry",
]


@dataclass
class Config:
    data_dir: Path
    out_dir: Path
    zip_path: Path
    extract_zip: bool
    sentiment_half_life: float
    clinical_trial_events_path: Path


def ensure_data_dir(config: Config) -> Path:
    data_dir = config.data_dir
    if config.extract_zip and config.zip_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(config.zip_path, "r") as zf:
            zf.extractall(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # If extraction created one nested folder, auto-descend.
    children = [p for p in data_dir.iterdir() if p.is_dir()]
    csv_count_here = len(list(data_dir.glob("*.csv")))
    if csv_count_here == 0 and len(children) == 1:
        nested = children[0]
        if len(list(nested.glob("*.csv"))) > 0:
            data_dir = nested

    return data_dir


def load_price(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "price_technicals.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def validate_price_alignment(price_df: pd.DataFrame) -> None:
    counts = price_df.groupby("ticker")["date"].nunique()
    min_cnt, max_cnt = int(counts.min()), int(counts.max())
    if min_cnt != max_cnt:
        print("WARNING: Date counts differ by ticker:")
        print(counts.to_string())
    else:
        print(f"Date alignment OK: each ticker has {min_cnt} rows.")


def load_sentiment(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "sentiment_daily.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def apply_sentiment_decay(sentiment_df: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """
    Exponential decay forward-fill:
    - filing days (n_filings > 0): use true sentiment_pos/neg
    - non-filing days: carry prior values with decay
    - set neutral as 1 - (pos + neg), clipped to [0, 1]
    """
    decay = math.exp(math.log(0.5) / half_life)
    out: List[pd.DataFrame] = []

    for ticker, grp in sentiment_df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        last_pos = 0.0
        last_neg = 0.0
        dec_pos = []
        dec_neg = []
        dec_neu = []
        dec_net = []

        for _, row in g.iterrows():
            filings = float(row.get("n_filings", 0) or 0)
            pos_raw = float(row.get("sentiment_pos", 0) or 0)
            neg_raw = float(row.get("sentiment_neg", 0) or 0)

            if filings > 0:
                last_pos = max(0.0, pos_raw)
                last_neg = max(0.0, neg_raw)
            else:
                last_pos *= decay
                last_neg *= decay

            neu = max(0.0, min(1.0, 1.0 - (last_pos + last_neg)))
            net = last_pos - last_neg

            dec_pos.append(last_pos)
            dec_neg.append(last_neg)
            dec_neu.append(neu)
            dec_net.append(net)

        g["sent_pos"] = dec_pos
        g["sent_neg"] = dec_neg
        g["sent_neu"] = dec_neu
        g["sent_net"] = dec_net
        out.append(g[["date", "ticker", "sent_pos", "sent_neg", "sent_neu", "sent_net", "n_filings"]])

    return pd.concat(out, ignore_index=True)


def load_fda_events(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "fda_event_calendar.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["event_type"] = df["event_type"].astype(str).str.upper()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_hist_reaction_map(fda_df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    train_end = pd.Timestamp(TRAIN_END)
    train_events = fda_df[fda_df["date"] <= train_end].copy()
    grouped = (
        train_events.groupby(["ticker", "event_type"], as_index=False)[["px_1d_ret_pct", "px_5d_ret_pct"]]
        .mean(numeric_only=True)
        .fillna(0.0)
    )

    reaction_map: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for _, row in grouped.iterrows():
        reaction_map[(row["ticker"], row["event_type"])] = (
            float(row["px_1d_ret_pct"]),
            float(row["px_5d_ret_pct"]),
        )
    return reaction_map


def build_fda_daily_features(price_df: pd.DataFrame, fda_df: pd.DataFrame) -> pd.DataFrame:
    reaction_map = build_hist_reaction_map(fda_df)
    rows: List[dict] = []

    for ticker, pgrp in price_df.groupby("ticker", sort=False):
        events = fda_df[fda_df["ticker"] == ticker].sort_values("date").copy()
        event_dates = events["date"].tolist()

        for as_of in pgrp["date"].tolist():
            nearest = None
            dte = 999

            for idx, ed in enumerate(event_dates):
                delta = (ed - as_of).days
                if delta >= 0:
                    dte = min(delta, 90)
                    nearest = events.iloc[idx]
                    break

            event_type = None
            area = None
            hist1 = 0.0
            hist5 = 0.0
            is_window = 0

            if nearest is not None:
                event_type = str(nearest["event_type"]).upper()
                area = str(nearest["therapeutic_area"])
                hist1, hist5 = reaction_map.get((ticker, event_type), (0.0, 0.0))
                is_window = int(dte <= 5)

            rec = {
                "date": as_of,
                "ticker": ticker,
                "days_to_event": int(dte),
                "is_event_window": is_window,
                "hist_1d_ret": hist1,
                "hist_5d_ret": hist5,
            }

            for ev in EVENT_TYPES:
                rec[f"evt_{ev}"] = int(event_type == ev)

            for ta in THERAPEUTIC_AREAS:
                key = ta.replace(" ", "_")
                rec[f"ta_{key}"] = int(area == ta)

            rows.append(rec)

    return pd.DataFrame(rows)


def build_xph_processed(data_dir: Path) -> pd.DataFrame:
    xph = pd.read_csv(data_dir / "xph_benchmark.csv")
    xph = xph.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    xph["date"] = pd.to_datetime(xph["date"])
    xph["ticker"] = "XPH"
    xph = xph.sort_values("date").reset_index(drop=True)
    return xph


def load_clinical_trial_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Clinical trial file must contain at least: date, ticker")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if "event_type" not in df.columns:
        df["event_type"] = ""
    if "trial_phase" not in df.columns:
        df["trial_phase"] = ""
    return df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_clinical_trial_daily_features(price_df: pd.DataFrame, ct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe trial features using only known events up to each trading day.

    Features:
      - ct_phase2_events_last_5d
      - ct_phase3_events_last_5d
      - ct_results_posted_last_5d
      - ct_terminated_last_20d
      - ct_events_last_20d
    """
    rows: List[pd.DataFrame] = []
    for ticker, pgrp in price_df.groupby("ticker", sort=False):
        trading_dates = pd.Index(sorted(pgrp["date"].unique()))
        base = pd.DataFrame(index=trading_dates)
        base["ct_phase2_event"] = 0
        base["ct_phase3_event"] = 0
        base["ct_results_posted_event"] = 0
        base["ct_terminated_event"] = 0
        base["ct_any_event"] = 0

        tdf = ct_df[ct_df["ticker"] == ticker].copy()
        for _, r in tdf.iterrows():
            event_date = pd.Timestamp(r["date"])
            idx = trading_dates.searchsorted(event_date, side="left")
            if idx >= len(trading_dates):
                continue
            aligned_date = trading_dates[idx]

            phase_text = str(r.get("trial_phase", "")).upper()
            event_type = str(r.get("event_type", "")).upper()
            is_phase2 = "PHASE2" in phase_text or "PHASE 2" in phase_text
            is_phase3 = "PHASE3" in phase_text or "PHASE 3" in phase_text
            is_results = "RESULTS_POSTED" in event_type
            is_terminated = (
                "TERMINATED" in event_type
                or "WITHDRAWN" in event_type
                or "SUSPENDED" in event_type
            )

            base.at[aligned_date, "ct_any_event"] += 1
            if is_phase2:
                base.at[aligned_date, "ct_phase2_event"] += 1
            if is_phase3:
                base.at[aligned_date, "ct_phase3_event"] += 1
            if is_results:
                base.at[aligned_date, "ct_results_posted_event"] += 1
            if is_terminated:
                base.at[aligned_date, "ct_terminated_event"] += 1

        out = pd.DataFrame(
            {
                "date": trading_dates,
                "ticker": ticker,
                "ct_phase2_events_last_5d": base["ct_phase2_event"].rolling(window=5, min_periods=1).sum().values,
                "ct_phase3_events_last_5d": base["ct_phase3_event"].rolling(window=5, min_periods=1).sum().values,
                "ct_results_posted_last_5d": base["ct_results_posted_event"].rolling(window=5, min_periods=1).sum().values,
                "ct_terminated_last_20d": base["ct_terminated_event"].rolling(window=20, min_periods=1).sum().values,
                "ct_events_last_20d": base["ct_any_event"].rolling(window=20, min_periods=1).sum().values,
            }
        )
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def split_train_test(unified: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = unified[(unified["date"] >= pd.Timestamp(TRAIN_START)) & (unified["date"] <= pd.Timestamp(TRAIN_END))]
    test = unified[(unified["date"] >= pd.Timestamp(TEST_START)) & (unified["date"] <= pd.Timestamp(TEST_END))]
    return train.copy(), test.copy()


def to_feature_config(unified: pd.DataFrame, out_dir: Path, half_life: float) -> None:
    price_features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_pct",
        "bb_width",
        "log_return",
        "volume_sma20",
        "volume_ratio",
    ]
    sentiment_features = ["sent_pos", "sent_neg", "sent_neu", "sent_net", "n_filings"]
    fda_features = [
        c
        for c in unified.columns
        if c in {"days_to_event", "is_event_window", "hist_1d_ret", "hist_5d_ret"}
        or c.startswith("evt_")
        or c.startswith("ta_")
    ]
    clinical_trial_features = [
        c
        for c in unified.columns
        if c.startswith("ct_")
    ]

    payload = {
        "tickers": MODEL_TICKERS,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "sentiment_half_life_days": half_life,
        "price_features": price_features,
        "sentiment_features": sentiment_features,
        "fda_features": sorted(fda_features),
        "clinical_trial_features": sorted(clinical_trial_features),
        "target_column": "close",
    }
    (out_dir / "feature_config.json").write_text(json.dumps(payload, indent=2))


def run_pipeline(config: Config) -> None:
    data_dir = ensure_data_dir(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    price = load_price(data_dir)
    # Keep the 8 model tickers; XPH is handled separately as benchmark.
    price = price[price["ticker"].isin(MODEL_TICKERS)].copy()
    validate_price_alignment(price)

    sentiment = load_sentiment(data_dir)
    sentiment = sentiment[sentiment["ticker"].isin(MODEL_TICKERS)].copy()
    sentiment_enh = apply_sentiment_decay(sentiment, config.sentiment_half_life)

    fda = load_fda_events(data_dir)
    fda = fda[fda["ticker"].isin(MODEL_TICKERS)].copy()
    fda_daily = build_fda_daily_features(price, fda)

    ct_daily = None
    if config.clinical_trial_events_path.exists():
        ct = load_clinical_trial_events(config.clinical_trial_events_path)
        ct = ct[ct["ticker"].isin(MODEL_TICKERS)].copy()
        ct_daily = build_clinical_trial_daily_features(price, ct)
        print(
            f"Loaded clinical trial events from {config.clinical_trial_events_path} "
            f"({len(ct):,} rows)."
        )
    else:
        print(
            f"Clinical trial file not found at {config.clinical_trial_events_path}. "
            "Continuing without ct_* features."
        )

    unified = price.merge(sentiment_enh, on=["date", "ticker"], how="left")
    unified = unified.merge(fda_daily, on=["date", "ticker"], how="left")
    if ct_daily is not None:
        unified = unified.merge(ct_daily, on=["date", "ticker"], how="left")

    # Fill defaults for unmatched FDA rows.
    if "days_to_event" in unified.columns:
        unified["days_to_event"] = unified["days_to_event"].fillna(999)
    for col in unified.columns:
        if col in {"date", "ticker", "days_to_event"}:
            continue
        if pd.api.types.is_numeric_dtype(unified[col]):
            unified[col] = unified[col].fillna(0.0)

    unified = unified.sort_values(["date", "ticker"]).reset_index(drop=True)
    train, test = split_train_test(unified)
    xph = build_xph_processed(data_dir)

    unified.to_csv(config.out_dir / "unified_dataset.csv", index=False)
    train.to_csv(config.out_dir / "train_dataset.csv", index=False)
    test.to_csv(config.out_dir / "test_dataset.csv", index=False)
    xph.to_csv(config.out_dir / "xph_processed.csv", index=False)
    to_feature_config(unified, config.out_dir, config.sentiment_half_life)

    print("\nPipeline complete.")
    print(f"data_dir: {data_dir}")
    print(f"out_dir:  {config.out_dir}")
    print(f"unified rows: {len(unified):,}")
    print(f"train rows:   {len(train):,}")
    print(f"test rows:    {len(test):,}")
    print(f"xph rows:     {len(xph):,}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="PharmaTrade-MM Phase 1 data pipeline")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing source CSVs")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for processed files")
    parser.add_argument("--zip-path", default=ZIP_PATH, help="Zip file path (optional extraction)")
    parser.add_argument(
        "--extract-zip",
        action="store_true",
        default=EXTRACT_ZIP,
        help="Extract zip file into data dir before processing",
    )
    parser.add_argument(
        "--no-extract-zip",
        action="store_false",
        dest="extract_zip",
        help="Skip zip extraction",
    )
    parser.add_argument(
        "--sentiment-half-life",
        type=float,
        default=5.0,
        help="Half-life (days) for sentiment decay forward-fill",
    )
    parser.add_argument(
        "--clinical-trial-events",
        default="clinical_trials/clinical_trial_event_calendar_fda_matched.csv",
        help="Path to clinical trial event calendar CSV",
    )
    args = parser.parse_args()

    return Config(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        zip_path=Path(args.zip_path),
        extract_zip=bool(args.extract_zip),
        sentiment_half_life=float(args.sentiment_half_life),
        clinical_trial_events_path=Path(args.clinical_trial_events),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
