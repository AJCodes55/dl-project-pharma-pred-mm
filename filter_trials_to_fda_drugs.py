#!/usr/bin/env python3
"""
Filter clinical trial files to keep only rows linked to FDA-event drugs.

Inputs:
- All data DL project/fda_event_calendar.csv
- clinical_trials/raw_trials.csv
- clinical_trials/clinical_trial_event_calendar.csv

Outputs:
- clinical_trials/raw_trials_fda_matched.csv
- clinical_trials/clinical_trial_event_calendar_fda_matched.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


STOP_TOKENS = {
    "full",
    "approval",
    "accelerated",
    "adcom",
    "denied",
    "vote",
    "expansion",
    "ra",
    "cd",
    "psa",
    "hcc",
    "nsclc",
    "escc",
    "1l",
    "2l",
    "3l",
    "covid",
    "vaccine",
    "mrd",
}


def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-\+\/\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cleanup_alias(a: str) -> str:
    a = norm_text(a)
    if not a:
        return ""
    parts = [p for p in re.split(r"\s+", a) if p and p not in STOP_TOKENS]
    return " ".join(parts).strip()


def aliases_from_drug_name(drug_name: str) -> Set[str]:
    raw = norm_text(drug_name)
    aliases: Set[str] = set()
    if not raw:
        return aliases

    # Remove suffix descriptors after dash.
    main = raw.split(" - ")[0].strip()
    aliases.add(main)

    # Parse parenthetical brand/generic names.
    parens = re.findall(r"\((.*?)\)", main)
    for p in parens:
        aliases.add(p.strip())
    main_wo_parens = re.sub(r"\(.*?\)", " ", main).strip()
    aliases.add(main_wo_parens)

    # Split combinations.
    for chunk in re.split(r"[\+\/]", main_wo_parens):
        aliases.add(chunk.strip())
    for p in parens:
        for chunk in re.split(r"[\+\/]", p):
            aliases.add(chunk.strip())

    cleaned: Set[str] = set()
    for a in aliases:
        c = cleanup_alias(a)
        if len(c) >= 4:
            cleaned.add(c)
    return cleaned


def build_ticker_aliases(fda_df: pd.DataFrame) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for ticker, grp in fda_df.groupby("ticker"):
        aset: Set[str] = set()
        for dn in grp["drug_name"].dropna().astype(str).tolist():
            aset |= aliases_from_drug_name(dn)
        # active_ingredient is mostly empty in your file, but keep it if present.
        if "active_ingredient" in grp.columns:
            for ai in grp["active_ingredient"].dropna().astype(str).tolist():
                c = cleanup_alias(ai)
                if len(c) >= 4:
                    aset.add(c)
        out[str(ticker).upper()] = aset
    return out


def row_matches_aliases(text: str, aliases: Set[str]) -> bool:
    txt = norm_text(text)
    if not txt or not aliases:
        return False
    return any(a in txt for a in aliases)


def filter_raw(raw_df: pd.DataFrame, ticker_aliases: Dict[str, Set[str]]) -> pd.DataFrame:
    cols = ["interventions", "brief_title", "official_title"]
    for c in cols:
        if c not in raw_df.columns:
            raw_df[c] = ""
    keep = []
    for _, r in raw_df.iterrows():
        t = str(r.get("ticker", "")).upper()
        aliases = ticker_aliases.get(t, set())
        text = " | ".join([str(r.get(c, "")) for c in cols])
        keep.append(row_matches_aliases(text, aliases))
    return raw_df[pd.Series(keep, index=raw_df.index)].copy()


def filter_events(ev_df: pd.DataFrame, ticker_aliases: Dict[str, Set[str]]) -> pd.DataFrame:
    if "interventions" not in ev_df.columns:
        ev_df["interventions"] = ""
    keep = []
    for _, r in ev_df.iterrows():
        t = str(r.get("ticker", "")).upper()
        aliases = ticker_aliases.get(t, set())
        text = str(r.get("interventions", ""))
        keep.append(row_matches_aliases(text, aliases))
    return ev_df[pd.Series(keep, index=ev_df.index)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter clinical trial rows to FDA-drug matches")
    parser.add_argument("--fda", default="All data DL project/fda_event_calendar.csv")
    parser.add_argument("--raw", default="clinical_trials/raw_trials.csv")
    parser.add_argument("--events", default="clinical_trials/clinical_trial_event_calendar.csv")
    parser.add_argument("--out-dir", default="clinical_trials")
    args = parser.parse_args()

    fda = pd.read_csv(args.fda)
    raw = pd.read_csv(args.raw)
    events = pd.read_csv(args.events)

    fda["ticker"] = fda["ticker"].astype(str).str.upper()
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    events["ticker"] = events["ticker"].astype(str).str.upper()

    ticker_aliases = build_ticker_aliases(fda)
    print("FDA alias counts by ticker:")
    for t in sorted(ticker_aliases):
        print(f"  {t}: {len(ticker_aliases[t])} aliases")

    raw_f = filter_raw(raw, ticker_aliases).reset_index(drop=True)
    events_f = filter_events(events, ticker_aliases).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / "raw_trials_fda_matched.csv"
    events_out = out_dir / "clinical_trial_event_calendar_fda_matched.csv"
    raw_f.to_csv(raw_out, index=False)
    events_f.to_csv(events_out, index=False)

    print("\nFilter results")
    print("--------------")
    print(f"raw_trials: {len(raw):,} -> {len(raw_f):,}")
    print(f"event_calendar: {len(events):,} -> {len(events_f):,}")
    if len(events_f) > 0:
        print("\nMatched event rows by ticker:")
        print(events_f["ticker"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
