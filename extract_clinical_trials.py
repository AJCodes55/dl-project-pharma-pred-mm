#!/usr/bin/env python3
"""
Extract clinical trial data for PharmaTrade-MM from ClinicalTrials.gov API v2.

Outputs:
1) raw_trials.csv
2) clinical_trial_event_calendar.csv

Why two files?
- raw_trials.csv keeps wide trial metadata for audit/debug.
- clinical_trial_event_calendar.csv is event-formatted for feature engineering.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Sponsor aliases to improve recall beyond ticker symbols.
TICKER_ALIASES: Dict[str, List[str]] = {
    "ABBV": ["AbbVie", "Abbvie"],
    "AMGN": ["Amgen"],
    "BIIB": ["Biogen"],
    "BMY": ["Bristol Myers Squibb", "Bristol-Myers Squibb", "Celgene"],
    "GILD": ["Gilead"],
    "JNJ": ["Johnson & Johnson", "Janssen", "Janssen Pharmaceuticals"],
    "MRK": ["Merck", "MSD"],
    "PFE": ["Pfizer"],
}


def _safe_get(d: dict, path: Iterable[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _date_from_struct(st: Optional[dict]) -> Optional[str]:
    if not isinstance(st, dict):
        return None
    # v2 usually exposes "date" and may include type info; we only need the date token.
    return st.get("date")


def normalize_date(x) -> pd.Timestamp:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT
    # Handles YYYY, YYYY-MM, YYYY-MM-DD.
    return pd.to_datetime(s, errors="coerce")


def fetch_studies_for_term(term: str, page_size: int = 1000, sleep_s: float = 0.1) -> List[dict]:
    studies: List[dict] = []
    page_token = None

    while True:
        params = {"query.term": term, "format": "json", "pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        batch = payload.get("studies", [])
        studies.extend(batch)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
        time.sleep(sleep_s)

    return studies


def infer_ticker_from_text(text: str) -> Optional[str]:
    txt = (text or "").lower()
    for ticker, aliases in TICKER_ALIASES.items():
        for alias in aliases:
            if alias.lower() in txt:
                return ticker
    return None


def extract_trial_row(study: dict) -> dict:
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    design = ps.get("designModule", {})
    arms = ps.get("armsInterventionsModule", {})
    cond = ps.get("conditionsModule", {})

    nct_id = ident.get("nctId")
    brief_title = ident.get("briefTitle")
    official_title = ident.get("officialTitle")

    lead_sponsor = _safe_get(sponsor_mod, ["leadSponsor", "name"], "")
    collaborators = sponsor_mod.get("collaborators", []) or []
    collab_names = [c.get("name", "") for c in collaborators if isinstance(c, dict)]

    phases = design.get("phases", []) or []
    if isinstance(phases, str):
        phases = [phases]
    phase_text = "|".join(phases)

    study_type = design.get("studyType")
    overall_status = status.get("overallStatus")

    start_date = _date_from_struct(status.get("startDateStruct"))
    primary_completion_date = _date_from_struct(status.get("primaryCompletionDateStruct"))
    completion_date = _date_from_struct(status.get("completionDateStruct"))
    results_first_posted_date = _date_from_struct(status.get("resultsFirstPostDateStruct"))
    last_update_posted_date = _date_from_struct(status.get("lastUpdatePostDateStruct"))

    interventions = arms.get("interventions", []) or []
    intervention_names = [x.get("name", "") for x in interventions if isinstance(x, dict)]

    conditions = cond.get("conditions", []) or []
    if not isinstance(conditions, list):
        conditions = [str(conditions)]

    ticker_text = " | ".join(
        [
            str(lead_sponsor or ""),
            str(brief_title or ""),
            str(official_title or ""),
            " | ".join(collab_names),
            " | ".join(intervention_names),
        ]
    )
    ticker = infer_ticker_from_text(ticker_text)

    return {
        "nct_id": nct_id,
        "ticker": ticker,
        "brief_title": brief_title,
        "official_title": official_title,
        "lead_sponsor": lead_sponsor,
        "collaborators": "|".join(collab_names),
        "study_type": study_type,
        "overall_status": overall_status,
        "phases": phase_text,
        "interventions": "|".join(intervention_names),
        "conditions": "|".join([str(c) for c in conditions]),
        "start_date": start_date,
        "primary_completion_date": primary_completion_date,
        "completion_date": completion_date,
        "results_first_posted_date": results_first_posted_date,
        "last_update_posted_date": last_update_posted_date,
        "source": "clinicaltrials.gov_api_v2",
    }


def build_event_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    date_events: List[Tuple[str, str]] = [
        ("start_date", "TRIAL_START"),
        ("primary_completion_date", "PRIMARY_COMPLETION"),
        ("completion_date", "STUDY_COMPLETION"),
        ("results_first_posted_date", "RESULTS_POSTED"),
    ]

    for _, r in raw_df.iterrows():
        base = {
            "ticker": r["ticker"],
            "nct_id": r["nct_id"],
            "trial_phase": r["phases"],
            "study_status": r["overall_status"],
            "lead_sponsor": r["lead_sponsor"],
            "interventions": r["interventions"],
            "source": r["source"],
        }

        for col, event_type in date_events:
            d = r[col]
            if pd.notna(d):
                rows.append(
                    {
                        **base,
                        "date": pd.to_datetime(d),
                        "event_type": event_type,
                        "event_date_source_col": col,
                        "is_inferred_status_date": 0,
                    }
                )

        # If terminated/withdrawn/suspended, we add a status-change event using last update date as proxy.
        status_txt = str(r["overall_status"] or "").upper()
        if any(x in status_txt for x in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]):
            d = r["last_update_posted_date"]
            if pd.notna(d):
                rows.append(
                    {
                        **base,
                        "date": pd.to_datetime(d),
                        "event_type": f"STATUS_{status_txt}",
                        "event_date_source_col": "last_update_posted_date",
                        "is_inferred_status_date": 1,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["date", "ticker", "event_type"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract trial events from ClinicalTrials.gov")
    parser.add_argument("--out-dir", default="clinical_trials", help="Output directory")
    parser.add_argument("--start-date", default="2012-01-01", help="Filter event date >= start")
    parser.add_argument("--end-date", default="2023-12-31", help="Filter event date <= end")
    parser.add_argument(
        "--require-interventional",
        action="store_true",
        default=True,
        help="Keep only interventional studies",
    )
    parser.add_argument(
        "--include-phase4",
        action="store_true",
        default=True,
        help="Include phase 4 studies (default True)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull by alias term and dedupe by nct_id later.
    all_studies: List[dict] = []
    for ticker, aliases in TICKER_ALIASES.items():
        for alias in aliases:
            print(f"Fetching studies for term: {alias}")
            try:
                studies = fetch_studies_for_term(alias)
            except Exception as e:
                print(f"WARNING: fetch failed for {alias}: {e}")
                continue
            print(f"  fetched {len(studies)}")
            all_studies.extend(studies)

    if not all_studies:
        raise RuntimeError("No studies fetched. Check network/access and try again.")

    raw = pd.DataFrame([extract_trial_row(s) for s in all_studies])
    raw = raw.dropna(subset=["nct_id"]).drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

    # Normalize dates
    for col in [
        "start_date",
        "primary_completion_date",
        "completion_date",
        "results_first_posted_date",
        "last_update_posted_date",
    ]:
        raw[col] = raw[col].apply(normalize_date)

    # Keep only mapped tickers for this project.
    raw = raw[raw["ticker"].isin(TICKER_ALIASES.keys())].copy()

    print(f"Rows after nct dedupe: {len(raw):,}")

    # Optional filters
    if args.require_interventional:
        before = len(raw)
        raw = raw[raw["study_type"].fillna("").str.upper().eq("INTERVENTIONAL")].copy()
        print(f"Rows after interventional filter: {len(raw):,} (dropped {before - len(raw):,})")

    # Keep phase 1/2/3 (+ optional 4)
    # API values are usually PHASE1/PHASE2/PHASE3/PHASE4 (often no space).
    # We support both with and without spaces, and EARLY_PHASE1.
    phase_text = raw["phases"].fillna("").str.upper()
    phase_mask = phase_text.str.contains(r"EARLY[_\s]*PHASE[_\s]*1|PHASE[_\s]*1|PHASE[_\s]*2|PHASE[_\s]*3", regex=True)
    if args.include_phase4:
        phase_mask = phase_mask | phase_text.str.contains(r"PHASE[_\s]*4", regex=True)
    before = len(raw)
    raw = raw[phase_mask].copy()
    print(f"Rows after phase filter: {len(raw):,} (dropped {before - len(raw):,})")

    raw = raw.sort_values(["ticker", "nct_id"]).reset_index(drop=True)
    raw_path = out_dir / "raw_trials.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} ({len(raw):,} rows)")

    events = build_event_rows(raw)
    if events.empty:
        print("No event rows generated from raw trials.")
        events_path = out_dir / "clinical_trial_event_calendar.csv"
        events.to_csv(events_path, index=False)
        print(f"Saved empty {events_path}")
        return

    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    events = events[(events["date"] >= start) & (events["date"] <= end)].copy()
    events = events.sort_values(["date", "ticker", "event_type"]).reset_index(drop=True)

    events_path = out_dir / "clinical_trial_event_calendar.csv"
    events.to_csv(events_path, index=False)
    print(f"Saved {events_path} ({len(events):,} rows)")

    # Quick QA summary
    print("\nQA summary")
    print("----------")
    print("Rows by ticker:")
    print(events["ticker"].value_counts().sort_index().to_string())
    print("\nRows by event_type:")
    print(events["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
