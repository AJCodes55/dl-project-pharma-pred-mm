#!/usr/bin/env python3
"""
Collect SEC 10-Q/10-K filings and build Phase 6 strict inputs:
- sec_filings/filings_metadata.csv
- sec_filings/text/<filing_id>.txt

This script aligns filing dates to the project window from feature_config.json:
train_start ... test_end (inclusive).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect SEC filings for Phase 6 strict pipeline")
    p.add_argument("--project-root", default="", help="Optional project root")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--company-name", required=True, help="SEC user-agent company name")
    p.add_argument("--email", required=True, help="SEC user-agent contact email")
    p.add_argument("--forms", default="10-Q,10-K", help="Comma-separated SEC form types")
    p.add_argument("--output-root", default="sec_filings", help="Output folder for metadata/text files")
    p.add_argument(
        "--download-root",
        default="sec_filings/raw_downloads",
        help="Folder where sec-edgar-downloader stores raw filings",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing text/metadata outputs")
    return p.parse_args()


def resolve_path(project_root: Path | None, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute() or project_root is None:
        return p
    return project_root / p


def _clean_form_name(form: str) -> str:
    return form.upper().replace("/", "-").strip()


def _parse_filed_date_from_submission(submission_text: str) -> Optional[pd.Timestamp]:
    m = re.search(r"FILED AS OF DATE:\s*(\d{8})", submission_text)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
    m = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", submission_text)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", errors="coerce").normalize()
    return None


def _select_best_text_source(folder: Path) -> Optional[Path]:
    preferred = [
        folder / "filing-details.html",
        folder / "filing-details.htm",
        folder / "full-submission.txt",
    ]
    for p in preferred:
        if p.exists():
            return p

    # Fallback to first text-like document.
    candidates = sorted(folder.glob("*.htm")) + sorted(folder.glob("*.html")) + sorted(folder.glob("*.txt"))
    return candidates[0] if candidates else None


def _extract_filing_date_and_source(folder: Path) -> Tuple[Optional[pd.Timestamp], Optional[Path]]:
    submission = folder / "full-submission.txt"
    filed_at = None
    if submission.exists():
        txt = submission.read_text(encoding="utf-8", errors="ignore")
        filed_at = _parse_filed_date_from_submission(txt)
    src = _select_best_text_source(folder)
    return filed_at, src


def _build_filing_id(ticker: str, form_type: str, filed_at: pd.Timestamp, accession: str) -> str:
    form_clean = _clean_form_name(form_type).replace("-", "")
    date_str = filed_at.strftime("%Y%m%d")
    acc_clean = re.sub(r"[^0-9A-Za-z]+", "", accession)
    return f"{ticker.lower()}_{form_clean.lower()}_{date_str}_{acc_clean.lower()}"


def _collect_for_ticker_form(
    raw_root: Path,
    ticker: str,
    form_type: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[Dict]:
    out: List[Dict] = []
    form_dir = raw_root / "sec-edgar-filings" / ticker / form_type
    if not form_dir.exists():
        return out

    for filing_folder in sorted([p for p in form_dir.iterdir() if p.is_dir()]):
        accession = filing_folder.name
        filed_at, src = _extract_filing_date_and_source(filing_folder)
        if filed_at is None or pd.isna(filed_at) or src is None:
            continue
        filed_at = pd.Timestamp(filed_at).normalize()
        if filed_at < start_date or filed_at > end_date:
            continue
        out.append(
            {
                "ticker": ticker,
                "form_type": form_type,
                "filed_at": filed_at,
                "accession": accession,
                "source_path": str(src),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else None

    feature_config_path = resolve_path(project_root, args.feature_config_path)
    output_root = resolve_path(project_root, args.output_root)
    download_root = resolve_path(project_root, args.download_root)

    output_text_dir = output_root / "text"
    output_root.mkdir(parents=True, exist_ok=True)
    output_text_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(feature_config_path.read_text())
    tickers = [str(t).upper() for t in cfg["tickers"]]
    start_date = pd.Timestamp(cfg["train_start"]).normalize()
    end_date = pd.Timestamp(cfg["test_end"]).normalize()
    forms = [_clean_form_name(f) for f in args.forms.split(",") if f.strip()]
    if not forms:
        raise ValueError("No forms provided. Example: --forms 10-Q,10-K")

    try:
        from sec_edgar_downloader import Downloader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sec-edgar-downloader is required. Install with:\n"
            "pip install sec-edgar-downloader"
        ) from exc

    dl = Downloader(
        args.company_name,
        args.email,
        str(download_root),
    )

    print("Downloading SEC filings...")
    print("Project date range:", start_date.date(), "to", end_date.date())
    print("Tickers:", tickers)
    print("Forms:", forms)

    # SEC `before` is exclusive; add one day for inclusive end_date.
    before_exclusive = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    after_inclusive = start_date.strftime("%Y-%m-%d")
    for t in tickers:
        for form in forms:
            try:
                dl.get(form, t, after=after_inclusive, before=before_exclusive)
            except Exception as exc:
                print(f"[WARN] download failed for {t} {form}: {exc}")

    rows: List[Dict] = []
    for t in tickers:
        for form in forms:
            rows.extend(
                _collect_for_ticker_form(
                    raw_root=download_root,
                    ticker=t,
                    form_type=form,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

    if not rows:
        raise RuntimeError(
            "No filings collected. Check internet access, SEC rate limits, and sec-edgar-downloader output."
        )

    df = pd.DataFrame(rows).sort_values(["ticker", "filed_at", "form_type", "accession"]).drop_duplicates(
        subset=["ticker", "form_type", "filed_at", "accession"], keep="first"
    )

    metadata_rows: List[Dict] = []
    for _, r in df.iterrows():
        filing_id = _build_filing_id(
            ticker=r["ticker"],
            form_type=r["form_type"],
            filed_at=pd.Timestamp(r["filed_at"]),
            accession=str(r["accession"]),
        )
        dst_name = f"{filing_id}.txt"
        dst_path = output_text_dir / dst_name

        if dst_path.exists() and not args.overwrite:
            pass
        else:
            src = Path(str(r["source_path"]))
            if not src.exists():
                continue
            # Keep raw content; downstream pipeline handles html cleanup.
            shutil.copyfile(src, dst_path)

        metadata_rows.append(
            {
                "filing_id": filing_id,
                "ticker": str(r["ticker"]),
                "filed_at": pd.Timestamp(r["filed_at"]).strftime("%Y-%m-%d"),
                "form_type": str(r["form_type"]),
                "text_path": dst_name,
                "source_path": str(r["source_path"]),
            }
        )

    if not metadata_rows:
        raise RuntimeError("No metadata rows built after file copy. Check downloaded raw filings.")

    meta = pd.DataFrame(metadata_rows).sort_values(["ticker", "filed_at", "form_type", "filing_id"]).reset_index(drop=True)
    metadata_path = output_root / "filings_metadata.csv"
    meta.to_csv(metadata_path, index=False)

    print("\nSaved strict Phase 6 SEC inputs:")
    print(" -", metadata_path)
    print(" -", output_text_dir)
    print("Rows:", len(meta), "| Tickers:", meta["ticker"].nunique(), "| Date range:", meta["filed_at"].min(), "to", meta["filed_at"].max())


if __name__ == "__main__":
    main()
