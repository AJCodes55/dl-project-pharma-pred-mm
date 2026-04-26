#!/usr/bin/env python3
"""
Phase 6 strict SEC pipeline:
- load SEC 10-Q/10-K filing metadata + text
- extract MD&A-like section
- build FinBERT CLS embeddings per filing
- optional PCA compression
- forward-fill filing embeddings to daily ticker panel (no lookahead)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 FinBERT SEC embedding pipeline")
    p.add_argument("--project-root", default="", help="Optional project root.")
    p.add_argument("--feature-config-path", default="processed/feature_config.json")
    p.add_argument("--train-path", default="processed/train_dataset.csv")
    p.add_argument("--test-path", default="processed/test_dataset.csv")
    p.add_argument("--filings-metadata-path", default="sec_filings/filings_metadata.csv")
    p.add_argument("--filings-text-root", default="sec_filings/text")
    p.add_argument("--output-filings-emb-path", default="processed/sec_filing_embeddings.csv")
    p.add_argument("--output-daily-emb-path", default="processed/sec_daily_embeddings.csv")
    p.add_argument("--output-feature-config-path", default="processed/sec_embedding_feature_config.json")
    p.add_argument("--model-name", default="ProsusAI/finbert")
    p.add_argument("--chunk-size", type=int, default=420)
    p.add_argument("--chunk-overlap", type=int, default=80)
    p.add_argument("--pca-dim", type=int, default=64, help="0 disables PCA compression.")
    p.add_argument("--max-chunks-per-filing", type=int, default=128)
    return p.parse_args()


def resolve_path(project_root: Path | None, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute() or project_root is None:
        return p
    return project_root / p


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html_tags(text: str) -> str:
    # Fast, dependency-free text fallback for HTML/XBRL.
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _normalize_whitespace(text)


def _extract_mda_section(raw_text: str) -> str:
    txt = raw_text
    low = txt.lower()
    # Heuristic MD&A window extraction for 10-Q/10-K.
    candidates = [
        "management's discussion and analysis",
        "management discussion and analysis",
        "item 2. management",
        "item 7. management",
    ]
    start = -1
    for pat in candidates:
        idx = low.find(pat)
        if idx >= 0:
            start = idx
            break
    if start < 0:
        return _normalize_whitespace(txt)

    end_markers = [
        "quantitative and qualitative disclosures",
        "controls and procedures",
        "financial statements and supplementary data",
        "item 3.",
        "item 7a",
        "item 8.",
    ]
    end = len(txt)
    tail_low = low[start + 1 :]
    for marker in end_markers:
        idx = tail_low.find(marker)
        if idx >= 0:
            end = min(end, start + 1 + idx)
    return _normalize_whitespace(txt[start:end])


def _load_filing_text(row: pd.Series, filings_text_root: Path) -> str:
    text_path = row.get("text_path")
    if isinstance(text_path, str) and text_path.strip():
        p = Path(text_path.strip())
        if not p.is_absolute():
            p = filings_text_root / p
    else:
        filing_id = str(row.get("filing_id", "")).strip()
        if not filing_id:
            raise ValueError("filing_id missing and text_path missing.")
        # Try common extensions.
        cands = [
            filings_text_root / f"{filing_id}.txt",
            filings_text_root / f"{filing_id}.html",
            filings_text_root / f"{filing_id}.htm",
        ]
        p = next((x for x in cands if x.exists()), cands[0])

    if not p.exists():
        raise FileNotFoundError(f"Filing text not found: {p}")
    raw = p.read_text(encoding="utf-8", errors="ignore")
    cleaned = _strip_html_tags(raw)
    return _extract_mda_section(cleaned)


def _chunk_token_ids(token_ids: Sequence[int], chunk_size: int, overlap: int, max_chunks: int) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    stride = chunk_size - overlap
    out = []
    i = 0
    n = len(token_ids)
    while i < n and len(out) < max_chunks:
        out.append(list(token_ids[i : i + chunk_size]))
        i += stride
    return out if out else [list(token_ids[:chunk_size])]


def _compute_cls_embedding(
    text: str,
    tokenizer,
    model,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> np.ndarray:
    import torch

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]
    chunks = _chunk_token_ids(ids, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
    if not chunks:
        raise ValueError("No token chunks produced for filing text.")
    cls_rows = []
    with torch.no_grad():
        for c in chunks:
            cls_id = getattr(tokenizer, "cls_token_id", None)
            sep_id = getattr(tokenizer, "sep_token_id", None)
            if cls_id is None or sep_id is None:
                raise ValueError("Tokenizer missing CLS/SEP token ids.")

            # Keep room for [CLS] and [SEP]
            chunk_ids = list(c)[:510]
            input_ids = [int(cls_id)] + [int(x) for x in chunk_ids] + [int(sep_id)]
            attn = [1] * len(input_ids)
            tok_type = [0] * len(input_ids)
            enc = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
                "token_type_ids": torch.tensor([tok_type], dtype=torch.long),
            }
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            cls_rows.append(cls)
    if not cls_rows:
        raise ValueError("No CLS vectors produced for filing.")
    mat = np.stack(cls_rows, axis=0)
    return mat.mean(axis=0)


def _load_and_validate_metadata(path: Path, tickers: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"SEC filings metadata not found: {path}. "
            "Expected columns: filing_id,ticker,filed_at,form_type,(optional)text_path."
        )
    df = pd.read_csv(path)
    required = {"filing_id", "ticker", "filed_at", "form_type"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["ticker"].isin(tickers)].copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"]).dt.normalize()
    df["form_type"] = df["form_type"].astype(str).str.upper()
    df = df[df["form_type"].isin(["10-Q", "10-K"])].copy()
    df = df.sort_values(["ticker", "filed_at", "filing_id"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No 10-Q/10-K rows found after filtering metadata.")
    return df


def _build_filing_embeddings(
    meta: pd.DataFrame,
    filings_text_root: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> Tuple[pd.DataFrame, int]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for strict Phase 6 SEC embeddings. "
            "Install with: pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    rows = []
    skipped = 0
    error_samples: List[str] = []
    for _, r in meta.iterrows():
        try:
            txt = _load_filing_text(r, filings_text_root=filings_text_root)
            if not txt:
                skipped += 1
                continue
            vec = _compute_cls_embedding(
                txt,
                tokenizer=tokenizer,
                model=model,
                chunk_size=chunk_size,
                overlap=overlap,
                max_chunks=max_chunks,
            )
            row = {
                "filing_id": str(r["filing_id"]),
                "date": pd.Timestamp(r["filed_at"]),
                "ticker": str(r["ticker"]),
                "form_type": str(r["form_type"]),
            }
            for i, v in enumerate(vec):
                row[f"sec_emb_{i:03d}"] = float(v)
            rows.append(row)
        except Exception as exc:
            skipped += 1
            if len(error_samples) < 5:
                filing_id = str(r.get("filing_id", "unknown"))
                ticker = str(r.get("ticker", "unknown"))
                error_samples.append(f"{ticker}/{filing_id}: {type(exc).__name__}: {exc}")

    if not rows:
        detail = "\n".join(error_samples) if error_samples else "No detailed errors captured."
        raise RuntimeError("No SEC filing embeddings were produced.\nSample errors:\n" + detail)
    if error_samples:
        print("Embedding warnings (sample):")
        for e in error_samples:
            print(" -", e)
    emb = pd.DataFrame(rows).sort_values(["ticker", "date", "filing_id"]).reset_index(drop=True)
    return emb, skipped


def _apply_train_only_pca(
    emb: pd.DataFrame,
    train_end: pd.Timestamp,
    pca_dim: int,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    emb_cols = [c for c in emb.columns if c.startswith("sec_emb_")]
    if pca_dim <= 0 or pca_dim >= len(emb_cols):
        return emb.copy(), emb_cols, {"pca_applied": False, "output_dim": len(emb_cols)}

    train_mask = emb["date"] <= train_end
    train_rows = emb.loc[train_mask, emb_cols]
    if train_rows.empty:
        raise ValueError("No training-period filings available for PCA fit.")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(train_rows.to_numpy(dtype=np.float32))
    transformed = pca.transform(emb[emb_cols].to_numpy(dtype=np.float32))

    out = emb[["filing_id", "date", "ticker", "form_type"]].copy()
    cols = [f"sec_emb_{i:03d}" for i in range(pca_dim)]
    for i, c in enumerate(cols):
        out[c] = transformed[:, i].astype(np.float32)
    meta = {
        "pca_applied": True,
        "output_dim": pca_dim,
        "explained_variance_ratio_sum": float(np.asarray(pca.explained_variance_ratio_).sum()),
    }
    return out, cols, meta


def _build_daily_forward_filled_panel(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    emb_cols: List[str],
) -> pd.DataFrame:
    panel = (
        pd.concat([train_df[["date", "ticker"]], test_df[["date", "ticker"]]], ignore_index=True)
        .drop_duplicates()
        .copy()
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ticker"] = panel["ticker"].astype(str).str.upper()

    out_rows = []
    for t in sorted(panel["ticker"].unique()):
        p = panel[panel["ticker"] == t].sort_values("date").copy()
        f = emb_df[emb_df["ticker"] == t].sort_values("date").copy()
        if f.empty:
            for c in emb_cols:
                p[c] = 0.0
            p["sec_days_since_filing"] = 999.0
            p["sec_recent_filing_flag_30d"] = 0.0
            out_rows.append(p)
            continue

        merged = pd.merge_asof(
            p[["date"]].sort_values("date"),
            f[["date"] + emb_cols].sort_values("date"),
            on="date",
            direction="backward",
        )
        merged["ticker"] = t
        for c in emb_cols:
            merged[c] = merged[c].fillna(0.0)

        filing_dates = f["date"].sort_values().to_numpy()
        d_arr = merged["date"].to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(filing_dates, d_arr, side="right") - 1
        days_since = np.full(len(merged), 999.0, dtype=np.float32)
        valid = idx >= 0
        if valid.any():
            last_dates = filing_dates[idx[valid]]
            delta = (d_arr[valid] - last_dates) / np.timedelta64(1, "D")
            days_since[valid] = delta.astype(np.float32)
        merged["sec_days_since_filing"] = days_since
        merged["sec_recent_filing_flag_30d"] = (merged["sec_days_since_filing"] <= 30).astype(np.float32)
        out_rows.append(merged)

    return pd.concat(out_rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else None

    feature_config_path = resolve_path(project_root, args.feature_config_path)
    train_path = resolve_path(project_root, args.train_path)
    test_path = resolve_path(project_root, args.test_path)
    metadata_path = resolve_path(project_root, args.filings_metadata_path)
    filings_text_root = resolve_path(project_root, args.filings_text_root)
    out_filings_path = resolve_path(project_root, args.output_filings_emb_path)
    out_daily_path = resolve_path(project_root, args.output_daily_emb_path)
    out_cfg_path = resolve_path(project_root, args.output_feature_config_path)

    out_filings_path.parent.mkdir(parents=True, exist_ok=True)
    out_daily_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(feature_config_path.read_text())
    tickers = list(cfg["tickers"])
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])
    train_end = pd.Timestamp(cfg["train_end"]).normalize()

    meta = _load_and_validate_metadata(metadata_path, tickers=tickers)
    filing_emb_raw, skipped = _build_filing_embeddings(
        meta,
        filings_text_root=filings_text_root,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        max_chunks=args.max_chunks_per_filing,
    )
    filing_emb, emb_cols, pca_meta = _apply_train_only_pca(
        filing_emb_raw,
        train_end=train_end,
        pca_dim=args.pca_dim,
    )
    filing_emb.to_csv(out_filings_path, index=False)

    daily = _build_daily_forward_filled_panel(
        train_df=train_df,
        test_df=test_df,
        emb_df=filing_emb,
        emb_cols=emb_cols,
    )
    sec_features = emb_cols + ["sec_days_since_filing", "sec_recent_filing_flag_30d"]
    daily.to_csv(out_daily_path, index=False)

    out_cfg = {
        "sec_features": sec_features,
        "source_mode": "finbert_cls_mda",
        "filings_metadata_path": str(metadata_path),
        "filings_text_root": str(filings_text_root),
        "daily_embeddings_path": str(out_daily_path),
        "model_name": args.model_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "max_chunks_per_filing": args.max_chunks_per_filing,
        "num_filings_total": int(len(meta)),
        "num_filings_embedded": int(len(filing_emb)),
        "num_filings_skipped": int(skipped),
        "train_end_for_pca": str(train_end.date()),
        **pca_meta,
    }
    out_cfg_path.write_text(json.dumps(out_cfg, indent=2))

    print("Saved strict Phase 6 SEC artifacts:")
    print(" -", out_filings_path)
    print(" -", out_daily_path)
    print(" -", out_cfg_path)
    print("SEC feature count:", len(sec_features))
    print("Embedded filings:", len(filing_emb), "| skipped:", skipped)


if __name__ == "__main__":
    main()
