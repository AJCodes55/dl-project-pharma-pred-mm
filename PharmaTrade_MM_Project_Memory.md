# PharmaTrade-MM — Project Memory & Context Document
**Last updated:** April 4, 2026  
**Project:** Deep Learning & Multimodal Systems — PharmaTrade-MM  
**Team:** AAM (Ashutosh Jaiswal, Abhee Anshul, Monisha Kohli)

---

## 1. Project Overview

We are building **PharmaTrade-MM**, a domain-specialized multimodal RL agent that trades eight major pharma stocks using three input streams:

1. **Price + Technical Indicators** (OHLCV, RSI, MACD, Bollinger Bands)
2. **Pharma News Sentiment** via FinBERT
3. **Structured FDA Event Calendar** encoding regulatory decisions, drug pipeline stage, therapeutic area, and historical price reactions

The agent uses **PPO (Proximal Policy Optimization)** with a **CVaR-augmented reward** and **event-aware position sizing** to handle tail-risk around FDA binary events.

**Core thesis:** Forward-looking FDA catalysts (scheduled months in advance, can move a stock 30–50% in one session) are far stronger signals for pharma stocks than backward-looking quarterly SEC filings (as used by Nawathe et al. 2024).

### Tickers Traded
`PFE, JNJ, MRK, ABBV, BMY, AMGN, GILD, BIIB`

### Time Periods
- **Training:** 2018-02-20 to 2021-12-31
- **Testing / Backtest:** 2022-01-03 to 2023-12-29

---

## 2. Architecture (From Proposal)

### Multimodal State Encoder
- **Price branch:** 2-layer LSTM, hidden size 128, inputs OHLCV + all technicals
- **Sentiment branch:** 1-layer LSTM, hidden size 64, inputs daily FinBERT sentiment (pos, neg, neu, net, n_filings)
- **FDA branch:** Small MLP, inputs FDA event vector (days-to-event, event type one-hot, therapeutic area one-hot, historical reaction) → 64-dim output
- **Fusion:** Cross-attention (price attends over sentiment), then concatenate with FDA embedding → project to 256-dim state

### PPO Actor-Critic
- **Actor:** Outputs Buy/Sell/Hold probabilities
- **Critic:** Outputs value estimate
- **Reward:** Incremental Sharpe ratio − CVaR tail-risk penalty − drawdown penalty − 0.1% transaction cost per trade
- **Position scaling:** In the 5 days before any scheduled FDA decision, a scaling factor reduces exposure

### Ablation Variants
| Ablation | Modalities | Notes |
|----------|-----------|-------|
| (1) | Price only | Baseline |
| (2) | Price + Sentiment | Tests sentiment value |
| (3) | Price + Sentiment + SEC Filings | Replicates Nawathe et al. — **build last** |
| (4) | Price + Sentiment + Basic FDA | Minimal FDA features |
| (5) | Price + Sentiment + Rich FDA + CVaR | **Full model** |

### Evaluation Baselines
- Buy-and-hold
- Equal-weight monthly rebalance
- 20-day momentum strategy
- XPH Pharma ETF (passive benchmark)

### Primary Metrics
Annualized Sharpe Ratio, Maximum Drawdown, Sortino Ratio, Cumulative Return vs. benchmark, Alpha vs. XPH ETF, Win Rate

### Interpretability
Plot agent position size vs. days-to-FDA-event — validates whether the agent learned to reduce exposure pre-event.

---

## 3. Data Inventory

All data is stored in a zip file uploaded to Colab. All sources are free and publicly accessible.

### 3a. Price + Technicals

**File:** `price_technicals.csv` (consolidated) + individual `price_{TICKER}.csv` files  
**Rows:** 13,284 (1,476 per ticker × 9 tickers including XPH)  
**Date range:** 2018-02-20 to 2023-12-29  
**Columns (19):**
```
date, ticker, open, high, low, close, volume, rsi, macd, macd_signal, macd_diff,
bb_upper, bb_middle, bb_lower, bb_pct, bb_width, log_return, volume_sma20, volume_ratio
```

**Also available:** `pharma_price_technicals.csv` (12,072 rows, slightly different column naming — uses `Date, Close, High, Low, Open, Volume, RSI, MACD, MACD_Signal, MACD_Hist, BB_Upper, BB_Mid, BB_Lower, Ticker`). We use `price_technicals.csv` as the primary file since it has more features (bb_pct, bb_width, log_return, volume_sma20, volume_ratio).

### 3b. Sentiment Daily

**File:** `sentiment_daily.csv`  
**Rows:** 11,808 (1,476 per ticker × 8 tickers)  
**Date range:** 2018-02-20 to 2023-12-29  
**Columns (7):**
```
date, ticker, sentiment_pos, sentiment_neg, sentiment_neu, n_filings, net_sentiment
```

**Key characteristic:** Most days have `n_filings = 0` with sentiment scores at zero. Actual filing days per ticker:

| Ticker | Filing Days |
|--------|------------|
| ABBV | 16 |
| AMGN | 8 |
| BIIB | 15 |
| BMY | 28 |
| GILD | 14 |
| JNJ | 16 |
| MRK | 14 |
| PFE | 19 |

**Decision:** We apply **exponential decay forward-fill** (half-life = 5 days) so the agent gets a fading memory of recent filings rather than an abrupt zero signal on non-filing days.

**Per-ticker sentiment files** (`sentiment_ABBV.csv`, etc.) — These are sparse filing-day-only summaries (10–31 rows each). Not used directly; `sentiment_daily.csv` is the primary file.

### 3c. FDA Event Calendar

**File:** `fda_event_calendar.csv`  
**Rows:** 38 events  
**Date range:** 2018-04-16 to 2023-10-06  
**Columns (12):**
```
date, ticker, event_type, drug_name, therapeutic_area, outcome, source,
application_type, active_ingredient, sponsor_raw, px_1d_ret_pct, px_5d_ret_pct
```

**Event types:** APPROVAL, ADCOM, CRL  
**Outcomes:** APPROVED, POSITIVE, NEGATIVE, REJECTED  
**Therapeutic areas:** Bone Disease, Cardiology, Dermatology, Immunology, Infectious Disease, Neurology, Oncology, Psychiatry

**Events per ticker:**

| Ticker | Events |
|--------|--------|
| ABBV | 5 |
| AMGN | 5 |
| BIIB | 5 |
| BMY | 6 |
| GILD | 4 |
| JNJ | 4 |
| MRK | 5 |
| PFE | 4 |

**Full event list:**

| Date | Ticker | Type | Drug | Outcome | 1d Ret% | 5d Ret% |
|------|--------|------|------|---------|---------|---------|
| 2018-04-16 | BMY | APPROVAL | Opdivo - HCC | APPROVED | -7.79 | -13.21 |
| 2018-06-13 | MRK | APPROVAL | Keytruda - 1L NSCLC | APPROVED | 0.24 | -1.10 |
| 2018-12-14 | PFE | ADCOM | Tafamidis (Vyndaqel) | POSITIVE | -1.73 | -5.92 |
| 2019-03-20 | BMY | ADCOM | Opdivo+Yervoy - NSCLC 1L | POSITIVE | -0.83 | -3.27 |
| 2019-04-02 | AMGN | APPROVAL | Evenity (romosozumab) | APPROVED | 0.34 | 0.63 |
| 2019-04-11 | MRK | APPROVAL | Keytruda - cervical cancer | APPROVED | -1.21 | -9.44 |
| 2019-05-03 | PFE | APPROVAL | Tafamidis (Vyndaqel) | APPROVED | 0.93 | 0.17 |
| 2019-05-15 | ABBV | APPROVAL | Skyrizi (risankizumab) | APPROVED | 0.43 | 4.02 |
| 2019-07-01 | JNJ | APPROVAL | Esketamine (Spravato) | APPROVED | 0.06 | 1.53 |
| 2019-07-30 | ABBV | APPROVAL | Rinvoq (upadacitinib) | APPROVED | 0.00 | -3.59 |
| 2019-10-16 | GILD | APPROVAL | Filgotinib - ADCOM denied | NEGATIVE | 0.08 | 1.96 |
| 2020-03-04 | ABBV | ADCOM | Rinvoq - RA expansion | POSITIVE | 4.77 | -3.11 |
| 2020-03-05 | BMY | APPROVAL | Zeposia (ozanimod) | APPROVED | -1.59 | -14.20 |
| 2020-10-22 | GILD | APPROVAL | Veklury (remdesivir) full | APPROVED | 0.76 | -2.79 |
| 2020-11-06 | BIIB | ADCOM | Aducanumab (Aduhelm) | NEGATIVE | 0.00 | -24.42 |
| 2021-01-15 | ABBV | APPROVAL | Skyrizi - PsA | APPROVED | -0.99 | -1.01 |
| 2021-02-27 | JNJ | ADCOM | Janssen COVID-19 Vaccine | POSITIVE | 0.54 | -0.67 |
| 2021-02-27 | JNJ | APPROVAL | Janssen COVID-19 Vaccine | APPROVED | 0.54 | -0.67 |
| 2021-03-26 | BMY | APPROVAL | Breyanzi (lisocabtagene) | APPROVED | 1.83 | 1.37 |
| 2021-05-28 | AMGN | APPROVAL | Lumakras (sotorasib) | APPROVED | 1.12 | 0.80 |
| 2021-06-07 | BIIB | APPROVAL | Aducanumab (Aduhelm) | APPROVED | 38.34 | 41.94 |
| 2021-06-07 | MRK | APPROVAL | Keytruda - TMB-H solid tumors | APPROVED | -1.77 | 2.68 |
| 2021-08-20 | GILD | CRL | Filgotinib RA | REJECTED | 1.07 | -0.54 |
| 2021-08-23 | PFE | APPROVAL | Comirnaty full approval | APPROVED | 2.48 | -4.02 |
| 2021-12-23 | MRK | APPROVAL | Molnupiravir (Lagevrio) | APPROVED | -0.56 | 0.63 |
| 2022-01-14 | PFE | APPROVAL | Paxlovid | APPROVED | -1.06 | -7.20 |
| 2022-05-19 | BMY | APPROVAL | Opdivo+Yervoy - ESCC | APPROVED | -1.38 | 0.74 |
| 2022-06-17 | ABBV | APPROVAL | Rinvoq - CD | APPROVED | -0.63 | 10.05 |
| 2022-07-22 | GILD | APPROVAL | Sunlenca (lenacapavir) | APPROVED | -0.34 | -2.07 |
| 2022-09-16 | BMY | APPROVAL | Sotyktu (deucravacitinib) | APPROVED | -0.36 | -1.49 |
| 2022-10-13 | MRK | CRL | Vericiguat - sNDA | REJECTED | 2.29 | 2.79 |
| 2022-12-14 | AMGN | ADCOM | Lumakras full approval | NEGATIVE | -0.42 | -2.20 |
| 2022-12-15 | AMGN | CRL | Lumakras full approval | REJECTED | -1.84 | -2.17 |
| 2022-12-20 | BIIB | ADCOM | Lecanemab (Leqembi) | POSITIVE | 1.10 | -3.64 |
| 2023-01-06 | BIIB | APPROVAL | Lecanemab accelerated | APPROVED | 2.82 | 6.06 |
| 2023-07-06 | BIIB | APPROVAL | Lecanemab full approval | APPROVED | -0.31 | -2.77 |
| 2023-08-04 | JNJ | APPROVAL | Tecvayli (teclistamab) | APPROVED | -0.94 | 1.88 |
| 2023-10-06 | AMGN | APPROVAL | Blincyto - MRD+ | APPROVED | 0.90 | 7.18 |

### 3d. FDA Timing Features

**File:** `fda_timing_features.csv`  
**Rows:** 2,386  
**Date range (as_of_date):** 2018-01-16 to 2023-10-06  
**days_to_event range:** 0 to 90  
**Columns (14):**
```
date, ticker, event_type, drug_name, therapeutic_area, outcome, source,
application_type, active_ingredient, sponsor_raw, px_1d_ret_pct, px_5d_ret_pct,
days_to_event, as_of_date
```

This is the pre-expanded version of the FDA calendar — for each event, it creates one row per day in the 90-day countdown window. `as_of_date` is the trading day, `date` is the event date, `days_to_event` is the countdown. This is used to build the daily FDA feature vector.

### 3e. XPH Benchmark

**File:** `xph_benchmark.csv`  
**Rows:** 501  
**Date range:** 2022-01-03 to 2023-12-29 (test period only)  
**Columns:** Date, Close, High, Low, Open, Volume

Also available: `price_XPH.csv` with full technicals for 2018–2023 (1,476 rows, same format as other price files).

---

## 4. Data Processing Decisions

### 4a. Sentiment Decay Forward-Fill
- **Problem:** On ~98% of trading days, `n_filings = 0` and sentiment is zero — the agent gets no signal.
- **Solution:** Exponential decay with **half-life = 5 trading days**. On filing days, use actual FinBERT scores. On subsequent days, decay the scores exponentially. Neutral sentiment renormalized to `1 - (pos_decayed + neg_decayed)`.
- **Effect:** Dramatically increases non-zero sentiment coverage while letting the signal naturally fade.

### 4b. FDA Feature Engineering
For each (ticker, trading_day), we extract:
- `days_to_event` — countdown to nearest upcoming FDA event (capped at 90, set to 999 if no upcoming event)
- `event_type` one-hot — APPROVAL, ADCOM, CRL
- `therapeutic_area` one-hot — Bone Disease, Cardiology, Dermatology, Immunology, Infectious Disease, Neurology, Oncology, Psychiatry
- `hist_1d_ret` — historical average 1-day return for that (ticker, event_type) pair, computed **only from training data** (no lookahead)
- `hist_5d_ret` — same for 5-day return
- `is_event_window` — binary flag, 1 if within 5 trading days of event

### 4c. Unified Dataset Structure
The unified dataset merges price (spine) + enhanced sentiment + FDA features via left joins on (date, ticker). All NaN from unmatched FDA rows filled with defaults (999 for days_to_event, 0 for everything else).

**Feature groups for the model:**
- **Price + Technical (17):** open, high, low, close, volume, rsi, macd, macd_signal, macd_diff, bb_upper, bb_middle, bb_lower, bb_pct, bb_width, log_return, volume_sma20, volume_ratio
- **Sentiment (5):** sent_pos, sent_neg, sent_neu, sent_net, n_filings
- **FDA (variable, ~15+):** days_to_event, is_event_window, hist_1d_ret, hist_5d_ret, evt_ADCOM, evt_APPROVAL, evt_CRL, ta_Bone_Disease, ta_Cardiology, ta_Dermatology, ta_Immunology, ta_Infectious_Disease, ta_Neurology, ta_Oncology, ta_Psychiatry

---

## 5. SEC Filing Embedding Pipeline (Ablation 3) — DEFERRED

**Decision:** Build this AFTER ablations 1, 2, 4, 5 are working. Not step 0.

**Plan (Option A — FinBERT CLS Embeddings):**
1. Identify all 10-Q and 10-K filings for 8 tickers on EDGAR (2018–2023), ~160–190 filings total
2. Download and extract MD&A (Management Discussion & Analysis) sections from HTML/XBRL
3. Chunk MD&A text into ~400–450 token windows (FinBERT limit = 512 tokens)
4. Run FinBERT, extract [CLS] embedding (768-dim) from last hidden layer per chunk
5. Mean-pool chunk embeddings → one 768-dim vector per filing
6. Forward-fill on daily timeline (each day uses most recent available filing embedding)
7. Optionally PCA to 64/128 dims, or let encoder MLP learn the compression (768 → 64)

**Time estimate:** ~half day for EDGAR download/parse, ~1 hour for FinBERT embedding on Colab GPU.

**Libraries:** `edgartools` or `sec-edgar-downloader`, Hugging Face Transformers (FinBERT), SEC rate limit = 10 req/sec.

---

## 6. Execution Plan — All Phases

### Phase 1 — Data Pipeline & Validation ✅ CODE WRITTEN
- Load all CSVs, validate date alignment across tickers
- Apply sentiment decay forward-fill (half-life = 5 days)
- Engineer FDA features (nearest-event extraction, one-hot encoding, historical reactions from train only)
- Merge into unified dataset, train/test split
- Save: `unified_dataset.csv`, `train_dataset.csv`, `test_dataset.csv`, `xph_processed.csv`, `feature_config.json`
- **Status:** Code written, awaiting Colab execution and output validation.

### Phase 2 — Gym Trading Environment
- Custom OpenAI Gym env stepping through one trading day at a time
- Portfolio state: cash, positions, portfolio value
- Actions: Buy/Sell/Hold per stock
- Transaction cost: 0.1% per trade
- Reward: Incremental Sharpe − CVaR penalty − drawdown penalty
- FDA event-aware position scaling (reduce exposure within 5 days of event)
- Sanity check with random agent

### Phase 3 — Price-Only Baseline (Ablation 1)
- Activate only the price LSTM branch (2-layer, hidden 128)
- PPO actor-critic on top via Stable-Baselines3
- Train, backtest on 2022–2023
- Build all classical baselines (buy-hold, equal-weight, momentum, XPH)
- Compute full metric suite

### Phase 4 — Add Sentiment (Ablation 2)
- Activate sentiment LSTM branch (1-layer, hidden 64)
- Implement cross-attention fusion (price attends over sentiment)
- Train and evaluate

### Phase 5 — Add FDA Calendar (Ablations 4 & 5)
- Activate FDA MLP branch → 64-dim
- Ablation 4: basic FDA features (days-to-event + event type only)
- Ablation 5: full rich FDA + CVaR reward active
- Build interpretability plot: agent position size vs. days-to-FDA-event

### Phase 6 — SEC Filing Baseline (Ablation 3)
- Build EDGAR pipeline (see Section 5 above)
- Swap FDA branch for SEC embedding branch
- Train and evaluate
- **Trigger:** Start this when ablations 1, 2, 4, 5 are all working

### Phase 7 — Full Evaluation & Analysis
- Run all 5 ablation variants + classical baselines through same backtest
- Comparison table: Sharpe, drawdown, Sortino, cumulative return, alpha, win rate
- Portfolio performance curves
- FDA event proximity interpretability plot
- Statistical significance (bootstrap confidence intervals on Sharpe)
- Case studies: specific FDA events where agent behavior was notable

### Phase 8 — Report, Presentation, Code Cleanup
- Full course report: related work, methodology, experiments, results, analysis
- Slides with ablation results, portfolio curves, interpretability plot
- GitHub repo with README and reproducibility guide
- Live demo of trained agent inference (if compute permits)

---

## 7. Compute & Libraries

- **Compute:** TACC / Google Colab Pro (A100 GPU). Each ablation ~20–40 min on A100.
- **Libraries:** PyTorch, Stable-Baselines3 (PPO), Hugging Face Transformers (FinBERT), yfinance, QuantStats, OpenAI Gym
- **Development:** Running code in Google Colab, data uploaded as zip file

---

## 8. Key References

1. Nawathe et al. (2024) — Multimodal Deep RL for Portfolio Optimization (arXiv:2412.17293)
2. Yang et al. (2023) — FinBERT: Pretrained LM for Financial Communications (arXiv:2006.08097)
3. Liu et al. (2024) — SAPPO: Sentiment-Augmented PPO for Algorithmic Trading
4. Fatemi et al. (2025) — Cross-Modal Temporal Fusion for Financial Market Forecasting (arXiv:2504.13522)
5. Loughran & McDonald (2011) — When Is a Liability Not a Liability? (Journal of Finance)
6. Schulman et al. (2017) — Proximal Policy Optimization Algorithms (arXiv:1707.06347)

---

## 9. Files Produced So Far

| File | Description | Status |
|------|------------|--------|
| `phase1_data_pipeline.py` | Phase 1 complete data pipeline script | Written, awaiting Colab run |
| `unified_dataset.csv` | Merged price + sentiment + FDA features | Will be produced by Phase 1 |
| `train_dataset.csv` | Training split (2018–2021) | Will be produced by Phase 1 |
| `test_dataset.csv` | Test split (2022–2023) | Will be produced by Phase 1 |
| `xph_processed.csv` | XPH benchmark with standardized columns | Will be produced by Phase 1 |
| `feature_config.json` | Feature names, tickers, split dates, config | Will be produced by Phase 1 |

---

## 10. Colab Setup Notes

**Paths to change in every script:**
```python
ZIP_PATH = "/content/pharma_data.zip"       # where zip is uploaded
DATA_DIR = "/content/pharma_data/"           # extraction target
OUT_DIR  = "/content/processed/"             # output directory
EXTRACT_ZIP = True                           # set True first run
```

**Zip handling:** Data is in a zip file uploaded to Colab. Phase 1 script handles extraction. If zip creates a nested subfolder, uncomment and adjust the `DATA_DIR` reassignment line after extraction.
