#!/usr/bin/env python3
"""
UI-based live demo app for PharmaTrade-MM Phase 8.

Run with:
    streamlit run phase8_live_demo_ui.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PharmaTrade Terminal", page_icon="💹", layout="wide")

ACTION_LABELS = {0: "Sell", 1: "Hold", 2: "Buy"}
PRICE_FIELDS = ["open", "high", "low", "close"]


def resolve(base: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs).expanduser()
    return p if p.is_absolute() else base / p


def run_replay(project_root: Path, device: str, seed: int, skip_plot: bool) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "phase8_live_demo.py",
        "--project-root",
        str(project_root),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if skip_plot:
        cmd.append("--skip-plot")
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, out


def load_results(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(out_dir / "phase8_live_demo_metrics.csv")
    perf = pd.read_csv(out_dir / "phase8_live_demo_perf.csv", parse_dates=["date"])
    action_by_ticker = pd.read_csv(out_dir / "phase8_live_demo_action_by_ticker.csv")
    action_totals = pd.read_csv(out_dir / "phase8_live_demo_action_totals.csv")
    return metrics, perf, action_by_ticker, action_totals


def has_results(out_dir: Path) -> bool:
    needed = [
        out_dir / "phase8_live_demo_metrics.csv",
        out_dir / "phase8_live_demo_perf.csv",
        out_dir / "phase8_live_demo_action_by_ticker.csv",
        out_dir / "phase8_live_demo_action_totals.csv",
    ]
    return all(p.exists() for p in needed)


@st.cache_data(show_spinner=False)
def load_market_data(project_root: Path) -> pd.DataFrame:
    market_path = resolve(project_root, "processed/test_dataset.csv")
    if not market_path.exists():
        raise FileNotFoundError(f"Missing market data file: {market_path}")
    df = pd.read_csv(market_path, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_model_actions(project_root: Path) -> pd.DataFrame:
    demo_actions = resolve(project_root, "results/live_demo/phase8_live_demo_actions.csv")
    phase6_actions = resolve(project_root, "results/phase6_sec/phase6_sec_actions.csv")
    chosen = demo_actions if demo_actions.exists() else phase6_actions
    if not chosen.exists():
        return pd.DataFrame()
    actions = pd.read_csv(chosen, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return actions


def ensure_paper_state(tickers: list[str], initial_cash: float) -> None:
    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash = float(initial_cash)
    if "paper_positions" not in st.session_state:
        st.session_state.paper_positions = {t: 0 for t in tickers}
    if "paper_trade_log" not in st.session_state:
        st.session_state.paper_trade_log = []
    if "paper_initial_cash" not in st.session_state:
        st.session_state.paper_initial_cash = float(initial_cash)


def reset_paper_state(tickers: list[str], initial_cash: float) -> None:
    st.session_state.paper_cash = float(initial_cash)
    st.session_state.paper_positions = {t: 0 for t in tickers}
    st.session_state.paper_trade_log = []
    st.session_state.paper_initial_cash = float(initial_cash)


def latest_prices_on_date(market_df: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    day = market_df[market_df["date"] == selected_date].copy()
    cols = ["ticker", "open", "high", "low", "close", "volume"]
    return day[cols].sort_values("ticker").reset_index(drop=True)


def mark_to_market_value(market_df: pd.DataFrame, selected_date: pd.Timestamp, positions: dict[str, int]) -> float:
    day = market_df[market_df["date"] == selected_date].set_index("ticker")
    total = 0.0
    for ticker, qty in positions.items():
        if qty <= 0 or ticker not in day.index:
            continue
        total += float(qty) * float(day.loc[ticker, "close"])
    return total


def model_action_for_ticker_date(actions_df: pd.DataFrame, ticker: str, selected_date: pd.Timestamp) -> str:
    if actions_df.empty:
        return "N/A (run replay first)"
    row = actions_df[actions_df["date"] == selected_date]
    if row.empty:
        return "N/A (no action for this date)"
    col = f"act_{ticker}"
    if col not in row.columns:
        return "N/A"
    action_code = int(row.iloc[0][col])
    return ACTION_LABELS.get(action_code, str(action_code))


def annualized_sharpe_from_returns(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    s = float(daily_returns.std(ddof=1))
    if s <= 1e-12:
        return 0.0
    return float((daily_returns.mean() / s) * np.sqrt(252.0))


def max_drawdown_from_equity(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    dd = (equity - peaks) / peaks.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def dedupe_series_index_last(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if series.index.has_duplicates:
        series = series.groupby(level=0).last()
    return series.sort_index()


def simulate_buy_hold_equal(
    close_prices: pd.DataFrame,
    tickers: list[str],
    initial_cash: float,
) -> pd.Series:
    px0 = close_prices.iloc[0]
    w = np.ones(len(tickers), dtype=float) / max(1, len(tickers))
    alloc = initial_cash * w
    shares = alloc / px0.to_numpy(dtype=float)
    equity = (close_prices.to_numpy(dtype=float) * shares.reshape(1, -1)).sum(axis=1)
    return pd.Series(equity, index=close_prices.index, name="buy_hold_equal")


def run_agent_walkforward_simulation(
    project_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float,
    device: str,
    seed: int,
) -> tuple[bool, dict]:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        return False, {"error": f"stable-baselines3 missing: {exc}"}

    try:
        from phase5_fda_ppo import run_ppo_backtest
        from phase6_multimodal_env import make_phase6_sequence_env_from_processed
        from phase6_multimodal_policy import Phase6MultimodalExtractor  # noqa: F401
    except Exception as exc:
        return False, {"error": f"Phase6 modules failed to import: {exc}"}

    dataset_path = resolve(project_root, "results/phase6_sec/_test_scaled.csv")
    feature_config_path = resolve(project_root, "processed/feature_config.json")
    sec_feature_config_path = resolve(project_root, "results/phase6_sec/phase6_sec_feature_config.json")
    model_path = resolve(project_root, "results/phase6_sec/ppo_phase6_sec.zip")

    for p in [dataset_path, feature_config_path, sec_feature_config_path, model_path]:
        if not p.exists():
            return False, {"error": f"Missing required file: {p}"}

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if df.empty:
        return False, {"error": "No rows in selected date range."}

    # Require enough history for sequence features and enough days for meaningful replay.
    unique_days = sorted(df["date"].drop_duplicates().tolist())
    if len(unique_days) < 25:
        return False, {"error": "Select at least ~25 trading days for stable replay."}

    tmp_path = resolve(project_root, "results/live_demo/_tmp_walkforward_scaled.csv")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False)

    env = make_phase6_sequence_env_from_processed(
        dataset_path=tmp_path,
        feature_config_path=feature_config_path,
        sec_feature_config_path=sec_feature_config_path,
        window_size=20,
        tickers=None,
        initial_cash=float(initial_cash),
        transaction_cost=0.001,
        trade_fraction=0.10,
    )

    model = PPO.load(str(model_path), device=device)
    perf_df, actions_df = run_ppo_backtest(env, model, seed=seed)
    perf_df["date"] = pd.to_datetime(perf_df["date"])
    perf_df = perf_df.sort_values("date").reset_index(drop=True)

    eq = perf_df.set_index("date")["portfolio_value"]
    eq = dedupe_series_index_last(eq)
    ret = eq.pct_change().fillna(0.0)

    tickers = sorted(df["ticker"].astype(str).unique().tolist())
    close_px = (
        df.pivot(index="date", columns="ticker", values="close")
        .sort_index()
        .reindex(columns=tickers)
        .dropna()
    )
    bh_eq = simulate_buy_hold_equal(close_px, tickers=tickers, initial_cash=float(initial_cash))
    bh_eq = dedupe_series_index_last(bh_eq)
    bh_ret = bh_eq.pct_change().fillna(0.0)

    out = {
        "perf_df": perf_df,
        "actions_df": actions_df,
        "agent_equity": eq,
        "agent_returns": ret,
        "bh_equity": bh_eq,
        "bh_returns": bh_ret,
        "sim_df": df,
    }
    return True, out


st.markdown(
    """
<style>
div[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 10%, #0d1b2a 0%, #0b1320 35%, #070b12 100%);
}
div[data-testid="stMetric"] {
    background: rgba(17, 25, 40, 0.75);
    border: 1px solid rgba(0, 255, 163, 0.20);
    border-radius: 10px;
    padding: 8px 10px;
}
div[data-testid="stMetricLabel"] p {
    color: #9fb3c8;
}
div[data-testid="stMetricValue"] {
    color: #e8f1ff;
}
.finance-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(0, 255, 163, 0.35);
    color: #83ffd2;
    font-size: 0.8rem;
    margin-right: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("PharmaTrade Terminal")
st.caption("RL Trading Desk | Multi-Asset Pharma Basket | SEC + FDA Aware Policy")

default_root = Path(__file__).resolve().parent
with st.sidebar:
    st.header("Demo Controls")
    root_input = st.text_input("Project root", value=str(default_root))
    device = st.selectbox("Replay device", ["auto", "cpu", "cuda", "mps"], index=0)
    seed = st.number_input("Replay seed", min_value=-1, max_value=10_000, value=-1, step=1)
    skip_plot = st.checkbox("Skip static PNG plot", value=False)
    run_now = st.button("Run/Refresh Model Replay", type="primary", use_container_width=True)
    st.divider()
    initial_cash = st.number_input("Paper account initial cash ($)", min_value=10_000, max_value=10_000_000, value=1_000_000, step=10_000)
    reset_paper = st.button("Reset Paper Portfolio", use_container_width=True)

project_root = Path(root_input).expanduser()
out_dir = resolve(project_root, "results/live_demo")

if run_now:
    with st.spinner("Running deterministic replay from saved model..."):
        ok, logs = run_replay(project_root, device=device, seed=int(seed), skip_plot=skip_plot)
    if ok:
        st.success("Replay completed successfully.")
    else:
        st.error("Replay failed. Check logs below.")
    with st.expander("Replay logs", expanded=not ok):
        st.code(logs if logs.strip() else "No logs captured.", language="text")

try:
    market_df = load_market_data(project_root)
except Exception as exc:
    st.error(str(exc))
    st.stop()

tickers = sorted(market_df["ticker"].astype(str).unique().tolist())
ensure_paper_state(tickers, float(initial_cash))
if reset_paper:
    reset_paper_state(tickers, float(initial_cash))
    st.success("Paper portfolio reset.")

actions_df = load_model_actions(project_root)

tab_replay, tab_live_sim, tab_trade = st.tabs(
    ["Trading Performance", "Live Market Simulation", "Paper Trading Desk"]
)

with tab_replay:
    st.subheader("Model Performance Board")
    if not has_results(out_dir):
        st.warning("No replay outputs found yet. Use sidebar button: Run/Refresh Model Replay.")
    else:
        metrics_df, perf_df, action_by_ticker_df, action_totals_df = load_results(out_dir)
        m = metrics_df.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Portfolio", f"${perf_df['portfolio_value'].iloc[-1]:,.0f}")
        col2.metric("Cumulative Return", f"{m['cumulative_return_pct']:.2f}%")
        col3.metric("Annualized Sharpe", f"{m['ann_sharpe']:.3f}")
        col4.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")

        st.markdown("**Equity Curve (Test Replay)**")
        st.line_chart(perf_df[["date", "portfolio_value"]].set_index("date"), height=320)

        st.markdown("**Daily Return and Reward**")
        st.line_chart(perf_df[["date", "daily_return", "reward"]].set_index("date"), height=260)

        st.markdown("**Action Distribution**")
        if not action_by_ticker_df.empty:
            bars = action_by_ticker_df.set_index("ticker")[["sell_ratio", "hold_ratio", "buy_ratio"]]
            st.bar_chart(bars, height=280)
        if not action_totals_df.empty:
            st.dataframe(action_totals_df, use_container_width=True)

        with st.expander("Replay Metrics Table", expanded=False):
            st.dataframe(metrics_df, use_container_width=True)

with tab_live_sim:
    st.subheader("Historical Walk-Forward Simulation")
    st.caption("Allocate capital to the RL strategy and replay day-by-day market execution on historical out-of-sample data.")

    sim_min = market_df["date"].min().date()
    sim_max = market_df["date"].max().date()
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns([1, 1, 1, 1])
    sim_start = pd.Timestamp(sim_c1.date_input("Simulation start", value=sim_min, min_value=sim_min, max_value=sim_max))
    sim_end = pd.Timestamp(sim_c2.date_input("Simulation end", value=sim_max, min_value=sim_min, max_value=sim_max))
    sim_cash = float(
        sim_c3.number_input(
            "Initial capital ($)",
            min_value=10_000,
            max_value=10_000_000,
            value=1_000_000,
            step=10_000,
        )
    )
    sim_seed = int(sim_c4.number_input("Simulation seed", min_value=1, max_value=10_000, value=7, step=1))
    run_sim = st.button("Run Agent Live Simulation", type="primary")

    if run_sim:
        with st.spinner("Simulating agent walk-forward on historical environment..."):
            ok, payload = run_agent_walkforward_simulation(
                project_root=project_root,
                start_date=min(sim_start, sim_end),
                end_date=max(sim_start, sim_end),
                initial_cash=sim_cash,
                device=device,
                seed=sim_seed,
            )
        if not ok:
            st.error(payload.get("error", "Simulation failed"))
        else:
            st.session_state.live_sim_payload = payload
            st.success("Simulation complete.")

    sim_payload = st.session_state.get("live_sim_payload")
    if sim_payload:
        agent_eq = sim_payload["agent_equity"]
        agent_ret = sim_payload["agent_returns"]
        bh_eq = sim_payload["bh_equity"]
        bh_ret = sim_payload["bh_returns"]
        actions_df = sim_payload["actions_df"]

        agent_final = float(agent_eq.iloc[-1])
        bh_final = float(bh_eq.iloc[-1])
        profit = agent_final - sim_cash
        profit_pct = (agent_final / sim_cash - 1.0) * 100.0
        alpha_vs_bh_pct = (agent_final / bh_final - 1.0) * 100.0 if bh_final > 0 else 0.0

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Initial Capital", f"${sim_cash:,.0f}")
        a2.metric("Agent Final Value", f"${agent_final:,.0f}")
        a3.metric("Agent Profit", f"${profit:,.0f}")
        a4.metric("Agent Return", f"{profit_pct:.2f}%")
        a5.metric("Vs Buy&Hold", f"{alpha_vs_bh_pct:.2f}%")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Agent Sharpe", f"{annualized_sharpe_from_returns(agent_ret):.3f}")
        r2.metric("Agent Max Drawdown", f"{max_drawdown_from_equity(agent_eq) * 100.0:.2f}%")
        r3.metric("Buy&Hold Return", f"{(bh_final / sim_cash - 1.0) * 100.0:.2f}%")
        r4.metric("Buy&Hold Sharpe", f"{annualized_sharpe_from_returns(bh_ret):.3f}")

        compare = pd.concat(
            [
                dedupe_series_index_last(agent_eq).rename("agent"),
                dedupe_series_index_last(bh_eq).rename("buy_hold_equal"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        st.markdown("**Equity Growth: Agent vs Buy & Hold**")
        st.line_chart(compare, height=340)

        merged_actions = actions_df.copy()
        merged_actions["date"] = pd.to_datetime(merged_actions["date"])
        st.markdown("**Recent Agent Decisions (last 15 days)**")
        st.dataframe(merged_actions.tail(15).reset_index(drop=True), use_container_width=True)

        sim_df = sim_payload.get("sim_df", pd.DataFrame()).copy()
        if not sim_df.empty:
            sim_df["date"] = pd.to_datetime(sim_df["date"])
            recent_dates = set(merged_actions.tail(15)["date"].tolist())
            recent_df = sim_df[sim_df["date"].isin(recent_dates)].copy()

            # Build long actions view for merge with ticker-level regulatory context.
            long_actions = []
            for _, row in merged_actions.tail(15).iterrows():
                d = pd.Timestamp(row["date"])
                for c in row.index:
                    if not c.startswith("act_"):
                        continue
                    ticker = c.replace("act_", "")
                    long_actions.append(
                        {
                            "date": d,
                            "ticker": ticker,
                            "agent_action": ACTION_LABELS.get(int(row[c]), str(int(row[c]))),
                        }
                    )
            action_long_df = pd.DataFrame(long_actions)

            fda_cols = [c for c in recent_df.columns if c.startswith("evt_")]
            sec_flag_col = "sec_recent_filing_flag_30d" if "sec_recent_filing_flag_30d" in recent_df.columns else None
            sec_days_col = "sec_days_since_filing" if "sec_days_since_filing" in recent_df.columns else None
            is_event_col = "is_event_window" if "is_event_window" in recent_df.columns else None

            if fda_cols or sec_flag_col or sec_days_col or is_event_col:
                recent_df["fda_event_type"] = ""
                if fda_cols:
                    def _fda_label(r):
                        active = [x.replace("evt_", "") for x in fda_cols if float(r.get(x, 0.0)) > 0.5]
                        return ",".join(active) if active else ""

                    recent_df["fda_event_type"] = recent_df.apply(_fda_label, axis=1)

                keep_cols = ["date", "ticker", "fda_event_type"]
                if is_event_col:
                    keep_cols.append(is_event_col)
                if sec_flag_col:
                    keep_cols.append(sec_flag_col)
                if sec_days_col:
                    keep_cols.append(sec_days_col)
                context_df = recent_df[keep_cols].copy()

                context_df = action_long_df.merge(context_df, on=["date", "ticker"], how="left")

                def _relevance_reason(r):
                    reasons = []
                    if is_event_col and float(r.get(is_event_col, 0.0)) > 0.5:
                        evt = str(r.get("fda_event_type", "")).strip()
                        reasons.append(f"FDA event window{f' ({evt})' if evt else ''}")
                    if sec_flag_col and float(r.get(sec_flag_col, 0.0)) > 0.5:
                        d = r.get(sec_days_col, None) if sec_days_col else None
                        if d is not None and pd.notna(d):
                            reasons.append(f"Recent SEC filing ({float(d):.0f}d ago)")
                        else:
                            reasons.append("Recent SEC filing")
                    return "; ".join(reasons)

                context_df["regulatory_context"] = context_df.apply(_relevance_reason, axis=1)
                context_df = context_df[context_df["regulatory_context"].astype(str).str.len() > 0].copy()

                st.markdown("**FDA/SEC Context Supporting Recent Decisions (last 15 days)**")
                if context_df.empty:
                    st.info("No strong FDA/SEC trigger flags found in the latest 15-day window.")
                else:
                    show_cols = ["date", "ticker", "agent_action", "regulatory_context"]
                    if sec_days_col and sec_days_col in context_df.columns:
                        show_cols.append(sec_days_col)
                    st.dataframe(
                        context_df[show_cols].sort_values(["date", "ticker"]).reset_index(drop=True),
                        use_container_width=True,
                    )
    else:
        st.info("Set dates and initial capital, then click 'Run Agent Live Simulation'.")

with tab_trade:
    st.subheader("Paper Trading Desk")
    st.caption("Choose ticker and price stream, then place manual BUY/SELL orders against historical market snapshots.")

    min_date = market_df["date"].min().date()
    max_date = market_df["date"].max().date()
    date_range = st.slider("Chart date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    c1, c2, c3 = st.columns([1, 1, 1])
    selected_ticker = c1.selectbox("Ticker", tickers, index=0)
    selected_price_field = c2.selectbox("Price field", PRICE_FIELDS, index=3)
    ticker_dates = (
        market_df[(market_df["ticker"] == selected_ticker) & (market_df["date"] >= start_date) & (market_df["date"] <= end_date)]["date"]
        .sort_values()
        .dt.date
        .tolist()
    )
    if not ticker_dates:
        st.warning("No rows for selected range/ticker.")
        st.stop()
    selected_trade_date = pd.Timestamp(c3.selectbox("Trade date", ticker_dates, index=len(ticker_dates) - 1))

    ticker_slice = market_df[
        (market_df["ticker"] == selected_ticker) & (market_df["date"] >= start_date) & (market_df["date"] <= end_date)
    ][["date", selected_price_field]].set_index("date")
    st.line_chart(ticker_slice, height=320)

    day_row = market_df[(market_df["ticker"] == selected_ticker) & (market_df["date"] == selected_trade_date)].head(1)
    if day_row.empty:
        st.error("Could not find selected date/ticker row.")
        st.stop()
    exec_price = float(day_row.iloc[0]["close"])
    model_hint = model_action_for_ticker_date(actions_df, selected_ticker, selected_trade_date)

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Open", f"{float(day_row.iloc[0]['open']):.2f}")
    d2.metric("High", f"{float(day_row.iloc[0]['high']):.2f}")
    d3.metric("Low", f"{float(day_row.iloc[0]['low']):.2f}")
    d4.metric("Close (exec)", f"{exec_price:.2f}")
    d5.metric("Model Hint", model_hint)

    st.markdown("**Order Ticket**")
    o1, o2, o3 = st.columns([1, 1, 2])
    side = o1.radio("Side", ["BUY", "SELL"], horizontal=True)
    quantity = int(o2.number_input("Quantity", min_value=1, max_value=50_000, value=100, step=1))
    place_order = o3.button("Place Order", use_container_width=True, type="primary")

    if place_order:
        positions = st.session_state.paper_positions
        cash = float(st.session_state.paper_cash)
        notional = exec_price * quantity
        if side == "BUY":
            if notional > cash:
                st.error(f"Insufficient cash. Need ${notional:,.2f}, available ${cash:,.2f}.")
            else:
                st.session_state.paper_cash = cash - notional
                positions[selected_ticker] = int(positions.get(selected_ticker, 0)) + quantity
                st.session_state.paper_trade_log.append(
                    {
                        "date": selected_trade_date.date().isoformat(),
                        "ticker": selected_ticker,
                        "side": "BUY",
                        "quantity": quantity,
                        "price": exec_price,
                        "notional": notional,
                    }
                )
                st.success("Buy order executed.")
        else:
            held = int(positions.get(selected_ticker, 0))
            if quantity > held:
                st.error(f"Insufficient position. Trying to sell {quantity}, held {held}.")
            else:
                st.session_state.paper_cash = cash + notional
                positions[selected_ticker] = held - quantity
                st.session_state.paper_trade_log.append(
                    {
                        "date": selected_trade_date.date().isoformat(),
                        "ticker": selected_ticker,
                        "side": "SELL",
                        "quantity": quantity,
                        "price": exec_price,
                        "notional": notional,
                    }
                )
                st.success("Sell order executed.")

    holdings_value = mark_to_market_value(market_df, selected_trade_date, st.session_state.paper_positions)
    total_value = float(st.session_state.paper_cash) + holdings_value
    pnl_pct = (total_value / float(st.session_state.paper_initial_cash) - 1.0) * 100.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Cash", f"${st.session_state.paper_cash:,.2f}")
    p2.metric("Holdings Value", f"${holdings_value:,.2f}")
    p3.metric("Total Equity", f"${total_value:,.2f}")
    p4.metric("P/L vs Start", f"{pnl_pct:.2f}%")

    st.markdown("**Holdings**")
    day_prices = market_df[market_df["date"] == selected_trade_date].set_index("ticker")
    hold_rows = []
    for t in tickers:
        qty = int(st.session_state.paper_positions.get(t, 0))
        if qty <= 0:
            continue
        if t not in day_prices.index:
            continue
        px = float(day_prices.loc[t, "close"])
        hold_rows.append({"ticker": t, "quantity": qty, "last_close": px, "market_value": qty * px})
    holdings_df = pd.DataFrame(hold_rows)
    if holdings_df.empty:
        st.info("No open positions.")
    else:
        st.dataframe(holdings_df.sort_values("market_value", ascending=False), use_container_width=True)

    st.markdown("**Market Snapshot (selected date)**")
    st.dataframe(latest_prices_on_date(market_df, selected_trade_date), use_container_width=True)

    st.markdown("**Trade Log**")
    trade_log_df = pd.DataFrame(st.session_state.paper_trade_log)
    if trade_log_df.empty:
        st.info("No trades placed yet.")
    else:
        st.dataframe(trade_log_df.iloc[::-1].reset_index(drop=True), use_container_width=True)
