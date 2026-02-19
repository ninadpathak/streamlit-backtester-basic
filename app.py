from __future__ import annotations

from datetime import date, time, timedelta
from pathlib import Path
import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import (
    BacktestParams,
    TriggerValidationError,
    choose_daily_triggers,
    is_trading_day,
    load_holidays,
    parse_chartink_csv,
    run_backtest,
)
from upstox_data import UpstoxConfig, UpstoxDataClient, UpstoxDataError

APP_DIR = Path(__file__).resolve().parent
DEFAULT_HOLIDAY_PATH = APP_DIR / "nse_trading_holidays_2025_2026.csv"


@st.cache_data(show_spinner=False)
def _read_csv_uploaded(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(raw_bytes))


def _download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def _style_trade_rows(df: pd.DataFrame):
    def row_style(row: pd.Series) -> list[str]:
        pnl = float(row.get("net_pnl", 0.0))
        if pnl > 0:
            color = "background-color: #dcfce7; color: #14532d;"
        elif pnl < 0:
            color = "background-color: #fee2e2; color: #7f1d1d;"
        else:
            color = "background-color: #fef9c3; color: #713f12;"
        return [color] * len(row)

    return df.style.apply(row_style, axis=1)


def _normalize_plot_candles(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles
    data = candles.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    data = data.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    market_open = time(9, 15)
    market_close = time(15, 30)
    data = data[
        (data["timestamp"].dt.time >= market_open)
        & (data["timestamp"].dt.time <= market_close)
    ].reset_index(drop=True)
    return data


def _filter_trades_by_timeline(trades_df: pd.DataFrame, timeline_label: str) -> tuple[pd.DataFrame, str]:
    if trades_df.empty:
        return trades_df, "No trades in selected range."

    data = trades_df.copy()
    data["exit_timestamp_dt"] = pd.to_datetime(data["exit_timestamp"], errors="coerce")
    data["trigger_date_dt"] = pd.to_datetime(data["trigger_date"], errors="coerce")
    data = data.dropna(subset=["exit_timestamp_dt", "trigger_date_dt"]).reset_index(drop=True)
    if data.empty:
        return data, "No valid trade dates found."

    if timeline_label == "All time":
        start_date = data["trigger_date_dt"].dt.date.min()
        end_date = data["exit_timestamp_dt"].dt.date.max()
        return data, f"Tested range: {start_date} to {end_date}"

    months = int(timeline_label.split()[0])
    anchor = data["exit_timestamp_dt"].max()
    cutoff = anchor - pd.DateOffset(months=months)
    filtered = data[data["exit_timestamp_dt"] >= cutoff].reset_index(drop=True)
    if filtered.empty:
        return filtered, f"No trades found in last {months} months."
    start_date = filtered["trigger_date_dt"].dt.date.min()
    end_date = filtered["exit_timestamp_dt"].dt.date.max()
    return filtered, f"Tested range: {start_date} to {end_date} (last {months} months)"


def _compute_filtered_summary(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "total_net_pnl": 0.0,
            "total_gross_pnl": 0.0,
            "total_brokerage": 0.0,
            "avg_net_return_pct": 0.0,
        }
    wins = int((trades_df["net_pnl"] > 0).sum())
    return {
        "trades": int(len(trades_df)),
        "wins": wins,
        "win_rate_pct": round((wins / len(trades_df)) * 100, 2),
        "total_net_pnl": round(float(trades_df["net_pnl"].sum()), 2),
        "total_gross_pnl": round(float(trades_df["gross_pnl"].sum()), 2),
        "total_brokerage": round(float(trades_df["brokerage"].sum()), 2),
        "avg_net_return_pct": round(float(trades_df["net_return_pct_on_allocated"].mean()), 4),
    }


def _compute_extended_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "losses": 0,
            "loss_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "risk_reward_ratio": 0.0,
            "expectancy_per_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "max_drawdown": 0.0,
            "median_holding_hours": 0.0,
        }

    pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss_abs = float(abs(losses.sum()))
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    risk_reward_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    curve = (
        trades_df.assign(exit_date=pd.to_datetime(trades_df["exit_timestamp"], errors="coerce").dt.date)
        .dropna(subset=["exit_date"])
        .groupby("exit_date", as_index=False)["net_pnl"]
        .sum()
        .sort_values("exit_date")
    )
    curve["cum"] = curve["net_pnl"].cumsum()
    curve["peak"] = curve["cum"].cummax()
    curve["dd"] = curve["cum"] - curve["peak"]
    max_drawdown = float(curve["dd"].min()) if not curve.empty else 0.0

    entry_ts = pd.to_datetime(trades_df["entry_timestamp"], errors="coerce")
    exit_ts = pd.to_datetime(trades_df["exit_timestamp"], errors="coerce")
    holding_hours = ((exit_ts - entry_ts).dt.total_seconds() / 3600.0).dropna()

    losses_count = int((pnl <= 0).sum())
    return {
        "losses": losses_count,
        "loss_rate_pct": round((losses_count / len(trades_df)) * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(float(profit_factor), 3),
        "risk_reward_ratio": round(float(risk_reward_ratio), 3),
        "expectancy_per_trade": round(float(pnl.mean()), 2),
        "best_trade": round(float(pnl.max()), 2),
        "worst_trade": round(float(pnl.min()), 2),
        "max_drawdown": round(max_drawdown, 2),
        "median_holding_hours": round(float(holding_hours.median()) if not holding_hours.empty else 0.0, 2),
    }


def _compute_capital_metrics(trades_df: pd.DataFrame, net_pnl: float, gross_pnl: float) -> dict:
    if trades_df.empty:
        return {
            "required_starting_capital": 0.0,
            "ending_capital_after_net": 0.0,
            "ending_capital_after_gross": 0.0,
            "net_return_pct_on_starting_capital": 0.0,
        }

    events: list[tuple[pd.Timestamp, int, float]] = []
    for r in trades_df.itertuples(index=False):
        entry_ts = pd.to_datetime(r.entry_timestamp, errors="coerce")
        exit_ts = pd.to_datetime(r.exit_timestamp, errors="coerce")
        cap = float(r.capital_allocated)
        if pd.isna(entry_ts) or pd.isna(exit_ts):
            continue
        events.append((entry_ts, 0, cap))
        events.append((exit_ts, 1, -cap))

    events.sort(key=lambda x: (x[0], x[1]))
    running = 0.0
    peak = 0.0
    for _, _, delta in events:
        running += delta
        if running > peak:
            peak = running

    return {
        "required_starting_capital": round(peak, 2),
        "ending_capital_after_net": round(peak + float(net_pnl), 2),
        "ending_capital_after_gross": round(peak + float(gross_pnl), 2),
        "net_return_pct_on_starting_capital": round((float(net_pnl) / peak) * 100, 4) if peak > 0 else 0.0,
    }


def _identify_metric_trades(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"max_profit_idx": None, "max_drawdown_trade_idx": None}

    data = trades_df.copy().reset_index(drop=True)
    data["net_pnl_num"] = pd.to_numeric(data["net_pnl"], errors="coerce").fillna(0.0)
    data["exit_date"] = pd.to_datetime(data["exit_timestamp"], errors="coerce").dt.date

    max_profit_idx = int(data["net_pnl_num"].idxmax()) if not data.empty else None

    curve = (
        data.dropna(subset=["exit_date"])
        .groupby("exit_date", as_index=False)["net_pnl_num"]
        .sum()
        .sort_values("exit_date")
    )
    if curve.empty:
        return {"max_profit_idx": max_profit_idx, "max_drawdown_trade_idx": None}

    curve["cum"] = curve["net_pnl_num"].cumsum()
    curve["peak"] = curve["cum"].cummax()
    curve["dd"] = curve["cum"] - curve["peak"]
    trough_date = curve.loc[curve["dd"].idxmin(), "exit_date"]

    drawdown_candidates = data[data["exit_date"] == trough_date]
    if drawdown_candidates.empty:
        max_dd_trade_idx = None
    else:
        max_dd_trade_idx = int(drawdown_candidates["net_pnl_num"].idxmin())

    return {"max_profit_idx": max_profit_idx, "max_drawdown_trade_idx": max_dd_trade_idx}


def _ensure_state() -> None:
    defaults = {
        "result_ready": False,
        "trades_df": pd.DataFrame(),
        "skipped_df": pd.DataFrame(),
        "capital_timeline_df": pd.DataFrame(),
        "summary": {},
        "unresolved_symbols": [],
        "non_trading_count": 0,
        "symbol_to_key": {},
        "parsed_triggers": pd.DataFrame(),
        "validated_symbols_count": 0,
        "sync_last_message": "",
        "live_fetched_symbols_count": 0,
        "cached_symbols_count": 0,
        "failed_symbols_count": 0,
        "selected_trade_idx": 0,
        "selected_trade_note": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _resolve_symbols_for_csv(
    trigger_csv_bytes: bytes,
) -> tuple[pd.DataFrame, dict[str, str], list[str], int]:
    trigger_df_raw = _read_csv_uploaded(trigger_csv_bytes)
    parsed_triggers = parse_chartink_csv(trigger_df_raw)

    if not DEFAULT_HOLIDAY_PATH.exists():
        raise TriggerValidationError(f"Default holiday file not found: {DEFAULT_HOLIDAY_PATH}")
    holidays = load_holidays(DEFAULT_HOLIDAY_PATH)

    non_trading_count = int((~parsed_triggers["trigger_date"].apply(lambda d: is_trading_day(d, holidays))).sum())

    unique_symbols = sorted(parsed_triggers["symbol"].unique().tolist())
    # No auth token needed for Upstox historical data
    with UpstoxDataClient(UpstoxConfig(access_token="dummy")) as client:
        symbol_to_key, unresolved_symbols = client.resolve_symbols(unique_symbols)
    return parsed_triggers, symbol_to_key, unresolved_symbols, non_trading_count


def _sync_data_for_csv(
    trigger_csv_bytes: bytes,
) -> None:
    parsed_triggers, symbol_to_key, unresolved_symbols, non_trading_count = _resolve_symbols_for_csv(
        trigger_csv_bytes
    )
    st.session_state["sync_last_message"] = (
        f"Validated {len(symbol_to_key)} symbols for backtesting. "
        "Candles will be fetched from cache if available; missing data will be fetched from Upstox."
    )
    st.session_state["symbol_to_key"] = symbol_to_key
    st.session_state["unresolved_symbols"] = unresolved_symbols
    st.session_state["non_trading_count"] = non_trading_count
    st.session_state["parsed_triggers"] = parsed_triggers
    st.session_state["validated_symbols_count"] = len(symbol_to_key)


def _run_backtest_pipeline(
    trigger_csv_bytes: bytes,
    params: BacktestParams,
) -> None:
    from market_data_store import MarketDataStore

    parsed_triggers, symbol_to_key, unresolved_symbols, non_trading_count = _resolve_symbols_for_csv(
        trigger_csv_bytes
    )
    holidays = load_holidays(DEFAULT_HOLIDAY_PATH)
    live_fetched_symbols: set[str] = set()
    cached_symbols: set[str] = set()
    failed_symbols: set[str] = set()
    symbol_data_cache: dict[str, pd.DataFrame] = {}
    
    # Initialize market data store for caching
    cache_db_path = APP_DIR / ".cache" / "market_data.sqlite"
    store = MarketDataStore(cache_db_path)

    # No auth token needed for Upstox historical data
    with UpstoxDataClient(UpstoxConfig(access_token="dummy")) as client:
        selected_triggers = choose_daily_triggers(parsed_triggers, params.max_stocks_per_day)
        if not selected_triggers.empty:
            calendar_buffer_days = max(10, (params.holding_days * 3) + 7)
            ranges = (
                selected_triggers.groupby("symbol", as_index=False)["trigger_date"]
                .agg(["min", "max"])
                .reset_index()
                .rename(columns={"min": "min_date", "max": "max_date"})
            )

            progress = st.progress(0, text="Preparing candles from DB/API...")
            total_symbols = len(ranges)
            for i, row in enumerate(ranges.itertuples(index=False), start=1):
                symbol = str(row.symbol)
                key = symbol_to_key.get(symbol)
                if not key:
                    continue

                from_date = row.min_date
                to_date = row.max_date + timedelta(days=calendar_buffer_days)
                
                # Check if we have cached data first (for reporting)
                if store.has_coverage(key, from_date, to_date):
                    cached_symbols.add(symbol)
                
                try:
                    # Use cached fetcher - returns cached data if available, otherwise fetches and stores
                    data = client.fetch_5min_candles_cached(key, symbol, from_date, to_date, store)
                    if not data.empty:
                        live_fetched_symbols.add(symbol)
                        data = data.copy()
                        data["_date"] = pd.to_datetime(data["timestamp"], errors="coerce").dt.date
                    else:
                        failed_symbols.add(symbol)
                except UpstoxDataError as e:
                    failed_symbols.add(symbol)
                    st.warning(f"Failed to fetch data for {symbol}: {e}")
                except Exception as e:
                    failed_symbols.add(symbol)
                    st.warning(f"Unexpected error fetching {symbol}: {e}")
                symbol_data_cache[symbol] = data if not data.empty else pd.DataFrame()
                progress.progress(
                    int(i * 100 / max(total_symbols, 1)),
                    text=f"Prepared candles for {i}/{total_symbols} symbols",
                )
            progress.empty()

    def candle_fetcher(symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
        data = symbol_data_cache.get(symbol)
        if data is None or data.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        return data[(data["_date"] >= from_date) & (data["_date"] <= to_date)].drop(columns=["_date"], errors="ignore").reset_index(drop=True)

    trades_df, skipped_df, capital_timeline_df, summary = run_backtest(
        triggers_df=parsed_triggers,
        holidays=holidays,
        params=params,
        candle_fetcher=candle_fetcher,
    )

    st.session_state["result_ready"] = True
    st.session_state["trades_df"] = trades_df
    st.session_state["skipped_df"] = skipped_df
    st.session_state["capital_timeline_df"] = capital_timeline_df
    st.session_state["summary"] = summary
    st.session_state["unresolved_symbols"] = unresolved_symbols
    st.session_state["non_trading_count"] = non_trading_count
    st.session_state["symbol_to_key"] = symbol_to_key
    st.session_state["parsed_triggers"] = parsed_triggers
    # Track data source stats
    st.session_state["live_fetched_symbols_count"] = len(live_fetched_symbols - cached_symbols)
    st.session_state["cached_symbols_count"] = len(cached_symbols)
    st.session_state["failed_symbols_count"] = len(failed_symbols)


def _get_cached_symbol_count() -> int | None:
    """Get cached symbol count - wrapped for caching."""
    cache_db_path = APP_DIR / ".cache" / "market_data.sqlite"
    if not cache_db_path.exists():
        return 0
    try:
        from market_data_store import MarketDataStore
        store = MarketDataStore(cache_db_path)
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(DISTINCT instrument_key) FROM candles_5m").fetchone()[0]
            return count
    except Exception:
        return None


@st.cache_data(ttl=60)
def _get_cached_count_cached():
    """Cache the symbol count for 60 seconds to avoid repeated DB hits."""
    return _get_cached_symbol_count()


def main() -> None:
    st.set_page_config(
        page_title="Chartink Backtester",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "Chartink Backtester - Backtest your trading strategies with historical data",
        },
    )
    _ensure_state()

    # Clean professional header
    st.markdown("""
        <h1 style='margin-bottom: 0;'>Chartink Backtester</h1>
        <p style='color: #666; margin-top: 0.5rem; font-size: 1.05rem;'>Backtest intraday strategies using Upstox 5-minute candle data</p>
        <hr style='margin: 1rem 0; border: 0; border-top: 1px solid #e0e0e0;'>
    """, unsafe_allow_html=True)

    # File upload section - PRIORITIZED (appears first for fast render)
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Upload Strategy Signals")
            uploaded_trigger_csv = st.file_uploader(
                "Upload Chartink CSV export",
                type=["csv"],
                help="CSV with 'date' (dd-mm-yyyy) and 'symbol' columns",
                label_visibility="collapsed",
            )
        with col2:
            st.markdown("### Actions")
            st.markdown("<br>", unsafe_allow_html=True)
            cta1, cta2 = st.columns(2)
            with cta1:
                sync_data = st.button("Validate Symbols", use_container_width=True, type="secondary")
            with cta2:
                run = st.button("Run Backtest", use_container_width=True, type="primary")

    st.markdown("---")

    with st.sidebar:
        st.markdown("## Strategy Settings")
        
        with st.expander("Risk Parameters", expanded=True):
            stop_loss_pct = st.number_input(
                "Stop Loss %",
                min_value=0.1,
                max_value=50.0,
                value=3.0,
                step=0.1,
                format="%.1f",
            )
            take_profit_pct = st.number_input(
                "Take Profit %",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.1,
                format="%.1f",
            )
        
        with st.expander("Trade Setup", expanded=True):
            trade_direction = st.segmented_control(
                "Direction",
                options=["BUY", "SELL"],
                default="BUY",
                help="SELL = Short selling (intraday only)",
            ) or "BUY"
            
            # For SELL trades, force holding_days to 1
            if trade_direction == "SELL":
                holding_days = 1
                st.info("Intraday only: SELL trades auto-exit by EOD")
            else:
                holding_days = st.number_input(
                    "Holding Days",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1,
                    help="Trading days (not calendar days)",
                )
            
            entry_time = st.time_input(
                "Entry Time (IST)",
                value=time(9, 30),
                step=300,
            )

        with st.expander("Capital & Fees", expanded=False):
            capital_per_trade = st.number_input(
                "Capital per Trade (₹)",
                min_value=1000.0,
                max_value=50000000.0,
                value=100000.0,
                step=1000.0,
            )
            max_stocks_per_day = st.number_input(
                "Max Stocks/Day (0 = unlimited)",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
            )
            brokerage_per_side = st.number_input(
                "Brokerage per Side (₹)",
                min_value=0.0,
                max_value=5000.0,
                value=20.0,
                step=1.0,
            )

        with st.expander("Advanced", expanded=False):
            tie_break_rule = st.selectbox(
                "Same-candle TP/SL rule",
                options=["conservative", "optimistic"],
                index=0,
                help="Which exit to take when both hit in same 5m candle",
            )
            st.caption("Conservative = SL first | Optimistic = TP first")

        st.divider()
        st.markdown("### Data Cache")
        
        # Deferred cache stats lookup
        cached_count = _get_cached_count_cached()
        if cached_count == 0:
            st.caption("No cached data")
        elif cached_count is None:
            st.caption("Cache status unknown")
        else:
            st.caption(f"{cached_count} symbols cached")
        
        if st.button("Clear Cache", use_container_width=True):
            try:
                cache_db_path = APP_DIR / ".cache" / "market_data.sqlite"
                if cache_db_path.exists():
                    cache_db_path.unlink()
                    st.success("Cache cleared!")
                    st.rerun()
                else:
                    st.info("No cache to clear")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")

    # ───────────────────────────────────────────────────────────────────────────────
    # HANDLE BUTTON ACTIONS
    # ───────────────────────────────────────────────────────────────────────────────

    if sync_data:
        if uploaded_trigger_csv is None:
            st.error("Upload trigger CSV to sync data.")
            return
        try:
            _sync_data_for_csv(uploaded_trigger_csv.getvalue())
            st.success(st.session_state.get("sync_last_message", "Sync completed."))
        except TriggerValidationError as exc:
            st.error(f"Validation failed: {exc}")
            st.exception(exc)
            return
        except UpstoxDataError as exc:
            st.error(f"Upstox data error: {exc}")
            st.exception(exc)
            return
        except Exception as exc:
            st.error(f"Data sync failed: {exc}")
            st.code(traceback.format_exc(), language="text")
            return
    if run:
        if uploaded_trigger_csv is None:
            st.error("Upload trigger CSV to run backtest.")
            return

        params = BacktestParams(
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
            holding_days=int(holding_days),
            capital_per_trade=float(capital_per_trade),
            max_stocks_per_day=int(max_stocks_per_day),
            entry_time=entry_time,
            brokerage_per_side=float(brokerage_per_side),
            tie_break_rule=str(tie_break_rule),
            trade_direction=str(trade_direction),
        )

        try:
            with st.spinner("Running backtest..."):
                _run_backtest_pipeline(uploaded_trigger_csv.getvalue(), params)
        except TriggerValidationError as exc:
            st.error(f"Validation failed: {exc}")
            st.exception(exc)
            return
        except UpstoxDataError as exc:
            st.error(f"Upstox data error: {exc}")
            st.exception(exc)
            return
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.code(traceback.format_exc(), language="text")
            return

    if not st.session_state["result_ready"]:
        # Welcome/getting started card
        with st.container():
            st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 1rem;
                    color: white;
                    margin: 1rem 0;
                ">
                    <h3 style="margin-top: 0; color: white;">Getting Started</h3>
                    <ol style="margin-bottom: 0;">
                        <li><b>Upload</b> your Chartink CSV export</li>
                        <li><b>(Optional)</b> Click Validate to check symbols</li>
                        <li><b>Run Backtest</b> to see results</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
        
        unresolved_symbols = st.session_state.get("unresolved_symbols", [])
        if unresolved_symbols:
            st.warning(f"{len(unresolved_symbols)} symbols not found in Upstox.")
            with st.expander("View unresolved symbols"):
                st.dataframe(pd.DataFrame({"symbol": unresolved_symbols}), use_container_width=True, hide_index=True)
        return

    trades_df = st.session_state["trades_df"]
    skipped_df = st.session_state["skipped_df"]
    capital_timeline_df = st.session_state["capital_timeline_df"]
    summary = st.session_state["summary"]
    unresolved_symbols = st.session_state["unresolved_symbols"]
    non_trading_count = st.session_state["non_trading_count"]
    live_fetched_symbols_count = int(st.session_state.get("live_fetched_symbols_count", 0))

    # Status alerts
    if non_trading_count:
        st.warning(
            f"{non_trading_count} triggers on weekends/holidays were skipped "
            f"(strict same-day entry rule).",
        )

    if unresolved_symbols:
        with st.expander(f"{len(unresolved_symbols)} symbols not found in Upstox", expanded=False):
            st.dataframe(pd.DataFrame({"symbol": unresolved_symbols}), use_container_width=True, hide_index=True)
    
    # Data source summary
    cached_symbols_count = int(st.session_state.get("cached_symbols_count", 0))
    failed_symbols_count = int(st.session_state.get("failed_symbols_count", 0))
    if cached_symbols_count > 0 or live_fetched_symbols_count > 0 or failed_symbols_count > 0:
        source_parts = []
        if live_fetched_symbols_count > 0:
            source_parts.append(f"{live_fetched_symbols_count} from API")
        if cached_symbols_count > 0:
            source_parts.append(f"{cached_symbols_count} cached")
        if failed_symbols_count > 0:
            source_parts.append(f"{failed_symbols_count} failed")
        st.caption(" · ".join(source_parts))

    # Results header with timeline selector
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("## Results")
    with header_col2:
        timeline = st.selectbox(
            "Filter",
            options=["2 months", "3 months", "6 months", "All time"],
            index=0,
            label_visibility="collapsed",
        )
    filtered_trades, tested_range_label = _filter_trades_by_timeline(trades_df, timeline)
    filtered_summary = _compute_filtered_summary(filtered_trades)
    extended = _compute_extended_stats(filtered_trades)
    metric_trades = _identify_metric_trades(filtered_trades)
    capital_metrics = _compute_capital_metrics(
        filtered_trades,
        net_pnl=filtered_summary.get("total_net_pnl", 0.0),
        gross_pnl=filtered_summary.get("total_gross_pnl", 0.0),
    )

    parsed_triggers = st.session_state.get("parsed_triggers", pd.DataFrame())
    total_triggers = int(len(parsed_triggers)) if not parsed_triggers.empty else 0
    total_skipped = int(len(skipped_df))
    validated_symbols_count = int(st.session_state.get("validated_symbols_count", 0))

    st.caption(tested_range_label)
    
    # Key metrics in styled containers
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Starting Capital",
                f"₹{capital_metrics.get('required_starting_capital', 0.0):,.0f}",
            )
        with col2:
            net_pnl = filtered_summary.get('total_net_pnl', 0.0)
            delta = capital_metrics.get('net_return_pct_on_starting_capital', 0.0)
            st.metric(
                "Net P&L",
                f"₹{net_pnl:,.2f}",
                f"{delta:.2f}%",
                delta_color="normal",
            )
        with col3:
            st.metric(
                "Win Rate",
                f"{filtered_summary.get('win_rate_pct', 0.0):.1f}%",
                f"{filtered_summary.get('trades', 0)} trades",
            )
        with col4:
            st.metric(
                "Profit Factor",
                f"{extended.get('profit_factor', 0.0):.2f}",
            )

    with st.expander("Detailed Metrics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Returns", f"₹{filtered_summary.get('total_gross_pnl', 0.0):,.2f}")
        c2.metric("Brokerage", f"₹{filtered_summary.get('total_brokerage', 0.0):,.2f}")
        c3.metric("Ending Capital", f"₹{capital_metrics.get('ending_capital_after_net', 0.0):,.2f}")
        c4.metric("Expectancy", f"₹{extended.get('expectancy_per_trade', 0.0):,.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg Win", f"₹{extended.get('avg_win', 0.0):,.2f}")
        c6.metric("Avg Loss", f"₹{extended.get('avg_loss', 0.0):,.2f}")
        c7.metric("Best Trade", f"₹{extended.get('best_trade', 0.0):,.2f}")
        c8.metric("Worst Trade", f"₹{extended.get('worst_trade', 0.0):,.2f}")
        
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Max Drawdown", f"₹{extended.get('max_drawdown', 0.0):,.2f}")
        c10.metric("Risk/Reward", f"{extended.get('risk_reward_ratio', 0.0):.2f}")
        c11.metric("Median Hold", f"{extended.get('median_holding_hours', 0.0):.1f}h")
        c12.metric("Loss Rate", f"{extended.get('loss_rate_pct', 0.0):.1f}%")
    
    # Quick stats row
    st.markdown("---")
    q1, q2, q3, q4 = st.columns(4)
    q1.caption(f"Triggers: **{total_triggers}**")
    q2.caption(f"Skipped: **{total_skipped}**")
    q3.caption(f"From Cache: **{cached_symbols_count}**")
    q4.caption(f"Fetched Live: **{live_fetched_symbols_count}**")

    tab_summary, tab_trades, tab_logs = st.tabs(["Summary", "Trade Explorer", "Diagnostics"])

    with tab_summary:
        if not filtered_trades.empty:
            pnl_curve = (
                filtered_trades.assign(exit_date=pd.to_datetime(filtered_trades["exit_timestamp"]).dt.date)
                .groupby("exit_date", as_index=False)["net_pnl"]
                .sum()
                .sort_values("exit_date")
            )
            pnl_curve["cumulative_net_pnl"] = pnl_curve["net_pnl"].cumsum()

            fig_pnl = go.Figure()
            fig_pnl.add_trace(
                go.Scatter(
                    x=pnl_curve["exit_date"],
                    y=pnl_curve["cumulative_net_pnl"],
                    mode="lines+markers",
                    line={"color": "#0f766e", "width": 2},
                    marker={"size": 5},
                    name="Cumulative Net P&L",
                )
            )
            fig_pnl.update_layout(
                title=f"Cumulative Net P&L ({timeline})",
                height=360,
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
                xaxis_title="Date",
                yaxis_title="P&L (INR)",
            )
            st.plotly_chart(fig_pnl, width="stretch")
        else:
            st.info("No trades in selected timeline for PnL curve.")

    with tab_trades:
        if filtered_trades.empty:
            st.info("No trades available in selected timeline.")
        else:
            note = st.session_state.get("selected_trade_note", "")
            if note:
                st.success(note)
            
            display_cols = [c for c in filtered_trades.columns if c not in {"exit_timestamp_dt", "trigger_date_dt"}]
            st.caption("Green = Profit | Red = Loss")
            st.dataframe(_style_trade_rows(filtered_trades[display_cols]), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Trade Chart Explorer")
            
            picker_df = filtered_trades.reset_index(drop=True)
            options = [
                f"{i+1}. {r.symbol} | {pd.to_datetime(r.trigger_date).date()} | Net ₹{float(r.net_pnl):,.2f}"
                for i, r in enumerate(picker_df.itertuples(index=False))
            ]
            preselect_idx = int(st.session_state.get("selected_trade_idx", 0))
            if preselect_idx < 0 or preselect_idx >= len(options):
                preselect_idx = 0
            selected_label = st.selectbox("Select trade to visualize", options=options, index=preselect_idx, key="trade_picker")
            selected_idx = options.index(selected_label)
            st.session_state["selected_trade_idx"] = selected_idx
            selected_trade = picker_df.iloc[selected_idx]

            symbol_to_key: dict[str, str] = st.session_state.get("symbol_to_key", {})
            key = symbol_to_key.get(str(selected_trade["symbol"]))
            if not key:
                st.warning("Unable to fetch chart (missing symbol mapping).")
            else:
                chart_from = pd.to_datetime(selected_trade["trigger_date"]).date()
                chart_to = pd.to_datetime(selected_trade["exit_timestamp"]).date()
                # Use cached fetcher for chart data
                cache_db_path = APP_DIR / ".cache" / "market_data.sqlite"
                from market_data_store import MarketDataStore
                chart_store = MarketDataStore(cache_db_path)
                # No auth token needed for Upstox historical data
                with UpstoxDataClient(UpstoxConfig(access_token="dummy")) as client:
                    candles = client.fetch_5min_candles_cached(
                        key, str(selected_trade["symbol"]), chart_from, chart_to, chart_store
                    )

                candles = _normalize_plot_candles(candles)
                if candles.empty:
                    st.warning("No candles available for chart rendering on selected trade.")
                else:
                    entry_ts = pd.to_datetime(selected_trade["entry_timestamp"], errors="coerce", utc=True).tz_convert("Asia/Kolkata")
                    exit_ts = pd.to_datetime(selected_trade["exit_timestamp"], errors="coerce", utc=True).tz_convert("Asia/Kolkata")
                    entry_price = float(selected_trade["entry_price"])
                    exit_price = float(selected_trade["exit_price"])

                    fig = go.Figure()
                    fig.add_trace(
                        go.Candlestick(
                            x=candles["timestamp"],
                            open=candles["open"],
                            high=candles["high"],
                            low=candles["low"],
                            close=candles["close"],
                            name="OHLC",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_ts],
                            y=[entry_price],
                            mode="markers+text",
                            text=["Entry"],
                            textposition="top center",
                            marker={"color": "#16a34a", "size": 12, "symbol": "triangle-up"},
                            name="Entry",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[exit_ts],
                            y=[exit_price],
                            mode="markers+text",
                            text=["Exit"],
                            textposition="bottom center",
                            marker={"color": "#dc2626", "size": 12, "symbol": "triangle-down"},
                            name="Exit",
                        )
                    )
                    fig.update_layout(
                        title=f"{selected_trade['symbol']} | {chart_from} to {chart_to}",
                        height=620,
                        margin={"l": 10, "r": 10, "t": 45, "b": 10},
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        hovermode="x unified",
                    )
                    fig.update_xaxes(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),
                            dict(pattern="hour", bounds=[15.5, 9.25]),
                        ]
                    )
                    st.plotly_chart(fig, width="stretch")

    with tab_logs:
        st.info("Data is cached in SQLite. Missing data is fetched from Upstox API.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Skipped Records")
            if skipped_df.empty:
                st.caption("No records skipped")
            else:
                st.dataframe(skipped_df, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("### Capital Timeline")
            if capital_timeline_df.empty:
                st.caption("No timeline data")
            else:
                st.dataframe(capital_timeline_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Export Data")
        c1, c2, c3 = st.columns(3)
        with c1:
            _download_button(trades_df, "Download Trades CSV", "trades.csv")
        with c2:
            _download_button(skipped_df, "Download Skipped CSV", "skipped.csv")
        with c3:
            _download_button(capital_timeline_df, "Download Capital CSV", "capital_timeline.csv")


if __name__ == "__main__":
    main()
