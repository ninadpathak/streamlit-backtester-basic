from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_ENTRY_TIME = time(9, 30)


@dataclass(frozen=True)
class BacktestParams:
    stop_loss_pct: float
    take_profit_pct: float
    holding_days: int
    capital_per_trade: float
    max_stocks_per_day: int = 0  # 0 means no cap
    entry_time: time = DEFAULT_ENTRY_TIME
    brokerage_per_side: float = 20.0
    tie_break_rule: str = "conservative"  # conservative => SL first, optimistic => TP first

    @property
    def round_trip_brokerage(self) -> float:
        return self.brokerage_per_side * 2


@dataclass(frozen=True)
class TriggerRow:
    trigger_date: date
    symbol: str


class TriggerValidationError(Exception):
    pass


def load_holidays(holiday_csv_path: str | Path) -> set[date]:
    df = pd.read_csv(holiday_csv_path)
    required = {"date"}
    missing = required - set(df.columns)
    if missing:
        raise TriggerValidationError(f"Holiday CSV missing columns: {sorted(missing)}")
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    bad = int(dates.isna().sum())
    if bad:
        raise TriggerValidationError(f"Holiday CSV has {bad} invalid date rows. Expected YYYY-MM-DD")
    return set(dates.dt.date)


def is_trading_day(d: date, holidays: set[date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def parse_chartink_csv(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"date", "symbol"}
    missing = expected - set(df.columns)
    if missing:
        raise TriggerValidationError(f"Trigger CSV missing columns: {sorted(missing)}")

    parsed = df.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], format="%d-%m-%Y", errors="coerce")
    parsed["symbol"] = parsed["symbol"].astype(str).str.strip().str.upper()

    bad_dates = int(parsed["date"].isna().sum())
    if bad_dates:
        raise TriggerValidationError(
            f"Trigger CSV has {bad_dates} invalid date rows. Expected format dd-mm-yyyy"
        )

    parsed = parsed.dropna(subset=["symbol"])
    parsed = parsed[parsed["symbol"] != ""]
    parsed["trigger_date"] = parsed["date"].dt.date
    parsed = parsed.drop_duplicates(subset=["trigger_date", "symbol"]).sort_values(["trigger_date", "symbol"])
    return parsed[["trigger_date", "symbol"]].reset_index(drop=True)


def choose_daily_triggers(parsed_df: pd.DataFrame, max_stocks_per_day: int) -> pd.DataFrame:
    ordered = parsed_df.sort_values(["trigger_date", "symbol"])
    if max_stocks_per_day <= 0:
        return ordered.reset_index(drop=True)
    return (
        ordered.groupby("trigger_date", as_index=False, group_keys=False)
        .head(max_stocks_per_day)
        .reset_index(drop=True)
    )


def get_nth_trading_day(start_day: date, n: int, holidays: set[date]) -> date:
    if n <= 0:
        raise ValueError("n must be >= 1")
    d = start_day
    count = 0
    while True:
        if is_trading_day(d, holidays):
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _entry_timestamp(trigger_day: date, entry_time: time) -> datetime:
    return datetime.combine(trigger_day, entry_time).replace(tzinfo=IST)


def _normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Candle data missing columns: {sorted(missing)}")

    normalized = candles.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    normalized = normalized.dropna(subset=["timestamp"])
    normalized["timestamp"] = normalized["timestamp"].dt.tz_convert(IST)
    for c in ["open", "high", "low", "close"]:
        normalized[c] = pd.to_numeric(normalized[c], errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return normalized.reset_index(drop=True)


def run_backtest(
    triggers_df: pd.DataFrame,
    holidays: set[date],
    params: BacktestParams,
    candle_fetcher: Callable[[str, date, date], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    selected = choose_daily_triggers(triggers_df, params.max_stocks_per_day)

    trades: list[dict] = []
    skipped: list[dict] = []

    for row in selected.itertuples(index=False):
        trigger_day = row.trigger_date
        symbol = row.symbol

        if not is_trading_day(trigger_day, holidays):
            skipped.append(
                {
                    "trigger_date": trigger_day,
                    "symbol": symbol,
                    "reason": "TRIGGER_NOT_TRADING_DAY",
                }
            )
            continue

        entry_ts = _entry_timestamp(trigger_day, params.entry_time)
        exit_day_limit = get_nth_trading_day(trigger_day, params.holding_days, holidays)

        candles = candle_fetcher(symbol, trigger_day, exit_day_limit)
        if candles.empty:
            skipped.append({"trigger_date": trigger_day, "symbol": symbol, "reason": "NO_CANDLES"})
            continue

        try:
            candles = _normalize_candles(candles)
        except Exception as exc:
            skipped.append(
                {
                    "trigger_date": trigger_day,
                    "symbol": symbol,
                    "reason": f"BAD_CANDLE_FORMAT: {exc}",
                }
            )
            continue

        candles = candles[
            (candles["timestamp"].dt.date >= trigger_day)
            & (candles["timestamp"].dt.date <= exit_day_limit)
        ]

        entry_rows = candles[candles["timestamp"] == entry_ts]
        if entry_rows.empty:
            skipped.append(
                {
                    "trigger_date": trigger_day,
                    "symbol": symbol,
                    "reason": f"MISSING_ENTRY_CANDLE_{params.entry_time.strftime('%H%M')}",
                }
            )
            continue

        entry_price = float(entry_rows.iloc[0]["open"])
        if entry_price <= 0:
            skipped.append(
                {
                    "trigger_date": trigger_day,
                    "symbol": symbol,
                    "reason": "NON_POSITIVE_ENTRY_PRICE",
                }
            )
            continue

        quantity = int(params.capital_per_trade // entry_price)
        if quantity <= 0:
            skipped.append(
                {
                    "trigger_date": trigger_day,
                    "symbol": symbol,
                    "reason": "CAPITAL_TOO_LOW_FOR_1_SHARE",
                }
            )
            continue

        tp_price = entry_price * (1 + params.take_profit_pct / 100)
        sl_price = entry_price * (1 - params.stop_loss_pct / 100)

        monitored = candles[candles["timestamp"] >= entry_ts]

        exit_ts: datetime | None = None
        exit_price: float | None = None
        exit_reason: str | None = None

        for c in monitored.itertuples(index=False):
            hit_tp = float(c.high) >= tp_price
            hit_sl = float(c.low) <= sl_price

            if hit_tp and hit_sl:
                if params.tie_break_rule == "optimistic":
                    exit_price = tp_price
                    exit_reason = "TP_AND_SL_SAME_CANDLE_TP_FIRST"
                else:
                    exit_price = sl_price
                    exit_reason = "TP_AND_SL_SAME_CANDLE_SL_FIRST"
                exit_ts = c.timestamp
                break

            if hit_tp:
                exit_price = tp_price
                exit_reason = "TAKE_PROFIT"
                exit_ts = c.timestamp
                break

            if hit_sl:
                exit_price = sl_price
                exit_reason = "STOP_LOSS"
                exit_ts = c.timestamp
                break

        if exit_ts is None:
            forced = monitored[monitored["timestamp"].dt.date == exit_day_limit]
            if forced.empty:
                skipped.append(
                    {
                        "trigger_date": trigger_day,
                        "symbol": symbol,
                        "reason": "NO_CANDLES_ON_FORCED_EXIT_DAY",
                    }
                )
                continue
            last = forced.iloc[-1]
            exit_ts = last["timestamp"]
            exit_price = float(last["close"])
            exit_reason = "TIME_EXIT"

        gross_pnl = (exit_price - entry_price) * quantity
        brokerage = params.round_trip_brokerage
        net_pnl = gross_pnl - brokerage

        trades.append(
            {
                "symbol": symbol,
                "trigger_date": trigger_day,
                "entry_timestamp": entry_ts,
                "entry_price": round(entry_price, 6),
                "exit_timestamp": exit_ts,
                "exit_price": round(float(exit_price), 6),
                "quantity": quantity,
                "capital_allocated": params.capital_per_trade,
                "capital_deployed": round(quantity * entry_price, 2),
                "take_profit_price": round(tp_price, 6),
                "stop_loss_price": round(sl_price, 6),
                "exit_reason": exit_reason,
                "gross_pnl": round(gross_pnl, 2),
                "brokerage": round(brokerage, 2),
                "net_pnl": round(net_pnl, 2),
                "net_return_pct_on_allocated": round((net_pnl / params.capital_per_trade) * 100, 4),
            }
        )

    trades_df = pd.DataFrame(trades)
    skipped_df = pd.DataFrame(skipped)

    if trades_df.empty:
        capital_timeline = pd.DataFrame(columns=["timestamp", "capital_in_use"])
        summary = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_net_pnl": 0.0,
            "total_gross_pnl": 0.0,
            "total_brokerage": 0.0,
            "avg_net_return_pct": 0.0,
            "peak_capital_required": 0.0,
            "total_allocated_capital_across_trades": 0.0,
        }
        return trades_df, skipped_df, capital_timeline, summary

    events: list[tuple[datetime, int, float]] = []
    for t in trades_df.itertuples(index=False):
        events.append((t.entry_timestamp.to_pydatetime(), 0, params.capital_per_trade))
        events.append((t.exit_timestamp.to_pydatetime(), 1, -params.capital_per_trade))

    events.sort(key=lambda x: (x[0], x[1]))

    running = 0.0
    timeline_rows = []
    for ts, _, delta in events:
        running += delta
        timeline_rows.append({"timestamp": ts, "capital_in_use": running})

    capital_timeline = pd.DataFrame(timeline_rows)

    wins = int((trades_df["net_pnl"] > 0).sum())
    losses = int((trades_df["net_pnl"] <= 0).sum())

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins / len(trades_df)) * 100, 2),
        "total_net_pnl": round(float(trades_df["net_pnl"].sum()), 2),
        "total_gross_pnl": round(float(trades_df["gross_pnl"].sum()), 2),
        "total_brokerage": round(float(trades_df["brokerage"].sum()), 2),
        "avg_net_return_pct": round(float(trades_df["net_return_pct_on_allocated"].mean()), 4),
        "peak_capital_required": round(float(capital_timeline["capital_in_use"].max()), 2),
        "total_allocated_capital_across_trades": round(float(trades_df["capital_allocated"].sum()), 2),
    }

    return trades_df, skipped_df, capital_timeline, summary
