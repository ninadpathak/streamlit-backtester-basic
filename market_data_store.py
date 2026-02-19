from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class IngestionState:
    first_ts_ist: str | None
    last_ts_ist: str | None
    last_success_at: str | None
    status: str | None
    message: str | None


class MarketDataStore:
    def __init__(self, db_path: str | Path):
        raw_input = str(db_path).strip()
        if not raw_input:
            raw_input = "data/market_data.sqlite"
        raw_path = Path(raw_input).expanduser()
        if raw_path.is_absolute():
            self.db_path = raw_path
        else:
            self.db_path = (Path(__file__).resolve().parent / raw_path).resolve()
        if self.db_path.exists() and self.db_path.is_dir():
            self.db_path = self.db_path / "market_data.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
        except sqlite3.OperationalError as exc:
            raise sqlite3.OperationalError(
                f"{exc} (db_path={self.db_path}, cwd={Path.cwd()})"
            ) from exc
        # Avoid forcing journal/synchronous mode here: on some environments
        # those PRAGMAs can fail with "unable to open database file".
        conn.execute("PRAGMA busy_timeout=30000;")
        try:
            conn.execute("PRAGMA temp_store=MEMORY;")
        except sqlite3.OperationalError:
            pass
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS instruments (
                  instrument_key TEXT PRIMARY KEY,
                  symbol TEXT NOT NULL,
                  exchange TEXT,
                  segment TEXT,
                  instrument_type TEXT,
                  last_seen_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(symbol);

                CREATE TABLE IF NOT EXISTS candles_5m (
                  instrument_key TEXT NOT NULL,
                  ts_ist TEXT NOT NULL,
                  open REAL NOT NULL,
                  high REAL NOT NULL,
                  low REAL NOT NULL,
                  close REAL NOT NULL,
                  volume REAL,
                  oi REAL,
                  PRIMARY KEY (instrument_key, ts_ist),
                  FOREIGN KEY (instrument_key) REFERENCES instruments(instrument_key)
                );
                CREATE INDEX IF NOT EXISTS idx_candles_5m_inst_ts ON candles_5m(instrument_key, ts_ist);

                CREATE TABLE IF NOT EXISTS ingestion_state (
                  instrument_key TEXT NOT NULL,
                  interval TEXT NOT NULL DEFAULT '5m',
                  first_ts_ist TEXT,
                  last_ts_ist TEXT,
                  last_success_at TEXT,
                  status TEXT,
                  message TEXT,
                  PRIMARY KEY (instrument_key, interval),
                  FOREIGN KEY (instrument_key) REFERENCES instruments(instrument_key)
                );
                """
            )

    def upsert_instruments(self, symbol_to_key: dict[str, str]) -> None:
        now = datetime.utcnow().isoformat()
        rows = [(k, s, "NSE", "NSE_EQ", "EQ", now) for s, k in symbol_to_key.items()]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO instruments (instrument_key, symbol, exchange, segment, instrument_type, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(instrument_key) DO UPDATE SET
                  symbol=excluded.symbol,
                  exchange=excluded.exchange,
                  segment=excluded.segment,
                  instrument_type=excluded.instrument_type,
                  last_seen_at=excluded.last_seen_at
                """,
                rows,
            )

    def upsert_candles_5m(self, instrument_key: str, candles: pd.DataFrame) -> int:
        if candles.empty:
            return 0
        data = candles.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
        for c in ["open", "high", "low", "close", "volume", "oi"]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            else:
                data[c] = None
        data = data.dropna(subset=["timestamp", "open", "high", "low", "close"]).drop_duplicates(subset=["timestamp"])
        if data.empty:
            return 0

        rows = [
            (
                instrument_key,
                ts.isoformat(),
                float(o),
                float(h),
                float(l),
                float(c),
                None if pd.isna(v) else float(v),
                None if pd.isna(oi) else float(oi),
            )
            for ts, o, h, l, c, v, oi in zip(
                data["timestamp"],
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
                data["oi"],
                strict=False,
            )
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO candles_5m (instrument_key, ts_ist, open, high, low, close, volume, oi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(instrument_key, ts_ist) DO UPDATE SET
                  open=excluded.open,
                  high=excluded.high,
                  low=excluded.low,
                  close=excluded.close,
                  volume=excluded.volume,
                  oi=excluded.oi
                """,
                rows,
            )

            first_ts = data["timestamp"].min().isoformat()
            last_ts = data["timestamp"].max().isoformat()
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO ingestion_state (instrument_key, interval, first_ts_ist, last_ts_ist, last_success_at, status, message)
                VALUES (?, '5m', ?, ?, ?, 'success', NULL)
                ON CONFLICT(instrument_key, interval) DO UPDATE SET
                  first_ts_ist = CASE
                    WHEN ingestion_state.first_ts_ist IS NULL OR excluded.first_ts_ist < ingestion_state.first_ts_ist
                    THEN excluded.first_ts_ist ELSE ingestion_state.first_ts_ist END,
                  last_ts_ist = CASE
                    WHEN ingestion_state.last_ts_ist IS NULL OR excluded.last_ts_ist > ingestion_state.last_ts_ist
                    THEN excluded.last_ts_ist ELSE ingestion_state.last_ts_ist END,
                  last_success_at = excluded.last_success_at,
                  status='success',
                  message=NULL
                """,
                (instrument_key, first_ts, last_ts, now),
            )
        return len(rows)

    def get_ingestion_state(self, instrument_key: str) -> IngestionState:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT first_ts_ist, last_ts_ist, last_success_at, status, message
                FROM ingestion_state
                WHERE instrument_key = ? AND interval='5m'
                """,
                (instrument_key,),
            ).fetchone()
        if row is None:
            return IngestionState(None, None, None, None, None)
        return IngestionState(*row)

    def has_coverage(self, instrument_key: str, from_date: date, to_date: date) -> bool:
        state = self.get_ingestion_state(instrument_key)
        if not state.first_ts_ist or not state.last_ts_ist:
            return False
        first = pd.to_datetime(state.first_ts_ist, errors="coerce")
        last = pd.to_datetime(state.last_ts_ist, errors="coerce")
        if pd.isna(first) or pd.isna(last):
            return False
        return bool(first.date() <= from_date and last.date() >= to_date)

    def get_candles_5m(self, instrument_key: str, from_date: date, to_date: date) -> pd.DataFrame:
        from_ts = pd.Timestamp(from_date).tz_localize("Asia/Kolkata").isoformat()
        to_ts = (pd.Timestamp(to_date).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)).isoformat()
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT ts_ist as timestamp, open, high, low, close, volume, oi
                FROM candles_5m
                WHERE instrument_key = ? AND ts_ist >= ? AND ts_ist <= ?
                ORDER BY ts_ist
                """,
                conn,
                params=(instrument_key, from_ts, to_ts),
            )
        return df

    def set_ingestion_error(self, instrument_key: str, message: str) -> None:
        with self._connect() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO ingestion_state (instrument_key, interval, last_success_at, status, message)
                VALUES (?, '5m', NULL, 'error', ?)
                ON CONFLICT(instrument_key, interval) DO UPDATE SET
                  status='error',
                  message=excluded.message
                """,
                (instrument_key, f"{message[:1000]} @ {now}"),
            )

    def get_coverage_table(self, symbol_to_key: dict[str, str], lookback_days: int = 365) -> pd.DataFrame:
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        from_date = today - timedelta(days=lookback_days)
        rows: list[dict] = []
        for symbol, key in sorted(symbol_to_key.items()):
            state = self.get_ingestion_state(key)
            has_cov = False
            if state.first_ts_ist and state.last_ts_ist:
                first = pd.to_datetime(state.first_ts_ist, errors="coerce")
                last = pd.to_datetime(state.last_ts_ist, errors="coerce")
                if not pd.isna(first) and not pd.isna(last):
                    has_cov = bool(first.date() <= from_date and last.date() >= today)
            rows.append(
                {
                    "symbol": symbol,
                    "instrument_key": key,
                    "first_ts_ist": state.first_ts_ist,
                    "last_ts_ist": state.last_ts_ist,
                    "status": state.status or ("covered" if has_cov else "missing"),
                    "coverage_1y": "yes" if has_cov else "no",
                    "message": state.message,
                }
            )
        return pd.DataFrame(rows)
