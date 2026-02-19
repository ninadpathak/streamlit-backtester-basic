from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
HISTORICAL_V3_URL = "https://api.upstox.com/v3/historical-candle"


class UpstoxDataError(Exception):
    pass


@dataclass(frozen=True)
class UpstoxConfig:
    access_token: str
    instruments_url: str = INSTRUMENTS_URL
    historical_url: str = HISTORICAL_V3_URL
    cache_dir: str = ".cache/upstox"
    request_timeout_sec: int = 30
    max_retries: int = 3
    retry_sleep_sec: float = 1.5
    days_per_chunk: int = 28
    inter_chunk_sleep_sec: float = 0.0


class UpstoxDataClient:
    def __init__(self, config: UpstoxConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {config.access_token.strip()}",
            }
        )
        self._instrument_map_cache: dict[str, str] | None = None

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "UpstoxDataClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.config.request_timeout_sec, **kwargs)
                if resp.status_code in {408, 429} or resp.status_code >= 500:
                    raise UpstoxDataError(f"HTTP {resp.status_code}: {resp.text[:250]}")
                return resp
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_sleep_sec * attempt)

        raise UpstoxDataError(f"Request failed after retries: {last_exc}")

    def _read_json_or_gzip_json(self, payload: bytes) -> list[dict]:
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        return json.loads(payload.decode("utf-8"))

    def _load_instruments(self) -> list[dict]:
        cache_file = self.cache_dir / "nse_instruments.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        resp = self._request("GET", self.config.instruments_url)
        if resp.status_code != 200:
            raise UpstoxDataError(f"Failed to download instruments: HTTP {resp.status_code}")

        data = self._read_json_or_gzip_json(resp.content)
        cache_file.write_text(json.dumps(data))
        return data

    @staticmethod
    def _row_symbol(row: dict) -> str:
        for k in ["trading_symbol", "tradingsymbol", "symbol", "scrip_name"]:
            v = row.get(k)
            if v:
                return str(v).upper().strip()
        return ""

    @staticmethod
    def _row_instrument_key(row: dict) -> str:
        for k in ["instrument_key", "instrumentKey"]:
            v = row.get(k)
            if v:
                return str(v)
        return ""

    def _build_symbol_map(self) -> dict[str, str]:
        if self._instrument_map_cache is not None:
            return self._instrument_map_cache

        data = self._load_instruments()
        mapping: dict[str, str] = {}

        for row in data:
            symbol = self._row_symbol(row)
            key = self._row_instrument_key(row)
            if not symbol or not key:
                continue

            segment = str(row.get("segment", "")).upper()
            exchange = str(row.get("exchange", "")).upper()
            instrument_type = str(row.get("instrument_type", "")).upper()

            is_nse_eq = (
                "NSE_EQ" in segment
                or exchange == "NSE"
                or instrument_type in {"EQ", "EQUITY"}
            )

            if is_nse_eq and symbol not in mapping:
                mapping[symbol] = key

        self._instrument_map_cache = mapping
        return mapping

    def resolve_symbols(self, symbols: Iterable[str]) -> tuple[dict[str, str], list[str]]:
        wanted = [str(s).upper().strip() for s in symbols]
        symbol_map = self._build_symbol_map()
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for s in wanted:
            key = symbol_map.get(s)
            if key:
                resolved[s] = key
            else:
                missing.append(s)
        return resolved, missing

    def _chunk_dates(self, from_date: date, to_date: date) -> list[tuple[date, date]]:
        chunks: list[tuple[date, date]] = []
        cur = from_date
        while cur <= to_date:
            end = min(cur + timedelta(days=self.config.days_per_chunk - 1), to_date)
            chunks.append((cur, end))
            cur = end + timedelta(days=1)
        return chunks

    def fetch_5min_candles(self, instrument_key: str, from_date: date, to_date: date) -> pd.DataFrame:
        all_rows: list[list] = []

        for start, end in self._chunk_dates(from_date, to_date):
            encoded_key = quote(instrument_key, safe="")
            url = (
                f"{self.config.historical_url}/{encoded_key}/minutes/5/"
                f"{end.isoformat()}/{start.isoformat()}"
            )
            resp = self._request("GET", url)

            if resp.status_code != 200:
                raise UpstoxDataError(
                    f"Historical fetch failed for {instrument_key} {start}..{end}: "
                    f"HTTP {resp.status_code} {resp.text[:300]}"
                )

            payload = resp.json()
            candles = payload.get("data", {}).get("candles", [])
            if not isinstance(candles, list):
                raise UpstoxDataError(f"Unexpected candles payload for {instrument_key}")
            all_rows.extend(candles)

            if self.config.inter_chunk_sleep_sec > 0:
                time.sleep(self.config.inter_chunk_sleep_sec)

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])

        df = pd.DataFrame(all_rows)
        if df.shape[1] < 5:
            raise UpstoxDataError(f"Candle payload had unexpected shape: {df.shape}")

        col_map = {
            0: "timestamp",
            1: "open",
            2: "high",
            3: "low",
            4: "close",
            5: "volume",
            6: "oi",
        }
        df = df.rename(columns=col_map)
        df = df[[c for c in ["timestamp", "open", "high", "low", "close", "volume", "oi"] if c in df.columns]]
        return df
