from __future__ import annotations

import threading
from datetime import timedelta

import pandas as pd

from market_data_store import MarketDataStore
from upstox_data import UpstoxConfig, UpstoxDataClient

_LOCK = threading.Lock()
_IN_PROGRESS: set[tuple[str, str]] = set()  # (db_path, instrument_key)


def _worker(access_token: str, db_path: str, items: list[tuple[str, str]]) -> None:
    store = MarketDataStore(db_path)
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    from_date = today - timedelta(days=365)

    try:
        with UpstoxDataClient(UpstoxConfig(access_token=access_token)) as client:
            for _, instrument_key in items:
                try:
                    if store.has_coverage(instrument_key, from_date, today):
                        continue
                    candles = client.fetch_5min_candles(instrument_key, from_date, today)
                    store.upsert_candles_5m(instrument_key, candles)
                except Exception as exc:
                    store.set_ingestion_error(instrument_key, str(exc))
    finally:
        with _LOCK:
            for _, instrument_key in items:
                _IN_PROGRESS.discard((db_path, instrument_key))


def queue_background_sync(access_token: str, db_path: str, symbol_to_key: dict[str, str]) -> int:
    items: list[tuple[str, str]] = []
    with _LOCK:
        for symbol, instrument_key in symbol_to_key.items():
            token = (db_path, instrument_key)
            if token in _IN_PROGRESS:
                continue
            _IN_PROGRESS.add(token)
            items.append((symbol, instrument_key))

    if not items:
        return 0

    t = threading.Thread(target=_worker, args=(access_token, db_path, items), daemon=True)
    t.start()
    return len(items)
