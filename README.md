# Chartink Backtester (Upstox 5m)

Streamlit app to backtest Chartink trigger CSVs using Upstox 5-minute historical candles.

## Rules implemented

- Entry: trigger date at configurable IST entry time (default `09:30`, strict same day)
- Scope: process all trigger rows, with optional `Max Stocks per Day` cap (`0` means no cap)
- Exit: stop loss / take profit / max holding period (`trading days`)
- Brokerage: configurable per side (default `20`, so `40` round trip)
- Capital model: fixed capital reserved per trade (default `100000`)

## Input CSV format

Chartink export with at least:

- `date` in `dd-mm-yyyy` (example: `01-07-2025`)
- `symbol` (example: `INFY`)

Optional columns are ignored.

## Holiday handling

Default holiday file: `nse_trading_holidays_2025_2026.csv`

- Weekends + these holidays are treated as non-trading days
- Trigger rows on non-trading days are skipped due to strict same-day entry rule
- Holiday upload in UI is intentionally removed; app always uses the bundled holiday file above

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. Upload trigger CSV
2. Optional: click `Sync 1Y Data to DB` (shows progress and DB coverage table)
3. Click `Run Backtest` (runs immediately; missing symbol coverage is queued for background sync)

## Required credential

Use a valid Upstox `access_token` in the app sidebar.

## Notes

- 5-minute candle storage uses SQLite at `data/market_data.sqlite` (DB-first reads, API fetch for missing ranges).
- Instrument mapping uses Upstox NSE instrument master (`NSE.json.gz`) and caches locally under `.cache/upstox`.
# streamlit-backtester-basic
