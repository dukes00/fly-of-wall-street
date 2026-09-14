# T6 — Market Data Layer: Validation Report

Cache: `data/market/{SYMBOL}_1m.parquet`; contract: int64-ns UTC timestamps, OHLC float64, volume int64, regular session only (13:30-20:00 UTC), sorted, no duplicate timestamps.

| Symbol | Bars | First ts (UTC) | Last ts (UTC) | Gaps | Out-of-session |
|---|---:|---|---|---:|---:|

**PARTIAL CACHE** — no file for: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V, UNH, XOM, LLY, JNJ, PG, MA, HD, AVGO, CVX, ABBV, SPY, ^GSPC.

_Session filter: `is_regular_session` — NYSE holidays 2024-2027 hardcoded (source: nyse.com hours-calendars)._