"""T6: market data layer — NYSE calendar, session filter, parquet cache loader.

Cache contract (shared with the fetch pipeline):
- One parquet per symbol at ``data/market/{SYMBOL}_1m.parquet``.
- Columns exactly: ``timestamp`` (int64, nanoseconds, UTC), ``open``, ``high``,
  ``low``, ``close`` (float64), ``volume`` (int64).
- US regular-session bars only (13:30-20:00 UTC), sorted by timestamp, no
  duplicate timestamps. Gaps are left as gaps.

NYSE holiday table source: NYSE "Holidays & Trading Hours" page
(https://www.nyse.com/markets/hours-calendars), cross-checked against
pandas-market-calendars for 2024-2027. Hardcoded for determinism; no network.
"""

from __future__ import annotations

import argparse
import time as _time
from datetime import UTC, date, time
from pathlib import Path

import pandas as pd

from fruitfly.__main__ import register_command

# Basket symbols. ``^GSPC`` (S&P 500 index) is cached under the file name
# GSPC to keep cache paths shell-safe.
BASKET: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "PG", "MA", "HD", "AVGO",
    "CVX", "ABBV", "SPY", "^GSPC",
]

CACHE_DIR = Path("data/market")
REPORT_PATH = Path("reports/t6-data.md")

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def _symbol_stem(symbol: str) -> str:
    """Cache file stem per symbol ("^GSPC" -> "GSPC")."""
    return symbol.lstrip("^")


def cache_path(symbol: str) -> Path:
    """Parquet cache path for ``symbol``."""
    return CACHE_DIR / f"{_symbol_stem(symbol)}_1m.parquet"


# ---------------------------------------------------------------------------
# NYSE regular-session calendar (hardcoded, 2024-2027)
# ---------------------------------------------------------------------------

# (month, day) of full closures per year. Sources: NYSE holidays page and
# pandas-market-calendars XNYS schedule. Good Fridays included; 2024 also
# closed Jun 19 (Juneteenth, Wed) and Dec 25. Early closes (Jul 3, Jul 4
# eve, Dec 24, Black Friday) are NOT full holidays and remain regular here;
# the session filter keeps only the standard 13:30-20:00 UTC window.
_HOLIDAYS: dict[int, frozenset[date]] = {
    2024: frozenset(
        date(2024, m, d)
        for m, d in [
            (1, 1), (1, 15), (2, 19), (3, 29), (5, 27), (6, 19),
            (7, 4), (9, 2), (11, 28), (12, 25),
        ]
    ),
    2025: frozenset(
        date(2025, m, d)
        for m, d in [
            (1, 1), (1, 9), (2, 17), (4, 18), (5, 26), (6, 19),
            (7, 4), (9, 1), (11, 27), (12, 25),
        ]
    ),
    2026: frozenset(
        date(2026, m, d)
        for m, d in [
            (1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19),
            (7, 3), (9, 7), (11, 26), (12, 25),
        ]
    ),
    2027: frozenset(
        date(2027, m, d)
        for m, d in [
            (1, 1), (1, 18), (2, 15), (3, 26), (5, 31), (6, 18),
            (7, 5), (9, 6), (11, 25), (12, 24),
        ]
    ),
}

#: Regular session window in UTC: 09:30-16:00 ET == 13:30-20:00 UTC (EST).
_SESSION_OPEN_UTC = time(13, 30)
_SESSION_CLOSE_UTC = time(20, 0)


def is_regular_session(ts: pd.Timestamp) -> bool:
    """True when ``ts`` (tz-aware; naive treated as UTC) falls in a NYSE
    regular session: a weekday, not a hardcoded holiday, inside
    13:30-20:00 UTC (09:30-16:00 ET). Half days (e.g. Jul 3) are NOT
    special-cased: bars past 16:00 ET on those days should not occur in
    regular-session-only caches anyway."""
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    if ts.weekday() >= 5:
        return False
    holidays = _HOLIDAYS.get(ts.year)
    if holidays is None:
        # Outside the hardcoded table range: fall back to weekday-only
        # weekends rule and be conservative about nothing else.
        holidays = frozenset()
    if ts.date() in holidays:
        return False
    return _SESSION_OPEN_UTC <= ts.time() <= _SESSION_CLOSE_UTC


def _session_mask(index: pd.DatetimeIndex) -> pd.Series[bool]:
    utc = index.tz_convert("UTC")
    weekday_ok = utc.weekday < 5
    holiday_ok = pd.Series(
        [d not in _HOLIDAYS.get(d.year, frozenset()) for d in utc.date],
        index=index,
    )
    t = utc.time
    time_ok = pd.Series(
        [_SESSION_OPEN_UTC <= x <= _SESSION_CLOSE_UTC for x in t], index=index
    )
    return weekday_ok & holiday_ok & time_ok


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_bars(
    symbols: list[str], start: str | None = None, end: str | None = None
) -> dict[str, pd.DataFrame]:
    """Load cached bars per symbol from the parquet cache.

    Returns ``{symbol: DataFrame}`` with a tz-aware UTC DatetimeIndex,
    contract columns, out-of-session bars dropped, duplicates dropped,
    sorted. ``start``/``end`` are inclusive ISO timestamps. Symbols with
    no cache file yield an empty frame (still present in the dict).
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = cache_path(symbol)
        if not path.exists():
            out[symbol] = pd.DataFrame(columns=_COLUMNS[1:])
            continue
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        index = pd.DatetimeIndex(ts)
        df = df.set_index(index).drop(columns=["timestamp"])
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df[_session_mask(df.index)]
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        out[symbol] = df
    return out


# ---------------------------------------------------------------------------
# Fetch (yfinance)
# ---------------------------------------------------------------------------


def _chunk_window(
    start: pd.Timestamp, end: pd.Timestamp, days: int = 7
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end)`` into contiguous calendar windows of <= ``days``
    days. Yahoo caps 1m granularity fetches at 8 days per request."""
    if end <= start:
        return [(start, end)]
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    lo = start
    step = pd.Timedelta(days=days)
    while lo < end:
        hi = min(lo + step, end)
        chunks.append((lo, hi))
        lo = hi
    return chunks


def _fetch_history_with_retry(label: str, fn, tries: int = 6):
    """Retry ``fn`` on transient failures; raise on final exhaustion so a
    partial cache is never written."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - yfinance raises variety
            last = exc
            if attempt < tries - 1:
                _time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"fetch failed after {tries} attempts: {label}") from last


def fetch_symbols(
    symbols: list[str], days: int = 25, start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch 1-minute bars via yfinance, session-filter, write cache.

    Requires network. Yields contract-conformant parquet files under
    ``data/market/``. Yahoo caps 1m fetches at 8 days per request, so any
    window spanning more than 7 calendar days is fetched in <=7-day chunks
    and merged. Per-chunk failures retry, then raise — never partial data.
    """
    import yfinance as yf  # deferred: offline tests never need it

    now = pd.Timestamp.now(tz="UTC")
    if start is not None:
        win_start = pd.Timestamp(start, tz="UTC")
        win_end = pd.Timestamp(end, tz="UTC") if end is not None else now
    else:
        win_end = pd.Timestamp(end, tz="UTC") if end is not None else now
        win_start = win_end - pd.Timedelta(days=days)
    chunks = _chunk_window(win_start, win_end)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames: list[pd.DataFrame] = []
        for lo, hi in chunks:
            label = f"{symbol} {lo:%Y-%m-%d}..{hi:%Y-%m-%d}"
            raw = _fetch_history_with_retry(
                label,
                lambda lo=lo, hi=hi, symbol=symbol: yf.Ticker(symbol).history(
                    start=lo.to_pydatetime(), end=hi.to_pydatetime(),
                    interval="1m",
                ),
            )
            if raw is not None and not raw.empty:
                frames.append(raw)
        if not frames:
            result[symbol] = pd.DataFrame(columns=_COLUMNS[1:])
            continue
        df = _conform(pd.concat(frames))
        df.to_parquet(cache_path(symbol), index=False)
        result[symbol] = df
    return result


def _conform(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance OHLCV frame to the cache contract columns."""
    renamed = raw.rename(columns=str.lower)[
        ["open", "high", "low", "close", "volume"]
    ]
    df = renamed.astype(
        {"open": "float64", "high": "float64", "low": "float64",
         "close": "float64", "volume": "int64"}
    )
    idx = pd.DatetimeIndex(
        pd.to_datetime(df.index, utc=True), name="timestamp"
    )
    df.index = idx
    df = df[_session_mask(df.index)].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.reset_index()
    df["timestamp"] = df["timestamp"].astype("int64")  # ns since epoch
    return df[_COLUMNS]


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


def validate_cache(symbols: list[str] | None = None) -> str:
    """Render the T6 data-validation report from the current cache state."""
    symbols = symbols if symbols is not None else BASKET
    lines = [
        "# T6 — Market Data Layer: Validation Report",
        "",
        "Cache: `data/market/{SYMBOL}_1m.parquet`; contract: int64-ns UTC"
        " timestamps, OHLC float64, volume int64, regular session only"
        " (13:30-20:00 UTC), sorted, no duplicate timestamps.",
        "",
        "| Symbol | Bars | First ts (UTC) | Last ts (UTC) | Gaps |"
        " Out-of-session |",
        "|---|---:|---|---|---:|---:|",
    ]
    missing: list[str] = []
    for symbol in symbols:
        path = cache_path(symbol)
        if not path.exists():
            missing.append(symbol)
            continue
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        idx = pd.DatetimeIndex(ts)
        # Gap = interval between consecutive bars other than 1 minute.
        diffs = idx.to_series().diff().dt.total_seconds().dropna()
        gaps = int((diffs != 60).sum())
        oos = int((~_session_mask(idx)).sum())
        lines.append(
            f"| {symbol} | {len(df):,} | {idx.min()} | {idx.max()}"
            f" | {gaps:,} | {oos} |"
        )
    lines.append("")
    if missing:
        lines.append(
            "**PARTIAL CACHE** — no file for: " + ", ".join(missing) + "."
        )
    else:
        lines.append("Full basket present.")
    # Timezone proof: timestamps are int64 ns UTC; show one decoded sample.
    first_file = next(
        (cache_path(s) for s in symbols if cache_path(s).exists()), None
    )
    if first_file is not None:
        sample = pd.read_parquet(first_file).iloc[0]
        decoded = pd.Timestamp(sample["timestamp"], tz="UTC")
        lines.append(
            f"Timezone proof: `{first_file.name}` first timestamp "
            f"{int(sample['timestamp'])} ns decodes to {decoded} "
            f"(tz=UTC)."
        )
    lines.append("")
    lines.append("_Session filter: `is_regular_session` — NYSE holidays "
                 "2024-2027 hardcoded (source: nyse.com hours-calendars)._")
    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _register() -> None:
    def builder(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "fetch-data",
            help="Fetch 1-minute bars via yfinance into the parquet cache,"
            " then validate it.",
        )
        p.add_argument(
            "--symbols", default=",".join(BASKET),
            help="Comma-separated symbols (default: full basket).",
        )
        p.add_argument(
            "--days", type=int, default=25,
            help="Calendar days of 1m history (default 25).",
        )
        p.add_argument("--start", default=None, help="ISO start (overrides --days).")
        p.add_argument("--end", default=None, help="ISO end.")
        p.set_defaults(func=_run)

    def _run(args: argparse.Namespace) -> int:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        fetch_symbols(symbols, days=args.days, start=args.start, end=args.end)
        print(validate_cache(symbols))
        return 0

    register_command("fetch-data", builder)


_register()
