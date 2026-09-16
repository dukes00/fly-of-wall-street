"""Tests for the T6 market data layer (fully offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fruitfly.data import (
    BASKET,
    _conform,
    is_regular_session,
    load_bars,
    neverseen_basket,
    parse_basket_file,
    resolve_basket,
)

NS = 1_000_000_000


def _ns(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    return int(pd.Timestamp(year, month, day, hour, minute, tz="UTC").value)


def _bars(timestamps: list[int]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": np.array(timestamps, dtype="int64"),
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": np.full(n, 1000, dtype="int64"),
        }
    )


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    """Synthetic cache with an out-of-session bar and a duplicate timestamp.

    Sessions: Mon 2025-11-03 and Tue 2025-11-04 (DST ended Nov 2, 2025 —
    ET is UTC-5, so sessions are 14:30-21:00 UTC... but our contract keeps
    the fixed 13:30-20:00 UTC window; bars are built inside it). One
    pre-open bar (13:15 UTC) and one duplicate are intentionally planted.
    """
    monkeypatch.setattr("fruitfly.data.CACHE_DIR", tmp_path)
    rows = _bars(
        [
            # Mon Nov 3 2025: pre-open plant + 13:30..13:34
            _ns(2025, 11, 3, 13, 15),
            _ns(2025, 11, 3, 13, 30),
            _ns(2025, 11, 3, 13, 31),
            _ns(2025, 11, 3, 13, 32),
            _ns(2025, 11, 3, 13, 33),
            _ns(2025, 11, 3, 13, 34),
            # Tue Nov 4 2025: session with a duplicate ts
            _ns(2025, 11, 4, 13, 30),
            _ns(2025, 11, 4, 13, 30),
            _ns(2025, 11, 4, 13, 31),
        ]
    )
    (tmp_path / "AAPL_1m.parquet").write_bytes(b"")
    import pyarrow  # noqa: F401  # ensure parquet engine available

    rows.to_parquet(tmp_path / "AAPL_1m.parquet", index=False)
    return tmp_path


def test_loader_filters_out_of_session_and_duplicates(cache) -> None:
    frames = load_bars(["AAPL"])
    df = frames["AAPL"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert df.index.tz is not None
    # Pre-open 13:15 bar dropped; duplicate 13:30 Nov 4 collapsed.
    assert len(df) == 5 + 2
    assert df.index[0] == pd.Timestamp(2025, 11, 3, 13, 30, tz="UTC")
    assert pd.Timestamp(2025, 11, 3, 13, 15, tz="UTC") not in df.index


def test_loader_start_end_and_missing_symbol(cache) -> None:
    frames = load_bars(["AAPL", "NOPE"], start="2025-11-04", end=None)
    assert frames["AAPL"].index[0] == pd.Timestamp(2025, 11, 4, 13, 30, tz="UTC")
    assert len(frames["AAPL"]) == 2  # dup 13:30 collapsed
    assert frames["NOPE"].empty


def test_session_filter_dst_boundary() -> None:
    # DST ends Sun Nov 2 2025. Monday pre-market: 13:00 UTC = 09:00 ET -> out.
    assert not is_regular_session(pd.Timestamp(2025, 11, 3, 13, 0, tz="UTC"))
    # 13:30 UTC = 08:30 ET (EST, UTC-5) -> per fixed UTC window, in-session.
    assert is_regular_session(pd.Timestamp(2025, 11, 3, 13, 30, tz="UTC"))
    assert is_regular_session(pd.Timestamp(2025, 11, 3, 19, 59, tz="UTC"))
    assert not is_regular_session(pd.Timestamp(2025, 11, 3, 20, 1, tz="UTC"))


def test_session_filter_weekend_and_holiday() -> None:
    # Sat
    assert not is_regular_session(pd.Timestamp(2025, 11, 1, 15, 0, tz="UTC"))
    # Thanksgiving 2025-11-27 (hardcoded holiday)
    assert not is_regular_session(pd.Timestamp(2025, 11, 27, 15, 0, tz="UTC"))
    # Naive timestamp treated as UTC
    assert is_regular_session(pd.Timestamp(2025, 11, 3, 14, 0))


def test_basket_constant() -> None:
    assert len(BASKET) == 22
    assert BASKET[0] == "AAPL" and BASKET[-1] == "^GSPC"


def test_conform_drops_out_of_session_and_dedups() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [2.0, 2.0, 2.0],
            "Low": [0.5, 0.5, 0.5],
            "Close": [1.5, 1.5, 1.5],
            "Volume": [10, 10, 10],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp(2025, 11, 3, 13, 15, tz="UTC"),  # pre-open
                pd.Timestamp(2025, 11, 3, 13, 30, tz="UTC"),
                pd.Timestamp(2025, 11, 3, 13, 30, tz="UTC"),  # dup
            ]
        ),
    )
    df = _conform(raw)
    assert list(df.columns) == [
        "timestamp", "open", "high", "low", "close", "volume"
    ]
    assert len(df) == 1
    assert df["timestamp"].dtype == "int64"
    assert df["volume"].dtype == "int64"


def test_chunk_window_splits_into_contiguous_seven_day_windows() -> None:
    from fruitfly.data import _chunk_window

    start = pd.Timestamp(2026, 1, 1, tz="UTC")
    end = pd.Timestamp(2026, 1, 25, tz="UTC")
    wins = _chunk_window(start, end)
    assert all((hi - lo) <= pd.Timedelta(days=7) for lo, hi in wins)
    assert all(wins[i][1] == wins[i + 1][0] for i in range(len(wins) - 1))
    assert wins[0][0] == start and wins[-1][1] == end
    assert (len(wins), (wins[-1][1] - wins[-1][0]).days) == (4, 3)
    # windows within the Yahoo 1m cap need no splitting
    assert _chunk_window(start, start + pd.Timedelta(days=6)) == [
        (start, start + pd.Timedelta(days=6))
    ]


# ---------------------------------------------------------------------------
# TRAINING2 Phase 0 §3: basket files + the never-seen eval guard
# ---------------------------------------------------------------------------


class TestBasketFiles:
    def test_parse_basket_file_round_trip(self, tmp_path):
        f = tmp_path / "train40.txt"
        f.write_text(
            "# TRAINING2 train basket (committed artifact)\n"
            "AAPL\n"
            "\n"
            "   MSFT   # trailing comment\n"
            "AAPL\n"
            "NVDA\n"
        )
        # One symbol per line, '#' comments (full-line and trailing), blank
        # lines skipped, duplicates keep their first occurrence and order.
        assert parse_basket_file(f) == ["AAPL", "MSFT", "NVDA"]

    def test_resolve_basket_defaults_and_missing_guard_is_noop(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "fruitfly.data.NEVERSEEN_BASKET_PATH", tmp_path / "absent.txt"
        )
        assert neverseen_basket() == []
        assert resolve_basket(["AAA", "BBB"]) == ["AAA", "BBB"]
        # None -> the module BASKET, returned as a fresh list.
        resolved = resolve_basket()
        assert resolved == list(BASKET)
        assert resolved is not BASKET

    def test_resolve_basket_raises_on_neverseen_intersection(
        self, tmp_path, monkeypatch
    ):
        p = tmp_path / "eval20-neverseen.txt"
        p.write_text("# frozen eval basket — never train on these\nZZZZ\nQQQQ\n")
        monkeypatch.setattr("fruitfly.data.NEVERSEEN_BASKET_PATH", p)
        with pytest.raises(ValueError) as ei:
            resolve_basket(["AAPL", "ZZZZ", "QQQQ"])
        # The offenders are listed; clean symbols are not offenders.
        assert "ZZZZ" in str(ei.value) and "QQQQ" in str(ei.value)
        assert "AAPL" not in str(ei.value)
