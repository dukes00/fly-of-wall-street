"""T10 post-mortem tests (offline, synthetic run dir, synthetic bars).

Gates: lifespan segments (hatch -> death -> re-hatch -> alive) are split and
timestamped correctly; the cause-of-death section cites events.jsonl LINE
RANGES that verifiably contain the death event (and the final-session tail
for an alive fly); the documented grudge/favorite heuristic fires on a
crafted tape (grudge only after a shock, not for the no-shock control or the
normal reflex); the report renders end-to-end with the real
``scoreboard.compare_run`` embedded on synthetic bars; double render is
byte-identical; the CLI writes the file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fruitfly.__main__ import main
from fruitfly.postmortem import (
    _behavior_autopsy,
    _lives,
    _pairs,
    generate_postmortem,
)

SYMBOLS = ["AAA", "BBB", "CCC"]
START = "2026-08-17"

_CITE_RE = re.compile(r"events\.jsonl:(\d+)(?:-(\d+))?")


# --- synthetic bars (same recipe as the T8 scoreboard tests) -----------------


def make_bars(n_bars: int = 400, seed: int = 7) -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2026-08-17 13:30", periods=n_bars, freq="min", tz="UTC")
    out = {}
    for i, sym in enumerate(SYMBOLS):
        r = np.random.default_rng(seed + i)
        t = np.arange(n_bars)
        phase = r.uniform(0.0, 2.0 * np.pi)
        drift = 0.0006 * np.sin(2.0 * np.pi * t / 60.0 + phase)
        close = 100.0 * np.cumprod(1.0 + drift + r.normal(0.0, 0.002, n_bars))
        volume = (
            1_000_000.0 * (1.0 + 0.4 * np.sin(2.0 * np.pi * t / 30.0 + phase + 1.0))
            + r.integers(0, 100_000, n_bars)
        ).astype(np.int64)
        out[sym] = pd.DataFrame({"close": close, "volume": volume}, index=ts)
    return out


# --- synthetic run: death + re-hatch + crafted grudge/favorite tape ----------


def _tape() -> list[dict]:
    enc = lambda ts, tkr, bal, raw: {  # noqa: E731
        "type": "encounter", "ts": ts, "ticker": tkr, "intensity": 1.0,
        "balance": bal, "raw_balance": raw, "valence_readout": bal * 100,
        "mbon_rate_hz": 0.0,
    }
    dec = lambda ts, tkr, action, reason: {  # noqa: E731
        "type": "decision", "ts": ts, "ticker": tkr, "action": action,
        "reason": reason,
    }
    order = lambda ts, tkr, side, reason, shares, price, pnl: {  # noqa: E731
        "type": "order", "ts": ts, "ticker": tkr, "side": side,
        "reason": reason, "shares": shares, "price": price,
        "realized_pnl": pnl, "cash_after": 0.0, "n_positions_after": 1,
    }
    return [
        {"type": "hatch", "hatch_equity": 100000.0},                       # 1
        {"type": "wake", "ts": "2026-08-17T13:30:00+00:00"},               # 2
        enc("2026-08-17T13:31:00+00:00", "AAPL", 0.60, 0.60),              # 3
        dec("2026-08-17T13:31:00+00:00", "AAPL", "buy", "approach"),       # 4
        order("2026-08-17T13:31:00+00:00", "AAPL", "buy", "approach",
              100, 100.0, None),                                           # 5
        enc("2026-08-17T13:33:00+00:00", "AAPL", 0.55, 0.55),              # 6
        dec("2026-08-17T13:33:00+00:00", "AAPL", "buy", "approach"),       # 7
        enc("2026-08-17T13:35:00+00:00", "AAPL", 0.50, 0.50),              # 8
        dec("2026-08-17T13:35:00+00:00", "AAPL", "buy", "approach"),       # 9
        enc("2026-08-17T13:36:00+00:00", "UNH", -0.40, -0.70),             # 10 shock
        dec("2026-08-17T13:36:00+00:00", "UNH", "avoid", "avoid"),         # 11
        enc("2026-08-17T13:38:00+00:00", "UNH", -0.10, +0.02),             # 12 grudge
        dec("2026-08-17T13:38:00+00:00", "UNH", "avoid", "avoid"),         # 13
        enc("2026-08-17T13:40:00+00:00", "MSFT", 0.02, 0.02),              # 14 control
        dec("2026-08-17T13:40:00+00:00", "MSFT", "pass", "neutral"),       # 15
        enc("2026-08-17T13:42:00+00:00", "TSLA", -0.30, -0.30),            # 16 reflex
        dec("2026-08-17T13:42:00+00:00", "TSLA", "avoid", "avoid"),        # 17
        order("2026-08-17T19:45:00+00:00", "AAPL", "sell", "valence_flip",
              175, 99.0, -1200.0),                                         # 18
        {"type": "death", "ts": "2026-08-17T19:50:00+00:00",
         "equity": 45000.0, "hatch_equity": 100000.0},                     # 19
        order("2026-08-17T19:50:00+00:00", "AAPL", "sell",
              "death_liquidation", 175, 99.0, -1175.50),                   # 20
        {"type": "hatch", "hatch_equity": 45000.0},                        # 21
        enc("2026-08-17T19:55:00+00:00", "GOOGL", 0.10, 0.10),             # 22
        dec("2026-08-17T19:55:00+00:00", "GOOGL", "buy", "approach"),      # 23
        {"type": "sugar_shock", "ts": "2026-08-17T20:00:00+00:00",
         "reward": 0.0, "punishment": 0.001, "hunger": 0.01,
         "arousal": 0.06},                                                 # 24
        {"type": "sleep", "ts": "2026-08-17T20:00:00+00:00"},              # 25
    ]


_EQUITY = (
    "timestamp,equity,cash,n_positions\n"
    "2026-08-17T13:30:00+00:00,100000.00,100000.00,0\n"
    "2026-08-17T13:31:00+00:00,100000.00,97500.00,1\n"
    "2026-08-17T13:35:00+00:00,100100.00,90000.00,3\n"
    "2026-08-17T13:36:00+00:00,99500.00,90000.00,3\n"
    "2026-08-17T13:40:00+00:00,99400.00,90000.00,3\n"
    "2026-08-17T19:45:00+00:00,46200.00,46200.00,0\n"
    "2026-08-17T19:50:00+00:00,45000.00,45000.00,0\n"
    "2026-08-17T19:55:00+00:00,45100.00,45100.00,1\n"
    "2026-08-17T20:00:00+00:00,45100.00,45100.00,1\n"
)


@pytest.fixture()
def run_dir(tmp_path: Path) -> tuple[Path, int]:
    """Synthetic run dir; returns (dir, 1-based line number of the death)."""
    d = tmp_path / "backtest_42_2026-08-17_2026-08-17"
    d.mkdir()
    (d / "equity.csv").write_text(_EQUITY)
    tape = _tape()
    (d / "events.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in tape)
    )
    death_line = next(
        i for i, e in enumerate(tape, 1) if e.get("type") == "death"
    )
    return d, death_line


# --- structure: lives, pairs, autopsy ----------------------------------------


def test_lives_split_hatch_death_rehatch_alive(run_dir):
    d, _ = run_dir
    events = [
        (i, json.loads(line))
        for i, line in enumerate(
            (d / "events.jsonl").read_text().splitlines(), 1
        )
    ]
    lives = _lives(events)
    assert [life.death is None for life in lives] == [False, True]
    first, second = lives
    assert first.hatch_line == 1
    assert first.born_ts == "2026-08-17T13:30:00+00:00"
    assert first.death["equity"] == 45000.0
    assert first.death_line == 19
    # duplicate-ish re-hatch: new life born at the next timestamped event
    assert second.hatch_line == 21
    assert second.born_ts == "2026-08-17T19:55:00+00:00"
    assert second.death is None


def test_grudge_favorite_heuristic_on_crafted_tape(run_dir):
    d, _ = run_dir
    events = [
        (i, json.loads(line))
        for i, line in enumerate(
            (d / "events.jsonl").read_text().splitlines(), 1
        )
    ]
    grudges, approaches, buys = _behavior_autopsy(_pairs(events))
    # UNH: shock at line 10, then innate +0.02 (raw says approach) but the
    # learned balance is -0.10 and the decision is avoid -> one-trial grudge
    assert [(g.ticker, g.shock_line, g.enc_line, g.action) for g in grudges] == [
        ("UNH", 10, 12, "avoid")
    ]
    assert grudges[0].raw_balance == pytest.approx(0.02)
    assert grudges[0].balance == pytest.approx(-0.10)
    # AAPL: three positive-balance buys -> favorite; MSFT control: neutral
    # pass with NO pending shock -> no grudge; TSLA: avoid right after its
    # own shock with innately negative raw_balance -> the normal reflex,
    # not a grudge.
    assert approaches == {"AAPL": 3, "GOOGL": 1}
    assert buys["AAPL"] == 3
    assert all(g.ticker != "MSFT" for g in grudges)
    assert all(g.ticker != "TSLA" for g in grudges)


# --- report content -----------------------------------------------------------


def test_death_cause_citations_contain_death_event(run_dir):
    d, death_line = run_dir
    report = generate_postmortem(d)
    start = report.index("## Cause of death")
    section = report[start : report.index("## P&L vs scoreboard")]
    # 55% below hatch, peak -> death trajectory, realized losses, liquidation
    assert "55.00% below hatch" in section
    assert "100,100.00" in section
    assert f"{100.0 * (1.0 - 45000.0 / 100100.0):.2f}% down" in section
    assert "AAPL -1,200.00" in section
    assert "death_liquidation" in section
    # EVERY cited line range in the cause section must be inside the log, and
    # at least one must actually contain the death event.
    n_lines = len((d / "events.jsonl").read_text().splitlines())
    ranges = [
        (int(a), int(b or a))
        for a, b in _CITE_RE.findall(section)
    ]
    assert ranges, "cause section must cite events.jsonl line ranges"
    assert all(1 <= a <= b <= n_lines for a, b in ranges)
    assert any(a <= death_line <= b for a, b in ranges)


def test_lifespan_and_tallies_in_report(run_dir):
    d, _ = run_dir
    report = generate_postmortem(d)
    assert "| 1 | events.jsonl:1 | 100,000.00 " in report
    assert "| 2026-08-17T19:50:00+00:00 | events.jsonl:19 |" in report
    assert "alive at end of log" in report
    assert "encounter: 8" in report
    assert "deaths: 1, hatches: 2" in report


def test_grudge_favorite_tables_in_report(run_dir):
    d, _ = run_dir
    report = generate_postmortem(d)
    grudge_part = report[
        report.index("### Grudges") : report.index("### Favorites")
    ]
    fav_part = report[report.index("### Favorites") : report.index("## Event tallies")]
    assert "| UNH | 1 |" in grudge_part
    assert "MSFT" not in grudge_part
    assert "TSLA" not in grudge_part
    assert "| AAPL | 3 | 3 |" in fav_part
    assert "GOOGL" not in fav_part  # 1 approach trial < threshold


# --- end-to-end render with the REAL scoreboard on synthetic bars -------------


def test_report_renders_end_to_end_with_scoreboard(run_dir):
    d, _ = run_dir
    out = d.parent / "t10-postmortem.md"
    report = generate_postmortem(d, out, bars_by_symbol=make_bars())
    assert out.read_text(encoding="utf-8") == report
    assert report.startswith("# T10 Post-mortem — backtest_42_2026-08-17_2026-08-17")
    # final equity lead line: 45,100 on the original 100,000 hatch capital
    assert "45,100.00" in report and "-54.90%" in report
    assert "| Fly (this run) |" in report
    assert "| S&P 500 buy-and-hold |" in report
    assert "| Monkey-with-darts (seed 42" in report
    assert "| Logistic control" in report
    assert "SPIVA" in report


def test_double_render_byte_identical(run_dir):
    d, _ = run_dir
    bars = make_bars()
    r1 = generate_postmortem(d, bars_by_symbol=bars)
    r2 = generate_postmortem(d, bars_by_symbol=bars)
    assert r1 == r2
    out1, out2 = d.parent / "a.md", d.parent / "b.md"
    generate_postmortem(d, out1, bars_by_symbol=bars)
    generate_postmortem(d, out2, bars_by_symbol=bars)
    assert out1.read_bytes() == out2.read_bytes()


def test_cli_postmortem_writes_report(run_dir, monkeypatch):
    d, _ = run_dir
    # keep the CLI's default scoreboard path hermetic: compare_run resolves
    # load_bars from fruitfly.data at call time.
    monkeypatch.setattr(
        "fruitfly.data.load_bars", lambda *a, **k: make_bars()
    )
    out = d.parent / "cli-postmortem.md"
    assert main(["postmortem", "--run-dir", str(d), "--out", str(out)]) == 0
    written = out.read_text(encoding="utf-8")
    assert written == generate_postmortem(d, bars_by_symbol=make_bars())
    assert "## Cause of death" in written


def test_scoreboard_failure_degrades_gracefully(run_dir, monkeypatch):
    d, _ = run_dir

    def boom(*a, **k):
        raise RuntimeError("no market cache")

    monkeypatch.setattr("fruitfly.postmortem.compare_run", boom)
    report = generate_postmortem(d)
    assert "scoreboard could not be rendered" in report
    assert "no market cache" in report
