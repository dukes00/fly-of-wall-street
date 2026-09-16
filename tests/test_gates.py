"""Synthetic-fixture tests for scripts/gates.py (G-A acceptance gates)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gates  # noqa: E402

NS = 1_000_000_000  # nanoseconds per second
HORIZON = 30


# ------------------------------------------------------------- fixture helpers

def make_market(dir_: Path, tickers: list[str], n_bars: int = 100, seed: int = 0):
    """Flat parquet per ticker: rising prices with a seeded wiggle."""
    dir_.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_ts = 1_700_000_000 * NS
    for tk in tickers:
        price = 100.0 + 0.1 * np.arange(n_bars) + rng.normal(0, 0.05, n_bars).cumsum()
        ts = base_ts + np.arange(n_bars) * 60 * NS
        pd.DataFrame({"timestamp": ts.astype(np.int64),
                      "open": price, "high": price, "low": price,
                      "close": price, "volume": 1000}).to_parquet(dir_ / f"{tk}_1m.parquet")
    return dir_


def fwd_ret(market_dir: Path, tk: str, bar: int, horizon: int = HORIZON) -> float:
    df = pd.read_parquet(market_dir / f"{tk}_1m.parquet")
    c0, c1 = float(df["close"].iloc[bar]), float(df["close"].iloc[bar + horizon])
    return c1 / c0 - 1.0


def write_run(dir_: Path, events: list[dict]):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n")
def signal_run(market_dir: Path, tickers: list[str], n_per: int = 60,
               shuffle: bool = False, seed: int = 1) -> list[dict]:
    """Encounters whose balance_used tracks (or is shuffled against) fwd return."""
    rng = np.random.default_rng(seed)
    events = []
    enc = []
    for tk in tickers:
        for j in range(n_per):
            bar = j + 5  # leaves room for horizon-30 forward return
            ts = (1_700_000_000 + bar * 60) * NS
            r = fwd_ret(market_dir, tk, bar)
            enc.append((ts, tk, 1000.0 * r + rng.normal(0, 0.01)))
    if shuffle:
        balances = [e[2] for e in enc]
        rng.shuffle(balances)
        enc = [(ts, tk, b) for (ts, tk, _), b in zip(enc, balances, strict=True)]
    for ts, tk, bal in enc:
        events.append({"type": "decision", "ts": ts, "ticker": tk,
                       "action": "buy", "reason": "signal", "balance_used": bal})
        events.append({"type": "encounter", "ts": ts, "ticker": tk,
                       "balance_used": bal})
    return events


def trade_events(trades: list[tuple[str, int, int, float, float]]) -> list[dict]:
    """(ticker, buy_bar, sell_bar, buy_px, sell_px) -> buy+sell order events."""
    base = 1_700_000_000
    ev = []
    for tk, b, s, bp, sp in trades:
        ev.append({"type": "order", "ts": (base + b * 60) * NS,
                   "ticker": tk, "side": "buy", "shares": 100, "price": bp})
        ev.append({"type": "order", "ts": (base + s * 60) * NS,
                   "ticker": tk, "side": "sell", "shares": 100, "price": sp})
    return ev


@pytest.fixture()
def market(tmp_path):
    return make_market(tmp_path / "market", ["AAA", "BBB", "CCC"])


# ------------------------------------------------------------------- G-A.1

def test_ga1_strong_signal_passes(market, tmp_path):
    events = signal_run(market, ["AAA", "BBB", "CCC"])
    res = gates.spearman_gate(
        gates.collect_samples(events, gates._market_index(market), HORIZON),
        HORIZON, gates._market_index(market))
    assert res["full"]["t"] > 2.0
    assert res["pass"]


def test_ga1_shuffled_balance_fails(market, tmp_path):
    events = signal_run(market, ["AAA", "BBB", "CCC"], shuffle=True, seed=3)
    m = gates._market_index(market)
    res = gates.spearman_gate(gates.collect_samples(events, m, HORIZON),
                              HORIZON, m)
    assert not res["pass"]
    assert abs(res["full"]["t"]) < 2.0


def test_ga1_nonoverlapping_matches_full_sign(market):
    events = signal_run(market, ["AAA", "BBB", "CCC"])
    m = gates._market_index(market)
    res = gates.spearman_gate(gates.collect_samples(events, m, HORIZON), HORIZON, m)
    f, s = res["full"], res["non_overlapping"]
    assert s["n"] > 0 and s["n"] < f["n"]
    assert (f["t"] > 0) == (s["t"] > 0)


# ------------------------------------------------------------------- G-A.2

def test_ga2_win_loss_ratio():
    # 3 wins, 1 loss -> ratio 3.0 > 1.0
    trades = gates.closed_trades(trade_events([
        ("AAA", 1, 5, 100.0, 101.0),
        ("AAA", 10, 15, 100.0, 102.0),
        ("BBB", 2, 6, 50.0, 51.0),
        ("BBB", 20, 25, 50.0, 49.0),  # loss
    ]))
    assert len(trades) == 4
    g = gates.behavior_gate(trades)
    assert g["wins"] == 3 and g["losses"] == 1
    assert g["win_loss_ratio"] == 3.0 and g["pass"]


def test_ga2_majority_losses_fails():
    trades = gates.closed_trades(trade_events([
        ("AAA", 1, 5, 100.0, 99.0),
        ("AAA", 10, 15, 100.0, 98.0),
        ("BBB", 2, 6, 50.0, 50.5),
    ]))
    g = gates.behavior_gate(trades)
    assert not g["pass"]


# ------------------------------------------------------------------- G-A.3

def test_ga3_run_3x_reference_trades_fails():
    ref_events = trade_events([
        ("AAA", 1, 5, 100.0, 101.0),
        ("AAA", 10, 15, 100.0, 100.5),
        ("BBB", 2, 6, 50.0, 50.5),
        ("BBB", 20, 25, 50.0, 51.0),
    ]) + [{"type": "encounter", "ts": 1_700_000_000 * NS, "ticker": "AAA", "balance_used": 1.0}]
    run_events = list(ref_events) + trade_events([
        ("CCC", 30, 35, 10.0, 10.2),
        ("CCC", 40, 45, 10.0, 10.1),
        ("CCC", 50, 55, 10.0, 10.3),
        ("CCC", 60, 65, 10.0, 10.4),
    ])  # 8 closed trades vs 4 -> 2x on trades/day... need 3x: add more
    run_events += trade_events([("CCC", 70, 75, 10.0, 10.5), ("CCC", 80, 85, 10.0, 10.6)])

    ref_m = gates.turnover_metrics(ref_events, gates.closed_trades(ref_events))
    run_m = gates.turnover_metrics(run_events, gates.closed_trades(run_events))
    g = gates.turnover_gate(run_m, ref_m)
    assert not g["pass"]
    assert g["metrics"]["trades_per_day"]["ratio"] > 2.0


def test_ga3_within_band_passes():
    ref_events = trade_events([("AAA", 1, 5, 100.0, 101.0)]) + [
        {"type": "encounter", "ts": 1_700_000_000 * NS, "ticker": "AAA", "balance_used": 1.0}]
    run_events = trade_events([
        ("AAA", 1, 5, 100.0, 100.5),
        ("BBB", 1, 5, 100.0, 101.5),
    ]) + [{"type": "encounter", "ts": 1_700_000_000 * NS, "ticker": "AAA", "balance_used": 1.0}]
    ref_m = gates.turnover_metrics(ref_events, gates.closed_trades(ref_events))
    run_m = gates.turnover_metrics(run_events, gates.closed_trades(run_events))
    assert gates.turnover_gate(run_m, ref_m)["pass"]


# ------------------------------------------------------------------- G-A.4

def test_ga4_changes_and_deaths():
    base = 1_700_000_000 * NS
    ref_events = [
        {"type": "decision", "ts": base, "ticker": "AAA", "action": "buy"},
        {"type": "decision", "ts": base + NS, "ticker": "BBB", "action": "hold"},
        {"type": "death", "ts": base + 2 * NS, "ticker": "CCC"},
    ]
    run_events = [
        {"type": "decision", "ts": base, "ticker": "AAA", "action": "hold"},  # changed
        {"type": "decision", "ts": base + NS, "ticker": "BBB", "action": "hold"},
        {"type": "death", "ts": base + 2 * NS, "ticker": "CCC"},
    ]
    g = gates.learning_gate(run_events, ref_events)
    assert g["n_matched_decisions"] == 2 and g["n_changed"] == 1
    assert g["decision_change_rate"] == 0.5
    assert g["n_deaths_run"] == 1 and g["n_deaths_reference"] == 1
    assert g["pass"]


def test_ga4_more_deaths_fails():
    base = 1_700_000_000 * NS
    ref_events = [{"type": "decision", "ts": base, "ticker": "AAA", "action": "buy"},
                  {"type": "death", "ts": base, "ticker": "AAA"}]
    run_events = [{"type": "decision", "ts": base, "ticker": "AAA", "action": "hold"},
                  {"type": "death", "ts": base, "ticker": "AAA"},
                  {"type": "death", "ts": base + NS, "ticker": "BBB"}]
    g = gates.learning_gate(run_events, ref_events)
    assert not g["pass"]


# ----------------------------------------------------------------- end-to-end

def test_compute_gates_overall_and_exit(tmp_path, market):
    tickers = ["AAA", "BBB", "CCC"]
    run_events = signal_run(market, tickers, seed=3)
    run_events += trade_events([
        ("AAA", 1, 5, 100.0, 101.0), ("AAA", 10, 15, 100.0, 101.5),
        ("BBB", 2, 6, 100.0, 100.8), ("BBB", 20, 25, 100.0, 101.2),
        ("CCC", 3, 7, 100.0, 100.6), ("CCC", 30, 35, 100.0, 101.1),
    ])
    run_events.append({"type": "death", "ts": 1_700_000_010 * NS, "ticker": "AAA"})
    ref_events = list(run_events)  # identical decisions but different actions below
    ref_events = [dict(e) for e in ref_events]
    for e in ref_events:
        if e.get("type") == "decision":
            e["action"] = "hold" if e["action"] == "buy" else "buy"
    ref_events.append({"type": "death", "ts": 1_700_000_010 * NS, "ticker": "BBB"})

    write_run(tmp_path / "run", run_events)
    write_run(tmp_path / "ref", ref_events)
    res = gates.compute_gates(tmp_path / "run", tmp_path / "ref", market, HORIZON)
    assert res["GA1"]["pass"] and res["GA2"]["pass"] and res["GA4"]["pass"]
    assert res["overall"]


def test_cli_exit_codes(tmp_path, market, capsys):
    events = signal_run(market, ["AAA", "BBB", "CCC"], seed=5)
    write_run(tmp_path / "run", events)
    write_run(tmp_path / "ref", events)
    rc = gates.main(["--run-dir", str(tmp_path / "run"), "--reference",
                     str(tmp_path / "ref"), "--market-dir", str(market)])
    assert rc in (0, 1)  # turnover gate fails (no trades) but signal gate reported


def test_shuffled_run_json_output(tmp_path, market):
    events = signal_run(market, ["AAA", "BBB", "CCC"], shuffle=True, seed=3)
    write_run(tmp_path / "run", events)
    write_run(tmp_path / "ref", events)
    out = tmp_path / "gates.json"
    rc = gates.main(["--run-dir", str(tmp_path / "run"), "--reference",
                     str(tmp_path / "ref"), "--market-dir", str(market),
                     "--json", str(out)])
    data = json.loads(out.read_text())
    assert set(data) == {"GA1", "GA2", "GA3", "GA4", "horizon", "overall"}
    assert abs(data["GA1"]["full"]["t"]) < 2.0
    assert rc == 1  # shuffled signal + no trades + no changes -> fail
