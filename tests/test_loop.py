"""Tests for the T7 foraging backtest loop (offline, synthetic).

Never touches the real chassis cache or market parquet cache: the loop's
seams (``_load_chassis``, ``load_bars``, ``BASKET``) are monkeypatched to a
tiny synthetic chassis (30 nodes mirroring the stripped-chassis populations)
and two synthetic tickers of 1-minute bars. The real-CLI acceptance runs live
in the orchestrator, not here.

The hard gates: same seed -> byte-identical artifacts, the position cap is
never exceeded even mid-rotation, every event family of DESIGN §7 appears in
the log, different seeds genuinely diverge (the loop RNG's noise floor), and
death at the equity threshold kills the fly and hatches a fresh one (D14).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fruitfly import loop as loop_module
from fruitfly.connectome import Chassis
from fruitfly.loop import BacktestConfig, _plume_set, run_backtest

# --- synthetic chassis (mirrors reports/t2-connectome.md populations) -------

_COLS = ["bodyId", "type", "instance", "somaSide", "population", "region",
         "neurotransmitter", "sign"]

# (type, population, neurotransmitter, sign)
_ROWS = (
    [("DA1_adPN1", "uPN", "acetylcholine", 1), ("DA1_lPN2", "uPN", "acetylcholine", 1),
     ("VA1v_lPN1", "uPN", "gaba", -1), ("VA1v_adPN2", "uPN", "acetylcholine", 1)]
    + [(f"KC{k}", "KC", "acetylcholine", 1) for k in range(8)]
    + [("MBON01", "MBON", "acetylcholine", 1), ("MBON02", "MBON", "acetylcholine", 1),
       ("MBON03", "MBON", "acetylcholine", 1), ("MBON04", "MBON", "glutamate", -1),
       ("MBON05", "MBON", "gaba", -1), ("MBON06", "MBON", "glutamate", -1)]
    + [("PAM01", "PAM", "dopamine", 0), ("PAM02", "PAM", "dopamine", 0),
       ("PPL101", "PPL1", "dopamine", 0), ("PPL102", "PPL1", "dopamine", 0)]
)

# uPN -> KC (dense enough to spike KCs), KC -> MBON (every KC feeds BOTH an
_EDGES = [
    # uPN -> KC: four KCs sit well above threshold (all four fire every
    # encounter); noise jitter then perturbs per-KC spike counts.
    (0, 4, 300), (3, 4, 300),
    (0, 5, 300), (3, 5, 300),
    (0, 6, 350), (3, 6, 250),
    (0, 7, 250), (3, 7, 350),
    (0, 8, 150), (1, 9, 150), (3, 10, 150), (3, 11, 150),  # sub-threshold
    # KC -> MBON: asymmetric approach/avoidance fan-out per KC, so the
    # balance is graded and moves when different KC sets dominate.
    (4, 12, 150), (4, 15, 80),
    (5, 13, 60), (5, 16, 150),
    (6, 14, 120), (6, 17, 120),
    (7, 12, 100), (7, 16, 60),
]


def make_chassis(lc: bool = False) -> Chassis:
    rows = list(_ROWS)
    regions = (
        ["antennal-lobe"] * 4 + ["mushroom-body"] * 14 + ["protocerebrum"] * 4
    )
    edges = list(_EDGES)
    if lc:
        # Two LC looming cells with direct LC->MBON synapses onto approach
        # MBONs (DESIGN v0.6 §5 structural path). Appended after the PPL1
        # rows so every existing node index stays unchanged.
        n = len(_ROWS)
        rows += [("LC01", "LC-looming", "acetylcholine", 1),
                 ("LC02", "LC-looming", "acetylcholine", 1)]
        regions += ["optic-lobe", "optic-lobe"]
        edges += [(n, 12, 400), (n + 1, 13, 400)]
    n = len(rows)
    nodes = pd.DataFrame(
        [(10_000 + i, t, t, "L", p, r, nt, s)
         for i, (t, p, nt, s), r
         in zip(range(n), rows, regions, strict=True)],
        columns=_COLS,
    )
    pre = np.array([e[0] for e in edges], dtype=np.int64)
    post = np.array([e[1] for e in edges], dtype=np.int64)
    w = np.array([e[2] for e in edges], dtype=np.int64)
    adj = sp.csr_matrix((w, (pre, post)), shape=(n, n), dtype=np.int64)
    adj.sum_duplicates()
    return Chassis(
        nodes=nodes, adj=adj, meta={"glomeruli": ["DA1", "VA1v"], "n_glomeruli": 2}
    )


def make_bars(crash: bool = False) -> dict[str, pd.DataFrame]:
    """Two tickers x two sessions x five 1-min bars; every bar moves.

    With ``crash`` the first bar of BBB's second session gaps down -95%
    (used to drive equity through the death threshold synthetically).
    """
    frames: dict[str, pd.DataFrame] = {}
    for ticker, base, drift in (("AAA", 100.0, 1.0), ("BBB", 200.0, -0.5)):
        rows, stamps = [], []
        price = base
        for day in (18, 19):
            for minute in range(5):
                ts = pd.Timestamp(f"2026-08-{day} 13:3{minute}:00", tz="UTC")
                stamps.append(ts)
                if crash and ticker == "BBB" and day == 19 and minute == 0:
                    price = 10.0  # -95%: the synthetic crash
                else:
                    price = price + drift if day == 18 else price - drift * 0.4
                rows.append(
                    {
                        "open": price - drift * 0.2,
                        "high": price + 0.3,
                        "low": price - 0.3,
                        "close": price,
                        "volume": 5_000 + 1_000 * (minute % 2),
                    }
                )
        frames[ticker] = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps))
    return frames


def make_looming_bars() -> dict[str, pd.DataFrame]:
    """One mover whose day-2 open is a crash bar: a huge vertical range
    right after a quiet window -> LC looming currents fire (DESIGN v0.6 §5).

    AAA is the only ticker, so every plume-bearing bar samples AAA and the
    encounter that sees the crash bar is exactly 2026-08-19T13:30Z.
    """
    plan = [  # (day, minute, close, half_range)
        (18, 0, 100.0, 0.3), (18, 1, 100.5, 0.3), (18, 2, 101.0, 0.3),
        (19, 0, 92.0, 8.0),  # the crash bar: big range, quiet window
        (19, 1, 92.5, 0.3), (19, 2, 93.0, 0.3),
    ]
    rows, stamps = [], []
    for day, minute, close, hr in plan:
        stamps.append(pd.Timestamp(f"2026-08-{day} 13:3{minute}:00", tz="UTC"))
        rows.append({"open": close - 0.2, "high": close + hr,
                     "low": close - hr, "close": close, "volume": 5_000})
    return {"AAA": pd.DataFrame(rows, index=pd.DatetimeIndex(stamps))}


@pytest.fixture()
def patched_lc(monkeypatch, tmp_path):
    """Like ``patched`` but on the LC->MBON chassis + the crash tape."""
    import fruitfly.loop as loop

    monkeypatch.setattr(loop, "_load_chassis", lambda: make_chassis(lc=True))
    monkeypatch.setattr(loop, "_load_whole_chassis", lambda: make_chassis(lc=True))
    monkeypatch.setattr(
        loop, "load_bars", lambda symbols, start=None, end=None: make_looming_bars()
    )
    monkeypatch.setattr(loop, "BASKET", ["AAA"])
    return loop


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    """Wire the loop's seams to the synthetic chassis + bars."""
    import fruitfly.loop as loop

    monkeypatch.setattr(loop, "_load_whole_chassis", lambda: chassis)
    chassis = make_chassis()
    bars = make_bars()
    monkeypatch.setattr(loop, "_load_chassis", lambda: chassis)
    monkeypatch.setattr(loop, "load_bars", lambda symbols, start=None, end=None: bars)
    monkeypatch.setattr(loop, "BASKET", ["AAA", "BBB"])
    return loop


def _config(seed: int, out_dir: Path, **overrides) -> BacktestConfig:
    defaults = dict(ms_per_bar=20.0, dt_ms=0.5)
    return BacktestConfig(seed=seed, start="2026-08-18", end="2026-08-19",
                          out_dir=out_dir, **{**defaults, **overrides})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Determinism (hard gate)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_double_run_byte_identical(self, patched, tmp_path):
        result_a = run_backtest(_config(7, tmp_path / "a"))
        result_b = run_backtest(_config(7, tmp_path / "b"))
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(result_a.run_dir / name) == _sha(result_b.run_dir / name)

    def test_different_seeds_diverge(self, patched, tmp_path):
        # A big noise floor guarantees at least one threshold crossing flips.
        kwargs = dict(noise_sigma_mv=5.0)
        a = run_backtest(_config(1, tmp_path / "a", **kwargs))
        b = run_backtest(_config(2, tmp_path / "b", **kwargs))
        diverged = any(
            _sha(a.run_dir / name) != _sha(b.run_dir / name)
            for name in ("equity.csv", "events.jsonl")
        )
        assert diverged


class TestInvariants:
    def test_position_cap_never_exceeded(self, patched, tmp_path):
        cap = 2  # tightened below the default to prove the guard bites
        # Tight thresholds + a strong noise floor: encounters produce real
        # churn (buys and valence-flip sells), so the cap guard is exercised.
        result = run_backtest(
            _config(7, tmp_path / "run", position_cap=cap,
                    approach_thr=0.001, avoid_thr=0.001, noise_sigma_mv=10.0)
        )
        open_positions: set[str] = set()
        for event in _events(result.run_dir / "events.jsonl"):
            if event["type"] != "order":
                continue
            if event["reason"] == "death_liquidation":
                continue
            if event["side"] == "buy" and event["reason"] != "add":
                open_positions.add(event["ticker"])
            elif event["side"] == "sell":
                open_positions.discard(event["ticker"])
            assert len(open_positions) <= cap
        rows = (result.run_dir / "equity.csv").read_text().splitlines()[1:]
        assert all(int(row.split(",")[3]) <= cap for row in rows)

    def test_event_types_cover_the_daily_loop(self, patched, tmp_path):
        # Tight thresholds + strong noise: encounters produce orders.
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=10.0)
        )
        types = {event["type"] for event in _events(result.run_dir / "events.jsonl")}
        assert {"wake", "encounter", "decision", "order", "sugar_shock", "sleep",
                "anchor"} <= types

    def test_equity_csv_contract(self, patched, tmp_path):
        result = run_backtest(_config(7, tmp_path / "run"))
        lines = (result.run_dir / "equity.csv").read_text().splitlines()
        assert lines[0] == "timestamp,equity,cash,n_positions"
        assert len(lines) - 1 == result.n_bars == 10  # 2 days x 5 bars
        first = lines[1].split(",")
        assert first[0] == "2026-08-18T13:30:00+00:00"


class TestDeath:
    def test_death_liquidates_and_hatches_fresh(self, patched, monkeypatch, tmp_path):
        loop = patched

        # Crash tape: BBB gaps down -95% on day 2. With the shock exit
        # disabled (huge adverse threshold) the loss drives equity below
        # the -2% death line.
        crash_bars = make_bars(crash=True)
        monkeypatch.setattr(
            loop, "load_bars", lambda symbols, start=None, end=None: crash_bars
        )

        def always_approach(balance, held, cap_reached, approach_thr, avoid_thr):
            if held:
                return "add", "approach"
            if cap_reached:
                return "pass", "cap"
            return "buy", "approach"

        monkeypatch.setattr(loop, "_decide", always_approach)
        result = run_backtest(
            _config(
                7, tmp_path / "run", death_threshold=-0.02,
                shock_adverse_pct=1.0e9, position_cap=2,
            )
        )
        events = _events(result.run_dir / "events.jsonl")
        types = [event["type"] for event in events]
        assert types.count("death") >= 1
        assert types.count("hatch") >= 2  # initial hatch + the fresh fly
        death = next(e for e in events if e["type"] == "death")
        assert death["equity"] <= death["hatch_equity"] * (1.0 - 0.02)
        liquidations = [e for e in events if e.get("reason") == "death_liquidation"]
        last = _events(result.run_dir / "events.jsonl")[-1]
        # The run ends with the close-of-day settle: sleep, then the
        # v0.6 daily anchor refresh (T9).
        assert last["type"] == "anchor"
        rows = (result.run_dir / "equity.csv").read_text().splitlines()[1:]
        # On the death bar the book is flat: n_positions returns to 0.
        death_row = next(
            row.split(",") for row in rows if row.startswith(death["ts"])
        )
        assert int(death_row[3]) == 0
        # The dying fly's pending orders are cancelled with the death; the
        # liquidations themselves stay immediate at the death bar's close.
        cancels = [
            e for e in events if e["type"] == "cancel" and e["cancel_reason"] == "death"
        ]
        assert cancels, "the dying fly's pending orders must be cancelled"
        for liq in liquidations:
            assert liq["ts"] == death["ts"]



# ---------------------------------------------------------------------------
# Fill model: pessimistic next-bar execution (no look-ahead)
# ---------------------------------------------------------------------------


def _always_buy(balance, held, cap_reached, approach_thr, avoid_thr):
    """Decision stub: real churn every encounter, cap-aware."""
    if held:
        return "add", "approach"
    if cap_reached:
        return "pass", "cap"
    return "buy", "approach"


class TestFillModel:
    """The pessimistic next-bar execution model (default fill_mode)."""

    def _orders(self, result, skip_death_liquidation=True):
        orders = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "order"
        ]
        if skip_death_liquidation:
            # Death liquidation is an immediate close (semantics unchanged).
            orders = [o for o in orders if o["reason"] != "death_liquidation"]
        return orders


    def test_no_look_ahead_fills_on_the_next_bar(self, patched, monkeypatch,
                                                 tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=10.0)
        )
        frames = make_bars()
        orders = self._orders(result)
        assert orders, "the tight-threshold run must produce orders"
        for o in orders:
            decision = pd.Timestamp(o["decision_ts"])
            assert pd.Timestamp(o["ts"]) > decision, "same-bar fill = look-ahead"
            df = frames[o["ticker"]]
            later = df.index[df.index > decision]
            assert later.size, "execution bar must exist in the run window"
            # The execution bar is the ticker's own NEXT available bar.
            assert pd.Timestamp(o["ts"]) == later[0]

    def test_pessimistic_pricing_buy_high_sell_low(self, patched, monkeypatch,
                                                   tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=10.0)
        )
        frames = make_bars()
        orders = self._orders(result)
        assert orders
        for o in orders:
            bar = frames[o["ticker"]].loc[pd.Timestamp(o["ts"])]
            if o["side"] == "buy":
                assert o["price"] == round(float(bar["high"]), 6)
                assert o["price"] >= float(bar["open"])  # provably conservative
            else:
                assert o["price"] == round(float(bar["low"]), 6)
                assert o["price"] <= float(bar["open"])

    def test_gap_skipping_fills_on_next_available_bar(self, patched, monkeypatch,
                                                      tmp_path):
        loop = patched
        frames = make_bars()
        # AAA misses one minute: the next bar of the UNION timeline (13:31,
        # printed by BBB) is NOT AAA's next available bar.
        gap_ts = pd.Timestamp("2026-08-18 13:31:00", tz="UTC")
        assert gap_ts in frames["AAA"].index
        frames["AAA"] = frames["AAA"].drop(gap_ts)
        monkeypatch.setattr(loop, "load_bars",
                            lambda symbols, start=None, end=None: frames)
        # Every encounter samples AAA, so decisions land on the gap window.
        monkeypatch.setattr(loop, "_plume_set",
                            lambda frames, ts, top_k: [("AAA", 1.0)])
        monkeypatch.setattr(loop, "_decide", _always_buy)
        result = run_backtest(_config(7, tmp_path / "run"))
        aaa_orders = [o for o in self._orders(result) if o["ticker"] == "AAA"]
        assert aaa_orders
        # The 13:30 decision's naive "next timeline bar" is 13:31 (BBB
        # printed there); AAA itself next prints at 13:32 -> fill there.
        first = next(
            o for o in aaa_orders
            if o["decision_ts"] == "2026-08-18T13:30:00+00:00"
        )
        assert first["ts"] == "2026-08-18T13:32:00+00:00"
        for o in aaa_orders:
            decision = pd.Timestamp(o["decision_ts"])
            later = frames["AAA"].index[frames["AAA"].index > decision]
            assert pd.Timestamp(o["ts"]) == later[0]

    def test_pending_order_carries_across_session_close(self, patched,
                                                        monkeypatch, tmp_path):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _always_buy)
        result = run_backtest(_config(7, tmp_path / "run"))
        frames = make_bars()
        crossing = [
            o for o in self._orders(result)
            if o["decision_ts"] == "2026-08-18T13:34:00+00:00"
        ]
        assert crossing, "a decision on day 1's last bar must exist"
        day2_open = pd.Timestamp("2026-08-19 13:30:00", tz="UTC")
        for o in crossing:
            # Unfilled at the session close -> carried into the next session.
            assert pd.Timestamp(o["ts"]) == day2_open
            assert o["price"] == round(
                float(frames[o["ticker"]].at[day2_open, "high"]), 6
            )

    def test_pending_order_cancelled_at_run_end(self, patched, monkeypatch,
                                                tmp_path):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _always_buy)
        result = run_backtest(_config(7, tmp_path / "run"))
        events = _events(result.run_dir / "events.jsonl")
        cancels = [e for e in events if e["type"] == "cancel"]
        assert cancels
        assert all(e["cancel_reason"] == "run_end" for e in cancels)
        # The last bar's decision can never fill -> cancelled at run end.
        assert any(
            e["decision_ts"] == "2026-08-19T13:34:00+00:00" for e in cancels
        )
        # Cancels precede the close-of-day settle: the run still ends asleep.
        assert events[-1]["type"] == "anchor"

    def test_close_mode_is_the_documented_look_ahead_hatch(self, patched,
                                                          tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", fill_mode="close",
                    approach_thr=0.001, avoid_thr=0.001, noise_sigma_mv=10.0)
        )
        frames = make_bars()
        orders = self._orders(result)
        assert orders
        for o in orders:
            # Legacy semantics: same-bar close fill, decision_ts == ts.
            assert o["ts"] == o["decision_ts"]
            assert o["price"] == round(
                float(frames[o["ticker"]].at[pd.Timestamp(o["ts"]), "close"]), 6
            )

    def test_position_cap_counts_queued_buys(self, patched, monkeypatch,
                                             tmp_path):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _always_buy)
        result = run_backtest(_config(7, tmp_path / "run", position_cap=1))
        rows = (result.run_dir / "equity.csv").read_text().splitlines()[1:]
        # Queued buys count toward the cap, so fills can never exceed it.
        assert all(int(row.split(",")[3]) <= 1 for row in rows)

    def test_fees_default_zero_untouched(self, patched, monkeypatch, tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=10.0)
        )
        orders = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "order"
        ]
        assert orders
        cash = 100_000.0
        for o in orders:
            if o["side"] == "buy":
                cash -= o["shares"] * o["price"]  # no fee margin
            else:
                cash += o["shares"] * o["price"]
            assert o["cash_after"] == pytest.approx(round(cash, 6))

# ---------------------------------------------------------------------------
# Chassis config (whole-fly phase 1): validation + sim construction
# ---------------------------------------------------------------------------


class TestChassisConfig:
    """``BacktestConfig`` chassis/STD surface (defaults: D22 v0.6 verdict)."""

    def test_default_is_whole_with_calibrated_std(self):
        # D22 shootout verdict (2026-09-16): the whole fly is the live brain;
        # whole implies the calibrated T12b STD defaults.
        cfg = BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18")
        assert cfg.chassis == "whole"
        assert cfg.std_beta == 0.1
        assert cfg.std_tau_rec_ms == 500.0

    def test_fill_mode_default_and_validation(self):
        assert BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18"
        ).fill_mode == "pessimistic_next_bar"
        with pytest.raises(ValueError, match="fill_mode"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           fill_mode="market")

    def test_whole_chassis_gets_calibrated_std_defaults(self):
        cfg = BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                             chassis="whole")
        # The calibrated T12b point (reports/t12b-apl-std.md) is mandatory.
        assert cfg.std_beta == 0.1
        assert cfg.std_tau_rec_ms == 500.0

    def test_whole_chassis_honors_explicit_std(self):
        cfg = BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                             chassis="whole", std_beta=0.3, std_tau_rec_ms=200.0)
        assert cfg.std_beta == 0.3
        assert cfg.std_tau_rec_ms == 200.0

    def test_unknown_chassis_rejected(self):
        with pytest.raises(ValueError, match="chassis must be"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           chassis="larval")

    def test_partial_std_on_stripped_rejected(self):
        with pytest.raises(ValueError, match="together"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           chassis="stripped", std_beta=0.1)

    def test_std_ranges_validated(self):
        with pytest.raises(ValueError, match="std_beta"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           chassis="whole", std_beta=1.5)
        with pytest.raises(ValueError, match="std_tau_rec_ms"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           chassis="whole", std_tau_rec_ms=-1.0)


class TestChassisSimConstruction:
    """The loop builds the engine the config asks for (real code path)."""

    def test_default_run_builds_calibrated_std_sim(self, patched, monkeypatch, tmp_path):
        import fruitfly.loop as loop

        seen: dict = {}
        real = loop.LIFSim

        def spy(chassis, **kwargs):
            seen.update(kwargs)
            return real(chassis, **kwargs)

        monkeypatch.setattr(loop, "LIFSim", spy)
        run_backtest(_config(7, tmp_path / "run"))
        # Whole default: the calibrated STD beta/tau ship switched on.
        assert seen["std_beta"] == 0.1
        assert seen["std_tau_rec_ms"] == 500.0

    def test_whole_config_routes_whole_seam_and_calibrated_std(
        self, patched, monkeypatch, tmp_path
    ):
        import fruitfly.loop as loop

        chassis = make_chassis()
        seen: dict = {}
        real = loop.LIFSim

        def spy(sim_chassis, **kwargs):
            seen["chassis"] = sim_chassis
            seen.update(kwargs)
            return real(sim_chassis, **kwargs)

        monkeypatch.setattr(loop, "_load_whole_chassis", lambda: chassis)
        monkeypatch.setattr(loop, "LIFSim", spy)
        result = run_backtest(_config(7, tmp_path / "run", chassis="whole"))
        assert seen["chassis"] is chassis
        # Callers cannot forget the calibrated T12b STD parameters.
        assert seen["std_beta"] == 0.1
        assert seen["std_tau_rec_ms"] == 500.0
        assert result.n_bars > 0

    def test_whole_run_is_deterministic(self, patched, monkeypatch, tmp_path):
        import fruitfly.loop as loop

        monkeypatch.setattr(loop, "_load_whole_chassis", make_chassis)
        a = run_backtest(_config(7, tmp_path / "a", chassis="whole"))
        b = run_backtest(_config(7, tmp_path / "b", chassis="whole"))
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(a.run_dir / name) == _sha(b.run_dir / name)



# ---------------------------------------------------------------------------
# DESIGN v0.6 (2026-09-15): plume ranker, per-exit dopamine, daily anchor
# refresh, grudge_top_k, structural looming drive
# ---------------------------------------------------------------------------


def _counter_stub(do_close: bool):
    """Decision stub: buy the first AAA encounter (day 1, 13:32), close it
    on the next AAA encounter (13:34) when ``do_close``."""
    state = {"n": 0}

    def stub(balance, held, cap_reached, approach_thr, avoid_thr):
        n = state["n"]
        state["n"] += 1
        if n == 1:
            return "buy", "approach"
        if do_close and held and n == 3:
            return "sell", "valence_flip"
        return "pass", "neutral"

    return stub


class TestPlumeRankerV06:
    """D3 amendment: intensity = |mom20| x 20 + volume ratio."""

    def _frames(self):
        ts = pd.Timestamp("2026-08-18 13:30:00", tz="UTC")

        def frame(closes):
            idx = pd.DatetimeIndex(
                [ts + pd.Timedelta(minutes=i) for i in range(len(closes))]
            )
            return pd.DataFrame(
                {"open": closes, "high": [c + 0.1 for c in closes],
                 "low": [c - 0.1 for c in closes], "close": closes,
                 "volume": [1_000] * len(closes)},
                index=idx,
            )

        return {
            # persistent 20-bar mover, +1%/bar
            "MOVE": frame([100.0 * (1.01 ** k) for k in range(21)]),
            # one 4% jump after a flat stretch
            "SPIKE": frame([100.0] * 20 + [104.0]),
            # persistent DECLINER: direction-agnostic surface
            "DECL": frame([100.0 * (0.99 ** k) for k in range(21)]),
            # dead: ret1 == 0 AND mom20 == 0
            "DEAD": frame([100.0] * 21),
            # flat last bar on a mover: NOT odorless (mom20 carries it)
            "FLATLAST": frame([100.0 * (1.01 ** k) for k in range(20)]
                              + [100.0 * 1.01 ** 19]),
        }

    def test_persistent_mover_beats_one_bar_spike_and_is_direction_agnostic(self):
        last_ts = self._frames()["MOVE"].index[-1]
        ranked = _plume_set(self._frames(), last_ts, 5)
        tickers = [t for t, _ in ranked]
        # Persistent movers of either sign surface above the 1-bar spike.
        assert tickers.index("MOVE") < tickers.index("SPIKE")
        assert tickers.index("DECL") < tickers.index("SPIKE")
        # A flat last bar on an otherwise-moving ticker still smells; a
        # dead ticker does not.
        assert "FLATLAST" in tickers
        assert "DEAD" not in tickers
        # The intensity itself is the amended formula on the winner.
        move = dict(ranked)["MOVE"]
        assert move == pytest.approx(abs(1.01 ** 20 - 1.0) * 20.0 + 1.0)


class TestPerExitDopamine:
    def test_closed_profitable_trade_emits_credit_and_moves_weights(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=True))
        with_close = run_backtest(
            _config(7, tmp_path / "close", fill_mode="close")
        )
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=False))
        without_close = run_backtest(
            _config(7, tmp_path / "open", fill_mode="close")
        )

        events = _events(with_close.run_dir / "events.jsonl")
        credits = [e for e in events if e["type"] == "trade_credit"]
        assert len(credits) == 1
        credit = credits[0]
        assert credit["ticker"] == "AAA"
        assert credit["realized_pnl"] > 0.0  # a profitable close
        sells = [e for e in events
                 if e["type"] == "order" and e["side"] == "sell"]
        assert credit["ts"] == sells[0]["ts"]
        buys = [e for e in events
                if e["type"] == "order" and e["side"] == "buy"]
        notional = buys[0]["shares"] * buys[0]["price"]
        assert credit["reward"] == pytest.approx(
            min(1.0, credit["realized_pnl"] / notional), abs=1e-8
        )
        assert credit["punishment"] == 0.0

        # The run without the close emits no credit and ends with
        # different fly weights than the credited run.
        assert not [
            e for e in _events(without_close.run_dir / "events.jsonl")
            if e["type"] == "trade_credit"
        ]
        assert not np.array_equal(with_close.final_weights,
                                  without_close.final_weights)

    def test_credit_snapshot_is_real_intraday_eligibility(
        self, patched, monkeypatch, tmp_path
    ):
        """The per-exit credit moves weights through the ENTRY SNAPSHOT, not
        sequence noise: a reference run with the SAME trades but
        ``observe_trade`` neutralized (everything else identical) ends with
        different weights than the credited run. This is only possible if
        the snapshot taken at the buy was nonzero — the loop's intraday
        trace, not the always-zero live ``plasticity.eligibility``."""
        import fruitfly.neuromod as neuromod

        loop = patched
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=True))
        with_close = run_backtest(
            _config(7, tmp_path / "close", fill_mode="close")
        )
        credits = [
            e for e in _events(with_close.run_dir / "events.jsonl")
            if e["type"] == "trade_credit"
        ]
        assert len(credits) == 1
        # Same trades, same encounters, same daily ritual — the ONLY
        # difference is that the credit's three-factor update is a no-op.
        # A FRESH stub instance: the stub carries per-run state, so sharing
        # one across runs would change decisions for the wrong reason.
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=True))
        monkeypatch.setattr(
            neuromod.Plasticity,
            "observe_trade",
            lambda self, state, eligibility_snapshot: None,
        )
        noop_credit = run_backtest(
            _config(7, tmp_path / "noop", fill_mode="close")
        )
        assert not np.array_equal(
            with_close.final_weights, noop_credit.final_weights
        )


class TestDailyAnchorRefresh:
    def test_anchor_emitted_each_day_and_centering_uses_refreshed_anchor(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=True))
        result = run_backtest(_config(7, tmp_path / "run", fill_mode="close"))
        events = _events(result.run_dir / "events.jsonl")

        anchors = [e for e in events if e["type"] == "anchor"]
        assert [a["ts"] for a in anchors] == [
            "2026-08-18T20:00:00+00:00",
            "2026-08-19T20:00:00+00:00",
        ]
        # The refresh reflects the day's learning (the profitable close
        # moved weights, so the day-1 settle re-measures a new anchor).
        assert anchors[1]["balance"] != anchors[0]["balance"]

        # The day-1 settle refresh is what day-2 encounters center on: the
        # re-centering drifts back with the learned bias instead of going
        # stale on the hatch measurement.
        refreshed = anchors[0]["balance"]
        day2 = [e for e in events
                if e["type"] == "encounter" and e["ts"] > "2026-08-19"]
        assert day2
        for e in day2:
            assert e["balance"] == pytest.approx(
                e["raw_balance"] - refreshed, abs=2e-9
            )
        assert max(abs(e["balance"]) for e in day2) < 0.05


class TestStructuralLoomingDrive:
    CRASH_TS = "2026-08-19T13:30:00+00:00"

    def test_crash_bar_shifts_balance_used_via_lambda_struct(
        self, patched_lc, monkeypatch, tmp_path
    ):
        loop = patched_lc
        real_decide = loop._decide
        seen = []

        def spy(balance, held, cap_reached, approach_thr, avoid_thr):
            seen.append(balance)
            return real_decide(
                balance, held, cap_reached, approach_thr, avoid_thr
            )

        monkeypatch.setattr(loop, "_decide", spy)
        result = run_backtest(_config(7, tmp_path / "run"))
        enc = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "encounter"
        ]
        assert len(seen) == len(enc)
        for balance_arg, e in zip(seen, enc, strict=True):
            # The decision consumes balance_used, not the centered balance.
            assert balance_arg == pytest.approx(e["balance_used"], abs=5e-10)

        crash = next(e for e in enc if e["ts"] == self.CRASH_TS)
        assert crash["structural_score"] > 0.0
        # The crash bar drives the strongest LC looming response.
        assert crash["structural_score"] == max(
            e["structural_score"] for e in enc
        )
        for e in enc:
            assert e["balance_used"] == pytest.approx(
                e["balance"] + 0.05 * e["structural_score"], abs=2e-9
            )

        # lambda_struct = 0 removes the structural term entirely (same
        # seed -> identical encounters, identical structural scores).
        zero = run_backtest(_config(7, tmp_path / "zero", lambda_struct=0.0))
        zc = [
            e for e in _events(zero.run_dir / "events.jsonl")
            if e["type"] == "encounter"
        ]
        zcrash = next(e for e in zc if e["ts"] == self.CRASH_TS)
        assert zcrash["structural_score"] == crash["structural_score"]
        assert zcrash["balance_used"] == zcrash["balance"]

    def test_chassis_without_lc_path_keeps_structural_score_zero(
        self, patched, tmp_path
    ):
        result = run_backtest(_config(7, tmp_path / "run"))
        enc = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "encounter"
        ]
        assert enc
        assert all(e["structural_score"] == 0.0 for e in enc)
        assert all(e["balance_used"] == e["balance"] for e in enc)


class TestGrudgeTopK:
    def test_default_and_config_flow_to_plasticity(
        self, patched, monkeypatch, tmp_path
    ):
        defaults = BacktestConfig(seed=1, start="2026-08-18", end="2026-08-19")
        assert defaults.grudge_top_k == 16
        assert defaults.lambda_struct == 0.05

        loop = patched
        real = loop.Plasticity
        captured = []

        def spy(chassis, **kwargs):
            captured.append(kwargs.get("top_k"))
            return real(chassis, **kwargs)

        monkeypatch.setattr(loop, "Plasticity", spy)
        run_backtest(_config(7, tmp_path / "run"))
        assert captured and set(captured) == {16}
        captured.clear()
        run_backtest(_config(7, tmp_path / "k3", grudge_top_k=3))
        assert captured and set(captured) == {3}


class TestV06Determinism:
    def test_same_seed_double_run_byte_identical_with_v06_events(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=True))
        a = run_backtest(_config(7, tmp_path / "a", fill_mode="close"))
        monkeypatch.setattr(loop, "_decide", _counter_stub(do_close=True))
        b = run_backtest(_config(7, tmp_path / "b", fill_mode="close"))
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(a.run_dir / name) == _sha(b.run_dir / name)
        types = {e["type"] for e in _events(a.run_dir / "events.jsonl")}
        assert {"trade_credit", "anchor"} <= types

    def test_same_seed_double_run_byte_identical_on_lc_chassis(
        self, patched_lc, tmp_path
    ):
        a = run_backtest(_config(7, tmp_path / "a"))
        b = run_backtest(_config(7, tmp_path / "b"))
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(a.run_dir / name) == _sha(b.run_dir / name)
        types = {e["type"] for e in _events(a.run_dir / "events.jsonl")}
        assert {"anchor"} <= types

@pytest.mark.skipif(
    os.environ.get("FRUITFLY_WHOLE_SMOKE") != "1",
    reason="real whole-fly smoke (~8 min + 781 MB cache): set FRUITFLY_WHOLE_SMOKE=1",
)
def test_whole_fly_real_smoke(tmp_path):
    """One real single-day whole-fly backtest through the config path."""
    result = run_backtest(
        BacktestConfig(seed=7, start="2026-08-18", end="2026-08-18",
                       chassis="whole", out_dir=tmp_path / "smoke")
    )
    assert result.n_bars > 0


# ---------------------------------------------------------------------------
# TRAINING2 Phase 0: entry-credit ledger (Options A/B), daily_observe,
# id_scale, basket file — knobs at defaults must remain a byte-identical
# no-op (the pinned incumbent receipts below).
# ---------------------------------------------------------------------------

# Pre-change incumbent receipts for the synthetic fixture above (seed 7,
# 2026-08-18..19): sha256 of the byte-exact artifacts before the
# entry-credit machinery landed. Phase 0 hard gate: every new knob at its
# default reproduces these bytes.
INCUMBENT_EVENTS_SHA = (
    "62853788c1c12429ae379102e290fb9325fa2ba347731cdee9c7f0b5df50d061"
)
INCUMBENT_EQUITY_SHA = (
    "aa67356aebf74676ce3a8475e4cb880981b3e6179b3a7d5ba630bfeb0d24afea"
)

# Entry-credit events round prices to 6 decimals and rates to 9; the tanh
# mapping amplifies that payload rounding by ~1/r_scale, so event-level
# arithmetic assertions use a 1e-6 tolerance.
PAYLOAD_TOL = 1e-6

class TestGateGains:
    """reward_gain/punishment_gain reach the ``Plasticity`` constructor
    (TRAINING2 §4: the loop previously dropped them)."""

    def test_gate_gains_flow_to_plasticity(self, patched, monkeypatch, tmp_path):
        import fruitfly.neuromod as neuromod

        captured = []
        original = neuromod.Plasticity.__init__

        def spy(self, chassis, **kwargs):
            captured.append(kwargs)
            return original(self, chassis, **kwargs)

        monkeypatch.setattr(neuromod.Plasticity, "__init__", spy)
        run_backtest(_config(7, tmp_path / "run", reward_gain=1.5))
        assert captured
        assert all(
            k.get("reward_gain") == 1.5 and k.get("punishment_gain") == 1.0
            for k in captured
        )


def _pass_only(balance, held, cap_reached, approach_thr, avoid_thr):
    """Decision stub: never acts — every signal encounter stays open."""
    return "pass", "neutral"


def _buy_then_close_next_day_stub():
    """Decision stub: buy the 2nd encounter (day 1, 13:31), close the first
    held encounter on day 2 — a settle credit followed by a supersedes
    delta. Self-contained state; build one per run."""
    state = {"n": 0}

    def stub(balance, held, cap_reached, approach_thr, avoid_thr):
        n = state["n"]
        state["n"] += 1
        if n == 1:
            return "buy", "approach"
        if held and n >= 6:
            return "sell", "valence_flip"
        return "pass", "neutral"

    return stub


def _single_ticker(monkeypatch, loop):
    """Restrict the patched fixture to AAA only: every plume-bearing bar
    encounters AAA, so stub state machines are deterministic."""
    aaa = make_bars()["AAA"]
    monkeypatch.setattr(
        loop, "load_bars", lambda symbols, start=None, end=None: {"AAA": aaa}
    )
    monkeypatch.setattr(loop, "BASKET", ["AAA"])


class TestEntryCreditNoOp:
    """Phase 0 hard gate: defaults reproduce the incumbent byte-for-byte."""

    def test_defaults_reproduce_incumbent_receipt(self, patched, tmp_path):
        result = run_backtest(_config(7, tmp_path / "run"))
        assert _sha(result.run_dir / "events.jsonl") == INCUMBENT_EVENTS_SHA
        assert _sha(result.run_dir / "equity.csv") == INCUMBENT_EQUITY_SHA
        types = {e["type"] for e in _events(result.run_dir / "events.jsonl")}
        assert "entry_credit" not in types

    def test_explicit_noop_knobs_match_bare_defaults(self, patched, tmp_path):
        bare = run_backtest(_config(7, tmp_path / "bare"))
        explicit = run_backtest(
            _config(
                7, tmp_path / "explicit",
                entry_credit="off", trade_credit_mode="realized",
                mix_weight=0.5, daily_observe="on", id_scale=1.0,
                horizon_bars=30, r_scale=0.005, a_scale=0.003,
                baseline_alpha=0.05, miss_weight=0.25,
                avoid_correct_weight=0.5,
                reward_gain=1.0, punishment_gain=1.0,
            )
        )
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(bare.run_dir / name) == _sha(explicit.run_dir / name)

    def test_new_paths_are_seeded_deterministic(
        self, patched, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(patched, "_decide", _pass_only)
        a = run_backtest(_config(7, tmp_path / "a", entry_credit="forecast"))
        b = run_backtest(_config(7, tmp_path / "b", entry_credit="forecast"))
        assert _sha(a.run_dir / "events.jsonl") == _sha(b.run_dir / "events.jsonl")
        assert any(
            e["type"] == "entry_credit"
            for e in _events(a.run_dir / "events.jsonl")
        )


class TestAntiLookahead:
    """Hard invariant (TRAINING2 §2/§9): every entry credit is computed from
    bars strictly after the decision bar; a day's final-bar encounter gets
    r_fwd = 0 and zero credit."""

    def test_credit_timing_and_final_bar_zero_credit(
        self, patched, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(patched, "_decide", _pass_only)
        result = run_backtest(
            _config(7, tmp_path / "run", entry_credit="forecast")
        )
        credits = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "entry_credit"
        ]
        assert credits, "the fixture must produce signal-bearing encounters"
        for e in credits:
            assert pd.Timestamp(e["credit_ts"]) >= (
                pd.Timestamp(e["decision_ts"]) + pd.Timedelta(minutes=1)
            )
        # Day 1's last bar (13:34): no strictly-later bar in the day, so the
        # settle credit sees base == last_close and credits nothing.
        final_bar = [
            e for e in credits
            if e["decision_ts"] == "2026-08-18T13:34:00+00:00"
        ]
        assert final_bar
        for e in final_bar:
            assert e["r_fwd"] == 0.0
            assert e["reward"] == 0.0 and e["punishment"] == 0.0


class TestAvoidSideCredit:
    """Avoid-side credit (TRAINING2 §2A): a correct avoid (r_fwd < 0) earns
    positive reward at avoid_correct_weight; a missed run-up (r_fwd > 0) is
    punished at miss_weight."""

    def test_pass_encounters_credit_avoid_correctness(
        self, patched, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(patched, "_decide", _pass_only)
        result = run_backtest(
            _config(7, tmp_path / "run", entry_credit="forecast")
        )
        credits = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "entry_credit"
        ]
        assert credits
        negatives = [e for e in credits if e["r_fwd"] < 0.0]
        positives = [e for e in credits if e["r_fwd"] > 0.0]
        assert negatives and positives  # the fixture tape has both regimes
        for e in negatives:
            assert e["punishment"] == 0.0
            assert e["reward"] == pytest.approx(
                0.5 * max(0.0, math.tanh(abs(e["r_fwd"]) / 0.005)),
                abs=PAYLOAD_TOL,
            )
        for e in positives:
            assert e["reward"] == 0.0
            assert e["punishment"] == pytest.approx(
                0.25 * max(0.0, math.tanh(e["r_fwd"] / 0.005)),
                abs=PAYLOAD_TOL,
            )


class TestSingleCreditLastCreditWins:
    """A buy credited at settle, closed the next day: exactly one supersedes
    delta, and settle + delta net to the final outcome (TRAINING2 §2A)."""

    def test_settle_credit_superseded_by_close_delta(
        self, patched, monkeypatch, tmp_path
    ):
        _single_ticker(monkeypatch, patched)
        monkeypatch.setattr(patched, "_decide", _buy_then_close_next_day_stub())
        result = run_backtest(
            _config(7, tmp_path / "run", fill_mode="close",
                    entry_credit="forecast", trade_credit_mode="forecast")
        )
        credits = [
            e for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "entry_credit"
        ]
        settle_buys = [
            e for e in credits
            if e["mode"] == "settle" and e["action"] == "buy"
        ]
        deltas = [e for e in credits if "supersedes" in e]
        assert len(settle_buys) == 1 and len(deltas) == 1
        settle, delta = settle_buys[0], deltas[0]
        assert delta["mode"] == "forecast"
        # The delta carries the signed component differences; the net gate
        # input over the record equals the close outcome exactly.
        net_reward = settle["reward"] + delta["reward"]
        net_punishment = settle["punishment"] + delta["punishment"]
        expected = max(0.0, math.tanh(delta["r_fwd"] / 0.005))
        assert net_reward == pytest.approx(expected, abs=PAYLOAD_TOL)
        assert net_punishment == pytest.approx(
            max(0.0, -math.tanh(delta["r_fwd"] / 0.005)), abs=PAYLOAD_TOL
        )
        # The delta's decision bar is the buy's encounter bar; its credit
        # bar is the close fill.
        assert delta["decision_ts"] == settle["decision_ts"]
        assert delta["credit_ts"] > settle["credit_ts"]


class TestOptionBBaseline:
    """Option B (TRAINING2 §2B): the gate fires on the advantage vs a bucket
    EMA — the credit consumes the PRE-update baseline, the EMA updates after
    every credit, and ``initial_baselines`` warm-starts a restored fly."""

    A_SCALE = 0.02  # unsaturated tanh: the baseline shift is visible
    ALPHA = 0.05

    @staticmethod
    def _buckets(bars):
        return {
            (ticker, ts.isoformat()): loop_module._entry_bucket(
                loop_module._features_at(df, i)[0]
            )
            for ticker, df in bars.items()
            for i, ts in enumerate(df.index)
        }

    def test_pre_update_ema_sequence_and_round_trip(
        self, patched, monkeypatch, tmp_path
    ):
        _single_ticker(monkeypatch, patched)
        buckets = self._buckets(make_bars())
        monkeypatch.setattr(patched, "_decide", _buy_then_close_next_day_stub())

        def run(name, initial_baselines=None):
            return run_backtest(
                _config(7, tmp_path / name, fill_mode="close",
                        entry_credit="advantage", a_scale=self.A_SCALE,
                        initial_baselines=initial_baselines)
            )

        cold = run("cold")
        credits = [
            e for e in _events(cold.run_dir / "events.jsonl")
            if e["type"] == "entry_credit"
        ]
        assert any(e["action"] == "buy" for e in credits)
        # Sequential reconstruction: every settle-buy credit consumed the
        # PRE-update baseline; every credit (buy or avoid-side, settle or
        # close) advances the EMA. Close credits under the realized mode
        # carry a=None (their gate is the realized pair) but their r_fwd
        # still updates the baseline.
        b: dict[tuple[int, int, int], float] = {}
        for e in credits:
            bucket = buckets[("AAA", e["decision_ts"])]
            if e["mode"] == "settle" and e["action"] == "buy":
                assert e["a"] == pytest.approx(
                    math.tanh((e["r_fwd"] - b.get(bucket, 0.0)) / self.A_SCALE),
                    abs=PAYLOAD_TOL,
                )
            elif e["mode"] == "settle":
                assert e["a"] == pytest.approx(
                    math.tanh(e["r_fwd"] / 0.005), abs=PAYLOAD_TOL
                )
            b[bucket] = (
                (1.0 - self.ALPHA) * b.get(bucket, 0.0) + self.ALPHA * e["r_fwd"]
            )
        assert cold.baselines is not None and cold.baselines
        assert all(isinstance(k, tuple) and len(k) == 3 for k in cold.baselines)
        for bucket, value in cold.baselines.items():
            assert value == pytest.approx(b[bucket], abs=1e-9)

    def test_string_keys_are_coerced_like_the_meta_round_trip(self, tmp_path):
        config = BacktestConfig(
            seed=1, start="2026-08-18", end="2026-08-19",
            initial_baselines={"1|0|1": 0.25, (-1, 2, 0): -0.5},
        )
        assert config.initial_baselines == {
            (1, 0, 1): 0.25, (-1, 2, 0): -0.5,
        }


class TestBasketGuardInLoop:
    """The never-seen guard arms inside ``run_backtest`` (TRAINING2 §3.4)."""

    def test_basket_override_flows_to_load_bars(self, patched, monkeypatch, tmp_path):
        seen = {}

        def spy(symbols, start=None, end=None):
            seen["symbols"] = list(symbols)
            return {"AAA": make_bars()["AAA"]}

        monkeypatch.setattr(patched, "load_bars", spy)
        run_backtest(_config(7, tmp_path / "run", basket=["AAA"]))
        assert seen["symbols"] == ["AAA"]

    def test_neverseen_intersection_raises_before_loading(
        self, patched, monkeypatch, tmp_path
    ):
        guard = tmp_path / "eval20-neverseen.txt"
        guard.write_text("ZZZZ\n")
        monkeypatch.setattr("fruitfly.data.NEVERSEEN_BASKET_PATH", guard)

        def boom(symbols, start=None, end=None):
            raise AssertionError("load_bars must not run past the guard")

        monkeypatch.setattr(patched, "load_bars", boom)
        with pytest.raises(ValueError, match="ZZZZ"):
            run_backtest(_config(7, tmp_path / "run", basket=["AAA", "ZZZZ"]))


class TestDailyObserveAblation:
    """``daily_observe="off"`` skips ONLY the pooled-day ``observe`` call."""

    def test_observe_skipped_but_ritual_events_intact(
        self, patched, monkeypatch, tmp_path
    ):
        import fruitfly.neuromod as neuromod

        calls = []
        original = neuromod.Plasticity.observe

        def spy(self, state, pre, post):
            calls.append(state)
            return original(self, state, pre, post)

        monkeypatch.setattr(neuromod.Plasticity, "observe", spy)
        on = run_backtest(_config(7, tmp_path / "on"))
        on_calls = len(calls)
        calls.clear()
        off = run_backtest(
            _config(7, tmp_path / "off", entry_credit="forecast",
                    daily_observe="off")
        )
        assert on_calls == 2  # one pooled observe per settling day
        assert calls == []  # ablated: the D6 ritual observe never fires
        off_types = {e["type"] for e in _events(off.run_dir / "events.jsonl")}
        on_types = {e["type"] for e in _events(on.run_dir / "events.jsonl")}
        assert {"sugar_shock", "sleep", "anchor", "entry_credit"} <= off_types
        assert off_types - on_types <= {"entry_credit"}


class TestIdScale:
    """``id_scale`` attenuates the identity term everywhere, including the
    ``_innate_balance`` anchor sniff that bypasses ``encode_smell``."""

    def test_zero_id_scale_silences_the_anchor_sniff(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        captured = {}
        original = loop._innate_balance

        def spy(*args, **kwargs):
            balance = original(*args, **kwargs)
            captured.setdefault("profiles", []).append(kwargs.get("id_scale"))
            return balance

        monkeypatch.setattr(patched, "_innate_balance", spy)
        on = run_backtest(_config(7, tmp_path / "on", id_scale=1.0))
        off = run_backtest(_config(7, tmp_path / "off", id_scale=0.0))
        # The anchor is measured at hatch and refreshed at every sleep.
        assert captured["profiles"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        anchors_on = [
            e["balance"] for e in _events(on.run_dir / "events.jsonl")
            if e["type"] == "anchor"
        ]
        anchors_off = [
            e["balance"] for e in _events(off.run_dir / "events.jsonl")
            if e["type"] == "anchor"
        ]
        assert anchors_on
        # id_scale=0: the sniff is silent (no identity drive), so the raw
        # balance reads neutral — the documented silent-anchor degeneracy.
        assert all(a == pytest.approx(0.0, abs=1e-9) for a in anchors_off)


# ---------------------------------------------------------------------------
# TRAINING2 §5 (amendment A5): mechanical exit grid + T-1 exit-forecast
# gate. All six knobs default to a byte-identical no-op; the new paths
# queue through the same pessimistic next-bar book as the incumbent exits.
# ---------------------------------------------------------------------------


def _tape(closes: list[float], day: int = 18) -> pd.DataFrame:
    """One ticker's 1-min bars from 13:30 local minute offsets, one session:
    close per entry, high/low = close ± 0.3, flat volume (a volume_ratio
    of 1.0 keeps the plume ranker's arithmetic transparent)."""
    stamps = pd.DatetimeIndex(
        [
            pd.Timestamp(f"2026-08-{day} 13:{30 + m:02d}:00", tz="UTC")
            for m in range(len(closes))
        ]
    )
    rows = [
        {
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 5_000,
        }
        for close in closes
    ]
    return pd.DataFrame(rows, index=stamps)


def _buy_first_stub():
    """Decision stub: buy the first encounter, pass ever after."""
    state = {"n": 0}

    def stub(balance, held, cap_reached, approach_thr, avoid_thr):
        state["n"] += 1
        if state["n"] == 1:
            return "buy", "approach"
        return "pass", "neutral"

    return stub


def _buy_two_stub():
    """Decision stub: buy the first two encounters, pass ever after."""
    state = {"n": 0}

    def stub(balance, held, cap_reached, approach_thr, avoid_thr):
        state["n"] += 1
        if state["n"] <= 2:
            return "buy", "approach"
        return "pass", "neutral"

    return stub


def _sell_when_held_stub():
    """Decision stub: buy the first encounter, then SELL every held
    encounter (the valence-flip path, forced)."""
    state = {"n": 0}

    def stub(balance, held, cap_reached, approach_thr, avoid_thr):
        state["n"] += 1
        if state["n"] == 1:
            return "buy", "approach"
        if held:
            return "sell", "valence_flip"
        return "buy", "approach"

    return stub


def _single_bars(monkeypatch, loop, bars: dict[str, pd.DataFrame]) -> None:
    """Point the patched fixture's seams at an explicit bars dict."""
    monkeypatch.setattr(
        loop, "load_bars", lambda symbols, start=None, end=None: bars
    )
    monkeypatch.setattr(loop, "BASKET", list(bars))


def _orders(path: Path) -> list[dict]:
    return [e for e in _events(path) if e["type"] == "order"]


class TestExitGridNoOp:
    """Phase 0 hard gate for the §5 knobs: defaults stay byte-identical."""

    def test_exit_grid_defaults_reproduce_incumbent_receipt(
        self, patched, tmp_path
    ):
        result = run_backtest(_config(7, tmp_path / "run"))
        assert _sha(result.run_dir / "events.jsonl") == INCUMBENT_EVENTS_SHA
        assert _sha(result.run_dir / "equity.csv") == INCUMBENT_EQUITY_SHA

    def test_explicit_noop_exit_knobs_match_bare_defaults(
        self, patched, tmp_path
    ):
        bare = run_backtest(_config(7, tmp_path / "bare"))
        explicit = run_backtest(
            _config(
                7, tmp_path / "explicit",
                trailing_stop_pct=None, atr_stop_mult=None,
                valence_flip_exit=True, hunger_exit=True,
                exit_forecast=False, exit_score_thr=None,
                exit_score_every_bar=False,
            )
        )
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(bare.run_dir / name) == _sha(explicit.run_dir / name)

    def test_exit_grid_validation(self):
        base = dict(seed=1, start="2026-08-18", end="2026-08-18")
        with pytest.raises(ValueError):
            BacktestConfig(**base, trailing_stop_pct=0.0)
        with pytest.raises(ValueError):
            BacktestConfig(**base, trailing_stop_pct=-1.0)
        with pytest.raises(ValueError):
            BacktestConfig(**base, atr_stop_mult=0.0)
        with pytest.raises(ValueError):
            BacktestConfig(**base, exit_forecast=True)  # thr required
        with pytest.raises(ValueError):
            BacktestConfig(**base, exit_score_every_bar=True)


class TestTrailingStop:
    """``trailing_stop_pct`` fires at exactly the x% breach of the
    high-water mark (TRAINING2 §5.2) — hand-computable on a declining
    tape. Bar 0 is plume-silent (no return history), so the entry
    decision lands bar 1 and fills bar 2 at the pessimistic high 98.3;
    the hwm stays 98.3 and the 1% trail (97.317) is first breached at
    the 97.0 close (bar 3)."""

    def test_trailing_stop_fires_at_breach(self, patched, monkeypatch, tmp_path):
        loop = patched
        _single_bars(monkeypatch, loop, {"AAA": _tape([100, 99, 98, 97, 96, 95])})
        monkeypatch.setattr(loop, "_decide", _buy_first_stub())
        result = run_backtest(
            _config(7, tmp_path / "run", trailing_stop_pct=1.0)
        )
        orders = _orders(result.run_dir / "events.jsonl")
        sells = [o for o in orders if o["side"] == "sell"]
        assert len(sells) == 1
        sell = sells[0]
        assert sell["reason"] == "trailing_stop"
        # Breach bar: 97 <= 98.3 * (1 - 0.01) = 97.317; bar 2's 98 does
        # not breach. The queue lands the fill one bar later, at bar 4's
        # pessimistic low 95.7.
        assert sell["decision_ts"] == "2026-08-18T13:33:00+00:00"
        assert sell["price"] == pytest.approx(95.7)
        buys = [o for o in orders if o["side"] == "buy"]
        assert len(buys) == 1
        # The whole position closes at fill (no oversell, no undersell).
        assert sell["shares"] == buys[0]["shares"] == 25

class TestAtrStop:
    """``atr_stop_mult`` fires when price < entry - k·σ20·price with
    σ20 = build_features' 20-bar close-to-close return std-dev
    (TRAINING2 §5.2)."""

    def test_atr_stop_fires_on_synthetic_data(self, patched, monkeypatch, tmp_path):
        from fruitfly.senses import build_features

        loop = patched
        # 21 bars alternating 99.75/100.25 (σ20 ≈ 0.005), then the drop:
        # 98.75, 98.25. Bar 0 is plume-silent, so the entry decision lands
        # bar 1 and fills bar 2 at the pessimistic high 100.05; k=2 puts
        # the stop at ≈ 99.03 on the last flat bar (no fire) and the first
        # drop bar clears it. shock_adverse_pct=5 keeps the incumbent
        # shock stop out of the way (max dip ≈ -1.8%).
        closes = [
            99.75 if j % 2 == 0 else 100.25 for j in range(21)
        ] + [98.75, 98.25]
        df = _tape(closes)
        _single_bars(monkeypatch, loop, {"AAA": df})
        monkeypatch.setattr(loop, "_decide", _buy_first_stub())
        result = run_backtest(
            _config(
                7, tmp_path / "run", atr_stop_mult=2.0, shock_adverse_pct=5.0
            )
        )
        orders = _orders(result.run_dir / "events.jsonl")
        sells = [o for o in orders if o["side"] == "sell"]
        assert len(sells) == 1
        sell = sells[0]
        assert sell["reason"] == "atr_stop"
        assert sell["decision_ts"] == "2026-08-18T13:51:00+00:00"  # bar 21
        # Verify the trigger inequality at the fire bar and its absence on
        # the previous bar, using the loop's own feature builder.
        k = 2.0
        entry = [o for o in orders if o["side"] == "buy"][0]["price"]
        fire_i = 21
        for i in (20, fire_i):
            sigma = build_features(df.iloc[: i + 1])["volatility"]
            price = float(df["close"].iloc[i])
            if i < fire_i:
                assert price >= entry - k * sigma * price
            else:
                assert price < entry - k * sigma * price


class TestValenceFlipExit:
    """``valence_flip_exit=False`` suppresses ONLY the sell-on-avoid
    decision — nothing else in the log changes."""

    def test_suppresses_sell_on_avoid(self, patched, monkeypatch, tmp_path):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _sell_when_held_stub())
        on = run_backtest(_config(7, tmp_path / "on", valence_flip_exit=True))
        # A fresh stub per run: the stub counts encounters, so sharing
        # one instance across runs would silence the second run's buys.
        monkeypatch.setattr(loop, "_decide", _sell_when_held_stub())
        off = run_backtest(_config(7, tmp_path / "off", valence_flip_exit=False))
        on_orders = _orders(on.run_dir / "events.jsonl")
        off_orders = _orders(off.run_dir / "events.jsonl")
        assert any(
            o["side"] == "sell" and o["reason"] == "valence_flip"
            for o in on_orders
        )
        assert not [o for o in off_orders if o["side"] == "sell"]
        # Nothing else changes: the same buys fire (the forced-buy stub),
        # and the off-run's non-sell orders match the on-run's prefix.
        assert off_orders == on_orders[: len(off_orders)]


class TestHungerExit:
    """``hunger_exit=False`` keeps the largest winner open when the
    hunger drawdown trips (TRAINING2 §5.2)."""

    def test_hunger_off_keeps_largest_winner_open(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        # AAA rises (the winner), BBB declines (the drawdown source) —
        # the incumbent hunger exit closes AAA, the largest winner.
        # A fresh stub per run: the stub counts encounters, so sharing
        # one instance across runs would silence the second run's buys.
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        on = run_backtest(
            _config(7, tmp_path / "on", hunger_drawdown=1e-5, shock_adverse_pct=100.0)
        )
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        off = run_backtest(
            _config(
                7, tmp_path / "off", hunger_exit=False,
                hunger_drawdown=1e-5, shock_adverse_pct=100.0,
            )
        )
        on_orders = _orders(on.run_dir / "events.jsonl")
        off_orders = _orders(off.run_dir / "events.jsonl")
        hunger = [o for o in on_orders if o["reason"] == "hunger"]
        assert hunger, "the incumbent hunger exit must fire on this fixture"
        assert not [o for o in off_orders if o["reason"] == "hunger"]
        assert not [o for o in off_orders if o["side"] == "sell"]
        rows = (off.run_dir / "equity.csv").read_text().splitlines()
        assert rows[-1].split(",")[-1] == "2"  # both positions still open


class TestExitForecast:
    """T-1 exit-forecast gate (TRAINING2 §5.1): score = -balance_used
    from the encounter pipeline without taste; default cadence scores the
    encountered-held ticker only, ``exit_score_every_bar`` scores the
    rest; both respect the pending-order queue and position accounting."""

    def test_exit_forecast_fires_on_encountered_held_bar(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        _single_bars(
            monkeypatch, loop, {"AAA": _tape([100, 99, 98, 97, 96, 95])}
        )
        monkeypatch.setattr(loop, "_decide", _buy_first_stub())
        # exit_score_thr = -10: |balance_used| <= 2, so the score always
        # clears it — a hand-computable always-fire threshold.
        result = run_backtest(
            _config(
                7, tmp_path / "run", exit_forecast=True, exit_score_thr=-10.0
            )
        )
        orders = _orders(result.run_dir / "events.jsonl")
        exits = [o for o in orders if o["reason"] == "exit_forecast"]
        assert len(exits) == 1
        exit_order = exits[0]
        # Bar 2: the buy (decided bar 1 — bar 0 is plume-silent) filled
        # in step 1b, the encounter hits the held ticker, and the sell is
        # decided on that very bar.
        assert exit_order["decision_ts"] == "2026-08-18T13:32:00+00:00"
        encounters = {
            (e["ts"], e["ticker"])
            for e in _events(result.run_dir / "events.jsonl")
            if e["type"] == "encounter"
        }
        assert (exit_order["decision_ts"], exit_order["ticker"]) in encounters
        # Negative control: an unreachable threshold never sells. A fresh
        # stub per run — the stub counts encounters, so sharing one
        # instance across runs would silence the second run's buys.
        monkeypatch.setattr(loop, "_decide", _buy_first_stub())
        quiet = run_backtest(
            _config(7, tmp_path / "quiet", exit_forecast=True, exit_score_thr=10.0)
        )
        assert not [
            o for o in _orders(quiet.run_dir / "events.jsonl")
            if o["reason"] == "exit_forecast"
        ]

    def test_every_bar_scores_unencountered_held_tickers(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        every = run_backtest(
            _config(
                7, tmp_path / "every", exit_forecast=True,
                exit_score_thr=-10.0, exit_score_every_bar=True,
            )
        )
        events = _events(every.run_dir / "events.jsonl")
        encounters = {
            (e["ts"], e["ticker"]) for e in events if e["type"] == "encounter"
        }
        exits = [
            (o["decision_ts"], o["ticker"])
            for o in _orders(every.run_dir / "events.jsonl")
            if o["reason"] == "exit_forecast"
        ]
        assert exits
        # The every-bar cadence scores held tickers that are NOT the
        # encountered one — at least one exit off the encounter path.
        assert any(pair not in encounters for pair in exits)
        # Default cadence contrast: every exit sits on an encounter pair.
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        default = run_backtest(
            _config(
                7, tmp_path / "default", exit_forecast=True,
                exit_score_thr=-10.0,
            )
        )
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        default_events = _events(default.run_dir / "events.jsonl")
        default_encounters = {
            (e["ts"], e["ticker"])
            for e in default_events if e["type"] == "encounter"
        }
        default_exits = [
            (o["decision_ts"], o["ticker"])
            for o in _orders(default.run_dir / "events.jsonl")
            if o["reason"] == "exit_forecast"
        ]
        assert default_exits
        assert all(pair in default_encounters for pair in default_exits)

    def test_exit_forecast_respects_position_accounting(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        result = run_backtest(
            _config(
                7, tmp_path / "run", exit_forecast=True,
                exit_score_thr=-10.0, exit_score_every_bar=True,
            )
        )
        events = _events(result.run_dir / "events.jsonl")
        orders = [e for e in events if e["type"] == "order"]
        # No sell without a position (the queue is respected: one sell in
        # flight per ticker, and each sell closes the whole position).
        assert not [
            e for e in events
            if e["type"] == "cancel" and e["cancel_reason"] == "no_position"
        ]
        shares: dict[str, int] = {}
        for o in orders:
            if o["side"] == "buy":
                shares[o["ticker"]] = shares.get(o["ticker"], 0) + o["shares"]
            else:
                assert o["shares"] == shares.get(o["ticker"], 0)
                shares.pop(o["ticker"], None)
    def test_exit_forecast_is_seeded_deterministic(
        self, patched, monkeypatch, tmp_path
    ):
        loop = patched
        kwargs = dict(
            exit_forecast=True, exit_score_thr=-10.0, exit_score_every_bar=True
        )
        # A fresh stub per run: the stub counts encounters, so sharing
        # one instance across runs would silence the second run's buys.
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        a = run_backtest(_config(7, tmp_path / "a", **kwargs))
        monkeypatch.setattr(loop, "_decide", _buy_two_stub())
        b = run_backtest(_config(7, tmp_path / "b", **kwargs))
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(a.run_dir / name) == _sha(b.run_dir / name)


class TestExitForecastSign:
    """The T-1 readout's sign contract: ``_exit_forecast_readout`` returns
    ``-balance_used`` — the exit score is the NEGATED centered balance, so
    an avoid-leaning readout clears a positive threshold (TRAINING2 §5.1).
    ``exit_score_thr = -10`` in the integration tests above cannot tell a
    doubled negation from the real thing; this can."""

    def test_readout_negates_the_centered_balance(
        self, patched, monkeypatch, tmp_path
    ):
        from fruitfly.neuromod import Plasticity
        from fruitfly.sim import LIFSim

        loop = patched
        chassis = make_chassis()
        plasticity = Plasticity(chassis)
        sim = LIFSim(chassis, dt_ms=0.5, seed=1)
        # A fixed approach-heavy MBON drive: A = 1.2, R = 0.11 — the same
        # vector for every step, so the noise floor cannot move the score.
        drive = np.array([0.5, 0.4, 0.3, 0.05, 0.04, 0.02])
        monkeypatch.setattr(
            plasticity, "mbon_activation", lambda spikes: drive
        )
        upn_rows, _ = loop.upn_channels(chassis)
        df = _tape([100.0])
        config = BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18")
        kwargs = dict(
            chassis=chassis, plasticity=plasticity, sim=sim,
            upn_rows=upn_rows, w_struct=None,
            lc_rows=np.flatnonzero(chassis.nodes["population"].to_numpy() == "LC"),
            config=config, ticker="AAA", df=df, i=0,
            rng=np.random.Generator(np.random.PCG64(1)),
        )
        a_drive, r_drive = 1.2, 0.11
        balance = (a_drive - r_drive) / (a_drive + r_drive + 1e-9)
        # anchor 0: the centered balance is the raw (positive) balance, so
        # the exit score is its negation — negative.
        score_neutral = loop._exit_forecast_readout(**kwargs, anchor=0.0)
        assert score_neutral == pytest.approx(-balance, abs=1e-9)
        assert score_neutral < 0.0
        # anchor 1: the centering flips the balance negative, so the exit
        # score turns positive — and the two scores differ by exactly the
        # anchor shift (a doubled negation would not).
        score_shifted = loop._exit_forecast_readout(**kwargs, anchor=1.0)
        assert score_shifted == pytest.approx(-(balance - 1.0), abs=1e-9)
        assert score_shifted > 0.0


class TestExitGridCli:
    """The six §5 knobs pass through the backtest CLI."""

    @staticmethod
    def _parse(argv: list[str]):
        from fruitfly.__main__ import build_parser

        return build_parser().parse_args(
            ["backtest", "--seed", "3", "--start", "2026-08-18",
             "--end", "2026-08-19", *argv]
        )

    def test_flags_flow_to_backtest_config(self, monkeypatch, tmp_path):
        import fruitfly.loop as loop_module
        from fruitfly.loop import RunResult

        captured = {}

        def fake_run(config):
            captured["config"] = config
            return RunResult(
                run_dir=tmp_path, n_bars=1, n_events=0, n_orders=0,
                n_deaths=0, final_equity=0.0, wall_s=0.0,
            )

        monkeypatch.setattr(loop_module, "run_backtest", fake_run)
        argv = [
            "--trailing-stop-pct", "1.5", "--atr-stop-mult", "2.0",
            "--no-valence-flip-exit", "--no-hunger-exit",
            "--exit-forecast", "--exit-score-thr", "0.01",
            "--exit-score-every-bar",
        ]
        assert loop_module._cmd_backtest(self._parse(argv)) == 0
        cfg = captured["config"]
        assert cfg.trailing_stop_pct == 1.5
        assert cfg.atr_stop_mult == 2.0
        assert cfg.valence_flip_exit is False
        assert cfg.hunger_exit is False
        assert cfg.exit_forecast is True
        assert cfg.exit_score_thr == 0.01
        assert cfg.exit_score_every_bar is True

    def test_cli_defaults_are_the_no_op_grid(self):
        args = self._parse([])
        assert args.trailing_stop_pct is None
        assert args.atr_stop_mult is None
        assert args.valence_flip_exit is True
        assert args.hunger_exit is True
        assert args.exit_forecast is False
        assert args.exit_score_thr is None
        assert args.exit_score_every_bar is False
