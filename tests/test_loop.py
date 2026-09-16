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
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

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
