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
from fruitfly.loop import BacktestConfig, run_backtest

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


def make_chassis() -> Chassis:
    n = len(_ROWS)
    nodes = pd.DataFrame(
        [(10_000 + i, t, t, "L", p, r, nt, s)
         for i, (t, p, nt, s), r
         in zip(range(n), _ROWS,
                ["antennal-lobe"] * 4 + ["mushroom-body"] * 14
                + ["protocerebrum"] * 4,
                strict=True)],
        columns=_COLS,
    )
    pre = np.array([e[0] for e in _EDGES], dtype=np.int64)
    post = np.array([e[1] for e in _EDGES], dtype=np.int64)
    w = np.array([e[2] for e in _EDGES], dtype=np.int64)
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


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    """Wire the loop's seams to the synthetic chassis + bars."""
    import fruitfly.loop as loop

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
                    approach_thr=0.001, avoid_thr=0.001, noise_sigma_mv=5.0)
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
                    noise_sigma_mv=5.0)
        )
        types = {event["type"] for event in _events(result.run_dir / "events.jsonl")}
        assert {"wake", "encounter", "decision", "order", "sugar_shock", "sleep"} <= types

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
        assert liquidations, "the dying fly must liquidate its book"
        last = _events(result.run_dir / "events.jsonl")[-1]
        assert last["type"] == "sleep"
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

    def test_no_look_ahead_fills_on_the_next_bar(self, patched, tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=5.0)
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

    def test_pessimistic_pricing_buy_high_sell_low(self, patched, tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=5.0)
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
        assert events[-1]["type"] == "sleep"

    def test_close_mode_is_the_documented_look_ahead_hatch(self, patched,
                                                          tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", fill_mode="close",
                    approach_thr=0.001, avoid_thr=0.001, noise_sigma_mv=5.0)
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

    def test_fees_default_zero_untouched(self, patched, tmp_path):
        result = run_backtest(
            _config(7, tmp_path / "run", approach_thr=0.001, avoid_thr=0.001,
                    noise_sigma_mv=5.0)
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
    """``BacktestConfig`` chassis/STD surface (defaults must not move)."""

    def test_default_is_stripped_without_std(self):
        cfg = BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18")
        assert cfg.chassis == "stripped"
        assert cfg.std_beta is None
        assert cfg.std_tau_rec_ms is None

    def test_fill_mode_validation(self):
        assert BacktestConfig(
            seed=1, start="2026-08-18", end="2026-08-18"
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
                           std_beta=0.1)

    def test_std_ranges_validated(self):
        with pytest.raises(ValueError, match="std_beta"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           chassis="whole", std_beta=1.5)
        with pytest.raises(ValueError, match="std_tau_rec_ms"):
            BacktestConfig(seed=1, start="2026-08-18", end="2026-08-18",
                           chassis="whole", std_tau_rec_ms=-1.0)


class TestChassisSimConstruction:
    """The loop builds the engine the config asks for (real code path)."""

    def test_default_run_builds_std_off_sim(self, patched, monkeypatch, tmp_path):
        import fruitfly.loop as loop

        seen: dict = {}
        real = loop.LIFSim

        def spy(chassis, **kwargs):
            seen.update(kwargs)
            return real(chassis, **kwargs)

        monkeypatch.setattr(loop, "LIFSim", spy)
        run_backtest(_config(7, tmp_path / "run"))
        # STD off by default: the engine stays bit-identical to the pre-STD one.
        assert seen["std_beta"] is None
        assert seen["std_tau_rec_ms"] is None

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
