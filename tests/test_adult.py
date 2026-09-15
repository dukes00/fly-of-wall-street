"""Tests for the T15 adult run harness (offline, synthetic).

Never touches the real chassis cache or market parquet cache: the harness's
seams (``_load_chassis``, ``load_bars``, ``BASKET``) are monkeypatched to the
same tiny synthetic chassis and two-ticker bar tape the loop tests use, so
adult receipts can be compared BYTE-FOR-BYTE against ``run_backtest`` on
identical inputs.

The hard gates:
- parity: a fresh adult replay produces receipts byte-identical to
  ``run_backtest`` for the same seed/window/config — same files, same event
  types, same death/hatch semantics;
- determinism: same seed -> byte-identical receipts, different seed diverges;
- resume: a synthetic SIGKILL (feed raises mid-run, no cleanup) followed by a
  restart continues from the state file and yields byte-identical receipts,
  even with a torn tail appended to the receipt files after the kill;
- state: the state file round-trips every field, and a foreign chassis or
  larval artifact is refused by fingerprint;
- live seam: the live execution adapter submits exactly one well-formed
  market order to the broker and mirrors the actual fill, and the D7
  clock helper lands on real session opens (weekends skipped).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fruitfly.adult import (
    STATE_FILE,
    AdultConfig,
    AdultRun,
    FlyState,
    LiveExecution,
    ReplayFeed,
    SimulatedExecution,
    next_session_open,
)
from fruitfly.broker import OrderEvent
from fruitfly.connectome import Chassis
from fruitfly.loop import BacktestConfig, PendingOrder, run_backtest
from fruitfly.train import chassis_fingerprint, save_larval_weights

# --- synthetic chassis (mirrors tests/test_loop.py populations) -------------

_COLS = ["bodyId", "type", "instance", "somaSide", "population", "region",
         "neurotransmitter", "sign"]

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

_EDGES = [
    (0, 4, 300), (3, 4, 300),
    (0, 5, 300), (3, 5, 300),
    (0, 6, 350), (3, 6, 250),
    (0, 7, 250), (3, 7, 350),
    (0, 8, 150), (1, 9, 150), (3, 10, 150), (3, 11, 150),
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
    (drives equity through the death threshold synthetically)."""
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
def patched(monkeypatch):
    """Wire BOTH the loop's and the harness's seams to one synthetic world."""
    import fruitfly.adult as adult
    import fruitfly.loop as loop

    chassis = make_chassis()
    bars = make_bars()
    for mod in (loop, adult):
        monkeypatch.setattr(mod, "_load_chassis", lambda: chassis)
        monkeypatch.setattr(mod, "load_bars", lambda symbols, start=None, end=None: bars)
        monkeypatch.setattr(mod, "BASKET", ["AAA", "BBB"])
    return adult

def _config(out_dir: Path, **overrides) -> AdultConfig:
    defaults = dict(ms_per_bar=20.0, dt_ms=0.5, seed=7,
                    start="2026-08-18", end="2026-08-19")
    return AdultConfig(run_dir=out_dir, **{**defaults, **overrides})


def _backtest_config(adult_config: AdultConfig, out_dir: Path) -> BacktestConfig:
    cfg = adult_config.backtest_config(out_dir=out_dir)
    return cfg


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _receipt_shas(run_dir: Path) -> tuple[str, str]:
    return _sha(run_dir / "equity.csv"), _sha(run_dir / "events.jsonl")


# ---------------------------------------------------------------------------
# Parity: adult replay == backtest, byte for byte
# ---------------------------------------------------------------------------


class TestParity:
    def test_adult_replay_matches_backtest_byte_identical(self, patched, tmp_path):
        cfg = _config(tmp_path / "adult")
        adult_result = AdultRun(cfg).run()

        bt_result = run_backtest(_backtest_config(cfg, tmp_path / "backtest"))

        assert adult_result.n_bars == bt_result.n_bars == 10  # 2 sessions x 5 min
        adult_shas = _receipt_shas(adult_result.run_dir)
        backtest_shas = _receipt_shas(bt_result.run_dir)
        assert adult_shas == backtest_shas

        # Same event families as the loop, in the same order.
        types_a = [e["type"] for e in _events(adult_result.run_dir / "events.jsonl")]
        types_b = [e["type"] for e in _events(bt_result.run_dir / "events.jsonl")]
        assert types_a == types_b
        assert types_a[0] == "hatch" and types_a[-1] == "anchor"  # v0.6 daily refresh
        for expected in ("wake", "encounter", "decision", "order", "sugar_shock"):
            assert expected in types_a
        # v0.6: every encounter receipt carries the structural looming drive.
        enc = next(e for e in _events(adult_result.run_dir / "events.jsonl")
                   if e["type"] == "encounter")
        assert set(enc) >= {"structural_score", "balance_used", "raw_balance"}
        # Anchor refresh runs at every settle; last event of the run is one.
        anchors = [e for e in types_a if e == "anchor"]
        assert len(anchors) == 2  # two sessions -> two settles

    def test_receipts_have_the_shared_shape(self, patched, tmp_path):
        cfg = _config(tmp_path / "run")
        AdultRun(cfg).run()
        header = (cfg.run_directory() / "equity.csv").read_text().splitlines()[0]
        assert header == "timestamp,equity,cash,n_positions"

    def test_trade_credit_credits_opening_snapshot(self, patched, monkeypatch, tmp_path):
        """A realized sell emits a ``trade_credit`` built from the opening
        fill's eligibility snapshot (DESIGN v0.6 D6)."""
        import fruitfly.adult as adult

        def buy_then_sell(balance, held, cap_reached, approach_thr, avoid_thr):
            return ("sell", "avoid") if held else ("buy", "approach")

        monkeypatch.setattr(adult, "_decide", buy_then_sell)
        cfg = _config(tmp_path / "tc")
        AdultRun(cfg).run()
        events = _events(cfg.run_directory() / "events.jsonl")
        credits = [e for e in events if e["type"] == "trade_credit"]
        assert credits, "a buy-then-sell fly must emit a trade_credit"
        credit = credits[0]
        sells = [e for e in events if e["type"] == "order" and e["side"] == "sell"]
        assert credit["ts"] == sells[0]["ts"]
        assert credit["realized_pnl"] == sells[0]["realized_pnl"]
        buys = [e for e in events if e["type"] == "order" and e["side"] == "buy"]
        notional = buys[0]["shares"] * buys[0]["price"]
        assert credit["reward"] == round(
            min(1.0, max(0.0, credit["realized_pnl"]) / notional), 9
        )
        assert credit["punishment"] == round(
            min(1.0, max(0.0, -credit["realized_pnl"]) / notional), 9
        )


# ---------------------------------------------------------------------------
# Determinism (hard gate)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_double_run_byte_identical(self, patched, tmp_path):
        cfg_a = _config(tmp_path / "a")
        cfg_b = _config(tmp_path / "b")
        AdultRun(cfg_a).run()
        AdultRun(cfg_b).run()
        assert _receipt_shas(cfg_a.run_directory()) == _receipt_shas(cfg_b.run_directory())

    def test_different_seed_diverges(self, patched, tmp_path):
        cfg_a = _config(tmp_path / "a", seed=7)
        cfg_b = _config(tmp_path / "b", seed=8)
        AdultRun(cfg_a).run()
        AdultRun(cfg_b).run()
        assert _receipt_shas(cfg_a.run_directory()) != _receipt_shas(cfg_b.run_directory())


# ---------------------------------------------------------------------------
# Kill-and-resume
# ---------------------------------------------------------------------------


class _KillAfter(Exception):
    """Synthetic SIGKILL: propagates out of run() with no cleanup."""


class _KillingFeed:
    """Wraps the replay feed and dies after ``kill_after`` yielded bars."""

    def __init__(self, inner, kill_after: int) -> None:
        self._inner = inner
        self._kill_after = kill_after

    def frames(self):
        return self._inner.frames()

    def bars(self):
        for i, (ts, frames) in enumerate(self._inner.bars()):
            if i == self._kill_after:
                raise _KillAfter(f"killed before bar {i}")
            yield ts, frames


class TestResume:
    def _reference(self, patched, tmp_path):
        cfg = _config(tmp_path / "ref")
        AdultRun(cfg).run()
        return cfg, _receipt_shas(cfg.run_directory())

    def test_resume_after_kill_is_byte_identical(self, patched, tmp_path):
        _, expected = self._reference(patched, tmp_path)

        cfg = _config(tmp_path / "victim")
        run_dir = cfg.run_directory()
        # Phase 1: die mid-run (7 bars in, inside day 1) with no cleanup.
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=7)).run()
        state_path = run_dir / STATE_FILE
        assert state_path.exists()
        partial = (run_dir / "events.jsonl").read_text().splitlines()
        assert len(partial) < len(
            (tmp_path / "ref" / "events.jsonl").read_text().splitlines()
        )

        # Phase 2: restart the same run directory — continues from state.
        result = AdultRun(cfg).run()
        assert result.resumed is True
        assert _receipt_shas(run_dir) == expected

    def test_resume_drops_torn_receipt_tail(self, patched, tmp_path):
        """A SIGKILL mid-write leaves torn lines AFTER the persisted cursor;
        resume must truncate them, not resume-on-top-of-them."""
        _, expected = self._reference(patched, tmp_path)

        cfg = _config(tmp_path / "victim")
        run_dir = cfg.run_directory()
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=7)).run()
        # The torn tail: half-flushed lines from the dying process.
        with (run_dir / "events.jsonl").open("a") as f:
            f.write('{"type": "encou')  # torn event line, no newline
        with (run_dir / "equity.csv").open("a") as f:
            f.write("2026-08-18T13:37:00+00:00,999.99,1.00,3\n")  # phantom row

        AdultRun(cfg).run()
        assert _receipt_shas(run_dir) == expected

    def test_resume_across_persist_every_n(self, patched, tmp_path):
        """persist_every > 1: resume re-processes the un-persisted bars from
        scratch (deterministic), so the receipts still match the reference."""
        _, expected = self._reference(patched, tmp_path)

        cfg = _config(tmp_path / "victim", persist_every=3)
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=7)).run()

        AdultRun(cfg).run()
        assert _receipt_shas(cfg.run_directory()) == expected

    def test_resume_refuses_config_change(self, patched, tmp_path):
        cfg = _config(tmp_path / "victim")
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=3)).run()

        tampered = AdultConfig(**{**cfg.__dict__, "seed": 99})
        with pytest.raises(ValueError, match="constitution"):
            AdultRun(tampered).run()

    def test_resume_refuses_missing_receipts(self, patched, tmp_path):
        cfg = _config(tmp_path / "victim")
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=3)).run()
        (cfg.run_directory() / "events.jsonl").unlink()
        with pytest.raises(ValueError, match="pair"):
            AdultRun(cfg).run()


    def test_resume_carries_pending_orders(self, patched, monkeypatch, tmp_path):
        """Pending pessimistic next-bar orders survive the state file: the
        resumed fly's receipts match the uninterrupted reference byte for
        byte (orders queued before the kill keep waiting)."""
        import fruitfly.adult as adult
        monkeypatch.setattr(adult, "_decide", _always_buy)

        cfg = _config(tmp_path / "ref")
        AdultRun(cfg).run()
        expected = _receipt_shas(cfg.run_directory())

        victim = _config(tmp_path / "victim")
        run_dir = victim.run_directory()
        inner = ReplayFeed(victim)
        with pytest.raises(_KillAfter):
            AdultRun(victim, feed=_KillingFeed(inner, kill_after=7)).run()
        state = FlyState.load(run_dir / STATE_FILE)
        assert state.pending, "an order must be pending at the kill"
        assert all(o.side == "buy" for o in state.pending)

        result = AdultRun(victim).run()
        assert result.resumed is True
        assert _receipt_shas(run_dir) == expected


def _always_buy(balance, held, cap_reached, approach_thr, avoid_thr):
    """Decision stub: real churn every encounter, cap-aware."""
    if held:
        return "add", "approach"
    if cap_reached:
        return "pass", "cap"
    return "buy", "approach"


class TestSimulatedExecution:
    """The replay broker seam runs the loop's pessimistic next-bar model."""

    def test_pending_fills_at_execution_bar_high_and_low(self):
        frames = make_bars()
        execution = SimulatedExecution(frames=frames)
        ts = pd.Timestamp("2026-08-18 13:30:00", tz="UTC")
        nxt = pd.Timestamp("2026-08-18 13:31:00", tz="UTC")

        execution.book.submit(
            PendingOrder(decision_ts=ts, ticker="AAA", side="buy",
                         reason="approach", shares=10)
        )
        fills = execution.book.drain(nxt)
        assert len(fills) == 1
        o, price = fills[0]
        assert o.ticker == "AAA" and o.shares == 10
        assert price == float(frames["AAA"].at[nxt, "high"])
        assert price != float(frames["AAA"].at[nxt, "close"])  # not the close

        execution.book.submit(
            PendingOrder(decision_ts=ts, ticker="BBB", side="sell",
                         reason="avoid")
        )
        fills = execution.book.drain(nxt)
        assert len(fills) == 1
        o, price = fills[0]
        assert price == float(frames["BBB"].at[nxt, "low"])

    def test_pending_order_skips_gap_and_cancel_all(self):
        frames = make_bars()
        gap_ts = pd.Timestamp("2026-08-18 13:31:00", tz="UTC")
        frames["AAA"] = frames["AAA"].drop(gap_ts)
        execution = SimulatedExecution(frames=frames)
        ts = pd.Timestamp("2026-08-18 13:30:00", tz="UTC")
        execution.book.submit(
            PendingOrder(decision_ts=ts, ticker="AAA", side="buy",
                         reason="approach", shares=10)
        )
        # 13:31 is the next union-timeline bar but AAA does not print there.
        assert execution.book.drain(gap_ts) == []
        nxt = pd.Timestamp("2026-08-18 13:32:00", tz="UTC")
        assert execution.book.drain(nxt)
        assert not execution.book.pending_orders()

    def test_close_mode_fills_at_reference_price(self):
        execution = SimulatedExecution(fill_mode="close")
        ts = pd.Timestamp("2026-08-18 13:31:00", tz="UTC")
        assert execution.execute(ts, "AAA", "buy", 10, 424.0) == 424.0


# ---------------------------------------------------------------------------
# Death -> hatch (D14), with loop parity
# ---------------------------------------------------------------------------


class TestDeath:
    def test_death_hatches_fresh_and_matches_backtest(self, patched, monkeypatch, tmp_path):
        import fruitfly.adult as adult
        import fruitfly.loop as loop

        crash_bars = make_bars(crash=True)
        for mod in (loop, adult):
            monkeypatch.setattr(mod, "load_bars", lambda symbols, start=None, end=None: crash_bars)

        def always_approach(balance, held, cap_reached, approach_thr, avoid_thr):
            if held:
                return "add", "approach"
            if cap_reached:
                return "pass", "cap"
            return "buy", "approach"

        for mod in (loop, adult):
            monkeypatch.setattr(mod, "_decide", always_approach)

        cfg = _config(tmp_path / "crash", death_threshold=-0.02, position_cap=2,
                      shock_adverse_pct=1.0e9)
        result = AdultRun(cfg).run()
        bt = run_backtest(_backtest_config(cfg, tmp_path / "backtest"))

        # Parity through a death: identical receipts, byte for byte.
        assert _receipt_shas(result.run_dir) == _receipt_shas(bt.run_dir)

        events = _events(result.run_dir / "events.jsonl")
        bt_events = _events(bt.run_dir / "events.jsonl")
        bt_types = [e["type"] for e in bt_events]
        types = [e["type"] for e in events]
        assert types.count("death") >= 1
        # loop.py v0.6 emits one hatch per death (brain reset; the anchor
        # re-calibration follows without a second hatch); adult mirrors that
        # exactly — and the full event stream is byte-identical anyway via
        # the receipt-sha check above.
        assert types.count("hatch") == bt_types.count("hatch") == 2  # initial + fresh fly

        death = next(e for e in events if e["type"] == "death")
        assert death["equity"] <= death["hatch_equity"] * (1.0 - 0.02)
        liquidations = [e for e in events if e.get("reason") == "death_liquidation"]
        assert liquidations, "the dying fly must liquidate its book"

        # After death the book is flat on the death bar.
        rows = (result.run_dir / "equity.csv").read_text().splitlines()[1:]
        death_row = next(r.split(",") for r in rows if r.startswith(death["ts"]))
        assert int(death_row[3]) == 0


# ---------------------------------------------------------------------------
# State file: round-trip and fingerprint guards
# ---------------------------------------------------------------------------


class TestState:
    def test_state_round_trip(self, patched, tmp_path):
        cfg = _config(tmp_path / "run")
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=5)).run()

        path = cfg.run_directory() / STATE_FILE
        before = FlyState.load(path)
        before.save(tmp_path / "copy.npz")
        after = FlyState.load(tmp_path / "copy.npz")

        for name in (
            "chassis_fingerprint", "config_echo", "larval_fingerprint", "cash",
            "hatch_equity", "equity", "cur_day", "pointer", "pointer_drawn",
            "anchor", "daily_signal_encounters", "day_realized", "day_abs_ret_sum",
            "day_abs_ret_n", "hunger_armed", "rng_state", "n_bars", "n_events",
            "n_orders", "n_deaths", "last_ts", "pending",
        ):
            assert getattr(after, name) == getattr(before, name), name
        assert set(after.positions) == set(before.positions)
        for ticker, pos in before.positions.items():
            assert after.positions[ticker].shares == pos.shares
            assert after.positions[ticker].avg_cost == pos.avg_cost
        assert after.last_close == before.last_close
        for name in (
            "weights", "eligibility", "habituation", "daily_spikes",
            "daily_mbon_drive", "sim_v", "sim_refr", "sim_prev", "sim_spikes",
        ):
            assert np.array_equal(getattr(after, name), getattr(before, name)), name

    def test_resume_refuses_foreign_chassis(self, patched, tmp_path):
        cfg = _config(tmp_path / "run")
        inner = ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=3)).run()

        # Tamper the state file's chassis fingerprint -> foreign brain.
        path = cfg.run_directory() / STATE_FILE
        state = FlyState.load(path)
        state.chassis_fingerprint = "0" * 64
        state.save(path)

        with pytest.raises(ValueError, match="different chassis"):
            AdultRun(cfg).run()


class TestLarval:
    def test_larval_weights_injected_with_fingerprint(self, patched, tmp_path):
        import fruitfly.adult as adult

        chassis = make_chassis()
        artifact = save_larval_weights(
            tmp_path / "larval.npz",
            weights=np.full((8, 6), 0.5),  # (n_kc, n_mbon) of the synthetic chassis
            fingerprint=chassis_fingerprint(chassis),
            meta={"stage": "larval-test"},
        )

        # A run killed right after the first bar persists the injected brain.
        cfg = _config(tmp_path / "run", larval_weights=artifact)
        inner = adult.ReplayFeed(cfg)
        with pytest.raises(_KillAfter):
            AdultRun(cfg, feed=_KillingFeed(inner, kill_after=1)).run()

        state = FlyState.load(cfg.run_directory() / STATE_FILE)
        assert state.weights.shape == (8, 6)
        assert np.allclose(state.weights, np.full((8, 6), 0.5))
        assert state.larval_fingerprint is not None

    def test_larval_artifact_from_foreign_chassis_refused(self, patched, tmp_path):
        artifact = save_larval_weights(
            tmp_path / "foreign.npz",
            weights=np.full((8, 6), 0.5),
            fingerprint="deadbeef" * 8,  # trained on some other brain
            meta={"stage": "larval-test"},
        )
        cfg = _config(tmp_path / "run", larval_weights=artifact)
        with pytest.raises(ValueError, match="different chassis"):
            AdultRun(cfg).run()


# ---------------------------------------------------------------------------
# Live seam (offline: fake broker + clock)
# ---------------------------------------------------------------------------


class _FakeBroker:
    """Paper-broker stand-in: one resting order that fills on first poll."""

    def __init__(self, fill_price: float = 424.25) -> None:
        self.fill_price = fill_price
        self.submitted: list[OrderEvent] = []
        self._next_id = 1

    def submit(self, event: OrderEvent) -> str:
        self.submitted.append(event)
        order_id = str(self._next_id)
        self._next_id += 1
        return order_id

    def cancel(self, order_id: str) -> None:
        pass

    class _Order:
        def __init__(self, status: str, price: float) -> None:
            self.status = type("S", (), {"value": status})()
            self.filled_avg_price = price

    def _get_order(self, order_id: str):
        return self._Order("filled", self.fill_price)

    @property
    def _client(self):
        broker = self

        class _Client:
            def get_order_by_id(self, order_id: str):
                return broker._get_order(order_id)

        return _Client()


class TestLiveSeam:
    def test_live_execution_submits_and_mirrors_fill(self):
        broker = _FakeBroker(fill_price=424.25)
        execution = LiveExecution(broker, poll_s=0.0, timeout_s=5.0)
        ts = pd.Timestamp("2026-09-15 13:31:00", tz="UTC")

        fill = execution.execute(ts, "SPY", "buy", 3, 424.00)

        assert fill == 424.25  # the ACTUAL fill, not the bar-close reference
        assert len(broker.submitted) == 1
        event = broker.submitted[0]
        assert isinstance(event, OrderEvent)
        assert event.symbol == "SPY" and event.side == "buy"
        assert event.qty == 3 and event.type == "market" and event.tif == "day"

    def test_next_session_open_skips_weekend(self):
        # Friday evening after the close -> Monday's open.
        friday_night = pd.Timestamp("2026-09-11 21:00:00", tz="UTC")
        assert friday_night.day_name() == "Friday"
        nxt = next_session_open(friday_night)
        assert nxt == pd.Timestamp("2026-09-14 13:30:00", tz="UTC")
        assert nxt.day_name() == "Monday"

    def test_next_session_open_during_session_is_now(self):
        wednesday = pd.Timestamp("2026-09-16 14:07:00", tz="UTC")
        assert next_session_open(wednesday) == pd.Timestamp("2026-09-16 13:30:00", tz="UTC")
        # Just before the open: today's open.
        pre_open = pd.Timestamp("2026-09-16 13:00:00", tz="UTC")
        assert next_session_open(pre_open) == pd.Timestamp("2026-09-16 13:30:00", tz="UTC")


# ---------------------------------------------------------------------------
# Chassis config (whole-fly phase 1): pass-through to the loop config
# ---------------------------------------------------------------------------


class TestChassisPassThrough:
    def test_whole_config_gets_calibrated_std_and_passes_through(self):
        cfg = _config(Path("data/adult/unused"), chassis="whole")
        # The calibrated T12b defaults are applied at construction.
        assert cfg.std_beta == 0.1
        assert cfg.std_tau_rec_ms == 500.0
        bt = cfg.backtest_config()
        assert bt.chassis == "whole"
        assert bt.std_beta == 0.1
        assert bt.std_tau_rec_ms == 500.0
        # Resume parity: the state-file echo pins the constitution.
        echo = cfg._echo()
        assert echo["chassis"] == "whole"
        assert echo["std_beta"] == 0.1
        assert echo["std_tau_rec_ms"] == 500.0

    def test_default_is_stripped_without_std(self):
        cfg = _config(Path("data/adult/unused"))
        assert cfg.chassis == "stripped"
        assert cfg.std_beta is None and cfg.std_tau_rec_ms is None
        bt = cfg.backtest_config()
        assert bt.chassis == "stripped"
        assert bt.std_beta is None and bt.std_tau_rec_ms is None

    def test_unknown_chassis_rejected(self):
        with pytest.raises(ValueError, match="chassis must be"):
            _config(Path("data/adult/unused"), chassis="larval")
