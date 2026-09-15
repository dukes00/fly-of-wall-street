"""Tests for T9 larval training (offline, synthetic chassis + bars).

Covers the acceptance contract: deterministic artifact I/O (byte-identical
double save, no pickle), weight injection changing a run's decisions,
death -> hatch with reset plasticity, and the chassis-fingerprint guard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_loop import make_chassis

from fruitfly.neuromod import Plasticity
from fruitfly.train import (
    load_larval_weights,
    save_larval_weights,
    train_larval,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic chassis (from test_loop) + bars tailored per test
# ---------------------------------------------------------------------------


def make_bars(n_days: int = 2, crash_idx: int | None = None) -> dict[str, pd.DataFrame]:
    """``n_days`` sessions x five 1-min bars for two tickers.

    AAA is flat (zero 1-bar return -> odorless, never in the plume set);
    BBB drifts down every bar so an always-approach policy piles into it.
    ``crash_idx`` (flat bar number across the whole tape) gaps BBB down
    -95% to drive equity through the death threshold / form a grudge.
    """
    frames: dict[str, pd.DataFrame] = {}
    rows, stamps = [], []
    price = 200.0
    k = 0
    for day_index in range(n_days):
        day = 18 + day_index
        for minute in range(5):
            ts = pd.Timestamp(f"2026-08-{day} 13:3{minute}:00", tz="UTC")
            stamps.append(ts)
            if crash_idx is not None and k == crash_idx:
                price = 10.0  # -95%: the synthetic crash
            else:
                price -= 2.0
            k += 1
            rows.append(
                {
                    "open": price - 0.4,
                    "high": price + 0.3,
                    "low": price - 0.3,
                    "close": price,
                    "volume": 5_000 + 1_000 * (minute % 2),
                }
            )
    frames["BBB"] = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps))
    flat = [
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1_000}
        for _ in stamps
    ]
    frames["AAA"] = pd.DataFrame(flat, index=pd.DatetimeIndex(stamps))
    return frames


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    """Wire the loop's seams to the synthetic chassis + bars."""
    import fruitfly.loop as loop

    chassis = make_chassis()
    monkeypatch.setattr(loop, "_load_chassis", lambda: chassis)
    monkeypatch.setattr(loop, "load_bars",
                        lambda symbols, start=None, end=None: make_bars())
    monkeypatch.setattr(loop, "BASKET", ["AAA", "BBB"])
    return loop


def _config(out_dir: Path, start="2026-08-18", end="2026-08-19", **overrides):
    defaults = dict(ms_per_bar=20.0, dt_ms=0.5)
    from fruitfly.loop import BacktestConfig

    return BacktestConfig(seed=7, out_dir=out_dir, start=start, end=end,
                          **{**defaults, **overrides})


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Artifact I/O: determinism + round trip
# ---------------------------------------------------------------------------


class TestArtifact:
    def test_double_save_byte_identical_and_roundtrip(self, patched, tmp_path):
        train = train_larval(
            _config(tmp_path / "run"), out_path=tmp_path / "larval.npz"
        )
        assert (tmp_path / "larval.npz").exists()
        save_larval_weights(
            tmp_path / "larval2.npz", train.weights, train.fingerprint, train.meta
        )
        assert _sha(tmp_path / "larval.npz") == _sha(tmp_path / "larval2.npz")

        lw = load_larval_weights(tmp_path / "larval.npz", make_chassis())
        assert np.array_equal(lw.weights, train.weights)
        assert lw.fingerprint == train.fingerprint
        assert lw.meta["seed"] == "7"
        assert lw.meta["n_bars"] == str(train.run.n_bars)
        # explicit no-pickle read of the exact format we wrote
        with np.load(tmp_path / "larval.npz", allow_pickle=False) as z:
            assert z["weights"].dtype == np.float64

    def test_double_train_byte_identical(self, patched, tmp_path):
        a = train_larval(_config(tmp_path / "a"), out_path=tmp_path / "a.npz")
        b = train_larval(_config(tmp_path / "b"), out_path=tmp_path / "b.npz")
        assert _sha(tmp_path / "a.npz") == _sha(tmp_path / "b.npz")
        for name in ("equity.csv", "events.jsonl"):
            assert _sha(a.run.run_dir / name) == _sha(b.run.run_dir / name)

    def test_fingerprint_guard_rejects_foreign_chassis(self, patched, tmp_path):
        train_larval(_config(tmp_path / "run"), out_path=tmp_path / "larval.npz")
        other = make_chassis()
        other.nodes.loc[0, "bodyId"] = 999_999
        with pytest.raises(ValueError, match="different chassis"):
            load_larval_weights(tmp_path / "larval.npz", other)

    def test_initial_weights_shape_guard(self, patched, tmp_path):
        from fruitfly.loop import run_backtest

        bad = np.zeros((3, 3))
        with pytest.raises(ValueError, match="initial_weights shape"):
            run_backtest(_config(tmp_path / "run", initial_weights=bad))


# ---------------------------------------------------------------------------
# Injection: a trained brain decides differently than a fresh one
# ---------------------------------------------------------------------------


class TestInjection:
    def test_trained_weights_differ_from_baseline(self, patched, tmp_path):
        train = train_larval(
            _config(tmp_path / "run", approach_thr=0.0001, avoid_thr=0.0001,
                    noise_sigma_mv=5.0),
            out_path=tmp_path / "larval.npz",
        )
        fresh = Plasticity(make_chassis())
        assert not np.array_equal(train.weights, fresh.weights)

    def test_injection_changes_decisions(self, patched, tmp_path):
        loop = patched
        kwargs = dict(approach_thr=0.0001, avoid_thr=0.0001, noise_sigma_mv=5.0)
        train_larval(_config(tmp_path / "train", **kwargs),
                     out_path=tmp_path / "larval.npz")
        lw = load_larval_weights(tmp_path / "larval.npz", make_chassis())
        fresh = loop.run_backtest(_config(tmp_path / "fresh", **kwargs))
        injected = loop.run_backtest(
            _config(tmp_path / "injected", initial_weights=lw.weights, **kwargs)
        )

        def balances(run_dir: Path) -> list[float]:
            return [
                event["balance"]
                for event in _events(run_dir / "events.jsonl")
                if event["type"] == "encounter"
            ]

        assert balances(fresh.run_dir) != balances(injected.run_dir)

    def test_injection_changes_actions(self, patched, monkeypatch, tmp_path):
        loop = patched
        # One-trial grudge (DESIGN §6): a -95% shock on day 2 realizes a
        # huge loss; the trained brain then avoids where the fresh fly
        # approached. Crash must not kill (death_threshold near -1).
        monkeypatch.setattr(
            loop, "load_bars",
            lambda symbols, start=None, end=None: make_bars(3, crash_idx=5),
        )
        kwargs = dict(approach_thr=0.0001, avoid_thr=0.0001, noise_sigma_mv=5.0,
                      death_threshold=-0.9, shock_adverse_pct=1.0e9)
        train_larval(_config(tmp_path / "train", start="2026-08-18",
                             end="2026-08-20", **kwargs),
                     out_path=tmp_path / "larval.npz")
        lw = load_larval_weights(tmp_path / "larval.npz", make_chassis())
        fresh = loop.run_backtest(_config(tmp_path / "fresh",
                                          start="2026-08-18", end="2026-08-20",
                                          **kwargs))
        injected = loop.run_backtest(_config(tmp_path / "injected",
                                             start="2026-08-18",
                                             end="2026-08-20",
                                             initial_weights=lw.weights,
                                             **kwargs))

        def actions(run_dir: Path) -> list[tuple[str, str]]:
            return [
                (event["ts"], event["action"])
                for event in _events(run_dir / "events.jsonl")
                if event["type"] == "decision"
            ]

        fresh_actions = actions(fresh.run_dir)
        injected_actions = actions(injected.run_dir)
        assert fresh_actions != injected_actions
        flips = sum(1 for a, b in zip(fresh_actions, injected_actions,
                                      strict=True)
                    if a[1] != b[1])
        assert flips >= 1


# ---------------------------------------------------------------------------
# Death -> hatch with reset plasticity (DESIGN §9, D14)
# ---------------------------------------------------------------------------


class TestDeathHatch:
    @pytest.fixture()
    def crashed(self, patched, monkeypatch):
        """Crash tape + always-approach policy: equity dives below -50%."""
        loop = patched

        def always_approach(balance, held, cap_reached, approach_thr, avoid_thr):
            if held:
                return "add", "approach"
            if cap_reached:
                return "pass", "cap"
            return "buy", "approach"

        monkeypatch.setattr(loop, "_decide", always_approach)
        return loop

    def test_death_kills_and_hatches_with_reset_plasticity(self, crashed,
                                                           monkeypatch, tmp_path):
        loop = crashed
        # The -95% gap lands on the final bar: after the death hatch no
        # further learning happens, so the run's final weights must equal a
        # brand-new Plasticity exactly (the fresh-fly reset, observed).
        # position_cap 3 sizes the compounded book so post-liquidation
        # equity lands below -50% but still positive (paper margin keeps
        # cash negative; the death hatch equity must stay positive).
        monkeypatch.setattr(
            loop, "load_bars",
            lambda symbols, start=None, end=None: make_bars(2, crash_idx=9),
        )
        result = loop.run_backtest(
            _config(tmp_path / "run", position_cap=3, shock_adverse_pct=1.0e9)
        )
        events = _events(result.run_dir / "events.jsonl")
        types = [event["type"] for event in events]
        assert types.count("death") == 1
        assert types.count("hatch") >= 2  # initial hatch + the fresh fly
        liquidations = [
            e for e in events if e.get("reason") == "death_liquidation"
        ]
        assert liquidations, "the dying fly must liquidate its book"

        death = next(e for e in events if e["type"] == "death")
        assert death["equity"] <= death["hatch_equity"] * 0.5  # below -50%
        assert death["hatch_equity"] > 0.0
        assert result.n_deaths == 1

        fresh = Plasticity(make_chassis())
        assert np.array_equal(result.final_weights, fresh.weights)

    def test_death_midrun_hatches_and_continues(self, crashed, monkeypatch,
                                                tmp_path):
        loop = crashed
        # Death on the first bar of day 2 (crash on bar 6 of 10): the
        # hatched fly keeps foraging afterwards, with its own hatch equity.
        monkeypatch.setattr(
            loop, "load_bars",
            lambda symbols, start=None, end=None: make_bars(2, crash_idx=5),
        )
        result = loop.run_backtest(
            _config(tmp_path / "run", position_cap=2, shock_adverse_pct=1.0e9)
        )
        events = _events(result.run_dir / "events.jsonl")
        types = [event["type"] for event in events]
        assert types.count("death") >= 1
        deaths = [e for e in events if e["type"] == "death"]
        assert deaths and deaths[0]["equity"] <= deaths[0]["hatch_equity"] * 0.5
        assert deaths[0]["hatch_equity"] > 0.0
        # wake after the death hatch: the hatched fly keeps encountering
        assert types.count("encounter") > 5

