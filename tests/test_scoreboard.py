"""T8 scoreboard tests (offline, synthetic bars).

Gates: monkey-with-darts and the logistic control reproduce byte-identical
outputs at a fixed seed; the logistic control's features are EXACTLY the
shared smell feature builder's outputs (identity with
``fruitfly.senses.smell.build_features`` and with the smell channel's
modulation inputs); ``compare_run`` renders on a synthetic run_dir.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fruitfly.scoreboard import (
    SPIVA_ROW,
    compare_run,
    feature_matrix,
    logistic_control,
    monkey_darts,
    spx_buyhold,
)
from fruitfly.senses import FEATURE_NEUTRAL, FEATURES, encode_smell
from fruitfly.senses.smell import RSI_PERIOD, VOL_WINDOW, build_features

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import charts  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
START, END = "2026-08-17", "2026-08-21"


def make_bars(
    n_bars: int = 600, seed: int = 7, symbols: list[str] = SYMBOLS
) -> dict[str, pd.DataFrame]:
    """Deterministic synthetic 1-minute bars: sine-drift random walks.

    The sine drift gives the logistic control a learnable directional
    signal; volumes oscillate so ``volume_delta`` carries information.
    """
    ts = pd.date_range("2026-08-17 13:30", periods=n_bars, freq="min", tz="UTC")
    out = {}
    for i, sym in enumerate(symbols):
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


@pytest.fixture(scope="module")
def bars():
    return make_bars()


@pytest.fixture(scope="module")
def chassis():
    return charts.make_chassis()


# --- shared feature builder identity ---------------------------------------


def test_build_features_matches_definitions(bars):
    df = bars["AAA"]
    feats = build_features(df)
    assert set(feats) == set(FEATURES)
    close = df["close"].to_numpy(dtype=float)
    rets = np.diff(close) / close[:-1]
    assert feats["returns"] == rets[-1]
    assert feats["volatility"] == np.std(rets[-VOL_WINDOW:], ddof=1)
    deltas = np.diff(close[-(RSI_PERIOD + 1):])
    gain = np.maximum(deltas, 0.0).mean()
    loss = np.maximum(-deltas, 0.0).mean()
    expected_rsi = 100.0 - 100.0 / (1.0 + gain / loss) if loss > 0 else 100.0
    assert feats["rsi"] == expected_rsi
    volume = df["volume"].to_numpy(dtype=float)
    assert feats["volume_delta"] == volume[-1] / volume[-(VOL_WINDOW + 1):-1].mean() - 1.0


def test_build_features_neutral_on_short_history():
    one = pd.DataFrame({"close": [100.0], "volume": [1_000]})
    assert build_features(one) == dict(FEATURE_NEUTRAL)
    two = pd.DataFrame({"close": [100.0, 101.0], "volume": [1_000, 1_100]})
    feats = build_features(two)
    assert feats["returns"] == 0.01
    assert feats["rsi"] == FEATURE_NEUTRAL["rsi"]  # < RSI_PERIOD+1 bars
    assert feats["volatility"] == FEATURE_NEUTRAL["volatility"]
    assert feats["volume_delta"] == FEATURE_NEUTRAL["volume_delta"]


def test_feature_matrix_rows_are_build_features(bars):
    df = bars["AAA"]
    fm = feature_matrix(df)
    assert list(fm.columns) == list(FEATURES)
    assert fm.index.equals(df.index)
    for t in itertools.chain(range(0, 3), [VOL_WINDOW, 137, len(df) - 1]):
        direct = build_features(df.iloc[: t + 1])
        for f in FEATURES:
            assert fm.iloc[t][f] == direct[f], (t, f)


def test_smell_modulation_uses_shared_features(bars, chassis):
    """The builder's outputs route through encode_smell's modulation inputs."""
    feats = build_features(bars["AAA"].iloc[-60:])
    assert set(feats) == set(FEATURES)
    neutral = dict(FEATURE_NEUTRAL)
    base = encode_smell("AAPL", neutral, chassis)
    assert not np.array_equal(base, encode_smell("AAPL", feats, chassis))
    # Each built feature value provably feeds the same per-(feature, channel)
    # modulator term encode_smell applies to hand-built feature dicts.
    for f, v in feats.items():
        assert np.array_equal(
            encode_smell("AAPL", {f: v}, chassis),
            encode_smell("AAPL", {**neutral, f: v}, chassis),
        )


# --- determinism gates -------------------------------------------------------


def test_monkey_darts_byte_identical(bars):
    a = monkey_darts(bars, seed=42, capital=100_000.0)
    b = monkey_darts(bars, seed=42, capital=100_000.0)
    pd.testing.assert_series_equal(a, b, check_exact=True)
    assert a.to_csv().encode() == b.to_csv().encode()


def test_logistic_control_byte_identical(bars):
    a_eq, a_sum = logistic_control(bars, seed=42, capital=100_000.0)
    b_eq, b_sum = logistic_control(bars, seed=42, capital=100_000.0)
    pd.testing.assert_series_equal(a_eq, b_eq, check_exact=True)
    assert a_eq.to_csv().encode() == b_eq.to_csv().encode()
    assert a_sum == b_sum


def test_monkey_darts_seed_changes_path(bars):
    a = monkey_darts(bars, seed=1).to_numpy()
    b = monkey_darts(bars, seed=2).to_numpy()
    assert not np.array_equal(a, b)


def test_monkey_darts_shape_and_start(bars):
    curve = monkey_darts(bars, seed=3, capital=250_000.0, n_positions=2)
    grid = pd.DatetimeIndex(sorted(set().union(*(df.index for df in bars.values()))))
    assert curve.index.equals(grid)
    assert curve.iloc[0] == 250_000.0
    assert (curve > 0).all()


# --- logistic control behavior ------------------------------------------------


def test_logistic_control_summary_contract(bars):
    equity, summary = logistic_control(bars, seed=42, capital=100_000.0)
    assert summary["feature_builder"] == "fruitfly.senses.smell.build_features"
    assert summary["features"] == list(FEATURES)
    assert summary["threshold"] == 0.5
    assert set(summary["coefficients"]) == set(FEATURES)
    assert summary["train_rows"] > 0
    assert "test_accuracy" in summary
    grid = pd.DatetimeIndex(sorted(set().union(*(df.index for df in bars.values()))))
    assert equity.index[0] > grid[0]  # trading starts after the 70% split
    assert equity.iloc[0] == 100_000.0
    assert (equity > 0).all()


# --- S&P buy-and-hold ---------------------------------------------------------


def test_spx_buyhold_grid_and_start():
    curve = spx_buyhold(START, END, capital=50_000.0)
    grid = pd.bdate_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"))
    assert curve.index.equals(grid)
    assert curve.iloc[0] == 50_000.0
    assert curve.notna().all()
    assert (curve > 0).all()


# --- compare_run ---------------------------------------------------------------


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "backtest_7_2026-08-17_2026-08-21"
    d.mkdir()
    bars = make_bars(n_bars=600)
    grid = pd.DatetimeIndex(sorted(set().union(*(df.index for df in bars.values()))))
    equity = 100_000.0 * np.cumprod(1.0 + 0.0002 * np.sin(np.arange(len(grid))))
    pd.DataFrame(
        {
            "timestamp": grid.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "equity": equity,
            "cash": equity * 0.5,
            "n_positions": 3,
        }
    ).to_csv(d / "equity.csv", index=False)
    with open(d / "events.jsonl", "w", encoding="utf-8") as fh:
        for i, (etype, payload) in enumerate(
            [
                ("wake", {"session": "2026-08-17"}),
                ("decision", {"action": "approach"}),
                ("sugar_shock", {"magnitude": 0.4}),
                ("sleep", {"session": "2026-08-17"}),
                ("death", {"cause": "starvation"}),
                ("hatch", {"generation": 1}),
            ]
        ):
            fh.write(json.dumps({"type": etype, "ts": grid[i].isoformat(), **payload}) + "\n")
    return d, bars


def test_compare_run_renders_report(run_dir, tmp_path):
    d, bars = run_dir
    out = tmp_path / "t8-scoreboard.md"
    report = compare_run(d, START, END, capital=100_000.0, bars_by_symbol=bars, out_path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == report
    # Fly row parsed from the run's own equity.csv
    assert "| Fly (this run) |" in report
    for name in ("S&P 500 buy-and-hold", "Monkey-with-darts (seed 7", "Logistic control"):
        assert f"| {name}" in report
    # Same window/grid across benchmarks: final returns consistent with curves
    assert "Max drawdown %" in report
    # SPIVA static reference row with citation
    assert SPIVA_ROW["url"] in report
    assert SPIVA_ROW["accessed"] in report
    assert f"1y {SPIVA_ROW['underperforming_pct']['1y']:.2f}%" in report
    # Events parsed
    assert "deaths: 1, hatches: 1" in report
    assert "sugar_shock: 1" in report
    # Logistic control documentation
    assert "fruitfly.senses.smell.build_features" in report
    assert "threshold 0.5" in report
    # Daily SPX closes must actually map onto the intraday comparison grid
    # (as-of last close, not a constant back-filled series).
    spx = spx_buyhold(START, END, capital=100_000.0)
    fly_grid = pd.DatetimeIndex(
        pd.to_datetime(pd.read_csv(d / "equity.csv")["timestamp"], utc=True)
    )
    aligned = spx.reindex(spx.index.union(fly_grid)).ffill().bfill().reindex(fly_grid)
    expected = 100.0 * (aligned.iloc[-1] / aligned.iloc[0] - 1.0)
    assert f"| S&P 500 buy-and-hold | {expected:+.2f}% |" in report


def test_spiva_row_citation_complete():
    assert set(SPIVA_ROW["underperforming_pct"]) == {"1y", "3y", "5y", "15y"}
    assert SPIVA_ROW["url"].startswith("https://")
    assert SPIVA_ROW["source"].startswith("S&P Dow Jones Indices")
    assert "2025" in SPIVA_ROW["source"]
