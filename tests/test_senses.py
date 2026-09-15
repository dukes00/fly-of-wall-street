"""Tests for the T4 senses package (offline, synthetic chassis).

Covers the acceptance gates: population-exact dims, identity signatures
constant per ticker and pairwise-distinct, looming fires on a crash spike and
is silent on flat, monotonic sweet/bitter taste magnitude, byte-identical
golden-fixture reproduction, and cross-run determinism.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fruitfly.senses import (
    encode_smell,
    encode_taste,
    encode_vision,
    identity_profile,
)
from fruitfly.senses.smell import FEATURE_NEUTRAL, build_features
from fruitfly.senses.vision import LOOMING_GAIN
from fruitfly.sim import LIFSim

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import charts  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def chassis():
    return charts.make_chassis()


@pytest.fixture(scope="module")
def pop(chassis):
    return chassis.nodes["population"].to_numpy()


# --- vision ----------------------------------------------------------------


def test_vision_dims_and_targets(chassis, pop):
    out = encode_vision(charts.chart_up(), chassis)
    assert out.shape == (chassis.n_neurons,) == (27_115,)
    assert out.dtype == np.float64
    # nonzero entries confined to T4 + T5 + LC-looming
    allowed = (pop == "T4") | (pop == "T5") | (pop == "LC-looming")
    assert np.count_nonzero(out[~allowed]) == 0
    assert np.count_nonzero(out) > 0


def test_vision_direction_sign(chassis, pop):
    t4t5 = (pop == "T4") | (pop == "T5")
    up = encode_vision(charts.chart_up(), chassis)[t4t5]
    crash = encode_vision(charts.chart_crash(), chassis)[t4t5]
    assert up.sum() > 0.0  # bullish drift depolarizes the ON ensemble
    assert crash.sum() < 0.0  # crash drift depolarizes the OFF ensemble


def test_vision_looming_fires_on_crash_silent_on_flat(chassis, pop):
    lc = pop == "LC-looming"
    flat = encode_vision(charts.chart_flat(), chassis)[lc]
    crash = encode_vision(charts.chart_crash(), chassis)[lc]
    assert np.all(flat == 0.0)  # silent on flat
    assert np.all(crash > 0.0)  # every LC cell above its threshold fires
    assert crash.max() > 0.5 * LOOMING_GAIN  # strongly fired, near saturation


def test_vision_deterministic(chassis):
    a = encode_vision(charts.chart_crash(), chassis)
    b = encode_vision(charts.chart_crash(), chassis)
    assert a.tobytes() == b.tobytes()


def test_vision_rejects_bad_windows(chassis):
    with pytest.raises(ValueError):
        encode_vision(charts.chart_flat()[["open", "high", "low"]], chassis)
    with pytest.raises(ValueError):
        encode_vision(charts.chart_flat().iloc[:0], chassis)


def _crash_tape_3x() -> pd.DataFrame:
    """Quiet oscillating band; the last bar's range is 3x the quiet bars'.

    Window-normalized expansion lands at a realistic ~0.2 (r_last 0.3 vs
    r_earlier 0.1) — the regime where the pre-T12b gains left LC silent.
    """
    rows = []
    for i in range(47):
        mid = 104.5 + 4.5 * np.sin(2.0 * np.pi * i / 12.0)
        rows.append((mid - 0.25, mid + 0.5, mid - 0.5, mid + 0.25))
    rows.append((105.0, 105.0, 102.0, 102.2))  # crash bar: range 3.0 vs quiet 1.0
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_vision_lc_spike_yield_on_crash_tape(chassis, pop):
    """T12b regression: on a crash-like bar (3x range expansion) looming LC
    cells must actually SPIKE within a 500 ms encounter, not merely receive
    sub-threshold current. Measured at the old gains (30/40): 0 spikes."""
    from fruitfly.senses import vision as vision_mod

    drift, rng = vision_mod._per_column_signals(_crash_tape_3x())
    expansion = float(rng[-1]) - float(rng[:-1].mean())
    assert 0.0 < expansion < 0.5  # realistic 3x expansion, not a degenerate spike

    lc = pop == "LC-looming"
    inp = encode_vision(_crash_tape_3x(), chassis)
    spikes = LIFSim(chassis).step(inp, 500.0)["spikes"]
    assert int(spikes[lc].sum()) > 0

    # Gentle drift stays LC-silent: uniform ranges give exactly zero expansion.
    assert np.all(encode_vision(charts.chart_up(), chassis)[lc] == 0.0)


def test_vision_long_window_uses_last_48_bars(chassis):
    long = charts.chart_up(n_bars=200)
    assert np.array_equal(
        encode_vision(long, chassis), encode_vision(long.iloc[-48:], chassis)
    )


# --- smell -----------------------------------------------------------------


_TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "AMZN", "META", "AMD", "NFLX", "SPY"]

_NEUTRAL = {
    "returns": 0.0,
    "rsi": 50.0,
    "volatility": 0.0,
    "volume_delta": 0.0,
    "mom20": 0.0,
    "slope20": 0.0,
}


def test_smell_dims_exact(chassis):
    out = encode_smell("AAPL", _NEUTRAL, chassis)
    assert out.shape == (chassis.meta["population_dims"]["uPN"],) == (391,)
    assert out.dtype == np.float64
    assert np.all(out >= 0.0)


def test_smell_identity_constant_per_ticker_distinct_across(chassis):
    sigs = [encode_smell(t, _NEUTRAL, chassis) for t in _TICKERS]
    for t, sig in zip(_TICKERS, sigs, strict=False):
        assert np.array_equal(sig, encode_smell(t, _NEUTRAL, chassis))
    assert all(not np.array_equal(a, b) for a, b in itertools.combinations(sigs, 2))
    # the profile primitive itself: 88 dims, constant, distinct
    for t in _TICKERS:
        prof = identity_profile(t)
        assert prof.shape == (88,)
        assert np.array_equal(prof, identity_profile(t))
    assert all(
        not np.array_equal(identity_profile(a), identity_profile(b))
        for a, b in itertools.combinations(_TICKERS, 2)
    )


def test_smell_state_modulates(chassis):
    base = encode_smell("AAPL", _NEUTRAL, chassis)
    hot = encode_smell(
        "AAPL", {"returns": 0.05, "rsi": 80.0, "volatility": 0.03,
                 "volume_delta": 0.4, "mom20": 0.1, "slope20": 0.002},
        chassis,
    )
    assert not np.array_equal(base, hot)
    # missing features are neutral: subset == full dict with zeros
    assert np.array_equal(
        encode_smell("AAPL", {"volatility": 0.03}, chassis),
        encode_smell("AAPL", {**_NEUTRAL, "volatility": 0.03}, chassis),
    )




def test_build_features_mom20_slope20_exact():
    # 21-bar geometric ramp: mom20 must reach close[-21] (= index 0), so the
    # window is 21 bars — an off-by-one would use close[-20] instead.
    closes = [100.0 * 1.01**i for i in range(21)]
    feats = build_features(pd.DataFrame({"close": closes, "volume": [1_000.0] * 21}))
    assert feats["mom20"] == closes[-1] / closes[0] - 1.0
    assert feats["mom20"] == pytest.approx(1.01**20 - 1.0)
    # 20-bar linear ramp: relative OLS slope = 1 per bar over mean 110.5.
    lin = [100.0 + float(i) for i in range(21)]
    f2 = build_features(pd.DataFrame({"close": lin, "volume": [1_000.0] * 21}))
    window = np.asarray(lin[-20:])
    assert f2["slope20"] == float(np.polyfit(np.arange(20), window, 1)[0] / window.mean())


def test_build_features_mom20_slope20_warmup_neutral():
    lin = [100.0 + float(i) for i in range(20)]
    feats = build_features(pd.DataFrame({"close": lin, "volume": [1_000.0] * 20}))
    assert feats["mom20"] == FEATURE_NEUTRAL["mom20"]  # mom20 needs 21 bars
    window = np.asarray(lin[-20:])
    assert feats["slope20"] == float(np.polyfit(np.arange(20), window, 1)[0] / window.mean())


def test_build_features_trailing_only():
    """No look-ahead: every feature's window fits in the trailing 21 bars,
    so a 30-bar frame and its last-21 slice must agree exactly."""
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 30))
    volume = rng.uniform(1_000.0, 2_000.0, 30)
    df = pd.DataFrame({"close": close, "volume": volume})
    assert build_features(df) == build_features(df.iloc[-21:])


def test_smell_new_features_flow_through(chassis):
    """mom20/slope20 modulate the identity like the other features; the
    encode_smell output dim is unchanged (391)."""
    base = encode_smell("AAPL", _NEUTRAL, chassis)
    hot = encode_smell("AAPL", {**_NEUTRAL, "mom20": 0.15, "slope20": 0.003}, chassis)
    assert hot.shape == (391,)
    assert not np.array_equal(base, hot)
    assert not np.array_equal(base, encode_smell("AAPL", {**_NEUTRAL, "mom20": 0.15}, chassis))
    assert not np.array_equal(base, encode_smell("AAPL", {**_NEUTRAL, "slope20": 0.003}, chassis))


def test_smell_glomerulus_mapping_covers_all_channels(chassis, pop):
    from fruitfly.senses.smell import upn_channels

    rows, channels = upn_channels(chassis)
    assert rows.size == 391
    assert np.array_equal(rows, np.flatnonzero(pop == "uPN"))
    assert sorted(set(channels.tolist())) == list(range(88))



def test_taste_targets_and_sign(chassis, pop):
    sweet = encode_taste(2.0, chassis)
    bitter = encode_taste(-2.0, chassis)
    pam = pop == "PAM"
    ppl1 = pop == "PPL1"
    assert np.count_nonzero(sweet[pam]) == 316 and np.count_nonzero(sweet[~pam]) == 0
    assert np.all(sweet[pam] > 0.0)
    assert np.count_nonzero(bitter[ppl1]) == 16 and np.count_nonzero(bitter[~ppl1]) == 0
    assert np.all(bitter[ppl1] > 0.0)  # bitter current is positive drive on PPL1


def test_taste_monotonic_magnitude(chassis):
    pnls = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    mags = [np.abs(encode_taste(p, chassis)).max() for p in pnls]
    assert mags == sorted(mags)
    assert all(a < b for a, b in itertools.pairwise(mags))  # strictly monotonic



# --- golden fixtures -------------------------------------------------------


def test_golden_fixtures_byte_reproducible(chassis):
    with np.load(FIXTURES / "vision_golden.npz") as golden:
        assert golden.files == ["chart_flat", "chart_up", "chart_crash"]  # fixed order
        for name, chart in (
            ("chart_flat", charts.chart_flat),
            ("chart_up", charts.chart_up),
            ("chart_crash", charts.chart_crash),
        ):
            got = encode_vision(chart(), chassis)
            exp = golden[name]
            assert got.dtype == exp.dtype
            assert got.tobytes() == exp.tobytes()  # byte-identical arrays
