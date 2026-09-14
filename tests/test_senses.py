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
import pytest

from fruitfly.senses import (
    encode_smell,
    encode_taste,
    encode_vision,
    identity_profile,
)

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
    assert crash.max() > 0.5 * 40.0  # strongly fired, near saturation


def test_vision_deterministic(chassis):
    a = encode_vision(charts.chart_crash(), chassis)
    b = encode_vision(charts.chart_crash(), chassis)
    assert a.tobytes() == b.tobytes()


def test_vision_rejects_bad_windows(chassis):
    with pytest.raises(ValueError):
        encode_vision(charts.chart_flat()[["open", "high", "low"]], chassis)
    with pytest.raises(ValueError):
        encode_vision(charts.chart_flat().iloc[:0], chassis)


def test_vision_long_window_uses_last_48_bars(chassis):
    long = charts.chart_up(n_bars=200)
    assert np.array_equal(
        encode_vision(long, chassis), encode_vision(long.iloc[-48:], chassis)
    )


# --- smell -----------------------------------------------------------------


_TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "AMZN", "META", "AMD", "NFLX", "SPY"]

_NEUTRAL = {"returns": 0.0, "rsi": 50.0, "volatility": 0.0, "volume_delta": 0.0}


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
        "AAPL", {"returns": 0.05, "rsi": 80.0, "volatility": 0.03, "volume_delta": 0.4},
        chassis,
    )
    assert not np.array_equal(base, hot)
    # missing features are neutral: subset == full dict with zeros
    assert np.array_equal(
        encode_smell("AAPL", {"volatility": 0.03}, chassis),
        encode_smell("AAPL", {**_NEUTRAL, "volatility": 0.03}, chassis),
    )


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
