"""Tests for the smell identity-attenuation knob (TRAINING2-SPEC §3, A2).

``encode_smell(..., id_scale=...)`` scales the identity ``base[channels]``
term only; the state modulator is unchanged. Default 1.0 must be byte-
identical to the incumbent output; 0.0 zeroes the identity glomeruli (and,
the state entering purely multiplicatively, the whole output).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fruitfly.senses import encode_smell, identity_profile
from fruitfly.senses.smell import upn_channels

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import charts  # noqa: E402

_TICKERS = ["AAPL", "MSFT", "TSLA", "SPY"]

_NEUTRAL = {
    "returns": 0.0,
    "rsi": 50.0,
    "volatility": 0.0,
    "volume_delta": 0.0,
    "mom20": 0.0,
    "slope20": 0.0,
}
_HOT = {"returns": 0.05, "rsi": 80.0, "volatility": 0.03,
        "volume_delta": 0.4, "mom20": 0.1, "slope20": 0.002}


@pytest.fixture(scope="module")
def chassis():
    return charts.make_chassis()


def test_id_scale_default_is_byte_identical_noop(chassis):
    for ticker in _TICKERS:
        for features in (_NEUTRAL, _HOT):
            assert encode_smell(ticker, features, chassis, id_scale=1.0).tobytes() == (
                encode_smell(ticker, features, chassis).tobytes()
            )


def test_id_scale_zero_zeroes_identity(chassis):
    # identity channels gone — and with the state purely multiplicative on
    # the identity, the whole current vector collapses to zero
    for ticker in _TICKERS:
        for features in (_NEUTRAL, _HOT):
            out = encode_smell(ticker, features, chassis, id_scale=0.0)
            assert np.all(out == 0.0)


def test_id_scale_scales_identity_leaving_state_modulator_exact(chassis):
    # with all-neutral features the modulator is exactly 1, so the output is
    # exactly the scaled identity profile
    _, channels = upn_channels(chassis)
    base = identity_profile("AAPL")
    assert np.array_equal(
        encode_smell("AAPL", _NEUTRAL, chassis, id_scale=0.5),
        base[channels] * 0.5,
    )
    # with state features the output scales proportionally with id_scale
    full = encode_smell("AAPL", _HOT, chassis)
    half = encode_smell("AAPL", _HOT, chassis, id_scale=0.5)
    nonzero = full != 0.0
    assert np.allclose(half[nonzero], 0.5 * full[nonzero], rtol=1e-12, atol=0.0)
    assert not np.array_equal(full, half)
