"""Smell sense: a ticker + market-state features mixed as a virtual odor across
the 88 antennal-lobe glomeruli, delivered by the 391 uniglomerular uPNs.

Two components (DESIGN §3, D5):

- **Identity** — a fixed 88-glomerular base profile derived from BLAKE2b of
  the ticker string ("AAPL smells like AAPL"): constant per ticker for the
  ticker's lifetime, pairwise-distinct across tickers. Profile values are
  uniform in [0, 1).
- **State** — market features (returns, RSI, volatility, volume_delta) scale
  the identity multiplicatively, per glomerulus. Each (feature, channel) pair
  has a BLAKE2b-derived weight in [-0.25, 0.25]; the per-channel modulator is
  the product of ``(1 + w * s)`` over the four features, where each feature is
  squashed to [-1, 1] first (tanh after a documented linear scale: returns
  x10, RSI centered at 50 over +/-25, volatility x25, volume_delta x2). A
  missing feature is treated as 0.0 (neutral). With all-neutral features the
  modulator is exactly 1 and the output *is* the identity signature.

Glomeruli -> uPNs: each uPN node in the chassis carries its glomerulus in its
type name (``DA1_lPN`` -> ``DA1``; the same suffix rule the T2 extractor
uses); the channel index is the glomerulus's position in
``chassis.meta["glomeruli"]`` (88 channels, including "" for the 4 null-type
uPNs). A uPN fires its channel's base profile x modulator. Deterministic
fallback: if a uPN's glomerulus is missing from the meta list the encoder
raises — the mapping is data, not a choice made at encode time.

Output: float64 array of 391 dims, aligned to the uPN nodes **in chassis node
order** (``chassis.nodes[population == "uPN"]``). Nonnegative (identity >= 0,
modulator >= 0.75**4 > 0). Pure and deterministic.
"""

from __future__ import annotations

import math
import re

import numpy as np

from fruitfly.senses._hash import _MASK64, stable_u64, stable_uniform

#: Same uPN-type suffix rule as the T2 extractor ("DA1_lPN" -> "DA1").
_GLOM_RE = re.compile(r"_(adPN|lPN|vPN|lvPN)\d*$")

#: Fixed feature order (iteration order everywhere — never dict order).
FEATURES = ("returns", "rsi", "volatility", "volume_delta")

#: Linear pre-scale per feature before the tanh squash to [-1, 1].
FEATURE_SCALE = {
    "returns": 10.0,  # 10% move saturates
    "rsi": 1.0 / 25.0,  # (rsi - 50) / 25: RSI 75 -> +1
    "volatility": 25.0,  # 4% realized vol saturates
    "volume_delta": 2.0,  # 50% relative volume change saturates
}

#: Weight range per (feature, glomerulus) modulator term.
_MOD_WEIGHT = 0.25

#: Neutral (identity-preserving) value per feature when a key is missing.
FEATURE_NEUTRAL = {"returns": 0.0, "rsi": 50.0, "volatility": 0.0, "volume_delta": 0.0}

#: Number of glomerular channels (matches chassis.meta["glomeruli"]).
_N_CHANNELS = 88


def _rsi(value: float) -> float:
    """RSI enters on the 0-100 scale; recentre to [-1, 1] before squashing."""
    return (value - 50.0) * FEATURE_SCALE["rsi"]


def identity_profile(ticker: str) -> np.ndarray:
    """Fixed 88-glomerular base profile for a ticker (constant, pairwise-distinct).

    Uniform in [0, 1): one BLAKE2b uint64 per channel,
    ``stable_u64("smell-identity:<ticker>:<channel>")`` scaled to [0, 1).
    Deterministic across processes and platforms.
    """
    return np.array(
        [stable_u64(f"smell-identity:{ticker}:{i}") / _MASK64 for i in range(_N_CHANNELS)],
        dtype=np.float64,
    )


def _squash(feature: str, value: float) -> float:
    scaled = _rsi(value) if feature == "rsi" else value * FEATURE_SCALE[feature]
    return math.tanh(scaled)


def upn_channels(chassis) -> tuple[np.ndarray, np.ndarray]:
    """(uPN row indices in chassis order, glomerulus channel index per uPN)."""
    meta_gloms = list(chassis.meta["glomeruli"])
    channel_of = {g: i for i, g in enumerate(meta_gloms)}
    nodes = chassis.nodes
    upn_rows = np.flatnonzero(nodes["population"].to_numpy() == "uPN")
    types = nodes["type"].to_numpy()[upn_rows]
    channels = np.empty(upn_rows.size, dtype=np.int64)
    for i, t in enumerate(types):
        name = "" if t is None or (isinstance(t, float) and math.isnan(t)) else str(t)
        glom = _GLOM_RE.sub("", name)
        if glom not in channel_of:
            raise ValueError(
                f"uPN type {t!r} -> glomerulus {glom!r} not in chassis.meta['glomeruli']"
            )
        channels[i] = channel_of[glom]
    return upn_rows, channels


def encode_smell(ticker: str, features: dict, chassis) -> np.ndarray:
    """Encode ticker identity x market state into 391 uPN currents.

    ``features`` maps (a subset of) ``FEATURES`` to floats — returns as a
    fraction, RSI on the 0-100 scale, volatility as a std-dev fraction,
    volume_delta as a relative change. A missing key takes its neutral value
    (``FEATURE_NEUTRAL``: 0.0 for returns/volatility/volume_delta, 50.0 for
    RSI) and leaves the identity signature untouched.
    Returns float64, shape (n_uPN,) = (391,), aligned to uPN nodes in chassis
    order. Pure and deterministic.
    """
    upn_rows, channels = upn_channels(chassis)
    base = identity_profile(ticker)

    squashed = {
        f: _squash(f, float(features.get(f, FEATURE_NEUTRAL[f]))) for f in FEATURES
    }
    modulator = np.ones(channels.size, dtype=np.float64)
    for f in FEATURES:
        s = squashed[f]
        if s == 0.0:
            continue  # exact identity preservation on neutral features
        for k, ch in enumerate(channels):
            w = stable_uniform(f"smell-mod:{f}:{int(ch)}", -_MOD_WEIGHT, _MOD_WEIGHT)
            modulator[k] *= 1.0 + w * s
    return base[channels] * modulator
