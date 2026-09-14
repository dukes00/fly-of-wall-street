"""Sensory encoders (T4): market data -> chassis-node currents.

- :func:`encode_vision` — OHLC window -> T4/T5 direction + LC looming currents
  (full-length vector aligned to chassis node order).
- :func:`encode_smell` — ticker identity x market state -> 391 uPN currents.
- :func:`encode_taste` — unrealized P&L -> PAM (sweet) / PPL1 (bitter) currents.

All encoders are pure functions of their inputs: no RNG, no wall clock, no
dict-order dependence — byte-identical outputs across runs by construction
(every derived constant comes from BLAKE2b, see ``senses/_hash.py``).
"""

from __future__ import annotations

from fruitfly.senses.smell import (
    FEATURE_NEUTRAL,
    FEATURES,
    build_features,
    encode_smell,
    identity_profile,
)
from fruitfly.senses.taste import (
    BITTER_GAIN,
    SWEET_GAIN,
    bitter_targets,
    encode_taste,
    sweet_targets,
)
from fruitfly.senses.vision import (
    DIRECTION_GAIN,
    GRID_H,
    GRID_W,
    LOOMING_GAIN,
    encode_vision,
    render_ohlc,
)

__all__ = [
    "BITTER_GAIN",
    "DIRECTION_GAIN",
    "build_features",
    "FEATURE_NEUTRAL",
    "GRID_H",
    "GRID_W",
    "LOOMING_GAIN",
    "SWEET_GAIN",
    "bitter_targets",
    "encode_smell",
    "encode_taste",
    "encode_vision",
    "identity_profile",
    "render_ohlc",
    "sweet_targets",
]
