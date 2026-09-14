"""Vision sense: an OHLC candlestick window drives T4/T5 direction cells and
LC looming cells directly.

The stripped chassis has **zero** photoreceptor -> T4/T5 synapses (lamina and
medulla relays are excluded; see reports/t2-connectome.md), so this encoder
plays the role of the excluded optic-lobe circuitry: it computes motion and
looming itself and injects current straight into the downstream targets.

Render model (fixed, documented):

- The window is rendered to an ommatidia-resolution grayscale grid of
  ``GRID_W x GRID_H`` = 48 x 24 pixels (48 time columns, 24 price rows),
  float64 in [0, 1]. If the window has more than 48 bars, the **last** 48 are
  used; fewer bars are right-aligned (most recent at the right edge), with
  zero padding on the left.
- The vertical price scale spans [min(low), max(high)] over the rendered
  window. Prices map to rows by nearest-neighbor rounding: row =
  round((price - lo) / span * (GRID_H - 1)), clipped to [0, GRID_H - 1].
- Candle bodies (open..close) are drawn at intensity 1.0, wicks (high, low)
  at 0.5. A flat window (span == 0) renders every candle on the middle row.

Derived drive:

- **Direction (T4 + T5, 6,865 + 6,720 = 13,585 cells).** Per-candle drift
  ``m_j = (close_j - open_j) / span`` in [-1, 1] (0 when span == 0 or the
  column is padding). Every T4/T5 cell is deterministically assigned, from
  BLAKE2b of its ``bodyId``, a preferred direction (+1 up / -1 down), a
  receptive-field anchor column, and a spread (1-3 columns, triangle kernel
  normalized over the full width). Cell current = ``DIRECTION_GAIN * pref *
  sum_j w_j * m_j`` — ON cells depolarize on up-drift, OFF cells on down.
- **Looming (LC, 1,239 cells: LC4/LC21/LC10).** Vertical range expansion
  ``expansion = r_last - mean(r_earlier)``, where ``r_j =
  (high_j - low_j) / span``. Each LC cell has a BLAKE2b-derived firing
  threshold in [0.1, 0.5]; current = ``LOOMING_GAIN * max(0, expansion - t) /
  (1 - t)``. Silent on flat (expansion 0), fires on any vertical spike
  (crash or rally).

Output: float64 array aligned to ``chassis.nodes`` order (length
``len(chassis.nodes)``), zero everywhere except the T4/T5/LC rows. Units match
``LIFSim.step`` input currents (arbitrary, calibrated by the gain constants).
All computation is pure float64 numpy with fixed op order — byte-identical
across runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fruitfly.senses._hash import stable_u64, stable_uniform

#: Ommatidia-resolution render: 48 time columns x 24 price rows.
GRID_W = 48
GRID_H = 24

#: Current (LIFSim input units) per unit normalized drift, at full kernel weight.
DIRECTION_GAIN = 30.0
#: Current at full looming saturation (expansion == 1).
LOOMING_GAIN = 40.0

_BODY_COLS = ("open", "high", "low", "close")


def _ohlc(o: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pull OHLC columns case-insensitively; validate and return float64 arrays."""
    cols: dict[str, np.ndarray] = {}
    lower = {str(c).lower(): c for c in o.columns}
    for name in _BODY_COLS:
        if name not in lower:
            raise ValueError(f"OHLC window missing column {name!r}; got {list(o.columns)}")
        cols[name] = o[lower[name]].to_numpy(dtype=np.float64)
    n = cols["open"].shape[0]
    if n == 0:
        raise ValueError("OHLC window is empty")
    return cols


def render_ohlc(o: pd.DataFrame) -> np.ndarray:
    """Render an OHLC window to the fixed 48 x 24 grayscale grid (see module doc).

    Returns float64 array of shape ``(GRID_H, GRID_W)``, most recent candle in
    the rightmost occupied column.
    """
    cols = _ohlc(o)
    n = min(cols["open"].shape[0], GRID_W)
    lo = float(cols["low"][-n:].min())
    hi = float(cols["high"][-n:].max())
    span = hi - lo
    mid = (GRID_H - 1) // 2

    def row(price: float) -> int:
        if span <= 0.0:
            return mid
        return int(min(max(round((price - lo) / span * (GRID_H - 1)), 0), GRID_H - 1))

    grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    for j in range(n):
        k = GRID_W - n + j  # right-aligned: most recent candle at the right edge
        r_open, r_close = row(cols["open"][-n + j]), row(cols["close"][-n + j])
        r_hi, r_lo = row(cols["high"][-n + j]), row(cols["low"][-n + j])
        grid[min(r_open, r_close) : max(r_open, r_close) + 1, k] = 1.0
        for r in (r_hi, r_lo):
            if grid[r, k] == 0.0:
                grid[r, k] = 0.5
    return grid


def _per_column_signals(o: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-column drift m_j in [-1, 1] and range r_j in [0, 1], length GRID_W.

    Columns beyond the (right-aligned) window are zero. drift and range are 0
    on a degenerate flat window (span == 0).
    """
    cols = _ohlc(o)
    n = min(cols["open"].shape[0], GRID_W)
    lo = float(cols["low"][-n:].min())
    hi = float(cols["high"][-n:].max())
    span = hi - lo

    drift = np.zeros(GRID_W, dtype=np.float64)
    rng = np.zeros(GRID_W, dtype=np.float64)
    for j in range(n):
        k = GRID_W - n + j
        if span > 0.0:
            drift[k] = (cols["close"][-n + j] - cols["open"][-n + j]) / span
            rng[k] = (cols["high"][-n + j] - cols["low"][-n + j]) / span
    return drift, rng


def _direction_currents(body_ids: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Hash-assigned ON/OFF receptive fields over the 13,585 T4+T5 cells."""
    currents = np.empty(body_ids.size, dtype=np.float64)
    for i, body_id in enumerate(body_ids):
        key = f"vision-direction:{int(body_id)}"
        pref = 1.0 if stable_u64(key + ":pref") & 1 else -1.0
        anchor = stable_u64(key + ":anchor") % GRID_W
        spread = 1 + stable_u64(key + ":spread") % 3
        # Triangle kernel centered on the anchor column, normalized over the
        # full width (padding columns carry m_j = 0 and dilute accordingly).
        d = np.abs(np.arange(GRID_W, dtype=np.float64) - anchor)
        w = np.maximum(0.0, 1.0 - d / spread)
        w /= w.sum()
        currents[i] = DIRECTION_GAIN * pref * float(np.dot(w, drift))
    return currents


def _looming_currents(body_ids: np.ndarray, expansion: float) -> np.ndarray:
    """Thresholded range-expansion drive into the 1,239 LC looming cells."""
    currents = np.empty(body_ids.size, dtype=np.float64)
    for i, body_id in enumerate(body_ids):
        t = stable_uniform(f"vision-looming:{int(body_id)}", 0.1, 0.5)
        over = expansion - t
        currents[i] = LOOMING_GAIN * (over / (1.0 - t)) if over > 0.0 else 0.0
    return currents


def encode_vision(o: pd.DataFrame, chassis) -> np.ndarray:
    """Encode an OHLC window into T4/T5 direction + LC looming currents.

    Returns a float64 array aligned to ``chassis.nodes`` order; nonzero only at
    T4, T5 (direction cells) and LC-looming (looming cells) rows. Pure and
    deterministic: same inputs give byte-identical outputs.
    """
    drift, rng = _per_column_signals(o)
    n_valid = int(np.count_nonzero((drift != 0.0) | (rng != 0.0)))
    expansion = float(rng[-1]) - float(rng[:-1].mean()) if n_valid >= 2 else 0.0

    out = np.zeros(chassis.n_neurons, dtype=np.float64)
    pop = chassis.nodes["population"].to_numpy()
    body_ids = chassis.nodes["bodyId"].to_numpy()

    direction_mask = (pop == "T4") | (pop == "T5")
    out[direction_mask] = _direction_currents(body_ids[direction_mask], drift)
    looming_mask = pop == "LC-looming"
    out[looming_mask] = _looming_currents(body_ids[looming_mask], expansion)
    return out
