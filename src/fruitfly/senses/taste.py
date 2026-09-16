"""Taste sense: unrealized P&L of the open position as sweet / bitter gustatory
input (DESIGN §3, D15 — the only sense representing what the fly owns).

Documented chassis targets (population column of the node table):

- **Sweet** (sugar) -> the **PAM cluster** (316 reward dopaminergic neurons,
  PAM01-PAM15). Positive unrealized P&L drives PAM.
- **Bitter** (shock) -> the **PPL1 cluster** (16 avoidance dopaminergic
  neurons, PPL101-PPL108). Negative unrealized P&L drives PPL1.

PAM/PPL1 have transmitter sign 0 by design: they do not propagate generic LIF
spikes; their activity is read by the neuromodulation layer (T5) as the
dopamine gate for KC->MBON plasticity. Taste current is therefore the circuit's
way of *poking* the dopaminergic clusters, not of driving downstream LIF.

**Dead channel today (TRAINING2-SPEC A5, §5.1):** PAM/PPL1 are sign-0 nodes —
they integrate input and never propagate spikes (see neuromod.py:15-19) — and
the backtest loop reads no PAM/PPL1 spike activity, so the taste currents this
encoder drives into those rows are readout-inert: taste cannot influence any
decision today. The D15 sweet/bitter -> PAM/PPL1 mapping is acknowledged as a
documented dead channel pending a D15 revisit (exit forecasting is handled
explicitly by the T-1 exit-forecast gate instead).

Magnitude: ``gain * tanh(|pnl|)`` — exactly 0 at flat, strictly monotonic in
|pnl|, saturating at extreme moves. Per-node gains carry a deterministic
BLAKE2b-derived jitter in [0.8, 1.2] (ordering preserved: monotonic in |pnl|
at every target node, since tanh >= 0 and the jitter is positive).

Output: float64 array aligned to ``chassis.nodes`` order, zero everywhere
except the sweet (PAM, pnl > 0) or bitter (PPL1, pnl < 0) target rows. Pure
and deterministic.
"""

from __future__ import annotations

import numpy as np

from fruitfly.senses._hash import stable_uniform

#: Current (LIFSim input units) at |pnl| -> saturation, per target node.
SWEET_GAIN = 50.0
BITTER_GAIN = 50.0


def sweet_targets(chassis) -> np.ndarray:
    """Row indices of the sweet (PAM) target nodes, in chassis order."""
    return np.flatnonzero(chassis.nodes["population"].to_numpy() == "PAM")


def bitter_targets(chassis) -> np.ndarray:
    """Row indices of the bitter (PPL1) target nodes, in chassis order."""
    return np.flatnonzero(chassis.nodes["population"].to_numpy() == "PPL1")


def encode_taste(unrealized_pnl_pct: float, chassis) -> np.ndarray:
    """Encode unrealized P&L (%) into sweet/bitter currents.

    ``unrealized_pnl_pct`` > 0 drives PAM (sweet); < 0 drives PPL1 (bitter);
    exactly 0.0 returns an all-zero vector. Returns float64 aligned to
    ``chassis.nodes`` order. Pure and deterministic.
    """
    pnl = float(unrealized_pnl_pct)
    out = np.zeros(chassis.n_neurons, dtype=np.float64)
    if pnl == 0.0:
        return out

    magnitude = abs(float(np.tanh(pnl)))  # monotonic in |pnl|, saturating
    if pnl > 0.0:
        rows = sweet_targets(chassis)
        gain = SWEET_GAIN
    else:
        rows = bitter_targets(chassis)
        gain = BITTER_GAIN
    for i, body_id in enumerate(chassis.nodes["bodyId"].to_numpy()[rows]):
        jitter = stable_uniform(f"taste-gain:{int(body_id)}", 0.8, 1.2)
        out[rows[i]] = gain * jitter * magnitude
    return out
