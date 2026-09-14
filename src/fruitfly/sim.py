"""T3: leaky integrate-and-fire engine over the connectome chassis.

Recipe (DESIGN §2): one LIF compartment per neuron, weights from raw synapse
counts scaled by a single conductance constant, excitatory/inhibitory sign
from the presynaptic neuron's transmitter sign column. Propagation is a sparse
matvec against the chassis CSR adjacency; the whole population is updated
vectorized each substep. Seeded but RNG-free: the engine contains no
stochasticity — inputs come only from the caller's ``input_current`` vector
(noise is the loop layer's job), so identical calls always produce identical
spike logs and different seeds cannot diverge.

Membrane equation (explicit Euler, dt << tau_m):

    v[t+1] = v[t] + (dt/tau_m) * (V_REST + I_ext[t] - v[t]) + Σ_i s_i[t] * w[i, j]

``I_ext`` is the caller's input current: a tonic drive in mV-above-rest units
that parks the membrane at ``V_REST + I_ext``. Synapses are quantal: a
presynaptic spike at substep t raises the postsynaptic membrane instantly by
``w[i, j] = sign[i] * SYN_CONDUCTANCE_MV * synapses[i, j]`` mV at substep
t+1 (one-substep axonal delay); leak then relaxes it toward rest. The
quantal PSP is therefore ~``w`` mV regardless of ``dt_ms``, and a tonic
presynaptic rate ``r`` shifts the target's equilibrium by ``r * w * tau_m``.

Modulatory neurons (``sign == 0``: dopaminergic PAM/PPL1, histaminergic
photoreceptors, unresolved) integrate current and cross threshold like any
other node — their spikes are counted — but their outgoing weights are zero,
so they never propagate into the LIF layer; the neuromodulation layer (T5)
reads their activity instead.
"""

from __future__ import annotations

import numpy as np

from fruitfly.connectome import Chassis

#: Postsynaptic drive (mV) delivered per synapse-count unit per presynaptic
#: spike. Single documented gain constant turning "number of synapses" into
#: membrane drive; the only calibration knob between raw connectome and LIF.
SYN_CONDUCTANCE_MV = 0.01

#: Resting = reset potential (mV). A spike clamps the membrane back here.
V_REST = -65.0
V_RESET = -65.0
#: Spike threshold (mV).
V_TH = -50.0
#: Membrane time constant (ms).
TAU_M_MS = 20.0
#: Absolute refractory period (ms): membrane held at V_RESET, no spiking.
REFRACTORY_MS = 2.0

#: Up to this many neurons spiking in one substep, propagation uses an exact
#: row-slice accumulation over the spiked rows (cheaper than a full matvec
#: when activity is sparse); above it, a dense CSR matvec. Both paths are
#: deterministic, so the switch cannot break run-to-run reproducibility.
_ROW_LOOP_MAX = 1024


class LIFSim:
    """Vectorized LIF simulation over a :class:`~fruitfly.connectome.Chassis`.

    Parameters
    ----------
    chassis:
        Node table + pre→post CSR adjacency (weight = synapse count).
    dt_ms:
        Integration substep in ms. ``duration_ms`` passed to :meth:`step` is
        rounded to a whole number of substeps.
    seed:
        Stored for API compatibility with the loop layer. The engine is fully
        deterministic — no RNG is consulted anywhere — so any seed yields
        byte-identical dynamics.
    """

    def __init__(self, chassis: Chassis, dt_ms: float = 0.5, seed: int = 0):
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        self.chassis = chassis
        self.dt_ms = float(dt_ms)
        self.seed = int(seed)

        n = chassis.n_neurons
        self._n = n
        self._a = self.dt_ms / TAU_M_MS
        self._refr_steps = max(1, int(round(REFRACTORY_MS / self.dt_ms)))

        # Effective weight: w[i, j] = sign[i] * SYN_CONDUCTANCE_MV * synapses[i, j].
        # Sign belongs to the presynaptic neuron (its transmitter). sign == 0
        # rows vanish, which is exactly "modulatory nodes do not propagate".
        sign = chassis.nodes["sign"].to_numpy()
        w = chassis.adj.multiply(sign[:, None]).tocsr().astype(np.float64)
        w.data *= SYN_CONDUCTANCE_MV
        w.eliminate_zeros()  # sign-0 rows vanish outright, not as explicit 0s
        w.sort_indices()
        self._w = w
        self._w_indptr = w.indptr
        self._w_indices = w.indices
        self._w_data = w.data

        self._v = np.full(n, V_REST, dtype=np.float64)
        self._refr = np.zeros(n, dtype=np.int64)
        # Spike indicator of the previous substep, float64 for the matvec.
        self._prev = np.zeros(n, dtype=np.float64)
        # Scratch buffers, preallocated to keep the substep loop alloc-free.
        self._scratch = np.zeros(n, dtype=np.float64)
        self._syn = np.zeros(n, dtype=np.float64)
        self._zero_input = np.zeros(n, dtype=np.float64)

        self._spikes = np.zeros(n, dtype=np.int64)

    # ------------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def spikes(self) -> np.ndarray:
        """Per-node cumulative spike counts (int64, chassis node order)."""
        return self._spikes

    def reset(self, seed: int | None = None) -> None:
        """Zero all state; optionally update the (inert) seed."""
        if seed is not None:
            self.seed = int(seed)
        self._v.fill(V_REST)
        self._refr.fill(0)
        self._prev.fill(0.0)
        self._spikes.fill(0)

    # ------------------------------------------------------------------
    def step(
        self, input_current: np.ndarray | None, duration_ms: float
    ) -> dict:
        """Advance the simulation by ``duration_ms`` of sim time.

        Parameters
        ----------
        input_current:
            Per-node drive in mV-above-rest units (see module docstring),
            indexed like ``chassis.nodes``. ``None`` means no external input.
            Held constant across the substeps of this call.
        duration_ms:
            Simulated time; rounded to a whole number of ``dt_ms`` substeps.

        Returns
        -------
        dict with ``spikes`` (int64 per-node spike counts for this call only),
        ``total_spikes``, ``steps``, ``duration_ms`` (simulated) and
        ``mean_rate_hz`` (population mean over this call).
        """
        n = self._n
        if input_current is None:
            inp = self._zero_input
        else:
            inp = np.ascontiguousarray(input_current, dtype=np.float64)
            if inp.shape != (n,):
                raise ValueError(
                    f"input_current shape {inp.shape} != ({n},)"
                )
        steps = int(round(duration_ms / self.dt_ms))
        if steps < 0:
            raise ValueError("duration_ms must be non-negative")

        step_counts = np.zeros(n, dtype=np.int64)
        self._run_steps(inp, steps, step_counts)

        total = int(step_counts.sum())
        sim_ms = steps * self.dt_ms
        rate = total / (n * sim_ms / 1000.0) if sim_ms > 0 else 0.0
        return {
            "spikes": step_counts,
            "total_spikes": total,
            "steps": steps,
            "duration_ms": sim_ms,
            "mean_rate_hz": rate,
        }

    # ------------------------------------------------------------------
    def _run_steps(
        self, inp: np.ndarray, steps: int, step_counts: np.ndarray
    ) -> None:
        """Substep loop; touches only preallocated buffers."""
        a = self._a
        v = self._v
        refr = self._refr
        prev = self._prev
        syn = self._syn
        scratch = self._scratch
        spikes = self._spikes
        refr_steps = self._refr_steps
        w_indptr = self._w_indptr
        w_indices = self._w_indices
        w_data = self._w_data
        v_reset = V_RESET
        v_th = V_TH

        for _ in range(steps):
            # 1. Propagate last substep's spikes through the signed CSR.
            spiked_prev = np.flatnonzero(prev)
            if spiked_prev.size:
                if spiked_prev.size <= _ROW_LOOP_MAX:
                    # Exact row-slice accumulation, ascending row order.
                    syn.fill(0.0)
                    for i in spiked_prev:
                        s = w_indptr[i]
                        e = w_indptr[i + 1]
                        syn[w_indices[s:e]] += w_data[s:e] * prev[i]
                else:
                    syn[:] = prev @ self._w
            else:
                syn.fill(0.0)

            # 2. Deliver quantal PSPs (instant membrane jump), then leak-
            #    integrate the external drive: v += a*(V_REST + I_ext - v).
            v += syn
            np.subtract(v_reset, v, out=scratch)
            scratch += inp
            scratch *= a
            v += scratch

            # 3. Refractory clamp (membrane held at reset, no spiking).
            idx_refr = np.flatnonzero(refr)
            if idx_refr.size:
                v[idx_refr] = v_reset

            # 4. Threshold crossing.
            spiked_idx = np.flatnonzero(v >= v_th)
            if spiked_idx.size:
                v[spiked_idx] = v_reset
                refr[spiked_idx] = refr_steps
                step_counts[spiked_idx] += 1
                spikes[spiked_idx] += 1

            # 5. Bookkeeping for the next substep.
            prev.fill(0.0)
            prev[spiked_idx] = 1.0
            if idx_refr.size:
                refr[idx_refr] -= 1



__all__ = [
    "LIFSim",
    "SYN_CONDUCTANCE_MV",
    "V_REST",
    "V_RESET",
    "V_TH",
    "TAU_M_MS",
    "REFRACTORY_MS",
]

