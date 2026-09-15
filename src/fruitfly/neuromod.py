"""Neuromodulation and dopamine-gated plasticity (DESIGN.md §4, D6).

Internal-state channels mapped to the market (DESIGN §4 table):

=========  =======================  =========================================
Channel    Biological substrate     Market mapping
=========  =======================  =========================================
reward     PAM dopamine neurons     realized profit (sugar)
punishment PPL1 dopamine neurons    realized loss (shock)
hunger     starvation state         portfolio drawdown (raises reward drive)
arousal    octopamine               market volatility (scales gate salience)
sleep      consolidation + decay    market closed (regime forgetting)
=========  =======================  =========================================

PAM/PPL1 are sign-0 (modulatory) nodes: per the T3 LIF contract they
integrate input but never propagate spikes, so no per-neuron DAN activity is
read from spike arrays in v1. ``NeuromodState.reward`` / ``.punishment`` are
the population-level PAM / PPL1 drives derived by the caller (backtest loop)
from realized P&L; this module turns them into the dopamine gate.

Learning: three-factor rule at Kenyon-cell → MBON synapses. Eligibility is a
decaying trace of KC (pre) × MBON (post) co-activity; the dopamine gate
multiplies the trace. Reward gates potentiation of KC→approach-MBON weights
and depression of KC→avoid-MBON weights; punishment does the opposite —
one trial is enough when the gate is strong (biological grudge, not a bug).
Unpunished co-activity habituates: the KC-level habituation factor decays and
shrinks the eligibility contribution (stimuli that predict nothing fade).

KC→MBON weight view (documented once, reused everywhere):
``Plasticity.kc_index[i]`` is the chassis node index of weight-row ``i``;
``Plasticity.mbon_index[j]`` is the chassis node index of weight-column
``j``. Rows/columns follow chassis node order (nodes sorted by ``bodyId``),
so ``weights[i, j]`` is the learned strength of chassis synapse
``adj[kc_index[i], mbon_index[j]]``. ``baseline`` is that structural CSR
submatrix passed through ``log1p`` and scaled to mean 1.0 over its nonzero
support — monotone in synapse count, but the compression tames the heavy
tail (real KC→MBON counts: median 5, max 152) so the O(1) dopamine gate can
move even the strongest pairs. ``support`` (baseline > 0) masks the update —
plasticity never invents synapses the connectome lacks.

MBON valence (approach +1 / avoidance −1) is a documented judgment call: the
connectome annotates no behavioral valence, so valence is read from the
transmitter sign column (excitatory/unknown → approach, inhibitory →
avoidance). 26 of 97 MBONs are glutamatergic; revisit at calibration if the
readout looks inverted (same caveat as the glutamate sign in T2).

Determinism: float64 everywhere, no RNG (sleep's top-k tie-breaking is a
lexicographic sort over chassis node order), fixed op order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fruitfly.connectome import Chassis

__all__ = ["NeuromodState", "Plasticity"]


@dataclass(frozen=True)
class NeuromodState:
    """The five internal-state channels (DESIGN §4).

    ``reward`` / ``punishment`` are unbounded non-negative magnitudes (size of
    the realized profit / loss event). ``hunger``, ``arousal`` and ``sleep``
    are pressures clamped to [0, 1]. All channels default to zero (awake,
    fed, no recent P&L events). ``sleep`` is sleep pressure — informational
    here; consolidation is triggered explicitly via :meth:`Plasticity.sleep`.
    """

    reward: float = 0.0
    punishment: float = 0.0
    hunger: float = 0.0
    arousal: float = 0.0
    sleep: float = 0.0

    def __post_init__(self) -> None:
        for name in ("reward", "punishment", "hunger", "arousal", "sleep"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if name in ("reward", "punishment") and value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if name in ("hunger", "arousal", "sleep"):
                value = min(1.0, max(0.0, value))
            object.__setattr__(self, name, value)


class Plasticity:
    """Dopamine-gated three-factor plasticity over the KC→MBON weight view.

    Parameters (all keyword, deterministic defaults):
        learning_rate: η — dopamine-gated increment per unit eligibility.
        eligibility_decay: per-observe multiplicative decay of the trace.
        habituation_decay: per-observe decay of active-KC habituation factors.
        min_habituation: floor of the habituation factor (a habituated
            stimulus never fades completely; salience resets it).
        hunger_gain / arousal_gain: scale the reward drive / gate salience.
        reward_gain / punishment_gain: scale the two gate drives.
        sleep_decay: fraction of the learned delta retained per sleep.
        top_k: strongest learned associations preserved through sleep.
    """

    def __init__(
        self,
        chassis: Chassis,
        *,
        learning_rate: float = 1.0,
        eligibility_decay: float = 0.5,
        habituation_decay: float = 0.25,
        min_habituation: float = 0.05,
        hunger_gain: float = 1.0,
        arousal_gain: float = 0.5,
        reward_gain: float = 1.0,
        punishment_gain: float = 1.0,
        sleep_decay: float = 0.5,
        top_k: int = 64,
    ) -> None:
        population = chassis.nodes["population"].to_numpy()
        self.kc_index = np.flatnonzero(population == "KC")
        self.mbon_index = np.flatnonzero(population == "MBON")
        if self.kc_index.size == 0 or self.mbon_index.size == 0:
            raise ValueError("chassis has no KC and/or MBON population nodes")
        self._n_neurons = chassis.n_neurons

        # --- KC→MBON weight view (see module docstring for the mapping) ---
        structural = chassis.adj[self.kc_index][:, self.mbon_index].toarray()
        structural = np.asarray(structural, dtype=np.float64)
        support = structural > 0.0
        if not support.any():
            raise ValueError("chassis has no KC→MBON synapses; nothing to learn on")
        self.baseline = np.log1p(structural)
        self.baseline /= self.baseline[support].mean()
        self.support = support

        # MBON valence from transmitter sign (documented judgment call):
        # excitatory/unknown → approach (+1), inhibitory → avoidance (−1).
        sign = chassis.nodes["sign"].to_numpy()[self.mbon_index]
        self.mbon_valence = np.where(np.asarray(sign) < 0, -1.0, 1.0).astype(np.float64)

        # --- tunables ---
        self.learning_rate = float(learning_rate)
        self.eligibility_decay = float(eligibility_decay)
        self.habituation_decay = float(habituation_decay)
        self.min_habituation = float(min_habituation)
        self.hunger_gain = float(hunger_gain)
        self.arousal_gain = float(arousal_gain)
        self.reward_gain = float(reward_gain)
        self.punishment_gain = float(punishment_gain)
        self.sleep_decay = float(sleep_decay)
        self.top_k = int(top_k)

        # --- state ---
        self._weights = np.where(support, self.baseline, 0.0)
        self._eligibility = np.zeros_like(self._weights)
        self._habituation = np.ones(self.kc_index.size, dtype=np.float64)

    # ------------------------------------------------------------------ views
    @property
    def weights(self) -> np.ndarray:
        """Live (n_kc, n_mbon) float64 weight view; row/col mapping above."""
        return self._weights

    @property
    def eligibility(self) -> np.ndarray:
        """Live eligibility trace, same (n_kc, n_mbon) layout as ``weights``."""
        return self._eligibility

    @property
    def habituation(self) -> np.ndarray:
        """Per-KC habituation factors in [min_habituation, 1] (1 = novel)."""
        return self._habituation

    def kc_activity(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Slice a full node-length activity array down to the KC rows."""
        return self._slice(pre_spikes, self.kc_index, "pre_spikes")

    def eligibility_driver(
        self, pre_spikes: np.ndarray, post_spikes: np.ndarray
    ) -> np.ndarray:
        """The exact eligibility increment one presentation contributes.

        Habituation-gated KC (pre) × MBON (post) co-activity, masked to the
        structural support — the same driver ``observe`` accumulates (with
        the per-observe ``eligibility_decay`` applied by the caller, exactly
        as ``observe`` does to the live trace). Pure: reads the current
        habituation factors, mutates NOTHING (not habituation, not the live
        trace), so the caller can compose its own trace.

        The backtest loop composes intraday traces from these drivers
        because the live trace is consumed (zeroed) by ``sleep()`` — it is
        always all-zero intraday, and the per-exit ``observe_trade`` credit
        needs the trace as it stood at the opening encounter.

        Deterministic: float64, no RNG, fixed op order.
        """
        x = self.kc_activity(pre_spikes)
        y = self._slice(post_spikes, self.mbon_index, "post_spikes")
        return (x * self._habituation)[:, None] * y[None, :] * self.support

    # ------------------------------------------------------------------ gate
    def reward_drive(self, state: NeuromodState) -> float:
        """PAM drive: reward scaled up by hunger (drawdown → risk appetite)."""
        return self.reward_gain * state.reward * (1.0 + self.hunger_gain * state.hunger)

    def _gate(self, state: NeuromodState) -> float:
        """Net dopamine signal: PAM reward drive minus PPL1 punishment drive,
        amplified by arousal (octopamine scales salience of both)."""
        d = self.reward_drive(state) - self.punishment_gain * state.punishment
        return d * (1.0 + self.arousal_gain * state.arousal)

    # ---------------------------------------------------------------- observe
    def observe(
        self,
        state: NeuromodState,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> None:
        """One timestep of three-factor learning.

        ``pre_spikes`` / ``post_spikes`` are arrays indexed like the chassis
        nodes (e.g. the per-call spike counts from ``LIFSim.step``). The KC
        slice of ``pre_spikes`` and the MBON slice of ``post_spikes`` form the
        co-activity that accumulates into the eligibility trace; the dopamine
        gate ``d`` (from ``state``) multiplies the *updated* trace, so
        co-activity and a dopamine event in the same ``observe`` learn in one
        trial.
        """
        x = self.kc_activity(pre_spikes)

        d = self._gate(state)
        salient = abs(d) > 0.0

        # Salience (a dopamine event) resets the active KCs' habituation
        # factors before they gate this presentation — novelty wins.
        if salient:
            self._habituation[x > 0.0] = 1.0

        # Eligibility trace: habituation-gated pre×post co-activity, masked to
        # structural support, decaying per observe.
        driver = self.eligibility_driver(pre_spikes, post_spikes)
        eligibility = self.eligibility_decay * self._eligibility + driver

        if salient:
            # Three-factor update: gate × eligibility × MBON valence.
            # reward (d>0): approach weights up / avoid weights down;
            # punishment (d<0): approach down / avoid up.
            delta = (self.learning_rate * d) * (eligibility * self.mbon_valence[None, :])
            delta *= self.support
            self._weights += delta
            np.clip(self._weights, 0.0, None, out=self._weights)  # synapses ≥ 0
        self._eligibility = eligibility
        # Habituation: unpunished/unrewarded repetition fades active KCs
        # (after this presentation has contributed at full remaining strength).
        if not salient:
            self._habituation[x > 0.0] = np.maximum(
                self.min_habituation, self.habituation_decay * self._habituation[x > 0.0]
            )

    def observe_trade(
        self, state: NeuromodState, eligibility_snapshot: np.ndarray
    ) -> None:
        """Precise per-exit credit assignment (DESIGN §7.3, D6).

        The caller snapshots ``eligibility`` at the opening buy/add of a
        position and calls this at the close with a ``state`` built from the
        trade's realized P&L. Applies the SAME three-factor update as
        ``observe`` but with the snapshot as the trace — the dopamine gate
        credits the *opening* encounter, not whatever co-activity happened
        since. WITHOUT advancing the live eligibility trace or touching
        habituation: the snapshot is consumed, the ongoing perception is not.

        The diffuse daily ``observe`` at close-of-day (day realized P&L under
        one gate) remains the DESIGNED ritual; this is the precise
        trade-level complement. Both are bounded by the same gate scale.

        Deterministic: float64, no RNG, fixed op order.
        """
        snapshot = np.asarray(eligibility_snapshot, dtype=np.float64)
        if snapshot.shape != self._eligibility.shape:
            raise ValueError(
                "eligibility_snapshot must match the eligibility trace "
                f"(shape {self._eligibility.shape}), got {snapshot.shape}"
            )
        d = self._gate(state)
        if abs(d) > 0.0:
            # Three-factor update: gate × snapshot × MBON valence.
            delta = (self.learning_rate * d) * (snapshot * self.mbon_valence[None, :])
            delta *= self.support
            self._weights += delta
            np.clip(self._weights, 0.0, None, out=self._weights)  # synapses ≥ 0

    # ------------------------------------------------------------------ sleep
    def sleep(self) -> None:
        """Consolidation + regime forgetting (DESIGN §4, D7).

        Weights decay multiplicatively toward the structural baseline; the
        ``top_k`` strongest learned associations (largest |weight − baseline|,
        ties broken toward the lower chassis node index) are preserved by the
        consolidation pass. The eligibility trace is consumed (consolidated)
        and habituation factors reset — the fly wakes with fresh attention.
        """
        delta = self._weights - self.baseline
        flat_delta = delta.ravel()
        k = min(self.top_k, flat_delta.size)
        protect = np.zeros(flat_delta.size, dtype=bool)
        if k > 0:
            strength = -np.abs(flat_delta)
            order = np.lexsort((np.arange(flat_delta.size), strength))
            protect[order[:k]] = True
        decayed = self.baseline + self.sleep_decay * delta
        self._weights = np.where(
            protect.reshape(delta.shape), self._weights, decayed
        )
        self._eligibility = np.zeros_like(self._weights)
        self._habituation = np.ones_like(self._habituation)

    # ---------------------------------------------------------------- readout
    def mbon_activation(self, pre_spikes: np.ndarray) -> np.ndarray:
        """MBON drive (n_mbon,) elicited by KC activity: ``Wᵀ x``."""
        return self._weights.T @ self.kc_activity(pre_spikes)

    def valence_readout(self, pre_spikes: np.ndarray) -> float:
        """Net approach/avoid score for KC activity: Σ_j valence_j · drive_j.

        Positive → net approach, negative → net avoidance. This is the scalar
        the decision layer thresholds (DESIGN §5); one strong punishment on an
        active KC set flips its sign on re-exposure (one-trial learning).
        """
        return float(self.mbon_valence @ self.mbon_activation(pre_spikes))

    # ----------------------------------------------------------------- intern
    def _slice(self, arr: np.ndarray, index: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(arr, dtype=np.float64)
        if values.shape != (self._n_neurons,):
            raise ValueError(
                f"{name} must be indexed like the chassis nodes "
                f"(shape ({self._n_neurons},)), got {values.shape}"
            )
        return values[index]
