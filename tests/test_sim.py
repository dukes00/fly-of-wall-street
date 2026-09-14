"""T3: LIFSim tests — single-neuron dynamics, sign handling, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fruitfly.connectome import Chassis
from fruitfly.sim import REFRACTORY_MS, SYN_CONDUCTANCE_MV, TAU_M_MS, V_REST, V_TH, LIFSim

NODE_COLS = ["bodyId", "type", "instance", "somaSide", "population", "region",
             "neurotransmitter", "sign"]


def make_chassis(signs: list[int], edges: list[tuple[int, int, int]]) -> Chassis:
    """Synthetic chassis: n nodes with given signs, edges (pre, post, synapses)."""
    n = len(signs)
    nodes = pd.DataFrame(
        {
            "bodyId": np.arange(1000, 1000 + n, dtype=np.int64),
            "type": [f"n{i}" for i in range(n)],
            "instance": [f"n{i}" for i in range(n)],
            "somaSide": ["L"] * n,
            "population": ["test"] * n,
            "region": ["brain"] * n,
            "neurotransmitter": ["unknown"] * n,
            "sign": np.asarray(signs, dtype=np.int64),
        },
        columns=NODE_COLS,
    )
    if edges:
        pre, post, w = zip(*edges, strict=True)
        adj = sp.coo_matrix(
            (np.asarray(w, dtype=np.int64), (pre, post)), shape=(n, n), dtype=np.int64
        ).tocsr()
    else:
        adj = sp.csr_matrix((n, n), dtype=np.int64)
    adj.sum_duplicates()
    return Chassis(nodes=nodes, adj=adj, meta={})


def constant_input(n: int, idx: int | list[int], value: float) -> np.ndarray:
    inp = np.zeros(n)
    inp[np.atleast_1d(idx)] = value
    return inp


# ---------------------------------------------------------------------------
# Single-neuron spiking at threshold
# ---------------------------------------------------------------------------


class TestThresholdSpiking:
    def test_single_neuron_spikes_at_analytic_threshold(self):
        """Isolated neuron with constant drive I crosses V_TH exactly when the
        Euler recurrence v_k = V_REST + I*(1-(1-a)^k) first reaches V_TH."""
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        drive = 20.0
        a = 0.5 / TAU_M_MS
        k = 1
        while V_REST + drive * (1.0 - (1.0 - a) ** k) < V_TH:
            k += 1
        # Margin guard: the analytic crossing must be clean (no ulp ambiguity).
        assert V_REST + drive * (1.0 - (1.0 - a) ** (k - 1)) < V_TH - 1e-3
        assert V_REST + drive * (1.0 - (1.0 - a) ** k) > V_TH + 1e-3

        out = sim.step(constant_input(1, 0, drive), duration_ms=0.5 * (k - 1))
        assert out["total_spikes"] == 0
        out = sim.step(constant_input(1, 0, drive), duration_ms=0.5)
        assert out["spikes"][0] == 1
        assert sim.spikes[0] == 1

    def test_subthreshold_input_never_spikes(self):
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        # Steady state V_REST + 10 = -55 mV < V_TH.
        out = sim.step(constant_input(1, 0, 10.0), duration_ms=500.0)
        assert out["total_spikes"] == 0
        assert sim.spikes[0] == 0

    def test_equilibrium_parks_at_rest_plus_drive(self):
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        sim.step(constant_input(1, 0, 10.0), duration_ms=200.0)
        # 200 ms = 10 tau: residual leak transient ~ 10 * e^-10 mV.
        assert sim._v[0] == pytest.approx(V_REST + 10.0, abs=1e-3)

    def test_refractory_holds_membrane_and_limits_rate(self):
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        # Huge drive -> spikes as fast as refractory allows.
        sim.step(constant_input(1, 0, 1000.0), duration_ms=100.0)
        max_rate = 1000.0 / REFRACTORY_MS  # 500 Hz
        measured = sim.spikes[0] / 0.1
        assert 0 < measured <= max_rate
        # Ceiling: one spike per refractory window, not one per substep.
        assert sim.spikes[0] <= 100.0 / REFRACTORY_MS + 1

    def test_duration_rounds_to_whole_substeps(self):
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        out = sim.step(None, duration_ms=123.0)
        assert out["steps"] == 246
        assert out["duration_ms"] == pytest.approx(123.0)


# ---------------------------------------------------------------------------
# Sign handling: excitation, inhibition, modulatory
# ---------------------------------------------------------------------------


class TestSign:
    DT = 0.5

    def test_excitatory_presynaptic_spike_depolarizes_post(self):
        # 2000 synapses * 0.01 mV = 20 mV drive per presynaptic spike.
        # A one-substep 600 mV kick lifts node 0 exactly to V_TH; its spike
        # propagates and depolarizes the target on the following substep.
        sim = LIFSim(make_chassis([1, 1], [(0, 1, 2000)]), dt_ms=self.DT)
        sim.step(constant_input(2, 0, 600.0), duration_ms=2 * self.DT)
        assert sim.spikes[0] == 1
        # A single quantal PSP of 2000*0.01 = 20 mV lifts the target from
        # rest past V_TH (one substep later): it spikes in turn.
        assert sim.spikes[1] == 1

        # 32 synchronous presynaptic spikes -> quantal jumps sum to +640 mV
        # in one substep -> the postsynaptic neuron crosses threshold.
        n_pre = 32
        volley = np.zeros(n_pre + 1)
        volley[:n_pre] = 600.0
        sim2 = LIFSim(
            make_chassis([1] * (n_pre + 1), [(i, n_pre, 2000) for i in range(n_pre)]),
            dt_ms=self.DT,
        )
        sim2.step(volley, duration_ms=2 * self.DT)
        assert sim2.spikes[:n_pre].sum() == n_pre
        assert sim2.spikes[n_pre] == 1

    def test_inhibitory_input_prevents_spiking(self):
        # A single inhibitory presynaptic neuron firing at its refractory-
        # limited rate (drive 1000 -> a spike every 5 substeps = 400 Hz)
        # delivers quantal jumps of 4000*0.01 = 40 mV every 5 substeps onto
        # a target that also receives a subthreshold direct drive of 12 mV.
        # The 2.5 ms impulse period is far below tau_m, so the target rails
        # far below its -53 mV drive equilibrium: it never spikes.
        inp = np.array([1000.0, 12.0])
        sim = LIFSim(make_chassis([-1, 1], [(0, 1, 4000)]), dt_ms=self.DT)
        sim.step(inp, duration_ms=200.0)
        assert sim.spikes[0] == 80  # exactly one spike per 5-substep cycle
        assert sim.spikes[1] == 0   # target never spikes
        assert sim._v[1] < V_REST + 12.0  # railed far below by inhibition

        # Same circuit and inputs, excitatory pre: the target spikes —
        # inhibition is what prevented it above.
        sim_exc = LIFSim(make_chassis([1, 1], [(0, 1, 4000)]), dt_ms=self.DT)
        out = sim_exc.step(inp, duration_ms=200.0)
        assert out["spikes"][1] > 0

    def test_inhibitory_sign_multiplies_weight(self):
        """Same circuit, sign flipped: target goes the other way."""
        kick = constant_input(2, 0, 600.0)
        exc = LIFSim(make_chassis([1, 1], [(0, 1, 2000)]), dt_ms=self.DT)
        exc.step(kick, duration_ms=2 * self.DT)
        assert exc.spikes[1] == 1  # +20 mV PSP crosses threshold

        inh = LIFSim(make_chassis([-1, 1], [(0, 1, 2000)]), dt_ms=self.DT)
        inh.step(kick, duration_ms=2 * self.DT)
        assert inh.spikes[1] == 0
        # Quantal jump of -20 mV, then one leak step of +a*20 back.
        assert inh._v[1] == pytest.approx(
            V_REST - 20.0 * (1.0 - self.DT / TAU_M_MS)
        )

    def test_sign_zero_integrates_but_does_not_propagate(self):
        ch = make_chassis([0, 1], [(0, 1, 2000)])
        sim = LIFSim(ch, dt_ms=self.DT)
        sim.step(constant_input(2, 0, 600.0), duration_ms=50.0)
        # Modulatory node crosses threshold and is counted...
        assert sim.spikes[0] > 0
        # ...but the target feels nothing at all.
        assert sim.spikes[1] == 0
        assert sim._v[1] == V_REST

    def test_modulatory_rows_absent_from_weight_matrix(self):
        ch = make_chassis([0, 1, -1], [(0, 1, 5), (2, 1, 5)])
        sim = LIFSim(ch, dt_ms=0.5)
        assert sim._w[0].nnz == 0
        assert sim._w[2].data[0] == pytest.approx(-5 * SYN_CONDUCTANCE_MV)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    N = 39  # divisible by 3 for the [1, -1, 0] sign pattern

    def _pattern_input(self, rng_seed: int) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)  # test-owned RNG, not the engine's
        return rng.uniform(0.0, 40.0, size=self.N)

    def _run(self, seed: int, rng_seed: int, n_calls: int = 20):
        ch = make_chassis(
            [1, -1, 0] * (self.N // 3),
            [(i, (i * 7 + 3) % self.N, (i % 11) * 97) for i in range(self.N)],
        )
        sim = LIFSim(ch, dt_ms=0.5, seed=seed)
        log = []
        for t in range(n_calls):
            inp = self._pattern_input(rng_seed + t)
            out = sim.step(inp, duration_ms=5.0)
            log.append(out["spikes"].tobytes())
        return sim.spikes.tobytes(), b"".join(log)

    def test_same_seed_double_run_byte_identical(self):
        log_a = self._run(seed=0, rng_seed=42)
        log_b = self._run(seed=0, rng_seed=42)
        assert log_a == log_b

    def test_engine_has_no_rng_different_seeds_identical(self):
        """The engine consults no RNG: seed only changes bookkeeping, so runs
        under different seeds must produce identical spike logs."""
        assert self._run(seed=0, rng_seed=7) == self._run(seed=12345, rng_seed=7)

    def test_reset_restores_identical_dynamics(self):
        ch = make_chassis([1, -1, 0] * (self.N // 3), [])
        sim = LIFSim(ch, dt_ms=0.5)
        inp = self._pattern_input(3)
        first = [sim.step(inp, duration_ms=5.0)["spikes"].tobytes() for _ in range(10)]
        cum_first = sim.spikes.tobytes()
        sim.reset(seed=99)
        second = [sim.step(inp, duration_ms=5.0)["spikes"].tobytes() for _ in range(10)]
        assert first == second
        assert sim.spikes.tobytes() == cum_first

    def test_dense_matvec_path_matches_row_loop_path(self, monkeypatch):
        """The >_ROW_LOOP_MAX fallback must agree with the row loop exactly."""
        import fruitfly.sim as sim_mod

        n_pre = 1100
        signs = [1] * (n_pre + 1)
        edges = [(i, n_pre, 2000) for i in range(n_pre)]
        volley = np.zeros(n_pre + 1)
        volley[:n_pre] = 600.0

        # Default: 1100 spiked rows > _ROW_LOOP_MAX -> dense matvec path.
        dense = LIFSim(make_chassis(signs, edges), dt_ms=0.5)
        out_dense = dense.step(volley, duration_ms=1.0)  # spike + propagation
        assert dense.spikes[:n_pre].sum() == n_pre
        assert out_dense["spikes"][n_pre] == 1

        # Force the row-loop path on the identical input; results must be
        # bit-identical (both sum presynaptic contributions in ascending
        # row order).
        monkeypatch.setattr(sim_mod, "_ROW_LOOP_MAX", 10_000)
        row = LIFSim(make_chassis(signs, edges), dt_ms=0.5)
        out_row = row.step(volley, duration_ms=1.0)  # spike + propagation
        assert out_row["spikes"].tobytes() == out_dense["spikes"].tobytes()
        assert row._v.tobytes() == dense._v.tobytes()


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class TestApi:
    def test_step_returns_contract_dict(self):
        sim = LIFSim(make_chassis([1, 1], [(0, 1, 10)]), dt_ms=0.5)
        out = sim.step(np.array([20.0, 0.0]), duration_ms=10.0)
        assert set(out) >= {"spikes", "total_spikes", "mean_rate_hz"}
        assert out["spikes"].shape == (2,)
        assert out["spikes"].dtype == np.int64
        assert out["total_spikes"] == int(out["spikes"].sum())

    def test_none_input_runs(self):
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        out = sim.step(None, duration_ms=10.0)
        assert out["total_spikes"] == 0

    def test_bad_input_shape_raises(self):
        sim = LIFSim(make_chassis([1, 1], []), dt_ms=0.5)
        with pytest.raises(ValueError):
            sim.step(np.zeros(3), duration_ms=1.0)

    def test_bad_dt_raises(self):
        with pytest.raises(ValueError):
            LIFSim(make_chassis([1], []), dt_ms=0.0)

    def test_spikes_property_is_cumulative(self):
        sim = LIFSim(make_chassis([1], []), dt_ms=0.5)
        sim.step(constant_input(1, 0, 30.0), duration_ms=10.0)
        c1 = sim.spikes.copy()
        sim.step(constant_input(1, 0, 30.0), duration_ms=10.0)
        assert (sim.spikes >= c1).all()
        assert sim.spikes.sum() > c1.sum()
