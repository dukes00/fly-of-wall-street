"""Tests for the T5 neuromodulation / plasticity module (offline, synthetic).

The synthetic chassis mirrors the stripped-chassis population semantics:
uPN → KC → MBON plus PAM/PPL1 modulatory nodes. MBON transmitter sign sets
valence (+1 approach / −1 avoidance), per the documented judgment call in
``fruitfly.neuromod``. All scenarios are seeded by construction — no RNG —
so every determinism check is a byte comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fruitfly.connectome import Chassis
from fruitfly.neuromod import NeuromodState, Plasticity

# --- synthetic chassis ------------------------------------------------------

_COLS = [
    "bodyId",
    "type",
    "instance",
    "somaSide",
    "population",
    "region",
    "neurotransmitter",
    "sign",
]

# (bodyId, type, population, region, neurotransmitter, sign)
_ROWS = [
    (100, "DA1_adPN1", "uPN", "antennal-lobe", "acetylcholine", 1),
    (101, "VA1v_lPN2", "uPN", "antennal-lobe", "gaba", -1),
    (102, "KCab-c1", "KC", "mushroom-body", "acetylcholine", 1),
    (103, "KCab-p2", "KC", "mushroom-body", "acetylcholine", 1),
    (104, "KCa'b'-a3", "KC", "mushroom-body", "acetylcholine", 1),
    (105, "KCg-s4", "KC", "mushroom-body", "acetylcholine", 1),
    (106, "MBON01", "MBON", "mushroom-body", "acetylcholine", 1),  # approach
    (107, "MBON02", "MBON", "mushroom-body", "acetylcholine", 1),  # approach
    (108, "MBON03", "MBON", "mushroom-body", "gaba", -1),  # avoidance
    (109, "MBON04", "MBON", "mushroom-body", "glutamate", -1),  # avoidance
    (110, "PAM01", "PAM", "protocerebrum", "dopamine", 0),
    (111, "PAM02", "PAM", "protocerebrum", "dopamine", 0),
    (112, "PPL101", "PPL1", "protocerebrum", "dopamine", 0),
]

# KC rows of the weight view / MBON columns (chassis node indices).
KC0, KC1, KC2, KC3 = 2, 3, 4, 5
MBON0, MBON1, MBON2, MBON3 = 6, 7, 8, 9
N = len(_ROWS)


def _nodes() -> pd.DataFrame:
    nodes = pd.DataFrame(
        _ROWS, columns=["bodyId", "type", "population", "region", "neurotransmitter", "sign"]
    )
    nodes["instance"] = nodes["type"]
    nodes["somaSide"] = None
    return nodes[_COLS]


def make_chassis(edges: list[tuple[int, int, int]] | None = None) -> Chassis:
    if edges is None:
        edges = _EDGES
    pre = np.array([e[0] for e in edges], dtype=np.int32)
    post = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.int64)
    adj = sp.csr_matrix((w, (pre, post)), shape=(N, N))
    adj.sum_duplicates()
    return Chassis(nodes=_nodes(), adj=adj, meta={})
_EDGES = [
        (0, KC0, 5),
        (1, KC3, 3),
        (KC0, MBON0, 2),
        (KC0, MBON2, 1),
        (KC1, MBON0, 2),
        (KC1, MBON2, 1),
        (KC2, MBON1, 1),
        (KC2, MBON3, 2),
        (KC3, MBON3, 1),
]


@pytest.fixture()
def chassis() -> Chassis:
    return make_chassis()


def spikes(active: dict[int, float]) -> np.ndarray:
    v = np.zeros(N, dtype=np.float64)
    for idx, count in active.items():
        v[idx] = count
    return v


# expected baseline: log1p(synapses) scaled to mean 1.0 over the 7-edge support
_MEAN = (3 * np.log1p(2.0) + 4 * np.log1p(1.0)) / 7.0
W00, W02 = np.log1p(2.0) / _MEAN, np.log1p(1.0) / _MEAN  # KC0 → MBON0 (appr) / MBON2 (avoid)
W10, W12 = W00, W02  # KC1 mirrors KC0
W21, W23 = W02, W00  # KC2 → MBON1 (approach) / MBON3 (avoid)
W33 = W02  # KC3 → MBON3


# --- weight view mapping -----------------------------------------------------


def test_weight_view_mapping(chassis):
    p = Plasticity(chassis)
    assert p.kc_index.tolist() == [KC0, KC1, KC2, KC3]
    assert p.mbon_index.tolist() == [MBON0, MBON1, MBON2, MBON3]
    assert p.weights.shape == (4, 4)
    assert p.weights.dtype == np.float64
    # support is exactly the KC→MBON CSR submatrix, valence from sign column
    structural = chassis.adj[p.kc_index][:, p.mbon_index].toarray()
    assert np.array_equal(p.support, structural > 0)
    assert p.mbon_valence.tolist() == [1.0, 1.0, -1.0, -1.0]
    # baseline is monotone in raw synapse counts (log1p, mean 1 over support)
    expected = np.log1p(structural) / np.log1p(structural[structural > 0]).mean()
    assert np.allclose(p.baseline, expected * (structural > 0))
    assert np.allclose(p.weights, p.baseline * p.support)
    assert p.weights[0, 1] == 0.0  # no structural KC0→MBON1 synapse


def test_requires_kc_mbon_synapses():
    with pytest.raises(ValueError, match="KC→MBON"):
        Plasticity(make_chassis(edges=[]))


# --- state channel semantics --------------------------------------------------


def test_state_validation():
    s = NeuromodState()
    assert (s.reward, s.punishment, s.hunger, s.arousal, s.sleep) == (0, 0, 0, 0, 0)
    assert NeuromodState(hunger=1.7, arousal=-3.0).hunger == 1.0  # clamped
    assert NeuromodState(arousal=-3.0).arousal == 0.0
    with pytest.raises(ValueError):
        NeuromodState(reward=-0.1)
    with pytest.raises(ValueError):
        NeuromodState(punishment=-1.0)
    with pytest.raises(ValueError):
        NeuromodState(reward=float("nan"))


def test_hunger_raises_reward_drive(chassis):
    p = Plasticity(chassis)
    fed = p.reward_drive(NeuromodState(reward=1.0))
    starved = p.reward_drive(NeuromodState(reward=1.0, hunger=0.8))
    assert starved == pytest.approx(1.8 * fed)
    assert p.reward_drive(NeuromodState(hunger=0.9)) == pytest.approx(0.0)


def test_arousal_amplifies_gate(chassis):
    pre, post = spikes({KC2: 1.0}), spikes({MBON1: 1.0})
    calm = Plasticity(chassis, learning_rate=1.0)
    alert = Plasticity(chassis, learning_rate=1.0)
    calm.observe(NeuromodState(reward=1.0), pre, post)
    alert.observe(NeuromodState(reward=1.0, arousal=1.0), pre, post)
    assert alert.weights[2, 1] > calm.weights[2, 1]  # octopamine scales learning


def test_zero_gate_leaves_weights_unchanged(chassis):
    p = Plasticity(chassis)
    p.observe(NeuromodState(), spikes({KC2: 1.0}), spikes({MBON1: 1.0}))
    p.observe(NeuromodState(hunger=1.0, arousal=1.0), spikes({KC2: 1.0}), spikes({MBON1: 1.0}))
    assert np.array_equal(p.weights, p.baseline * p.support)


# --- one-trial learning -------------------------------------------------------


def test_one_shock_flips_readout_to_avoidance(chassis):
    p = Plasticity(chassis, learning_rate=2.0)
    re_exposure_pre = spikes({KC0: 1.0, KC1: 1.0})
    assert p.valence_readout(re_exposure_pre) > 0  # starts net-approach

    # one shock while the KC set is co-active with an approach MBON
    p.observe(NeuromodState(punishment=1.0), re_exposure_pre, spikes({MBON0: 1.0}))

    # re-exposure with no neuromodulation: readout flipped to avoidance
    assert p.valence_readout(re_exposure_pre) < 0
    # the punished KC→approach-MBON synapses depressed to the floor; the
    # structurally untouched KC→avoid-MBON synapse keeps its baseline weight
    assert p.weights[0, 0] == 0.0
    assert p.weights[1, 0] == 0.0
    assert p.weights[0, 2] == pytest.approx(W02)


def test_one_sugar_event_flips_readout_to_approach(chassis):
    p = Plasticity(chassis, learning_rate=2.0)
    pre = spikes({KC2: 1.0})
    assert p.valence_readout(pre) < 0  # KC2's approach weight is weaker

    p.observe(NeuromodState(reward=1.0), pre, spikes({MBON1: 1.0}))
    assert p.valence_readout(pre) > 0
    assert p.weights[2, 1] == pytest.approx(W21 + 2.0)


def test_punishment_strengthens_avoid_mbons(chassis):
    p = Plasticity(chassis, learning_rate=2.0)
    pre = spikes({KC2: 1.0})
    post = spikes({MBON3: 1.0})  # avoidance-valence MBON
    p.observe(NeuromodState(punishment=1.0), pre, post)
    assert p.weights[2, 3] == pytest.approx(W23 + 2.0)  # d<0 × valence −1 → up


# --- habituation ---------------------------------------------------------------


def test_habituation_fades_eligibility(chassis):
    p = Plasticity(chassis)
    pre, post = spikes({KC2: 1.0}), spikes({MBON1: 1.0})
    initial = p.weights.copy()
    traces = []
    for _ in range(6):
        p.observe(NeuromodState(), pre, post)  # stimulus predicts nothing
        traces.append(float(p.eligibility[2, 1]))
    assert all(a > b for a, b in zip(traces, traces[1:], strict=False))  # monotone fade
    assert traces[0] == pytest.approx(1.0)
    assert traces[-1] < 0.15
    assert np.array_equal(p.weights, initial)  # no dopamine → no plasticity


def test_salience_resets_habituation(chassis):
    p = Plasticity(chassis)
    pre, post = spikes({KC2: 1.0}), spikes({MBON1: 1.0})
    for _ in range(3):
        p.observe(NeuromodState(), pre, post)
    assert p.habituation[2] < 1.0
    p.observe(NeuromodState(reward=1.0), pre, post)
    assert p.habituation[2] == 1.0  # salient event restores full attention


def test_habituation_is_stimulus_specific(chassis):
    p = Plasticity(chassis)
    for _ in range(3):
        p.observe(NeuromodState(), spikes({KC2: 1.0}), spikes({MBON1: 1.0}))
    assert p.habituation[2] < 1.0
    assert p.habituation[0] == 1.0  # other KCs keep fresh attention


# --- sleep: consolidation + regime forgetting ----------------------------------


def test_sleep_decays_but_consolidates_top_k(chassis):
    p = Plasticity(chassis, learning_rate=2.0, top_k=1)
    p.observe(
        NeuromodState(punishment=1.0), spikes({KC0: 1.0, KC1: 1.0}), spikes({MBON0: 1.0})
    )
    assert p.weights[0, 0] == 0.0 and p.weights[1, 0] == 0.0

    p.sleep()
    # strongest association (tie broken toward the lower chassis node index)
    assert p.weights[0, 0] == 0.0  # consolidated — survives sleep
    assert p.weights[1, 0] == pytest.approx(W10 + 0.5 * (0.0 - W10))  # 0.7: decayed
    assert p.weights[0, 2] == pytest.approx(W02)  # untouched → back at baseline
    assert not p.eligibility.any()
    assert np.all(p.habituation == 1.0)


def test_sleep_with_default_top_k_keeps_recent_learning(chassis):
    p = Plasticity(chassis, learning_rate=2.0)
    p.observe(
        NeuromodState(punishment=1.0), spikes({KC0: 1.0, KC1: 1.0}), spikes({MBON0: 1.0})
    )
    after_shock = p.weights.copy()
    p.sleep()  # default top_k=64 > 7 nonzero pairs → everything consolidated
    assert np.array_equal(p.weights, after_shock)


def test_sleep_restores_fresh_attention(chassis):
    p = Plasticity(chassis)
    pre, post = spikes({KC2: 1.0}), spikes({MBON1: 1.0})
    for _ in range(3):
        p.observe(NeuromodState(), pre, post)
    p.sleep()
    assert np.all(p.habituation == 1.0)
    assert not p.eligibility.any()


# --- readout helpers ------------------------------------------------------------


def test_mbon_activation_matches_weights(chassis):
    p = Plasticity(chassis)
    pre = spikes({KC2: 1.0})
    act = p.mbon_activation(pre)
    assert act.shape == (4,)
    assert act == pytest.approx(p.weights[2])


def test_observe_validates_input_shape(chassis):
    p = Plasticity(chassis)
    with pytest.raises(ValueError, match="pre_spikes"):
        p.observe(NeuromodState(), np.zeros(N - 1), np.zeros(N))
    with pytest.raises(ValueError, match="post_spikes"):
        p.observe(NeuromodState(), np.zeros(N), np.zeros(N + 1))


def test_kc_activity_slices_kc_rows(chassis):
    p = Plasticity(chassis)
    x = p.kc_activity(spikes({KC1: 2.0, MBON0: 9.0}))
    assert x.tolist() == [0.0, 2.0, 0.0, 0.0]  # MBON input ignored, KC rows only


# --- per-exit credit assignment (observe_trade) ---------------------------------


def test_observe_trade_matches_observe_on_same_trace(chassis):
    """Reward-gated snapshot credit equals what observe does on the same trace."""
    trade = Plasticity(chassis, learning_rate=2.0)
    pre, post = spikes({KC0: 1.0, KC1: 2.0}), spikes({MBON0: 1.0, MBON2: 3.0})
    trade.observe(NeuromodState(), pre, post)
    snapshot = trade.eligibility.copy()
    # Reference: the same co-activity driven through observe with the reward
    # gate fired in that same step, so its live trace equals the snapshot.
    ref = Plasticity(chassis, learning_rate=2.0)
    reward_state = NeuromodState(reward=1.0, arousal=0.5)
    ref.observe(reward_state, pre, post)
    trade.observe_trade(reward_state, snapshot)
    # Same gate × same trace × same valence rule → identical weights.
    assert trade.weights.tobytes() == ref.weights.tobytes()
    assert trade.weights[0, 0] > W00  # approach potentiated
    assert trade.weights[0, 2] < W02  # avoid depressed


def test_observe_trade_leaves_live_state_untouched(chassis):
    trade = Plasticity(chassis, learning_rate=2.0)
    trade.observe(NeuromodState(), spikes({KC0: 1.0}), spikes({MBON0: 1.0}))
    live_elig = trade.eligibility.copy()
    live_hab = trade.habituation.copy()
    weights_before = trade.weights.copy()
    trade.observe_trade(NeuromodState(), trade.eligibility.copy())  # zero gate → no-op
    trade.observe_trade(NeuromodState(reward=1.0), np.zeros_like(trade.eligibility))
    assert np.array_equal(trade.eligibility, live_elig)
    assert np.array_equal(trade.habituation, live_hab)
    assert np.array_equal(trade.weights, weights_before)  # zero trace → no delta


def test_observe_trade_validates_snapshot_shape(chassis):
    trade = Plasticity(chassis)
    with pytest.raises(ValueError, match="eligibility_snapshot"):
        trade.observe_trade(NeuromodState(), np.zeros((3, 3)))



# --- determinism (byte-identical double run) ------------------------------------


def _scenario(p: Plasticity) -> Plasticity:
    pre_shock = spikes({KC0: 1.0, KC1: 1.0})
    post_shock = spikes({MBON0: 1.0})
    pre_odor = spikes({KC2: 1.0})
    post_odor = spikes({MBON1: 1.0})
    p.observe(NeuromodState(punishment=1.0), pre_shock, post_shock)
    for _ in range(3):
        p.observe(NeuromodState(), pre_odor, post_odor)
    p.observe(NeuromodState(reward=2.0, hunger=0.5, arousal=0.3), pre_odor, post_odor)
    p.observe(NeuromodState(punishment=0.5, arousal=0.1), pre_shock, post_shock)
    p.sleep()
    return p


def test_determinism_byte_identical_double_run(chassis):
    a = _scenario(Plasticity(chassis, learning_rate=2.0))
    b = _scenario(Plasticity(chassis, learning_rate=2.0))
    assert a.weights.tobytes() == b.weights.tobytes()
    assert a.eligibility.tobytes() == b.eligibility.tobytes()
    assert a.habituation.tobytes() == b.habituation.tobytes()
