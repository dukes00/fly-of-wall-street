"""Tests for T12 weight transplant + behavioral validation (offline, synthetic).

Never touches the real chassis caches or the trained artifact: the transplant
is exercised on two tiny synthetic chassis sharing the MaleCNS bodyId/label
contract (sorted bodyIds, population column, ``glomeruli`` meta), and the
agreement metric on synthetic decision logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fruitfly.connectome import Chassis
from fruitfly.neuromod import Plasticity
from fruitfly.transplant import (
    action_bucket,
    agreement_table,
    label_whole_chassis,
    match_rate,
    transplant_weights,
)

_COLS = ["bodyId", "type", "instance", "somaSide", "population", "region",
         "neurotransmitter", "sign"]


def _nodes(rows: list[tuple[int, str, str, str, int]]) -> pd.DataFrame:
    """(bodyId, type, population, neurotransmitter, sign) -> node table."""
    return pd.DataFrame(
        [(b, t, t, "L", p, "r", nt, s) for b, t, p, nt, s in rows],
        columns=_COLS,
    )


def _chassis(nodes: pd.DataFrame, edges: list[tuple[int, int, int]]) -> Chassis:
    """Chassis from bodyId-keyed edges; edge weights are synapse counts."""
    n = len(nodes)
    pos = {int(b): i for i, b in enumerate(nodes["bodyId"].to_numpy())}
    pre = np.array([pos[e[0]] for e in edges], dtype=np.int64)
    post = np.array([pos[e[1]] for e in edges], dtype=np.int64)
    w = np.array([e[2] for e in edges], dtype=np.int64)
    adj = sp.csr_matrix((w, (pre, post)), shape=(n, n), dtype=np.int64)
    adj.sum_duplicates()
    return Chassis(
        nodes=nodes, adj=adj,
        meta={"glomeruli": ["DA1", "VA1v"], "n_glomeruli": 2},
    )


# bodyId: uPN 10-13, KC 100-103, MBON 200-202 (200/201 approach, 202 inhibit),
# PAM 300-301, PPL1 310-311.
_STRIPPED_ROWS = (
    [(10 + i, "PN", "uPN", "acetylcholine", 1) for i in range(4)]
    + [(100 + i, "KC", "KC", "acetylcholine", 1) for i in range(4)]
    + [(200, "MBON-a", "MBON", "acetylcholine", 1),
       (201, "MBON-b", "MBON", "acetylcholine", 1),
       (202, "MBON-c", "MBON", "gaba", -1)]
    + [(300, "PAM1", "PAM", "dopamine", 0), (301, "PAM2", "PAM", "dopamine", 0)]
    + [(310, "PPL1-1", "PPL1", "dopamine", 0), (311, "PPL1-2", "PPL1", "dopamine", 0)]
)
# KC 100 -> MBON 200/202, KC 101 -> MBON 200/201, KC 102 -> MBON 202; KC 103 none.
# 101 -> 200 exists ONLY in the stripped chassis (the whole fly lacks the
# synapse) — the drop path.
_STRIPPED_EDGES = [
    (10, 100, 300), (11, 101, 300),
    (100, 200, 10), (100, 202, 4),
    (101, 201, 20),
    (101, 200, 6),
    (102, 202, 5),
]

# Whole fly: every stripped bodyId (sorted, same relative order) PLUS
# bodyId 104 — a whole-fly KC absent from the stripped chassis — bodyId 203,
# a whole-fly MBON absent from the stripped chassis, and an unlabeled filler
# 400. 104 -> 203 is a real whole-fly synapse between two unmatched nodes;
# 102 -> 201 is a whole-fly synapse with NO stripped counterpart (the
# baseline-fill path).
_WHOLE_EDGES = [
    (10, 100, 300), (11, 101, 300),
    (100, 200, 10), (100, 202, 4),
    (101, 201, 20),
    (102, 202, 5),
    # whole-fly-only structure:
    (104, 203, 7),   # unmatched KC -> unmatched MBON
    (100, 203, 9),   # matched KC -> unmatched MBON
    (102, 201, 7),   # matched pair with no stripped synapse
]
_EXTRA_ROWS = [(104, "KC", "KC", "acetylcholine", 1),
               (203, "MBON-d", "MBON", "acetylcholine", 1),
               (400, "Filler", "whole", "gaba", -1)]


def make_stripped() -> Chassis:
    return _chassis(_nodes(list(_STRIPPED_ROWS)), _STRIPPED_EDGES)


def make_whole() -> Chassis:
    rows = sorted(_STRIPPED_ROWS + _EXTRA_ROWS)
    ids = [r[0] for r in rows]
    # sorted bodyId order: 10..13, 100..104, 200..203, 300,301,310,311,400
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    # population column is "whole" everywhere in the raw whole-fly cache;
    # labels arrive only via label_whole_chassis.
    rows = [(b, t, "whole", nt, s) for b, t, _p, nt, s in rows]
    return _chassis(_nodes(rows), _WHOLE_EDGES)


def _trained_on_support(stripped: Chassis) -> tuple[np.ndarray, np.ndarray]:
    """Trained weights on the stripped view plus its support mask.

    Baseline everywhere on support, then: (100→200) potentiated, (102→202)
    depressed to zero (a learned grudge is still a supported pair), and
    (101→200 — the stripped-only synapse) trained to 2.5.
    """
    p = Plasticity(stripped)
    w = p.weights.copy()
    w[0, 0] += 3.0
    w[2, 2] = 0.0
    w[1, 0] = 2.5
    return w, p.support


# ---------------------------------------------------------------------------
# bodyId matching / labeling
# ---------------------------------------------------------------------------


class TestLabeling:
    def test_labels_transfer_by_bodyid_and_unmatched_stay_whole(self):
        stripped, whole = make_stripped(), make_whole()
        labeled = label_whole_chassis(whole, stripped)
        pop = labeled.nodes.set_index("bodyId")["population"]
        # matched nodes carry the stripped labels ...
        assert pop.loc[100] == "KC" and pop.loc[202] == "MBON"
        assert pop.loc[10] == "uPN" and pop.loc[310] == "PPL1"
        # ... whole-fly-only nodes keep the raw whole-fly label ...
        assert pop.loc[104] == "whole" and pop.loc[203] == "whole"
        assert pop.loc[400] == "whole"
        # region follows the same rule
        region = labeled.nodes.set_index("bodyId")["region"]
        assert region.loc[100] == "r" and region.loc[104] == "brain"

    def test_meta_carries_stripped_glomeruli_and_adj_shared(self):
        stripped, whole = make_stripped(), make_whole()
        labeled = label_whole_chassis(whole, stripped)
        assert labeled.meta["glomeruli"] == stripped.meta["glomeruli"]
        assert labeled.adj is whole.adj  # no edge-table copy
        assert labeled.n_neurons == whole.n_neurons

    def test_match_rate(self):
        stripped_ids = np.array([1, 2, 3])
        assert match_rate(stripped_ids, np.array([1, 2, 3, 4])) == 1.0
        assert match_rate(stripped_ids, np.array([2, 3])) == pytest.approx(2 / 3)
        assert match_rate(np.array([], dtype=int), np.array([1])) == 1.0


# ---------------------------------------------------------------------------
# Transplant: shape, sparsity, per-pair fate
# ---------------------------------------------------------------------------


class TestTransplant:
    def test_shape_and_support_sparsity(self):
        stripped, whole = make_stripped(), make_whole()
        trained, _ = _trained_on_support(stripped)
        res = transplant_weights(trained, whole, stripped_chassis=stripped)
        p = res.plasticity
        # view covers exactly the labeled (matched) KC/MBON sets: the
        # whole-fly-only KC 104 / MBON 203 are structurally excluded
        assert p.weights.shape == (4, 3)
        assert list(p.kc_index) == [4, 5, 6, 7]      # node rows of bodyId 100-103
        assert list(p.mbon_index) == [9, 10, 11]     # node rows of bodyId 200-202
        # weights ONLY on structural support (plasticity never invents
        # synapses): the 101->200 trained mass has no whole-fly synapse
        live = p.weights.copy()
        assert live[1, 0] == 0.0
        assert np.all(live[~p.support] == 0.0)

    def test_trained_values_copied_verbatim_and_baseline_fill(self):
        stripped, whole = make_stripped(), make_whole()
        trained, _ = _trained_on_support(stripped)
        res = transplant_weights(trained, whole, stripped_chassis=stripped)
        p = res.plasticity
        # KC100 -> MBON200: potentiated value lands exactly (synapse counts
        # are identical between the chassis over matched pairs)
        assert p.weights[0, 0] == pytest.approx(trained[0, 0])
        # KC102 -> MBON202: depressed-to-zero trained value stays zero
        # (a learned grudge is copied, NOT refilled with baseline)
        assert p.weights[2, 2] == 0.0
        # KC102 -> MBON201: no stripped synapse (trained 0) but the whole fly
        # HAS one -> baseline weight from the whole-fly synapse count
        baseline = Plasticity(label_whole_chassis(whole, stripped)).baseline
        assert p.weights[2, 1] == pytest.approx(baseline[2, 1])
        assert res.report.pairs_baseline_filled == 1

    def test_drop_path_counts_missing_whole_support(self):
        stripped, whole = make_stripped(), make_whole()
        trained, _ = _trained_on_support(stripped)
        res = transplant_weights(trained, whole, stripped_chassis=stripped)
        # trained 2.5 at (101 -> 200): supported in the stripped chassis but
        # the whole fly has no such synapse -> dropped, reported
        assert res.report.pairs_dropped_no_support == 1
        assert res.plasticity.weights[1, 0] == 0.0

    def test_report_and_match_rates(self):
        stripped, whole = make_stripped(), make_whole()
        trained, _ = _trained_on_support(stripped)
        res = transplant_weights(trained, whole, stripped_chassis=stripped)
        r = res.report
        assert (r.n_kc_stripped, r.n_mbon_stripped) == (4, 3)
        assert (r.n_kc_whole, r.n_mbon_whole) == (4, 3)  # 104/203 unlabeled
        assert r.kc_match_rate == 1.0 and r.mbon_match_rate == 1.0
        assert r.pairs_total == 12
        assert r.pairs_copied == 3  # (100,200), (100,202), (101,201)
        assert r.synapse_consistent is False  # synthetic edge sets differ
        assert r.fingerprint_verified is False  # raw array, no provenance

    def test_unmatched_whole_neurons_excluded_not_driven(self):
        # The whole-fly-only KC 104 fires into 203, but neither carries a
        # stripped label, so the transplanted readout cannot see them: the
        # MBON drive over the transplanted view ignores their activity.
        stripped, whole = make_stripped(), make_whole()
        trained, _ = _trained_on_support(stripped)
        res = transplant_weights(trained, whole, stripped_chassis=stripped)
        spikes = np.zeros(whole.n_neurons)
        spikes[whole.nodes.index[whole.nodes.bodyId == 104][0]] = 50.0
        assert np.all(res.plasticity.mbon_activation(spikes) == 0.0)

    def test_fingerprint_mismatch_rejected(self, tmp_path):
        from fruitfly.train import (
            LarvalWeights,
            chassis_fingerprint,
            save_larval_weights,
        )

        stripped, whole = make_stripped(), make_whole()
        trained, _ = _trained_on_support(stripped)
        path = tmp_path / "w.npz"
        save_larval_weights(path, trained, chassis_fingerprint(stripped), {})
        res = transplant_weights(path, whole, stripped_chassis=stripped)
        assert res.report.fingerprint_verified is True
        foreign = LarvalWeights(weights=trained, fingerprint="deadbeef", meta={})
        with pytest.raises(ValueError, match="different chassis"):
            transplant_weights(foreign, whole, stripped_chassis=stripped)

    def test_shape_mismatch_rejected(self):
        stripped, whole = make_stripped(), make_whole()
        with pytest.raises(ValueError, match="does not match"):
            transplant_weights(np.zeros((3, 3)), whole, stripped_chassis=stripped)


# ---------------------------------------------------------------------------
# Agreement metric on synthetic decision logs
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_action_bucket(self):
        assert action_bucket("buy") == "BUY"
        assert action_bucket("add") == "BUY"
        assert action_bucket("sell") == "SELL"
        assert action_bucket("pass") == "PASS"

    def test_per_day_agreement_and_day_weighted_mean(self):
        a = [
            ("2026-09-11T14:30:00+00:00", "AAPL", "buy"),
            ("2026-09-11T14:31:00+00:00", "MSFT", "pass"),
            ("2026-09-11T14:32:00+00:00", "NVDA", "pass"),
            ("2026-09-14T14:30:00+00:00", "AAPL", "sell"),
        ]
        b = [
            ("2026-09-11T14:30:00+00:00", "AAPL", "buy"),    # agree
            ("2026-09-11T14:31:00+00:00", "MSFT", "pass"),   # agree
            ("2026-09-11T14:32:00+00:00", "GOOGL", "pass"),  # action agree,
            #                                                ticker diverged
            ("2026-09-14T14:30:00+00:00", "AAPL", "pass"),   # disagree
        ]
        rows, mean = agreement_table(a, b)
        assert [r["day"] for r in rows] == ["2026-09-11", "2026-09-14"]
        d1, d2 = rows
        assert d1["bars"] == 3
        assert d1["action_agreement"] == pytest.approx(1.0)
        assert d1["ticker_agreement"] == pytest.approx(2 / 3)
        assert d1["action_and_ticker_agreement"] == pytest.approx(2 / 3)
        assert d2["bars"] == 1
        assert d2["action_agreement"] == 0.0
        # day-weighted mean: (1.0 + 0.0) / 2, NOT bar-weighted (0.75)
        assert mean == pytest.approx(0.5)

    def test_disjoint_logs_agree_on_nothing(self):
        a = [("2026-09-11T14:30:00+00:00", "AAPL", "buy")]
        b = [("2026-09-14T14:30:00+00:00", "AAPL", "buy")]
        rows, mean = agreement_table(a, b)
        assert rows == [] and mean == 0.0

    def test_identical_logs_agree_perfectly(self):
        a = [
            ("2026-09-11T14:30:00+00:00", "AAPL", "sell"),
            ("2026-09-11T14:31:00+00:00", "MSFT", "add"),
        ]
        rows, mean = agreement_table(a, list(a))
        assert rows[0]["action_agreement"] == 1.0
        assert rows[0]["action_and_ticker_agreement"] == 1.0
        assert mean == 1.0
