"""Tests for the T2 connectome module (fully offline, synthetic fixtures).

Builds a tiny synthetic copy of the raw connectome feather files (same
schemas, a few dozen bodies) and exercises extraction, sign mapping, edge
aggregation, cache determinism and the loader round-trip — no 1 GB data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest

from fruitfly.connectome import (
    build_stripped_chassis,
    load_stripped_chassis,
    load_whole_fly,
)

# --- fixture ---------------------------------------------------------------

# (bodyId, type, class, superclass, somaSide, instance, population-or-None)
_ROWS = [
    (100, "R1-R6", "visual", "ol_sensory", "L", "R1-R6_L", "photoreceptor"),
    (101, "R7p", "visual", "ol_sensory", "R", "R7p_R", "photoreceptor"),
    (102, "R7R8_unclear", "visual", "ol_sensory", "R", "R78_R", "photoreceptor"),
    (200, "T4a", "visual", "ol_intrinsic", "L", "T4a_L", "T4"),
    (201, "T4b", "visual", "ol_intrinsic", "R", "T4b_R", "T4"),
    (202, "T5a", "visual", "ol_intrinsic", "R", "T5a_R", "T5"),  # absent from NT
    (300, "T5b", "visual", "ol_intrinsic", "L", "T5b_L", "T5"),
    (400, "LC4", "visual", "visual_projection", "L", "LC4_L", "LC-looming"),
    (401, "LC10a", "visual", "visual_projection", "L", "LC10a_L", "LC-looming"),
    (402, "LC10c-1", "visual", "visual_projection", "R", "LC10c-1_R", "LC-looming"),
    (403, "LC21", "visual", "visual_projection", "R", "LC21_R", "LC-looming"),
    (404, "LC40", "visual", "visual_projection", "R", "LC40_R", None),
    (500, "DA1_lPN", "ALPN", "cb_sensory", "L", "DA1_lPN_L", "uPN"),
    (501, "DA1_vPN", "ALPN", "cb_sensory", "R", "DA1_vPN_R", "uPN"),
    (502, "M_vPNml53", "ALPN", "cb_sensory", "L", "M_vPNml53_L", None),
    (600, "KCab-m", "Kenyon_Cell", "cb_intrinsic", "L", "KCab-m_L", "KC"),
    (601, "KCg-m", "Kenyon_Cell", "cb_intrinsic", "R", "KCg-m_R", "KC"),
    (700, "MBON01", "MBON", "cb_intrinsic", "L", "MBON01_L", "MBON"),
    (800, "PAM01", "DAN", "cb_intrinsic", "L", "PAM01_L", "PAM"),
    (801, "PPL101", "DAN", "cb_intrinsic", "L", "PPL101_L", "PPL1"),
    (802, "PPL201", "DAN", "cb_intrinsic", "L", "PPL201_L", None),
    (900, "DNg07", None, "descending_neuron", "L", "DNg07_L", "DN"),
    (901, "DNb02", None, "descending_neuron_tbc", "R", "DNb02_R", "DN"),
    # annotated but not chassis-selected: optic-lobe interneuron
    (950, "Mi1", "visual", "ol_intrinsic", "L", "Mi1_L", None),
]

# body: (consensus_nt, predicted_nt, celltype_predicted_nt)
# Tier order is consensus_nt -> predicted_nt -> celltype_predicted_nt.
_NT = {
    100: ("acetylcholine", "acetylcholine", "acetylcholine"),
    101: ("acetylcholine", "unclear", "acetylcholine"),
    102: ("unclear", "unclear", "unclear"),  # unresolved
    200: ("acetylcholine", "acetylcholine", "acetylcholine"),
    201: ("gaba", "unclear", "unclear"),
    300: ("glutamate", "glutamate", "glutamate"),  # inhibitory (documented)
    400: ("gaba", "gaba", "gaba"),
    401: ("acetylcholine", "acetylcholine", "acetylcholine"),
    402: ("acetylcholine", "acetylcholine", "acetylcholine"),
    403: ("unclear", "acetylcholine", "acetylcholine"),  # predicted fallback
    500: ("acetylcholine", "acetylcholine", "acetylcholine"),
    501: ("acetylcholine", "acetylcholine", "acetylcholine"),
    600: ("glutamate", "glutamate", "glutamate"),
    601: ("gaba", "unclear", "gaba"),
    700: ("gaba", "gaba", "gaba"),
    800: ("dopamine", "dopamine", "dopamine"),  # modulatory -> 0
    801: ("dopamine", "dopamine", "dopamine"),
    900: ("acetylcholine", "acetylcholine", "acetylcholine"),
    901: ("acetylcholine", "acetylcholine", "acetylcholine"),
    950: ("acetylcholine", "acetylcholine", "acetylcholine"),
    # 202: absent from the NT table entirely -> unknown
}

# (body_pre, body_post, weight) — bodyId space
_EDGES = [
    (100, 200, 5),  # R -> T4
    (101, 950, 3),  # R -> excluded Mi1: dropped
    (950, 200, 2),  # excluded -> T4: dropped
    (200, 400, 2),  # T4 -> LC4
    (500, 600, 3),  # uPN -> KC, duplicated below -> aggregated to 10
    (500, 600, 7),
    (501, 601, 1),
    (600, 700, 4),  # KC -> MBON
    (800, 600, 1),  # PAM -> KC (modulatory edge kept in graph, sign 0 node)
    (900, 900, 2),  # DN self-loop: kept
    (999, 100, 1),  # unknown body: dropped
    (100, 999, 1),  # unknown body: dropped
]

ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


@pytest.fixture()
def fixture_dir(tmp_path: Path) -> Path:
    """Synthetic raw connectome directory with the real table schemas."""
    ann = pd.DataFrame(
        {
            "bodyId": np.array([r[0] for r in _ROWS], dtype="int64"),
            "type": [r[1] for r in _ROWS],
            "class": [r[2] for r in _ROWS],
            "superclass": [r[3] for r in _ROWS],
            "somaSide": [r[4] for r in _ROWS],
            "instance": [r[5] for r in _ROWS],
        }
    )
    feather.write_feather(ann, tmp_path / ANNOTATIONS_FILE)

    nt = pd.DataFrame(
        {
            "body": np.array(sorted(_NT), dtype="int64"),
            "consensus_nt": [_NT[b][0] for b in sorted(_NT)],
            "predicted_nt": [_NT[b][1] for b in sorted(_NT)],
            "predicted_nt_confidence": np.full(len(_NT), 0.9),
            "celltype_predicted_nt": [_NT[b][2] for b in sorted(_NT)],
        }
    )
    feather.write_feather(nt, tmp_path / NT_FILE)

    w = pd.DataFrame(
        {
            "body_pre": np.array([e[0] for e in _EDGES], dtype="int64"),
            "body_post": np.array([e[1] for e in _EDGES], dtype="int64"),
            "weight": np.array([e[2] for e in _EDGES], dtype="int64"),
        }
    )
    feather.write_feather(w, tmp_path / WEIGHTS_FILE)
    return tmp_path


# --- population selection ---------------------------------------------------


def test_population_selection(fixture_dir):
    ch = build_stripped_chassis(fixture_dir, cache=False)
    dims = ch.meta["population_dims"]
    assert dims == {
        "DN": 2,
        "KC": 2,
        "LC-looming": 4,
        "MBON": 1,
        "PAM": 1,
        "PPL1": 1,
        "T4": 2,
        "T5": 2,
        "photoreceptor": 3,
        "uPN": 2,
    }
    # LC40 excluded (not looming-selective); multiglomerular M_ PN excluded;
    # PPL2 DAN excluded; Mi1 and unproofread fragments excluded.
    for body in (404, 502, 802, 950):
        assert body not in ch.nodes.bodyId.values
    assert set(ch.nodes.population.unique()) == set(dims)
    # nodes sorted by bodyId
    assert ch.nodes.bodyId.is_monotonic_increasing
    assert ch.adj.shape == (ch.n_neurons, ch.n_neurons)


def test_glomeruli(fixture_dir):
    ch = build_stripped_chassis(fixture_dir, cache=False)
    # DA1_lPN and DA1_vPN both map to the DA1 glomerulus
    assert ch.meta["n_glomeruli"] == 1
    assert ch.meta["glomeruli"] == ["DA1"]


# --- neurotransmitter signs -------------------------------------------------


def test_sign_mapping(fixture_dir):
    ch = build_stripped_chassis(fixture_dir, cache=False)
    sign = ch.nodes.set_index("bodyId")["sign"]
    nt = ch.nodes.set_index("bodyId")["neurotransmitter"]
    assert sign[100] == 1 and nt[100] == "acetylcholine"
    assert sign[101] == 1 and nt[101] == "acetylcholine"  # consensus tier
    assert sign[201] == -1 and nt[201] == "gaba"  # consensus tier
    assert sign[300] == -1 and nt[300] == "glutamate"  # documented choice
    assert sign[600] == -1
    assert sign[700] == -1
    assert sign[800] == 0 and nt[800] == "dopamine"  # modulatory
    assert sign[801] == 0
    assert sign[403] == 1 and nt[403] == "acetylcholine"  # predicted fallback
    assert sign[102] == 0 and nt[102] == "unclear"  # unresolved
    assert sign[202] == 0  # absent from NT table
    tier = ch.meta["nt_tier_counts"]
    assert tier["consensus_nt"] == 17
    assert tier["predicted_nt"] == 1  # 403: unclear consensus, ach prediction
    assert tier["none"] == 2  # 102 (all unclear) + 202 (absent)


# --- edges / adjacency ------------------------------------------------------


def test_edges(fixture_dir):
    ch = build_stripped_chassis(fixture_dir, cache=False)
    idx = {b: i for i, b in enumerate(ch.nodes.bodyId)}
    adj = ch.adj

    assert adj[idx[100], idx[200]] == 5  # R -> T4
    assert adj[idx[200], idx[400]] == 2
    assert adj[idx[500], idx[600]] == 10  # duplicate (3 + 7) aggregated
    assert adj[idx[501], idx[601]] == 1
    assert adj[idx[600], idx[700]] == 4
    assert adj[idx[800], idx[600]] == 1
    assert adj[idx[900], idx[900]] == 2  # self-loop kept
    # excluded bodies contribute no rows/columns
    for body in (404, 502, 802, 950, 999):
        assert body not in idx
    assert ch.meta["n_edges"] == 7
    assert ch.meta["total_synapses"] == 5 + 2 + 10 + 1 + 4 + 1 + 2
    assert adj.nnz == 7  # duplicate pair collapsed into one stored entry
    assert adj.format == "csr"
    assert np.all(adj.data >= 0)


def test_meta_dumps_chassis_neuron_count(fixture_dir):
    ch = build_stripped_chassis(fixture_dir, cache=False)
    assert ch.meta["n_neurons"] == ch.n_neurons == 20
    assert ch.meta["kind"] == "stripped-chassis"


# --- determinism / cache / loaders ------------------------------------------


def _cache_sha(data_dir: Path) -> dict[str, str]:
    names = [
        "stripped-chassis.parquet",
        "stripped-chassis-edges.parquet",
    ]
    # schema metadata (json, sort_keys) lives inside the nodes-file bytes,
    # so hashing the bytes covers meta stability too.
    return {n: hashlib.sha256((data_dir / n).read_bytes()).hexdigest() for n in names}


def _fingerprint(ch) -> str:
    h = hashlib.sha256()
    h.update(ch.nodes.to_csv(index=True).encode())
    for attr in ("indptr", "indices", "data"):
        h.update(getattr(ch.adj, attr).tobytes())
    h.update(json.dumps(ch.meta, sort_keys=True).encode())
    return h.hexdigest()


def test_cache_determinism(fixture_dir):
    a = build_stripped_chassis(fixture_dir, cache=True)
    b = build_stripped_chassis(fixture_dir, cache=True)
    assert _cache_sha(fixture_dir) == _cache_sha(fixture_dir)
    assert a.meta["n_neurons"] == b.meta["n_neurons"]
    assert (a.adj != b.adj).nnz == 0


def test_loader_roundtrip(fixture_dir):
    build_stripped_chassis(fixture_dir, cache=True)
    ch1 = load_stripped_chassis(fixture_dir)
    ch2 = load_stripped_chassis(fixture_dir)

    assert _fingerprint(ch1) == _fingerprint(ch2)
    assert ch1.meta["kind"] == "stripped-chassis"
    assert ch1.nodes.bodyId.is_monotonic_increasing
    # adjacency survives the round-trip
    idx = {b: i for i, b in enumerate(ch1.nodes.bodyId)}
    assert ch1.adj[idx[100], idx[200]] == 5


def test_load_whole_fly(fixture_dir):
    ch = load_whole_fly(fixture_dir)
    # all annotated (non-null superclass) neurons, not just the chassis set
    assert ch.meta["n_neurons"] == 24
    assert ch.meta["kind"] == "whole-fly"
    idx = {b: i for i, b in enumerate(ch.nodes.bodyId)}
    assert ch.adj[idx[950], idx[200]] == 2  # Mi1 edge present in whole fly
    assert ch.adj[idx[101], idx[950]] == 3  # R7p -> Mi1 edge present in whole fly
    assert ch.nodes.bodyId.is_monotonic_increasing
