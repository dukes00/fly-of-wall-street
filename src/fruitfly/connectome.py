"""T2: connectome extraction — stripped-chassis and whole-fly loaders.

Raw data (Apache Feather, see ``data/connectome/MANIFEST.md``):

- ``body-annotations-…feather`` — neuron table (type/class/superclass/…).
- ``body-neurotransmitters-…feather`` — predicted NT per body.
- ``connectome-weights-…feather`` — edge table (body_pre, body_post,
  weight = synapse count), 151.8M rows.

The stripped chassis (DESIGN.md §2, D10 phase A) is the union of the
task-relevant populations: retina + direction-selective / looming optic-lobe
cells + olfactory projection neurons + mushroom body + dopaminergic
reinforcement clusters + descending readout. Intermediate optic-lobe neurons
(L1/L2, Mi, Tm, …) are deliberately excluded — direction selectivity is
computed by the sensory encoders (T4) in the stripped mode; the whole-fly
mode keeps the full annotated graph.

Caching: extraction scans the 1 GB edge table in chunks. Results are cached
as parquet pairs under ``data/connectome/`` (gitignored):

- ``stripped-chassis.parquet`` / ``stripped-chassis-edges.parquet``
- ``whole-fly.parquet`` / ``whole-fly-edges.parquet``

The node parquet carries the :class:`Chassis` meta dict in its Arrow schema
metadata under ``b"fruitfly.connectome.meta"``.

Determinism: bodyIds are sorted; edges are aggregated (duplicate pre/post
pairs summed), sorted by (pre, post) and stored canonically, so the cache
bytes — and the resulting CSR — are hash-stable across runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.feather as feather
import scipy.sparse as sp

#: Raw data directory (relative paths, run from the repo root).
DATA_DIR = Path("data/connectome")

_ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
_NT = "body-neurotransmitters-male-cns-v1.0.feather"
_WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

CHASSIS_NODES = "stripped-chassis.parquet"
CHASSIS_EDGES = "stripped-chassis-edges.parquet"
WHOLE_NODES = "whole-fly.parquet"
WHOLE_EDGES = "whole-fly-edges.parquet"

_META_KEY = b"fruitfly.connectome.meta"

#: Neurotransmitter → LIF sign. Acetylcholine excites; GABA, glutamate and
#: glycine inhibit (glutamate is treated inhibitory per the Shiu et al. 2024
#: whole-brain convention — contested for some cell types, documented in the
#: T2 report). Dopamine, serotonin, octopamine, histamine, "unclear" and
#: anything unmapped → 0: modulators are handled by the neuromod layer, not
#: generic LIF.
NT_SIGN: dict[str, int] = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "glycine": -1,
}

# ---------------------------------------------------------------------------
# Population selection (stripped chassis)
# ---------------------------------------------------------------------------

#: (population, region, predicate) over the annotated-neuron frame. ``t`` is
#: the null-free ``type`` column, ``c`` the ``class`` column, ``s`` the
#: ``superclass`` column. Predicates are evaluated in order; a neuron matches
#: at most one population (first hit wins). Anchored LC regex deliberately
#: excludes LC40/LC41/LC43/LC44 (not looming-selective).
_POPULATIONS: list[tuple[str, str, str]] = [
    ("photoreceptor", "retina", r"^R[1-8]"),
    ("T4", "medulla", r"^T4"),
    ("LC-looming", "lobula", r"^LC(4|21)(_.*)?$|^LC10"),
    ("T5", "lobula", r"^T5"),
    ("uPN", "antennal-lobe", None),  # class == ALPN and not multiglomerular
    ("KC", "mushroom-body", None),  # class == Kenyon_Cell
    ("MBON", "mushroom-body", None),  # class == MBON
    ("PAM", "protocerebrum", r"^PAM"),  # within class == DAN
    ("PPL1", "protocerebrum", r"^PPL1"),  # within class == DAN
    ("DN", "brain", None),  # superclass in descending_neuron(_tbc)
]


def _population_labels(ann: pd.DataFrame) -> pd.DataFrame:
    """Assign (population, region) to annotated neurons; first match wins.

    ``ann`` must be the proofread-neuron frame (non-null ``superclass``).
    Returns a frame indexed like ``ann`` with columns ``population`` and
    ``region`` (NaN for neurons outside the chassis).
    """
    t = ann["type"].fillna("")
    c = ann["class"]
    s = ann["superclass"]
    population = pd.Series(pd.NA, index=ann.index, dtype="object")
    region = pd.Series(pd.NA, index=ann.index, dtype="object")
    for name, reg, pattern in _POPULATIONS:
        if name == "uPN":
            # Uniglomerular projection neurons: ALPN class minus the
            # multiglomerular ("M_"-prefixed) types.
            hit = (c == "ALPN") & ~t.str.startswith("M_")
        elif name == "KC":
            hit = c == "Kenyon_Cell"
        elif name == "MBON":
            hit = c == "MBON"
        elif name == "PAM" or name == "PPL1":
            hit = (c == "DAN") & t.str.match(pattern)
        elif name == "DN":
            hit = s.isin(["descending_neuron", "descending_neuron_tbc"])
        else:
            hit = t.str.match(pattern)
        hit &= population.isna()
        population[hit] = name
        region[hit] = reg
    return pd.DataFrame({"population": population, "region": region}, index=ann.index)


_GLOM_RE = re.compile(r"_(adPN|lPN|vPN|lvPN)\d*$")

def _glomerulus(uPN_type: str | float) -> str:
    """Glomerulus name from a uniglomerular PN type ("DA1_lPN" -> "DA1").

    Returns "" for neurons with a null ``type`` (contributed by the caller
    as-is; the empty string is filtered out of the glomerulus set).
    """
    if not isinstance(uPN_type, str):
        return ""
    return _GLOM_RE.sub("", uPN_type)


def _nt_signs(ann: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Neurotransmitter name + LIF sign per body, with fallback tiers.

    Tier 1: ``consensus_nt`` — the dataset's curated reconciliation, which
    fixes known predictor failures (e.g. Kenyon cells are predicted
    "dopamine" by the Shiu-style classifier but are cholinergic; the
    consensus says acetylcholine).
    Tier 2: body-level ``predicted_nt`` (Shiu-style per-body prediction).
    Tier 3: cell-type consensus ``celltype_predicted_nt``. Unresolved →
    unknown (sign 0).
    """
    nt = feather.read_table(data_dir / _NT).to_pandas()
    nt = nt.set_index("body")
    for col in ("consensus_nt", "predicted_nt", "celltype_predicted_nt"):
        if col not in nt.columns:
            raise ValueError(f"neurotransmitter table missing column {col!r}")

    body_id = ann["bodyId"]
    nt_name = pd.Series("unclear", index=ann.index, dtype="object")
    tier = pd.Series("none", index=ann.index, dtype="object")
    known = set(NT_SIGN) | {
        "dopamine",
        "serotonin",
        "octopamine",
        "histamine",
    }

    cons_nt = nt["consensus_nt"].reindex(body_id).to_numpy()
    body_nt = nt["predicted_nt"].reindex(body_id).to_numpy()
    cell_nt = nt["celltype_predicted_nt"].reindex(body_id).to_numpy()

    for array, name in (
        (cons_nt, "consensus_nt"),
        (body_nt, "predicted_nt"),
        (cell_nt, "celltype_predicted_nt"),
    ):
        series = pd.Series(array, index=ann.index)
        resolved = nt_name.isin(known)
        take = series.isin(known) & ~resolved
        nt_name[take] = series[take]
        tier[take] = name

    sign = nt_name.map(NT_SIGN).fillna(0).astype(int)
    return pd.DataFrame({"neurotransmitter": nt_name, "sign": sign, "nt_tier": tier})


# ---------------------------------------------------------------------------
# Edge extraction (chunked over the 1 GB edge table)
# ---------------------------------------------------------------------------


def _extract_edges(
    body_ids: np.ndarray, weights_path: Path, batch_rows: int = 1 << 22
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan the connectome-weights table in chunks; keep intra-``body_ids`` edges.

    Returns (pre_idx, post_idx, weight) in node-index space, duplicate
    (pre, post) pairs aggregated by summed weight, sorted canonically by
    (pre_idx, post_idx).
    """
    ids = np.asarray(body_ids, dtype=np.int64)
    n = ids.size
    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []

    dataset = pads.dataset(weights_path, format="feather")
    scanner = dataset.scanner(
        columns=["body_pre", "body_post", "weight"], batch_size=batch_rows
    )
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        pre = df["body_pre"].to_numpy(dtype=np.int64)
        post = df["body_post"].to_numpy(dtype=np.int64)
        ip = np.searchsorted(ids, pre)
        ip_ok = (ip < n) & (ids[np.minimum(ip, n - 1)] == pre)
        ii = np.searchsorted(ids, post[ip_ok])
        ii_ok = (ii < n) & (ids[np.minimum(ii, n - 1)] == post[ip_ok])
        pre_parts.append(ip[ip_ok][ii_ok].astype(np.int32))
        post_parts.append(ii[ii_ok].astype(np.int32))
        w_parts.append(df["weight"].to_numpy(dtype=np.int64)[ip_ok][ii_ok])

    if not pre_parts:
        e = np.empty(0, dtype=np.int32)
        return e, e.copy(), np.empty(0, dtype=np.int64)

    pre_idx = np.concatenate(pre_parts)
    post_idx = np.concatenate(post_parts)
    weight = np.concatenate(w_parts)

    # Canonical order + duplicate aggregation (sum). Vectorized: lexsort,
    # then sum runs of identical (pre, post).
    order = np.lexsort((post_idx, pre_idx))
    pre_idx, post_idx, weight = pre_idx[order], post_idx[order], weight[order]
    pair = (pre_idx.astype(np.int64) << np.int64(32)) | post_idx.astype(np.int64)
    new = np.empty(pair.size, dtype=bool)
    new[0] = True
    np.not_equal(pair[1:], pair[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    if starts.size != pair.size:  # duplicates exist -> aggregate
        sums = np.add.reduceat(weight, starts)
        pre_idx = pre_idx[starts]
        post_idx = post_idx[starts]
        weight = sums
    return pre_idx, post_idx, weight


# ---------------------------------------------------------------------------


@dataclass
class Chassis:
    """A connectome subgraph: node table + pre→post CSR adjacency.

    ``nodes`` columns: ``bodyId``, ``type``, ``instance``, ``somaSide``,
    ``population``, ``region``, ``neurotransmitter``, ``sign`` (1 / -1 / 0),
    sorted by ``bodyId``. ``adj[i, j]`` = synapse count from node i to node j.
    """

    nodes: pd.DataFrame
    adj: sp.csr_matrix
    meta: dict

    @property
    def n_neurons(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
def _chassis_from_frame(
    nodes: pd.DataFrame,
    meta: dict,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Chassis:
    """Assemble the Chassis: CSR from (pre_idx, post_idx, weight) node-index edges."""
    pre_idx, post_idx, weight = edges
    n = len(nodes)
    adj = sp.coo_matrix(
        (weight.astype(np.int64), (pre_idx.astype(np.int32), post_idx.astype(np.int32))),
        shape=(n, n),
        dtype=np.int64,
    ).tocsr()
    adj.sum_duplicates()
    return Chassis(nodes=nodes, adj=adj, meta=meta)


def _write_cache(
    nodes: pd.DataFrame,
    edges: tuple,
    meta: dict,
    cache_dir: Path,
    nodes_name: str,
    edges_name: str,
) -> None:
    """Write the parquet pair: node table (+ meta in schema metadata), edge table."""
    pre_idx, post_idx, weight = edges
    meta_bytes = json.dumps(meta, sort_keys=True).encode()
    schema = pa.Schema.from_pandas(nodes).with_metadata({_META_KEY: meta_bytes})
    cache_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = cache_dir / nodes_name
    with pa.OSFile(str(nodes_path), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        writer.write_table(pa.Table.from_pandas(nodes, schema=schema))
    edges_table = pa.table(
        {
            "body_pre": pa.array(nodes["bodyId"].to_numpy()[pre_idx], pa.int64()),
            "body_post": pa.array(nodes["bodyId"].to_numpy()[post_idx], pa.int64()),
            "pre_idx": pa.array(pre_idx, pa.int32()),
            "post_idx": pa.array(post_idx, pa.int32()),
            "weight": pa.array(weight, pa.int64()),
        }
    )
    edges_path = cache_dir / edges_name
    schema_e = edges_table.schema
    with pa.OSFile(str(edges_path), "wb") as sink, pa.ipc.new_file(sink, schema_e) as writer:
        writer.write_table(edges_table)


def _read_cache(cache_dir: Path, nodes_name: str, edges_name: str) -> Chassis | None:
    nodes_path = cache_dir / nodes_name
    edges_path = cache_dir / edges_name
    if not nodes_path.exists() or not edges_path.exists():
        return None
    table = feather.read_table(nodes_path, memory_map=True)
    meta_raw = table.schema.metadata or {}
    if _META_KEY not in meta_raw:
        raise ValueError(f"{nodes_path}: missing {_META_KEY!r} schema metadata")
    meta = json.loads(meta_raw[_META_KEY])
    nodes = table.to_pandas()
    e = feather.read_table(edges_path, memory_map=True).to_pandas()
    edges = (
        e["pre_idx"].to_numpy(dtype=np.int32),
        e["post_idx"].to_numpy(dtype=np.int32),
        e["weight"].to_numpy(dtype=np.int64),
    )
    return _chassis_from_frame(nodes, meta, edges)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


_PRED_DOC = {
    "uPN": "class == 'ALPN' and not type.startswith('M_') (multiglomerular excluded)",
    "KC": "class == 'Kenyon_Cell'",
    "MBON": "class == 'MBON'",
    "DN": "superclass in {'descending_neuron', 'descending_neuron_tbc'}",
}


def build_stripped_chassis(data_dir: Path | None = None, cache: bool = True) -> Chassis:
    """Extract the stripped chassis from the raw feather tables.

    Deterministic: sorted bodyIds, canonical edge order, stable cache bytes.
    With ``cache=True`` the result is written to the parquet pair consumed by
    :func:`load_stripped_chassis`.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    ann_all = feather.read_table(data_dir / _ANNOTATIONS).to_pandas()
    ann = ann_all[ann_all["superclass"].notna()].copy()

    labels = _population_labels(ann)
    selected = labels["population"].notna()
    ann = ann.loc[selected].reset_index(drop=True)
    labels = labels.loc[selected].reset_index(drop=True)

    nt = _nt_signs(ann, data_dir)
    body_ids = ann["bodyId"].to_numpy(dtype=np.int64)
    order = np.argsort(body_ids, kind="stable")
    body_ids = body_ids[order]

    nodes = pd.DataFrame(
        {
            "bodyId": body_ids,
            "type": ann["type"].to_numpy()[order],
            "instance": ann["instance"].to_numpy()[order],
            "somaSide": ann["somaSide"].to_numpy()[order],
            "population": labels["population"].to_numpy()[order],
            "region": labels["region"].to_numpy()[order],
            "neurotransmitter": nt["neurotransmitter"].to_numpy()[order],
            "sign": nt["sign"].to_numpy()[order],
        }
    )

    edges = _extract_edges(body_ids, data_dir / _WEIGHTS)
    pre_idx, post_idx, weight = edges
    total_syn = int(weight.sum())

    dims = nodes["population"].value_counts().sort_index().to_dict()
    glomeruli = sorted(
        {_glomerulus(t) for t in nodes.loc[nodes.population == "uPN", "type"]}
    )
    region_counts = nodes["region"].value_counts().sort_index().to_dict()
    sign_counts = nodes["sign"].value_counts().sort_index().to_dict()
    n = len(nodes)
    tier_counts = nt["nt_tier"].value_counts().sort_index().to_dict()

    meta = {
        "kind": "stripped-chassis",
        "n_neurons": n,
        "n_edges": int(pre_idx.size),
        "total_synapses": total_syn,
        "population_dims": {k: int(v) for k, v in dims.items()},
        "glomeruli": glomeruli,
        "n_glomeruli": len(glomeruli),
        "region_counts": {k: int(v) for k, v in region_counts.items()},
        "sign_counts": {str(k): int(v) for k, v in sign_counts.items()},
        "nt_tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "predicates": {name: (pat or _PRED_DOC[name]) for name, _, pat in _POPULATIONS},
        "sign_map": NT_SIGN,
        "adjacency": "pre->post CSR, weight = synapse count, self-loops kept",
        "note_glutamate": "glutamate treated inhibitory (Shiu et al. 2024 convention); contested",
    }
    chassis = _chassis_from_frame(nodes, meta, edges)
    if cache:
        _write_cache(nodes, edges, meta, data_dir, CHASSIS_NODES, CHASSIS_EDGES)
    return chassis


def build_whole_fly(data_dir: Path | None = None, cache: bool = True) -> Chassis:
    """Whole-fly graph: every proofread neuron (non-null superclass).

    Same node schema as the stripped chassis (population/region = ``whole``/
    ``brain``), same deterministic construction. Heavy: scans the full edge
    table. Cached as the ``whole-fly`` parquet pair.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    ann_all = feather.read_table(data_dir / _ANNOTATIONS).to_pandas()
    ann = ann_all[ann_all["superclass"].notna()].copy()

    nt = _nt_signs(ann, data_dir)
    body_ids = ann["bodyId"].to_numpy(dtype=np.int64)
    order = np.argsort(body_ids, kind="stable")
    body_ids = body_ids[order]

    nodes = pd.DataFrame(
        {
            "bodyId": body_ids,
            "type": ann["type"].to_numpy()[order],
            "instance": ann["instance"].to_numpy()[order],
            "somaSide": ann["somaSide"].to_numpy()[order],
            "population": "whole",
            "region": "brain",
            "neurotransmitter": nt["neurotransmitter"].to_numpy()[order],
            "sign": nt["sign"].to_numpy()[order],
        }
    )

    edges = _extract_edges(body_ids, data_dir / _WEIGHTS)
    pre_idx, post_idx, weight = edges
    meta = {
        "kind": "whole-fly",
        "n_neurons": len(nodes),
        "n_edges": int(pre_idx.size),
        "total_synapses": int(weight.sum()),
        "sign_map": NT_SIGN,
        "note_glutamate": "glutamate treated inhibitory (Shiu et al. 2024 convention); contested",
    }
    chassis = _chassis_from_frame(nodes, meta, edges)
    if cache:
        _write_cache(nodes, edges, meta, data_dir, WHOLE_NODES, WHOLE_EDGES)
    return chassis


# ---------------------------------------------------------------------------
# Loaders (contract API)
# ---------------------------------------------------------------------------


def load_stripped_chassis(path: str | Path | None = None) -> Chassis:
    """Load the stripped chassis (node table + pre→post CSR adjacency).

    Reads the cached parquet pair; falls back to extracting from the raw
    feather tables (and caching) when the cache is absent. ``path`` overrides
    the cache directory.
    """
    cache_dir = Path(path) if path else DATA_DIR
    chassis = _read_cache(cache_dir, CHASSIS_NODES, CHASSIS_EDGES)
    return chassis if chassis is not None else build_stripped_chassis(cache_dir, cache=True)


def load_whole_fly(path: str | Path | None = None) -> Chassis:
    """Load the whole-fly graph (all 166,700 proofread neurons).

    Reads its own cached parquet pair under ``path`` (default
    ``data/connectome``); falls back to a full extraction when absent.
    """
    cache_dir = Path(path) if path else DATA_DIR
    chassis = _read_cache(cache_dir, WHOLE_NODES, WHOLE_EDGES)
    return chassis if chassis is not None else build_whole_fly(cache_dir, cache=True)
