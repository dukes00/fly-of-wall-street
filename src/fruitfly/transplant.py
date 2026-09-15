"""T12: D19 weight transplant — larval (stripped-chassis) plasticity into the
whole-fly brain.

DESIGN §2 Phase B / D19: the KC→MBON weight matrix trained on the stripped
chassis (27,115 neurons, artifact ``data/fly-larval-weights.npz``) is copied
into the whole-fly chassis (166,700 neurons) for behavioral validation. The
direct-copy branch is the default; fine-tuning only if the ≥90% decision-
agreement gate fails (D19 is revocable — the validation report states which
branch was taken).

How the mapping works (both chassis come from the same MaleCNS v1.0 release):

- A trained artifact's row/column identity is *implicit*: rows follow the
  stripped chassis's KC node order (nodes sorted by ``bodyId``), columns the
  MBON node order (:class:`~fruitfly.neuromod.Plasticity` documents the view).
  The artifact's chassis fingerprint pins exactly which node table that is,
  so :func:`transplant_weights` verifies the fingerprint against the stripped
  chassis supplying the bodyIds and refuses a foreign artifact.
- Whole-fly nodes are matched to stripped nodes by ``bodyId``. Both node
  tables are drawn from the same annotation release and the stripped chassis
  is a bodyId subset of the whole fly (verified at transplant time and
  reported), so the match is exact by construction.
- The whole-fly chassis cache stores ``population="whole"`` for every node
  (T11) — the loop's sensory encoders, ``upn_channels`` and the
  KC/MBON/PAM/PPL1 readouts all key off the population column, so
  :func:`label_whole_chassis` transfers the stripped chassis's population and
  region labels onto the matched whole-fly bodyIds first. Unlabeled
  (unmatched) whole-fly nodes keep ``population="whole"``: they take part in
  the LIF dynamics through their structural synapses but are not driven by
  the encoders and not read by the KC→MBON view.
- **Absent-KC/MBON handling (documented per D19 review):** whole-fly
  KC/MBON-labeled nodes absent from the stripped chassis cannot occur when
  both caches come from the same release (the stripped builder labels every
  annotated Kenyon cell / MBON, and the whole fly is a superset) — the
  measured match rate is reported and must be 100% for the transplant to be
  meaningful. Defensively, the code still handles the case: a matched pair
  the artifact never had a synapse for (no stripped support) but the whole
  fly does have keeps its **baseline weight from the whole-fly synapse
  count** (log1p-compressed, mean-1-scaled — the fresh-:class:`Plasticity`
  initialization); a trained weight whose whole-fly synapse is missing
  (never observed in practice, checked and counted) is dropped — plasticity
  never invents synapses the connectome lacks.

Determinism: bodyId matching is searchsorted over the sorted node tables;
weight placement is a fixed-order numpy copy; the module has no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fruitfly.connectome import Chassis, load_stripped_chassis
from fruitfly.neuromod import Plasticity
from fruitfly.train import LarvalWeights, chassis_fingerprint, load_larval_weights

__all__ = [
    "TransplantReport",
    "TransplantResult",
    "label_whole_chassis",
    "match_rate",
    "transplant_weights",
]


# ---------------------------------------------------------------------------
# Chassis labeling
# ---------------------------------------------------------------------------


def label_whole_chassis(whole: Chassis, stripped: Chassis) -> Chassis:
    """Transfer stripped-chassis population/region labels onto whole-fly nodes.

    Whole-fly nodes are matched to stripped nodes by ``bodyId``; matched nodes
    take the stripped chassis's ``population`` and ``region``, unmatched nodes
    keep ``"whole"``/``"brain"``. The result's ``meta`` inherits the whole
    fly's meta plus the stripped chassis's glomerulus channel table (the
    smell encoder indexes identity profiles by that exact list). The node
    bodyIds and CSR adjacency are untouched, so chassis fingerprints are
    unchanged.

    Returns a new :class:`Chassis` sharing the whole fly's adjacency array
    (no copy of the 781 MB edge table).
    """
    whole_ids = whole.nodes["bodyId"].to_numpy()
    stripped_ids = stripped.nodes["bodyId"].to_numpy()

    population = np.full(len(whole_ids), "whole", dtype=object)
    region = np.full(len(whole_ids), "brain", dtype=object)

    # searchsorted both ways over sorted bodyIds (node tables are sorted).
    pos = np.searchsorted(stripped_ids, whole_ids)
    hit = (pos < len(stripped_ids))
    hit[hit] = stripped_ids[pos[hit]] == whole_ids[hit]
    population[hit] = stripped.nodes["population"].to_numpy()[pos[hit]]
    region[hit] = stripped.nodes["region"].to_numpy()[pos[hit]]

    nodes = whole.nodes.copy()
    nodes["population"] = population
    nodes["region"] = region

    meta = dict(whole.meta)
    meta["glomeruli"] = list(stripped.meta["glomeruli"])
    meta["n_glomeruli"] = len(meta["glomeruli"])
    meta["labeled_from"] = "stripped-chassis"
    return Chassis(nodes=nodes, adj=whole.adj, meta=meta)


def match_rate(stripped_ids: np.ndarray, whole_ids: np.ndarray) -> float:
    """Fraction of ``stripped_ids`` present in ``whole_ids`` (both sorted)."""
    if stripped_ids.size == 0:
        return 1.0
    pos = np.searchsorted(whole_ids, stripped_ids)
    inside = pos < whole_ids.size
    inside[inside] = whole_ids[pos[inside]] == stripped_ids[inside]
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransplantReport:
    """Measured transplant statistics (rendered into reports/t12-transplant.md)."""

    #: Stripped-chassis weight view dimensions.
    n_kc_stripped: int
    n_mbon_stripped: int
    #: Labeled whole-fly weight view dimensions.
    n_kc_whole: int
    n_mbon_whole: int
    #: bodyId match rates (stripped nodes found in the whole fly).
    kc_match_rate: float
    mbon_match_rate: float
    #: Matched (KC, MBON) pairs and their fate.
    pairs_total: int
    pairs_copied: int              #: trained value placed verbatim
    pairs_baseline_filled: int     #: no stripped support, whole support -> baseline
    pairs_dropped_no_support: int  #: trained > 0 but the whole fly lacks the synapse
    #: Sum of copied trained weight (baseline-filled/dropped pairs excluded).
    copied_mass: float
    #: Matched-pair synapse counts identical between the two chassis.
    synapse_consistent: bool
    #: Artifact fingerprint verified against the stripped chassis.
    fingerprint_verified: bool


@dataclass(frozen=True)
class TransplantResult:
    """Transplanted brain: a ready :class:`Plasticity` on the labeled whole fly."""

    #: Plasticity over the labeled whole-fly chassis with the transplanted
    #: weights already in place (its ``.weights`` view).
    plasticity: Plasticity
    #: The labeled whole-fly chassis (pass this to the loop's chassis seam).
    labeled_chassis: Chassis
    report: TransplantReport

    @property
    def weights(self) -> np.ndarray:
        """The transplanted KC→MBON weight view (live Plasticity weights)."""
        return self.plasticity.weights


# ---------------------------------------------------------------------------
# The transplant
# ---------------------------------------------------------------------------


def transplant_weights(
    source: str | Path | LarvalWeights | np.ndarray,
    whole_chassis: Chassis,
    *,
    stripped_chassis: Chassis | None = None,
) -> TransplantResult:
    """Copy trained stripped-chassis KC→MBON weights into the whole-fly brain.

    ``source`` is the larval artifact path (fingerprint-verified against the
    stripped chassis), an already-loaded :class:`~fruitfly.train.LarvalWeights`,
    or a raw ``(n_kc, n_mbon)`` weight array whose row/column identity is the
    stripped chassis's KC/MBON node order (no fingerprint to verify — the
    caller owns provenance).

    ``stripped_chassis`` defaults to the cached stripped chassis; it supplies
    the bodyId identity of the artifact's rows and columns.

    Returns a :class:`TransplantResult`: a :class:`Plasticity` built on the
    labeled whole-fly chassis (baseline-from-synapse-counts everywhere, then
    the trained values placed on matched pairs), ready for behavioral replay
    via the loop's chassis seam, plus the measured :class:`TransplantReport`.
    """
    stripped = stripped_chassis if stripped_chassis is not None else load_stripped_chassis()

    fingerprint_verified = False
    if isinstance(source, (str, Path)):
        larval = load_larval_weights(source, chassis=stripped, verify=True)
        trained = larval.weights
        fingerprint_verified = True
    elif isinstance(source, LarvalWeights):
        trained = source.weights
        expected = chassis_fingerprint(stripped)
        if source.fingerprint != expected:
            raise ValueError(
                "larval artifact was trained on a different chassis "
                f"(artifact {source.fingerprint[:12]}… != stripped "
                f"{expected[:12]}…)"
            )
        fingerprint_verified = True
    else:
        trained = np.asarray(source, dtype=np.float64)

    # --- identity of the artifact's rows/columns (stripped chassis order) ---
    s_pop = stripped.nodes["population"].to_numpy()
    s_kc_idx = np.flatnonzero(s_pop == "KC")
    s_mbon_idx = np.flatnonzero(s_pop == "MBON")
    if trained.ndim != 2 or trained.shape != (s_kc_idx.size, s_mbon_idx.size):
        raise ValueError(
            f"trained weights shape {trained.shape} does not match the stripped "
            f"chassis KC→MBON view {(s_kc_idx.size, s_mbon_idx.size)}"
        )

    # --- labeled whole fly + its weight view ---
    labeled = label_whole_chassis(whole_chassis, stripped)
    plasticity = Plasticity(labeled)
    w_pop = labeled.nodes["population"].to_numpy()
    w_kc_idx = np.flatnonzero(w_pop == "KC")
    w_mbon_idx = np.flatnonzero(w_pop == "MBON")

    w_ids = labeled.nodes["bodyId"].to_numpy()
    s_ids = stripped.nodes["bodyId"].to_numpy()
    # Row/column indices within each brain's weight view, per matched pair.
    # Both views follow chassis node order, so the relative order of the
    # matched subsets is identical.
    kc_hit = np.isin(s_ids[s_kc_idx], w_ids)
    mbon_hit = np.isin(s_ids[s_mbon_idx], w_ids)
    view_rows_s = np.flatnonzero(kc_hit)          # artifact row -> matched
    view_cols_s = np.flatnonzero(mbon_hit)        # artifact col -> matched
    # Position within the whole weight view: whole KC rows are w_kc_idx in
    # node order; the matched stripped KCs appear in the same relative order.
    wkc_pos = np.searchsorted(w_ids[w_kc_idx], s_ids[s_kc_idx[view_rows_s]])
    wmbon_pos = np.searchsorted(w_ids[w_mbon_idx], s_ids[s_mbon_idx[view_cols_s]])

    kc_match_rate = match_rate(s_ids[s_kc_idx], w_ids)
    mbon_match_rate = match_rate(s_ids[s_mbon_idx], w_ids)

    # Structural support of the matched block in each brain, and the
    # synapse-count consistency receipt (same release => identical counts).
    s_counts = stripped.adj[s_kc_idx][:, s_mbon_idx][view_rows_s][:, view_cols_s].toarray()
    w_counts = (
        labeled.adj[w_kc_idx][:, w_mbon_idx][wkc_pos][:, wmbon_pos].toarray()
    )
    s_support = s_counts > 0
    w_support = w_counts > 0
    synapse_consistent = bool(np.array_equal(s_counts, w_counts))

    # --- the copy (fixed order, deterministic) ---
    weights = plasticity.weights
    trained_block = trained[view_rows_s][:, view_cols_s]
    block = weights[np.ix_(wkc_pos, wmbon_pos)]
    block = np.where(s_support, trained_block, block)
    # Plasticity never invents synapses: drop trained mass where the whole
    # fly has no structural KC→MBON synapse (counted; empty in practice).
    dropped = int(np.count_nonzero((block > 0.0) & ~w_support))
    block = np.where(w_support, block, 0.0)
    weights[np.ix_(wkc_pos, wmbon_pos)] = block

    copied = int(np.count_nonzero((trained_block > 0.0) & s_support & w_support))
    baseline_filled = int(np.count_nonzero(~s_support & w_support))
    report = TransplantReport(
        n_kc_stripped=int(s_kc_idx.size),
        n_mbon_stripped=int(s_mbon_idx.size),
        n_kc_whole=int(w_kc_idx.size),
        n_mbon_whole=int(w_mbon_idx.size),
        kc_match_rate=kc_match_rate,
        mbon_match_rate=mbon_match_rate,
        pairs_total=int(wkc_pos.size * wmbon_pos.size),
        pairs_copied=copied,
        pairs_baseline_filled=baseline_filled,
        pairs_dropped_no_support=dropped,
        copied_mass=float(block.sum()),
        synapse_consistent=synapse_consistent,
        fingerprint_verified=fingerprint_verified,
    )
    return TransplantResult(
        plasticity=plasticity, labeled_chassis=labeled, report=report
    )


# ---------------------------------------------------------------------------
# Behavioral agreement (D19: replay held-out days through both brains)
# ---------------------------------------------------------------------------


#: Decision-action buckets compared by the agreement metric (D19 wording:
#: "BUY/SELL/pass decision agreement"; ``add`` is a BUY of an held position).
_ACTION_BUCKET = {"buy": "BUY", "add": "BUY", "sell": "SELL", "pass": "PASS"}


def action_bucket(action: str) -> str:
    """Coarsen a loop action to the D19 comparison buckets BUY/SELL/pass."""
    return _ACTION_BUCKET.get(action, action.upper())


def decision_log(events_path: str | Path) -> list[tuple[str, str, str]]:
    """``[(ts, ticker, action), ...]`` in file order from a run's events.jsonl.

    One decision per encounter bar, so ``ts`` is unique within a run.
    """
    import json

    out: list[tuple[str, str, str]] = []
    with open(events_path) as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") == "decision":
                out.append((event["ts"], event["ticker"], event["action"]))
    return out


def agreement_table(
    decisions_a: list[tuple[str, str, str]],
    decisions_b: list[tuple[str, str, str]],
) -> tuple[list[dict], float]:
    """Per-day decision agreement between two brains' replay logs.

    Decisions are keyed by bar timestamp (the encounter schedule is
    seed/market-determined and identical across brains; the ticker each brain
    rotates onto may differ, which the ticker-agreement column reports
    separately). The primary metric per D19 is the action-bucket agreement
    over the bars both brains decided on.

    Returns ``(per_day_rows, mean_action_agreement)`` where each row is
    ``{"day", "bars", "action_agreement", "ticker_agreement",
    "action_and_ticker_agreement"}`` and the mean weights every replayed day
    equally.
    """
    map_a = {ts: (ticker, action) for ts, ticker, action in decisions_a}
    map_b = {ts: (ticker, action) for ts, ticker, action in decisions_b}
    by_day: dict[str, list[str]] = {}
    for ts in map_a.keys() & map_b.keys():
        by_day.setdefault(ts[:10], []).append(ts)

    rows: list[dict] = []
    for day in sorted(by_day):
        stamps = sorted(by_day[day])
        n = len(stamps)
        if n == 0:
            continue
        same_action = same_ticker = same_both = 0
        for ts in stamps:
            ta, aa = map_a[ts]
            tb, ab = map_b[ts]
            same_action += action_bucket(aa) == action_bucket(ab)
            same_ticker += ta == tb
            same_both += ta == tb and action_bucket(aa) == action_bucket(ab)
        rows.append(
            {
                "day": day,
                "bars": n,
                "action_agreement": same_action / n,
                "ticker_agreement": same_ticker / n,
                "action_and_ticker_agreement": same_both / n,
            }
        )
    # Mean decision agreement over the replayed days, every day weighted
    # equally (the D19 headline number); the bar-weighted overall figure is
    # derivable from the per-day rows.
    mean = (
        sum(r["action_agreement"] for r in rows) / len(rows) if rows else 0.0
    )
    return rows, float(mean)
