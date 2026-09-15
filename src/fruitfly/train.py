"""T9: larval training — backtest-driven plasticity training + artifact I/O.

The larval stage (DESIGN §9) is backtest training: ``train_larval`` runs
``loop.run_backtest`` over a historical window at a fixed seed and persists
the surviving fly's KC→MBON weights to ``data/fly-larval-weights.npz``
(gitignored, like everything under ``data/``). The adult stage restores them
with ``load_larval_weights`` and passes them to ``run_backtest`` via
``BacktestConfig.initial_weights`` — the loop injects them into a fresh
``Plasticity`` before the first bar, so a trained brain makes different
decisions than a structural-baseline fly from bar one (the acceptance sanity
probe; see ``scripts/calibrate.py probe``).

What is persisted: the learned weight matrix only. The eligibility trace and
habituation factors are run-transient state and reset on restore (the adult
fly wakes with fresh attention, matching the post-sleep reset semantics);
the structural ``baseline`` is re-derived from the chassis, never trusted
from disk.

Artifact format (``.npz``, readable by ``np.load(..., allow_pickle=False)``):

===============  =============================================  ==============
member           content                                        dtype
===============  =============================================  ==============
``weights``      (n_kc, n_mbon) float64 KC→MBON weight view     float64
``fingerprint``  sha256 over the chassis bodyIds + CSR arrays   <U64 unicode
``meta_*``       provenance strings (seed, window, params, …)   unicode
===============  =============================================  ==============

Determinism: ``np.savez`` stamps zip members with the wall clock, so it is
NOT byte-reproducible. ``save_larval_weights`` therefore writes the same
.npz layout by hand — ``numpy.lib.format.write_array`` blobs stored with
``zipfile.ZIP_STORED`` under fixed member order and a fixed member timestamp
(1980-01-01) — making a same-seed double run byte-identical (verified in
``tests/test_train.py``). Key order is fixed by construction; no member ever
needs ``allow_pickle`` (strings are plain unicode arrays).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fruitfly import loop as _loop
from fruitfly.loop import BacktestConfig, RunResult

__all__ = [
    "LARVAL_WEIGHTS_PATH",
    "TRAIN_SEED",
    "LarvalWeights",
    "TrainResult",
    "chassis_fingerprint",
    "save_larval_weights",
    "load_larval_weights",
    "train_larval",
    "decision_map",
]

#: Default artifact path (relative to the repo root; gitignored via data/).
LARVAL_WEIGHTS_PATH = Path("data/fly-larval-weights.npz")

#: Default training seed (matches the T7 receipts so runs are comparable).
TRAIN_SEED = 7
#: Fixed zip member timestamp for deterministic archives (zip's floor).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Chassis fingerprint
# ---------------------------------------------------------------------------


def chassis_fingerprint(chassis) -> str:
    """sha256 over the chassis identity: node bodyIds + CSR adjacency arrays.

    Two ``Chassis`` objects with the same fingerprint have the same node
    table and the same synapse graph, so a trained weight view indexed by
    chassis node order transfers losslessly. Everything else (region labels,
    neurotransmitter annotations) is derivable from the same cache file and
    cannot drift independently.
    """
    h = hashlib.sha256()
    h.update(np.asarray(chassis.nodes["bodyId"]).tobytes())
    adj = chassis.adj.tocsr()
    h.update(np.asarray(adj.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(adj.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(adj.indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(adj.data).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Deterministic artifact I/O
# ---------------------------------------------------------------------------


def _npy_bytes(values) -> bytes:
    """Serialize one array to .npy bytes without pickling."""
    arr = np.asarray(values)
    if arr.dtype.kind not in "fiuU b":
        raise TypeError(f"unsupported artifact member dtype {arr.dtype}")
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def save_larval_weights(
    path: str | Path,
    weights: np.ndarray,
    fingerprint: str,
    meta: dict[str, str],
) -> Path:
    """Write a deterministic .npz artifact (see module docstring).

    Member order is fixed (``weights``, ``fingerprint``, then ``meta_*`` in
    sorted-key order) so the byte stream depends only on the content.
    """
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2-D (n_kc, n_mbon), got {w.shape}")
    members: list[tuple[str, bytes]] = [("weights", _npy_bytes(w))]
    members.append(("fingerprint", _npy_bytes(str(fingerprint))))
    for key in sorted(meta):
        members.append((f"meta_{key}", _npy_bytes(str(meta[key]))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, blob in members:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, blob)
    return path


@dataclass(frozen=True)
class LarvalWeights:
    """A loaded larval artifact: weights + chassis fingerprint + provenance."""

    weights: np.ndarray
    fingerprint: str
    meta: dict[str, str]


def load_larval_weights(
    path: str | Path = LARVAL_WEIGHTS_PATH,
    chassis=None,
    *,
    verify: bool = True,
) -> LarvalWeights:
    """Load the larval artifact; optionally verify it against ``chassis``.

    With ``chassis`` given (and ``verify``), a fingerprint mismatch raises —
    an artifact trained on a different brain must never be injected. Pass
    ``verify=False`` to inspect a foreign artifact without a chassis.
    """
    with np.load(path, allow_pickle=False) as z:
        weights = np.asarray(z["weights"], dtype=np.float64)
        fingerprint = z["fingerprint"].item()
        meta = {k[len("meta_"):]: z[k].item() for k in z.files
                if k.startswith("meta_")}
    if chassis is not None and verify:
        actual = chassis_fingerprint(chassis)
        if actual != fingerprint:
            raise ValueError(
                "larval artifact was trained on a different chassis "
                f"(artifact {fingerprint[:12]}… != current {actual[:12]}…)"
            )
    return LarvalWeights(weights=weights, fingerprint=fingerprint, meta=meta)


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one larval training run."""

    run: RunResult
    artifact: Path
    weights: np.ndarray
    fingerprint: str
    meta: dict[str, str]


def train_larval(
    config: BacktestConfig,
    out_path: str | Path = LARVAL_WEIGHTS_PATH,
) -> TrainResult:
    """Run the larval backtest and persist the trained KC→MBON weights.

    Uses the loop's chassis seam (``loop._load_chassis``) so tests can run
    the whole driver on a synthetic brain. The persisted weights are the
    last-hatched fly's (a death late in the window means the artifact carries
    the fresh hatch's baseline plus whatever it learned after hatching).
    """
    result = _loop.run_backtest(config)
    if result.final_weights is None:  # pragma: no cover - defensive
        raise RuntimeError("run_backtest returned no final weights")
    chassis = _loop._load_chassis()
    fingerprint = chassis_fingerprint(chassis)
    meta = {
        "seed": str(config.seed),
        "start": config.start,
        "end": config.end,
        "position_cap": str(config.position_cap),
        "death_threshold": str(config.death_threshold),
        "final_equity": f"{result.final_equity:.2f}",
        "n_bars": str(result.n_bars),
        "n_deaths": str(result.n_deaths),
    }
    artifact = save_larval_weights(out_path, result.final_weights, fingerprint, meta)
    return TrainResult(
        run=result,
        artifact=artifact,
        weights=result.final_weights,
        fingerprint=fingerprint,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Decision readout (sanity probe)
# ---------------------------------------------------------------------------


def decision_map(events_path: str | Path) -> dict[tuple[str, str], tuple[str, str]]:
    """``{(ts, ticker): (action, reason)}`` from a run's events.jsonl.

    The probe key is the encounter identity: same bar, same plume. Two runs
    over the same window produce comparable maps (the rotation order is
    seed-locked, so the encounter set matches).
    """
    import json

    out: dict[tuple[str, str], tuple[str, str]] = {}
    with open(events_path) as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") == "decision":
                out[(event["ts"], event["ticker"])] = (event["action"], event["reason"])
    return out


def _config_dict(config: BacktestConfig) -> dict:
    d = asdict(config)
    return {k: v for k, v in d.items() if not k.startswith("_")}
