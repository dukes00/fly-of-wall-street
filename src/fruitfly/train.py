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

Artifact meta contract (TRAINING2 A9, additive): ``train_larval`` records
the credit-knob echo (``META_KNOBS``), the trained ``basket`` (JSON list of
the loop's bound ``BASKET`` symbols), and — advantage mode only — the
Option-B ``initial_baselines`` (JSON ``"sign|vol|dir"`` int triple -> float,
for multi-run warm-starts). Existing loaders ignore unknown meta keys; a
legacy artifact without the new keys still loads unchanged.

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
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fruitfly import loop as _loop
from fruitfly.loop import BacktestConfig, RunResult

__all__ = [
    "LARVAL_WEIGHTS_PATH",
    "TRAIN_SEED",
    "META_KNOBS",
    "LarvalWeights",
    "TrainResult",
    "add_knob_args",
    "chassis_fingerprint",
    "decode_baselines",
    "encode_baselines",
    "knob_kwargs",
    "meta_knobs",
    "save_larval_weights",
    "load_larval_weights",
    "train_larval",
    "decision_map",
]


LARVAL_WEIGHTS_PATH = Path("data/fly-larval-weights.npz")

#: Default training seed (matches the T7 receipts so runs are comparable).
TRAIN_SEED = 7
#: Fixed zip member timestamp for deterministic archives (zip's floor).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

#: Artifact ``chassis`` meta label -> the Chassis ``meta["kind"]`` it must
#: match at load time (the fingerprint guard cannot distinguish two chassis
#: built on identical graphs; the label can).
_CHASSIS_KINDS = {"stripped": "stripped-chassis", "whole": "whole-fly"}


# ---------------------------------------------------------------------------
# TRAINING2 artifact meta contract (A9): credit knobs + basket + baselines
# ---------------------------------------------------------------------------


#: The TRAINING2 credit knobs ``train_larval`` records in the artifact meta
#: and ``scripts/eval_brains.py`` asserts against the eval runtime config.
#: Canonical order — both the meta writer and the report renderer emit in
#: this order. Additive: existing artifact loaders ignore unknown meta keys.
META_KNOBS: tuple[str, ...] = (
    "entry_credit", "trade_credit_mode", "mix_weight", "daily_observe",
    "horizon_bars", "r_scale", "a_scale", "baseline_alpha",
    "miss_weight", "avoid_correct_weight",
    "id_scale", "reward_gain", "punishment_gain",
)

#: CLI type of each pass-through knob (everything else is a string enum).
KNOB_TYPES: dict[str, type] = {
    "horizon_bars": int,
    "id_scale": float,
    "r_scale": float,
    "a_scale": float,
    "baseline_alpha": float,
    "miss_weight": float,
    "avoid_correct_weight": float,
    "mix_weight": float,
    "reward_gain": float,
    "punishment_gain": float,
}

#: Per-knob help for the shared CLI pass-through flags.
KNOB_HELP: dict[str, str] = {
    "entry_credit": "Entry-credit objective: off | forecast | advantage.",
    "trade_credit_mode": "Per-exit gate credit: realized | forecast | mix.",
    "mix_weight": "Blend weight of the forecast term in trade_credit_mode=mix.",
    "daily_observe": "Daily pooled sugar/shock ritual: on | off.",
    "horizon_bars": "Entry-forecast horizon in 1-min bars (default 30).",
    "r_scale": "Forecast tanh scale: tanh(r_fwd / r_scale).",
    "a_scale": "Advantage tanh scale (Option B).",
    "baseline_alpha": "Bucket-baseline EMA rate (Option B).",
    "miss_weight": "Punishment weight for missed winners.",
    "avoid_correct_weight": "Reward weight for correct avoids.",
    "id_scale": "Identity-profile scale in [0, 1] (1.0 = unattenuated).",
    "reward_gain": "Gate reward gain.",
    "punishment_gain": "Gate punishment gain.",
}


def add_knob_args(parser) -> None:
    """Add the TRAINING2 credit-knob pass-through flags to ``parser``.

    Every flag defaults to ``None`` (= keep the ``BacktestConfig`` no-op
    default), so the existing CLIs are bit-compatible at defaults.
    """
    for name in META_KNOBS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=KNOB_TYPES.get(name, str),
            default=None,
            help=KNOB_HELP.get(name, name) + " (default: BacktestConfig value).",
        )


def knob_kwargs(namespace) -> dict:
    """CLI namespace with ``None``-defaults -> ``BacktestConfig`` kwargs.

    Only knobs explicitly given on the command line are passed through;
    absent flags keep the config defaults (no-op pass-through).
    """
    return {
        name: getattr(namespace, name)
        for name in META_KNOBS
        if getattr(namespace, name, None) is not None
    }


def meta_knobs(config: BacktestConfig) -> dict[str, str]:
    """The A9 meta echo: every credit knob as its canonical meta string.

    Knobs the installed ``BacktestConfig`` does not carry yet are skipped
    (they cannot have influenced the run); the eval-side assertion skips
    the same set, so both sides stay consistent.
    """
    return {
        name: str(getattr(config, name))
        for name in META_KNOBS
        if hasattr(config, name)
    }


def meta_basket(config: BacktestConfig) -> list[str]:
    """The effective training basket.

    ``config.basket`` (the parsed ``--basket-file`` list) when given,
    otherwise the loop's bound ``BASKET`` module attribute at run time.
    """
    explicit = getattr(config, "basket", None)
    if explicit is not None:
        return [str(symbol) for symbol in explicit]
    return [str(symbol) for symbol in _loop.BASKET]


def encode_baselines(baselines) -> dict[str, float]:
    """Bucket baselines -> the JSON-safe meta form ``"sign|vol|dir"`` -> float."""
    out: dict[str, float] = {}
    for key, value in baselines.items():
        if isinstance(key, (tuple, list)):
            sign, vol, direction = (int(part) for part in key)
            key = f"{sign}|{vol}|{direction}"
        out[str(key)] = float(value)
    return out


def decode_baselines(raw: str | dict) -> dict[tuple[int, int, int], float]:
    """The meta form back into ``{(sign, vol, dir): float}``."""
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    out: dict[tuple[int, int, int], float] = {}
    for key, value in parsed.items():
        sign, vol, direction = (int(part) for part in str(key).split("|"))
        out[(sign, vol, direction)] = float(value)
    return out


def _meta_baselines(config: BacktestConfig, result: RunResult) -> str | None:
    """Option-B warm-start baselines for the meta (JSON string), or None."""
    if getattr(config, "entry_credit", "off") != "advantage":
        return None
    baselines = getattr(result, "baselines", None)
    if not baselines:
        return None
    return json.dumps(encode_baselines(baselines), sort_keys=True)




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
    an artifact trained on a different brain must never be injected — and so
    does a chassis-kind mismatch between the artifact's ``chassis`` meta
    label and the runtime chassis's ``meta["kind"]`` (fingerprints cannot
    distinguish two chassis built on identical graphs; labels can). Legacy
    artifacts without the label and chassis fixtures without a kind are
    checked by fingerprint only. Pass ``verify=False`` to inspect a foreign
    artifact without a chassis.
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
        artifact_chassis = meta.get("chassis")
        kind = chassis.meta.get("kind")
        if (
            artifact_chassis is not None
            and kind is not None
            and _CHASSIS_KINDS.get(artifact_chassis, artifact_chassis) != kind
        ):
            raise ValueError(
                "larval artifact was trained on a different chassis kind "
                f"(artifact chassis {artifact_chassis!r} != current {kind!r})"
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
    chassis = _loop._resolve_chassis(config)
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
        "chassis": config.chassis,
        "std_beta": str(config.std_beta),
        "std_tau_rec_ms": str(config.std_tau_rec_ms),
        **meta_knobs(config),
        "basket": json.dumps(meta_basket(config)),
    }
    baselines_meta = _meta_baselines(config, result)
    if baselines_meta is not None:
        meta["initial_baselines"] = baselines_meta
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
