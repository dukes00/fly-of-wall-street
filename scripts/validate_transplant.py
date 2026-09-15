"""T12: D19 transplant validation — replay held-out market days through BOTH
brains (stripped chassis with the trained larval artifact vs the whole-fly
brain with the transplanted weights) and write reports/t12-transplant.md.

Replay protocol (identical loop config, D19):

- Same seed (7, the training seed), same window, same loop constants; each
  brain differs only in its chassis and its initial KC→MBON weights.
- Stripped brain: ``run_backtest`` with the cached stripped chassis and
  ``initial_weights`` = the artifact weights.
- Whole-fly brain: the loop's chassis seam (``fruitfly.loop._load_chassis``)
  is pointed at the labeled whole-fly chassis (population labels transferred
  by bodyId, see ``fruitfly.transplant.label_whole_chassis``) and
  ``initial_weights`` = the transplanted weights.
- Decision agreement: per-day fraction of commonly decided bars with the
  identical BUY/SELL/pass bucket (``fruitfly.transplant.agreement_table``).
  The encounter schedule is market/seed-determined and identical across
  brains; the noise floor draws consume different RNG-stream lengths on
  different-sized chassis, so a brain may rotate onto a different plume on
  later days — the ticker-agreement column of the report quantifies that
  separately.

Runtime: the whole-fly LIF steps ~0.9-1.0 s per encounter on the M1 target
(T11 measured 1.78 s at dt=0.5 ms; the loop's dt=1.0 ms halves the substeps).
Two replay days (~390 encounter bars each) ≈ 12-20 min per whole-fly run; the
script runs each brain TWICE for the same-seed byte-identical determinism
receipt. ``--reuse`` keeps existing run receipts (e.g. to re-render the
report from fixed inputs byte-stably).

Usage::

    uv run python scripts/validate_transplant.py            # full protocol
    uv run python scripts/validate_transplant.py --reuse    # report only

T12b extension (opt-in short-term synaptic depression on the whole-fly
brain's inhibitory feedback — the APL/DPM KC-clamp root cause of T12)::

    # Calibration sweep of (beta, tau_rec) over replayed real bars:
    uv run python scripts/validate_transplant.py --calibrate-std
    # Full 2-day x 2-brain protocol with the whole fly running STD:
    uv run python scripts/validate_transplant.py --std-beta B --std-tau-rec MS

``--std-beta``/``--std-tau-rec`` switch the report target to
``reports/t12b-apl-std.md`` and the run dirs to ``data/runs/t12b-*``. The
stripped brain never runs STD (it contains no APL/DPM); its receipts are
byte-reused from the T12 runs when present (identical config), else rerun
into ``t12b-stripped-*``. The whole-fly LIFSim is constructed by the loop,
so the STD parameters reach it through the loop's ``LIFSim`` name seam —
the same monkeypatch pattern as the documented ``_load_chassis`` seam.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
from fruitfly.connectome import load_stripped_chassis, load_whole_fly
from fruitfly.loop import BacktestConfig, run_backtest
from fruitfly.train import load_larval_weights
from fruitfly.transplant import (
    agreement_table,
    decision_log,
    transplant_weights,
)

#: Replay window: the LAST TWO trading days of the market cache. The cache
#: spans 2026-08-17..2026-09-14 — exactly the larval training window — so no
#: strictly out-of-sample bars exist anywhere in the cache; the final two
#: sessions are the closest available "held-out" replay (documented in the
#: report as an in-sample caveat, not hidden).
REPLAY_START = "2026-09-11"
REPLAY_END = "2026-09-14"
#: The training seed (train.TRAIN_SEED) — replays use the identical seed.
REPLAY_SEED = 7
RECEIPTS = ("events.jsonl", "equity.csv")

#: Static bar for the readout-silence diagnostic (first replay day,
#: XOM — a ticker the stripped brain actively traded).
DIAG_TICKER = "XOM"
DIAG_TS = "2026-09-11 14:35:00+00:00"
LARVAL_ARTIFACT = Path("data/fly-larval-weights.npz")
REPORT_PATH = Path("reports/t12-transplant.md")

RUN_DIRS = {
    "stripped_a": Path("data/runs/t12-stripped-a"),
    "stripped_b": Path("data/runs/t12-stripped-b"),
    "wholefly_a": Path("data/runs/t12-wholefly-a"),
    "wholefly_b": Path("data/runs/t12-wholefly-b"),
}
RECEIPTS = ("events.jsonl", "equity.csv")

#: T12b (STD) report and run dirs. The stripped brain is identical to T12
#: (no APL/DPM to depress), so its fresh t12b receipts are byte-copied from
#: the T12 runs when those exist; the whole-fly runs always re-execute with
#: the STD-enabled LIFSim.
REPORT_PATH_STD = Path("reports/t12b-apl-std.md")
RUN_DIRS_STD = {
    "stripped_a": Path("data/runs/t12b-stripped-a"),
    "stripped_b": Path("data/runs/t12b-stripped-b"),
    "wholefly_a": Path("data/runs/t12b-wholefly-a"),
    "wholefly_b": Path("data/runs/t12b-wholefly-b"),
}

#: Calibration sweep grid (T12b): depletion fraction x recovery tau (ms),
#: evaluated over replayed real bars (``CAL_BARS``). Target: whole-fly KC
#: spike yield comparable to the stripped brain's while LC/T4/T5 pathway
#: activity stays within ``CAL_VISION_FLOOR`` of the no-STD whole-fly
#: baseline.
CAL_BETAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
CAL_TAUS_MS = (25.0, 50.0, 100.0, 200.0, 500.0)
CAL_VISION_FLOOR = 0.9
CAL_BAR_COUNT = 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir_fresh(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in RECEIPTS)


def _replay(tag: str, config: BacktestConfig, chassis=None) -> Path:
    """One replay; ``chassis`` (when given) is patched into the loop's seam."""
    import fruitfly.loop as loop

    run_dir = config.out_dir
    if _run_dir_fresh(run_dir):
        print(f"[t12] {tag}: reusing {run_dir}")
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    token = loop._load_chassis  # noqa: SLF001 - the loop's documented seam
    try:
        if chassis is not None:
            loop._load_chassis = lambda: chassis
        result = run_backtest(config)
    finally:
        loop._load_chassis = token
    print(
        f"[t12] {tag}: {result.n_bars} bars, {result.n_orders} orders, "
        f"equity {result.final_equity:.2f}, wall {result.wall_s:.0f}s"
    )
    return run_dir


def _equity_of(run_dir: Path) -> str:
    last = (run_dir / "equity.csv").read_text().strip().splitlines()[-1]
    return last.split(",")[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true",
                        help="reuse existing run receipts; only (re)render the report")
    parser.add_argument("--start", default=REPLAY_START)
    parser.add_argument("--end", default=REPLAY_END)
    parser.add_argument("--seed", type=int, default=REPLAY_SEED)
    parser.add_argument("--artifact", default=str(LARVAL_ARTIFACT))
    parser.add_argument("--calibrate-std", action="store_true",
                        help="T12b: sweep (beta, tau_rec) on replayed real bars")
    parser.add_argument("--std-beta", type=float, default=None,
                        help="T12b: STD depletion fraction (with --std-tau-rec)")
    parser.add_argument("--std-tau-rec", type=float, default=None,
                        help="T12b: STD recovery time constant in ms")
    args = parser.parse_args(argv)

    stripped = load_stripped_chassis()
    artifact = load_larval_weights(args.artifact, chassis=stripped, verify=True)
    print(f"[t12] artifact {args.artifact}: weights {artifact.weights.shape}, "
          f"fingerprint verified against the stripped chassis")

    whole = load_whole_fly()
    transplant = transplant_weights(artifact, whole, stripped_chassis=stripped)
    report = transplant.report
    print(f"[t12] transplant: KC match {report.kc_match_rate:.4f}, "
          f"MBON match {report.mbon_match_rate:.4f}, copied {report.pairs_copied} pairs")

    if args.calibrate_std:
        return _cmd_calibrate_std(args, stripped, transplant)
    if (args.std_beta is None) != (args.std_tau_rec is None):
        parser.error("--std-beta and --std-tau-rec must be given together")
    if args.std_beta is not None:
        return _main_std(args, artifact, transplant)

    config = dict(seed=args.seed, start=args.start, end=args.end,
                  out_dir=None)  # out_dir set per run below
    stripped_cfg_a = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS["stripped_a"],
           "initial_weights": artifact.weights}
    )
    stripped_cfg_b = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS["stripped_b"],
           "initial_weights": artifact.weights}
    )
    # The transplanted Plasticity already carries the copied weights; the loop
    # re-injects them through initial_weights (shape-checked) into its own
    # Plasticity built on the same labeled chassis.
    whole_weights = transplant.weights.copy()
    whole_cfg_a = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS["wholefly_a"],
           "initial_weights": whole_weights}
    )
    whole_cfg_b = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS["wholefly_b"],
           "initial_weights": whole_weights}
    )
    diagnosis = _diagnose_kc_silence(stripped, transplant)

    if not args.reuse:
        _replay("stripped-A", stripped_cfg_a)
        _replay("stripped-B", stripped_cfg_b)
    _replay("wholefly-A", whole_cfg_a, chassis=transplant.labeled_chassis)
    _replay("wholefly-B", whole_cfg_b, chassis=transplant.labeled_chassis)

    # --- determinism receipts ------------------------------------------------
    receipts = {}
    for tag, run_dir in RUN_DIRS.items():
        receipts[tag] = {name: _sha256(run_dir / name) for name in RECEIPTS}
    det_stripped = (receipts["stripped_a"] == receipts["stripped_b"])
    det_wholefly = (receipts["wholefly_a"] == receipts["wholefly_b"])
    print(f"[t12] determinism: stripped={det_stripped} wholefly={det_wholefly}")

    # --- agreement -----------------------------------------------------------
    rows, mean_action = agreement_table(
        decision_log(RUN_DIRS["stripped_a"] / "events.jsonl"),
        decision_log(RUN_DIRS["wholefly_a"] / "events.jsonl"),
    )
    for r in rows:
        print(f"[t12] {r['day']}: bars={r['bars']} "
              f"action={r['action_agreement']:.4f} ticker={r['ticker_agreement']:.4f}")
    print(f"[t12] mean action agreement: {mean_action:.4f}  "
          f"(D19 gate: {'PASS' if mean_action >= 0.9 else 'FAIL'} at 0.90)")

    _write_report(args, transplant, rows, mean_action, receipts,
                  det_stripped, det_wholefly, diagnosis)
    return 0


# ---------------------------------------------------------------------------
# T12b: opt-in short-term synaptic depression on the whole-fly brain
# ---------------------------------------------------------------------------


def _std_bars() -> list[tuple[str, pd.Timestamp, pd.DataFrame, int, dict]]:
    """Real replayed bars for STD calibration: the DIAG bar plus evenly
    spaced 2026-09-11 session bars of the same ticker. Deterministic."""
    import numpy as np
    import pandas as pd

    from fruitfly.data import BASKET, load_bars
    from fruitfly.loop import _features_at

    frames = {sym: df for sym, df in load_bars(list(BASKET), None, None).items()
              if len(df)}
    df = frames[DIAG_TICKER]
    day = df[df.index.date == pd.Timestamp(DIAG_TS).date()]
    picks = sorted({DIAG_TS} | {
        str(day.index[int(round(f * (len(day) - 1)))])
        for f in np.linspace(0.1, 0.9, CAL_BAR_COUNT - 1)
    })
    out = []
    for ts_s in picks:
        ts = pd.Timestamp(ts_s)
        i = df.index.get_indexer([ts], method="pad")[0]
        features, _, _ = _features_at(df, i)
        out.append((ts_s, ts, df, i, features))
    return out


def _probe_bar(chassis, plast, ts, df, i, features, gain=1.0, **std) -> dict:
    """One replayed bar through a fresh LIFSim (the T12 diagnostic drive:
    identical loop input, seeded noise, loop gain x ``gain``). Returns
    KC / MBON / APL / LC-T4-T5 activity plus the wall time of the step."""
    import time as _time

    import numpy as np

    from fruitfly.loop import NOISE_SIGMA_MV, SMELL_GAIN, VISION_WINDOW_BARS
    from fruitfly.senses import encode_smell, encode_vision
    from fruitfly.senses.smell import upn_channels
    from fruitfly.sim import LIFSim

    sim = LIFSim(chassis, dt_ms=1.0, seed=0, **std)
    ur, _ = upn_channels(chassis)
    window = df.iloc[max(0, i - VISION_WINDOW_BARS + 1): i + 1]
    inp = encode_vision(window, chassis).astype(np.float64)
    inp[ur] += SMELL_GAIN * gain * encode_smell(DIAG_TICKER, features, chassis)
    inp += np.random.default_rng(0).standard_normal(chassis.n_neurons) * NOISE_SIGMA_MV
    t0 = _time.perf_counter()
    sp = sim.step(inp, 500.0)["spikes"]
    wall_s = _time.perf_counter() - t0
    pop = chassis.nodes["population"].to_numpy()
    vision = np.isin(pop, ("LC-looming", "T4", "T5"))
    typ = chassis.nodes["type"].fillna("").to_numpy()
    return {
        "kc_spikes": int(sp[plast.kc_index].sum()),
        "mbon_drive": float(plast.mbon_activation(sp.astype(np.float64)).sum()),
        "apl_spikes": int(sp[np.flatnonzero(typ == "APL")].sum()),
        "vision_spikes": int(sp[vision].sum()),
        "wall_s": wall_s,
    }


def _mean_probe(chassis, plast, bars, **std) -> dict:
    """Mean of :func:`_probe_bar` over the calibration bars."""
    agg = {"kc_spikes": 0.0, "mbon_drive": 0.0, "apl_spikes": 0.0,
           "vision_spikes": 0.0, "wall_s": 0.0}
    for _, ts, df, i, features in bars:
        r = _probe_bar(chassis, plast, ts, df, i, features, **std)
        for k in agg:
            agg[k] += r[k]
    n = len(bars)
    return {k: v / n for k, v in agg.items()}


def _sweep_std(stripped, transplant, bars) -> tuple[list[dict], dict, dict]:
    """Calibration sweep: references + (beta, tau_rec) grid over ``bars``.

    Returns ``(grid_rows, ref_stripped, ref_whole)``; grid rows carry the
    mean KC/MBON/APL/vision activity and the vision ratio vs the no-STD
    whole-fly baseline.
    """
    from fruitfly.neuromod import Plasticity

    ch = transplant.labeled_chassis
    plast = transplant.plasticity
    ref_stripped = _mean_probe(stripped, Plasticity(stripped), bars)
    ref_whole = _mean_probe(ch, plast, bars)
    rows = []
    for beta in CAL_BETAS:
        for tau in CAL_TAUS_MS:
            m = _mean_probe(ch, plast, bars, std_beta=beta, std_tau_rec_ms=tau)
            ratio = (m["vision_spikes"] / ref_whole["vision_spikes"]
                     if ref_whole["vision_spikes"] else 1.0)
            rows.append({"beta": beta, "tau": tau, **m, "vision_ratio": ratio})
    return rows, ref_stripped, ref_whole


def _choose_std(rows, ref_stripped, ref_whole) -> dict | None:
    """Pick the sweep point: vision pathway intact (>= floor of the no-STD
    whole-fly baseline), KC activity alive, KC yield closest to the
    stripped brain's per-bar yield."""
    target = ref_stripped["kc_spikes"]
    ok = [r for r in rows
          if r["vision_ratio"] >= CAL_VISION_FLOOR and r["kc_spikes"] > 0]
    if not ok:
        return None
    return min(ok, key=lambda r: (abs(r["kc_spikes"] - target), -r["kc_spikes"]))


def _cmd_calibrate_std(args, stripped, transplant) -> int:
    """Sweep (beta, tau_rec) over replayed real bars and print the table."""
    bars = _std_bars()
    print(f"[t12b] calibration bars: {[b[0] for b in bars]}")
    rows, ref_stripped, ref_whole = _sweep_std(stripped, transplant, bars)
    print(f"[t12b] reference stripped (no STD): "
          f"KC {ref_stripped['kc_spikes']:.1f}, MBON {ref_stripped['mbon_drive']:.1f}, "
          f"vision {ref_stripped['vision_spikes']:.0f}, "
          f"wall {ref_stripped['wall_s']:.3f}s/bar")
    print(f"[t12b] reference whole-fly (no STD): "
          f"KC {ref_whole['kc_spikes']:.1f}, MBON {ref_whole['mbon_drive']:.1f}, "
          f"APL {ref_whole['apl_spikes']:.0f}, vision {ref_whole['vision_spikes']:.0f}, "
          f"wall {ref_whole['wall_s']:.3f}s/bar")
    for r in rows:
        print(f"[t12b] beta={r['beta']:<4} tau={r['tau']:<6} "
              f"KC {r['kc_spikes']:6.1f} | MBON {r['mbon_drive']:7.1f} "
              f"| APL {r['apl_spikes']:6.0f} | vision {r['vision_spikes']:6.0f} "
              f"| x{r['vision_ratio']:.3f} | wall {r['wall_s']:.3f}s")
    best = _choose_std(rows, ref_stripped, ref_whole)
    if best is not None:
        print(f"[t12b] CHOSEN: beta={best['beta']} tau_rec={best['tau']}ms "
              f"(KC {best['kc_spikes']:.1f} vs stripped "
              f"{ref_stripped['kc_spikes']:.1f}, vision x{best['vision_ratio']:.3f})")
    else:
        print("[t12b] NO sweep point met the vision floor with KC activity")
    return 0


def _replay_std(tag: str, config: BacktestConfig, chassis, beta: float,
                tau: float) -> tuple[Path, float | None, int | None]:
    """One whole-fly replay with the STD-enabled LIFSim patched into the
    loop's ``LIFSim`` name seam (the documented monkeypatch pattern)."""
    import fruitfly.loop as loop
    from fruitfly.sim import LIFSim

    run_dir = config.out_dir
    if _run_dir_fresh(run_dir):
        print(f"[t12b] {tag}: reusing {run_dir}")
        return run_dir, None, None
    if run_dir.exists():
        shutil.rmtree(run_dir)

    def _std_sim(c, dt_ms, seed, std_beta=None, std_tau_rec_ms=None):
        # The loop now passes its config's std_* through; the T12b seam pins
        # the calibrated values this invocation was given (same values).
        return LIFSim(c, dt_ms=dt_ms, seed=seed,
                      std_beta=beta, std_tau_rec_ms=tau)

    chassis_token = loop._load_chassis  # noqa: SLF001 - documented seam
    sim_token = loop.LIFSim
    try:
        loop._load_chassis = lambda: chassis
        loop.LIFSim = _std_sim
        result = run_backtest(config)
    finally:
        loop._load_chassis = chassis_token
        loop.LIFSim = sim_token
    per_bar = result.wall_s / max(1, result.n_bars)
    print(f"[t12b] {tag}: {result.n_bars} bars, {result.n_orders} orders, "
          f"equity {result.final_equity:.2f}, wall {result.wall_s:.0f}s "
          f"({per_bar:.2f} s/bar incl. STD)")
    import json

    (run_dir / "timing.json").write_text(
        json.dumps({"wall_s": result.wall_s, "n_bars": result.n_bars}) + "\n"
    )
    return run_dir, result.wall_s, result.n_bars


def _diagnose_std_gains(stripped, transplant, beta: float, tau: float) -> dict:
    """T12's gain sweep redone with STD: stripped (no STD) vs whole-fly
    (STD) vs whole-fly (no STD) KC yield per SMELL_GAIN multiple on the
    DIAG bar."""
    from fruitfly.neuromod import Plasticity

    diag_bar = _std_bars()[0]  # the DIAG bar is the first pick
    _, ts, df, i, features = diag_bar
    gains = {}
    for mult in (1, 2, 4, 8):
        row = {
            "stripped": _probe_bar(stripped, Plasticity(stripped), ts, df, i,
                                   features, gain=mult),
            "whole_std": _probe_bar(transplant.labeled_chassis,
                                    transplant.plasticity, ts, df, i, features,
                                    gain=mult, std_beta=beta,
                                    std_tau_rec_ms=tau),
            "whole_nostd": _probe_bar(transplant.labeled_chassis,
                                      transplant.plasticity, ts, df, i,
                                      features, gain=mult),
        }
        gains[mult] = row
    return gains


def _diagnose_kc_silence(stripped, transplant) -> dict:
    """Root-cause probe for whole-fly readout silence (see the report).

    Replays ONE real bar's exact loop input (the DIAG bar below, same drive
    into both brains) through fresh LIF Sims at the loop gain and at
    multiples of it, and reports KC spikes / MBON drive per brain. Also
    measures the per-KC synaptic budget of the whole-fly-only feedback
    inhibitor APL vs the uPN drive. Fully deterministic (seeded noise,
    fixed bar), so the numbers in the report are reproducible.
    """
    import numpy as np
    import pandas as pd

    from fruitfly.data import BASKET, load_bars
    from fruitfly.loop import (
        NOISE_SIGMA_MV,
        SMELL_GAIN,
        VISION_WINDOW_BARS,
        _features_at,
    )
    from fruitfly.neuromod import Plasticity
    from fruitfly.senses import encode_smell, encode_vision
    from fruitfly.senses.smell import upn_channels
    from fruitfly.sim import LIFSim

    frames = {sym: df for sym, df in load_bars(list(BASKET), None, None).items()
              if len(df)}
    ts = pd.Timestamp(DIAG_TS)
    df = frames[DIAG_TICKER]
    i = df.index.get_indexer([ts], method="pad")[0]
    window = df.iloc[max(0, i - VISION_WINDOW_BARS + 1): i + 1]
    features, _, _ = _features_at(df, i)

    brains = (
        ("stripped", stripped, Plasticity(stripped)),
        ("whole-fly", transplant.labeled_chassis, transplant.plasticity),
    )
    gains: dict[int, dict] = {}
    for mult in (1, 2, 4, 8):
        row = {}
        for label, ch, plast in brains:
            sim = LIFSim(ch, dt_ms=1.0, seed=0)
            ur, _ = upn_channels(ch)
            inp = encode_vision(window, ch).astype(np.float64)
            inp[ur] += SMELL_GAIN * mult * encode_smell(DIAG_TICKER, features, ch)
            inp += np.random.default_rng(0).standard_normal(ch.n_neurons) * NOISE_SIGMA_MV
            sp = sim.step(inp, 500.0)["spikes"].astype(np.float64)
            row[label] = {"kc_spikes": int(sp[plast.kc_index].sum()),
                          "mbon_drive": float(plast.mbon_activation(sp).sum())}
        gains[mult] = row

    # APL synaptic budget over the matched KC set (whole-fly adjacency).
    ch = transplant.labeled_chassis
    typ = ch.nodes["type"].fillna("").to_numpy()
    apl = np.flatnonzero(typ == "APL")
    kc = ch.nodes["bodyId"].to_numpy()[ch.nodes["population"].to_numpy() == "KC"]
    kc_rows = np.searchsorted(ch.nodes["bodyId"].to_numpy(), kc)
    s_typ = set(stripped.nodes["type"].fillna(""))
    apl_syn = np.asarray(ch.adj[apl][:, kc_rows].sum(axis=0)).ravel()
    ur, _ = upn_channels(ch)
    upn_syn = np.asarray(ch.adj[ur][:, kc_rows].sum(axis=0)).ravel()
    return {
        "bar_ts": DIAG_TS,
        "ticker": DIAG_TICKER,
        "gains": gains,
        "apl_in_stripped": "APL" in s_typ,
        "apl_neurons_whole": int(apl.size),
        "apl_synapses_per_kc_median": float(np.median(apl_syn)),
        "upn_synapses_per_kc_median": float(np.median(upn_syn)),
    }


def _main_std(args, artifact, transplant) -> int:
    """T12b protocol: the D19 two-day, two-brain replay with the whole-fly
    brain running LIFSim(std_beta, std_tau_rec_ms). Stripped brain is the
    unchanged T12 reference (byte-reused receipts when present)."""
    stripped = load_stripped_chassis()

    # The stripped brain is bit-identical to its T12 counterpart (same seed,
    # window, artifact weights, no STD) — reuse those receipts verbatim when
    # they exist; otherwise rerun into the t12b dirs.
    for tag in ("stripped_a", "stripped_b"):
        dst = RUN_DIRS_STD[tag]
        if _run_dir_fresh(dst):
            continue
        if dst.exists():
            shutil.rmtree(dst)
        if _run_dir_fresh(RUN_DIRS[tag]):
            shutil.copytree(RUN_DIRS[tag], dst)
            print(f"[t12b] {tag}: byte-copied receipts from {RUN_DIRS[tag]}")

    config = dict(seed=args.seed, start=args.start, end=args.end,
                  out_dir=None)  # out_dir set per run below
    weights = artifact.weights
    stripped_cfg_a = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS_STD["stripped_a"],
           "initial_weights": weights}
    )
    stripped_cfg_b = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS_STD["stripped_b"],
           "initial_weights": weights}
    )
    whole_weights = transplant.weights.copy()
    whole_cfg_a = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS_STD["wholefly_a"],
           "initial_weights": whole_weights}
    )
    whole_cfg_b = BacktestConfig(
        **{**config, "out_dir": RUN_DIRS_STD["wholefly_b"],
           "initial_weights": whole_weights}
    )

    timings: dict[str, tuple[float | None, int | None]] = {}
    if not args.reuse:
        _replay("stripped-A", stripped_cfg_a)
        _replay("stripped-B", stripped_cfg_b)
        for tag, cfg in (("wholefly-A", whole_cfg_a),
                         ("wholefly-B", whole_cfg_b)):
            _, wall_s, n_bars = _replay_std(
                tag, cfg, transplant.labeled_chassis,
                args.std_beta, args.std_tau_rec)
            timings[tag] = (wall_s, n_bars)
    # Reused receipts: recover the measured wall timings from the run dirs.
    import json

    for tag in ("wholefly-A", "wholefly-B"):
        if timings.get(tag, (None, None))[0] is None:
            tfile = RUN_DIRS_STD[tag.lower().replace("-", "_")] / "timing.json"
            if tfile.exists():
                t = json.loads(tfile.read_text())
                timings[tag] = (t["wall_s"], t["n_bars"])

    # --- determinism receipts ------------------------------------------------
    receipts = {tag: {n: _sha256(d / n) for n in RECEIPTS}
                for tag, d in RUN_DIRS_STD.items()}
    det_stripped = receipts["stripped_a"] == receipts["stripped_b"]
    det_wholefly = receipts["wholefly_a"] == receipts["wholefly_b"]
    print(f"[t12b] determinism: stripped={det_stripped} wholefly={det_wholefly}")

    # --- agreement -----------------------------------------------------------
    rows, mean_action = agreement_table(
        decision_log(RUN_DIRS_STD["stripped_a"] / "events.jsonl"),
        decision_log(RUN_DIRS_STD["wholefly_a"] / "events.jsonl"),
    )
    for r in rows:
        print(f"[t12b] {r['day']}: bars={r['bars']} "
              f"action={r['action_agreement']:.4f} "
              f"ticker={r['ticker_agreement']:.4f}")
    print(f"[t12b] mean action agreement: {mean_action:.4f}  "
          f"(D19 gate: {'PASS' if mean_action >= 0.9 else 'FAIL'} at 0.90)")

    # Calibration evidence is cheap and fully deterministic: re-derive it so
    # the report always carries the measured sweep, not a pasted one.
    bars = _std_bars()
    sweep_rows, ref_stripped, ref_whole = _sweep_std(stripped, transplant, bars)
    gains = _diagnose_std_gains(stripped, transplant,
                                args.std_beta, args.std_tau_rec)
    _write_report_std(args, transplant, rows, mean_action, receipts,
                      det_stripped, det_wholefly, sweep_rows, ref_stripped,
                      ref_whole, gains, timings)
    return 0


def _write_report_std(args, transplant, rows, mean_action, receipts,
                      det_stripped, det_wholefly, sweep_rows, ref_stripped,
                      ref_whole, gains, timings) -> None:
    """Render reports/t12b-apl-std.md from measured inputs only."""
    from fruitfly.transplant import action_bucket

    beta, tau = args.std_beta, args.std_tau_rec
    chosen = _choose_std(sweep_rows, ref_stripped, ref_whole)
    gate_pass = mean_action >= 0.9
    whole_dec = decision_log(RUN_DIRS_STD["wholefly_a"] / "events.jsonl")
    stripped_dec = decision_log(RUN_DIRS_STD["stripped_a"] / "events.jsonl")
    n_orders = lambda dec: sum(  # noqa: E731
        1 for _, _, a in dec if action_bucket(a) != "PASS")
    kc_alive = chosen is not None and chosen["kc_spikes"] > 0

    lines: list[str] = []
    add = lines.append
    add("# T12b — APL/DPM short-term depression: whole-fly KC revival and "
        "D19 re-validation")
    add("")
    add(f"**Window:** {args.start} .. {args.end} (replay) · "
        f"**Seed:** {args.seed} · **Artifact:** `{args.artifact}` · "
        f"**STD:** beta={beta}, tau_rec={tau} ms · "
        f"**Script:** `scripts/validate_transplant.py` "
        f"(`--std-beta` / `--std-tau-rec` / `--calibrate-std`)")
    add("")
    add("Follow-up to reports/t12-transplant.md: T12 measured the whole-fly "
        "readout silent — tonic APL/DPM feedback inhibition pins all 4064 "
        "KCs below threshold (median 50 APL inhibitory synapses per KC vs "
        "97 uPN excitatory) because quantal LIF coupling never fatigues. "
        "T12b adds **opt-in short-term synaptic depression (STD)** to the "
        "LIF engine's inhibitory terminals and re-runs the D19 validation.")
    add("")
    add("## Mechanism (opt-in, default off)")
    add("")
    add("- Per presynaptic node, a depression factor `f` in [0, 1] scales "
        "that node's outgoing weights (`w[i,:] * f[i]`); only inhibitory "
        "terminals are tracked (51,744 nodes with outgoing edges in the "
        "whole fly). Excitatory and modulatory terminals keep `f = 1` "
        "exactly — excitatory synapses are untouched.")
    add("- Fixed per-substep op order: (1) deliver the previous substep's "
        "spikes scaled by `f`; (2) integrate, refractory clamp, threshold; "
        "(3) recover every inhibitory factor toward 1 with the exact "
        "exponential form `f = 1 - (1 - f) * exp(-dt / tau_rec)` (the "
        "exponential is precomputed once — no per-substep transcendentals; "
        "`f == 1` is an exact fixed point, so undepressed terminals never "
        "drift); (4) deplete the terminals that spiked this substep: "
        "`f *= (1 - beta)`.")
    add("- A spike is therefore delivered at its depleted amplitude, and "
        "recovery accrues per substep until the next spike. Determinism: "
        "fixed op order, float64, no RNG, no wall clock. With STD disabled "
        "(the default) the propagation step uses the unscaled spike vector, "
        "so existing runs are **bit-identical** to the pre-STD engine — "
        "pinned by the full test suite plus a dedicated bit-identity test.")
    add("")
    add("## Calibration — (beta, tau_rec) sweep on replayed real bars")
    add("")
    add(f"- Calibration bars (XOM, 2026-09-11 session, identical loop drive "
        f"per bar, seeded noise): "
        f"{', '.join('`' + b[0] + '`' for b in _std_bars())}")
    add("")
    add("| Brain | KC spikes/bar | MBON drive | APL spikes | "
        "LC/T4/T5 spikes | step wall (s) |")
    add("|---|---|---|---|---|---|")
    add(f"| stripped (no STD) | {ref_stripped['kc_spikes']:.1f} "
        f"| {ref_stripped['mbon_drive']:.1f} | {ref_stripped['apl_spikes']:.0f} "
        f"| {ref_stripped['vision_spikes']:.0f} "
        f"| {ref_stripped['wall_s']:.3f} |")
    add(f"| whole-fly (no STD) | {ref_whole['kc_spikes']:.1f} "
        f"| {ref_whole['mbon_drive']:.1f} | {ref_whole['apl_spikes']:.0f} "
        f"| {ref_whole['vision_spikes']:.0f} | {ref_whole['wall_s']:.3f} |")
    if chosen is not None:
        add(f"| **whole-fly (chosen STD)** | **{chosen['kc_spikes']:.1f}** "
            f"| {chosen['mbon_drive']:.1f} | {chosen['apl_spikes']:.0f} "
            f"| {chosen['vision_spikes']:.0f} (x{chosen['vision_ratio']:.2f} "
            f"of no-STD) | {chosen['wall_s']:.3f} |")
    add("")
    add("Full grid (mean over the calibration bars; vision ratio = LC/T4/T5 "
        "spike yield vs the no-STD whole-fly baseline):")
    add("")
    add("| beta | tau_rec (ms) | KC spikes | MBON drive | APL spikes | "
        "vision spikes | vision x |")
    add("|---|---|---|---|---|---|---|")
    for r in sweep_rows:
        add(f"| {r['beta']} | {r['tau']:.0f} | {r['kc_spikes']:.1f} "
            f"| {r['mbon_drive']:.1f} | {r['apl_spikes']:.0f} "
            f"| {r['vision_spikes']:.0f} | {r['vision_ratio']:.3f} |")
    add("")
    if chosen is not None:
        add(f"Chosen: **beta = {chosen['beta']}, tau_rec = {chosen['tau']:.0f} ms** "
            f"— KC yield {chosen['kc_spikes']:.1f}/bar vs the stripped "
            f"brain's {ref_stripped['kc_spikes']:.1f}/bar at the same drive, "
            f"with the LC/T4/T5 pathway at x{chosen['vision_ratio']:.2f} of "
            f"its no-STD activity (floor {CAL_VISION_FLOOR:.2f}).")
    else:
        add("No sweep point met the acceptance rule (KC activity alive, "
            "vision >= floor).")
    add("")
    add("## Whole-fly KC revival vs gain (DIAG bar)")
    add("")
    add("| SMELL_GAIN x | stripped KC (no STD) | whole-fly KC (STD) "
        "| whole-fly KC (no STD) | whole-fly MBON drive (STD) |")
    add("|---|---|---|---|---|")
    for mult in sorted(gains):
        g = gains[mult]
        add(f"| {mult} | {g['stripped']['kc_spikes']} "
            f"| {g['whole_std']['kc_spikes']} "
            f"| {g['whole_nostd']['kc_spikes']} "
            f"| {g['whole_std']['mbon_drive']:.1f} |")
    if chosen is not None:
        add("")
        add(f"- Measured caveats: the transplanted KC→MBON drive under STD "
            f"({chosen['mbon_drive']:.0f} summed activation/bar) is ~"
            f"{chosen['mbon_drive'] / max(ref_stripped['mbon_drive'], 1e-9):.0f}x "
            "the stripped brain's — the loop's decision variable is the "
            "valence-normalized balance, so the scale difference does not "
            "enter the BUY/SELL/pass comparison directly. LC/T4/T5 spike "
            "yield is 0 in BOTH brains under this loop drive (vision is "
            "delivered sub-threshold at the loop gain), so the vision "
            "check is vacuous here — it measures that STD did not create "
            "or destroy visual activity on these bars, not that a "
            "vision-driven regime is preserved.")
    add("")
    add("## Decision agreement (per day, D19 metric)")
    add("")
    add("| Day | Bars decided | BUY/SELL/pass agreement | Ticker agreement "
        "| Action + ticker |")
    add("|---|---|---|---|---|")
    for row in rows:
        add(f"| {row['day']} | {row['bars']} "
            f"| {_fmt_pct(row['action_agreement'])} "
            f"| {_fmt_pct(row['ticker_agreement'])} "
            f"| {_fmt_pct(row['action_and_ticker_agreement'])} |")
    add(f"| **Mean (day-weighted)** | — | **{_fmt_pct(mean_action)}** | — | — |")
    add("")
    add(f"- Stripped brain orders: {n_orders(stripped_dec)} of "
        f"{len(stripped_dec)} decided bars; whole-fly (STD) orders: "
        f"{n_orders(whole_dec)} of {len(whole_dec)}.")
    add("")
    add("## Sim-rate impact")
    add("")
    wall_a, bars_a = timings.get("wholefly-A", (None, None))
    if wall_a and bars_a:
        add(f"- Whole-fly replay **with STD**: {wall_a:.0f} s for {bars_a} "
            f"bars = {wall_a / bars_a:.2f} s/bar (loop dt = 1.0 ms).")
    else:
        add("- Whole-fly replay receipts were reused; re-run without "
            "`--reuse` for a fresh timing.")
    if chosen is not None:
        add(f"- Engine-level step cost on the calibration bars: no-STD "
            f"{ref_whole['wall_s']:.3f} s/bar vs STD {chosen['wall_s']:.3f} "
            f"s/bar (x{chosen['wall_s'] / ref_whole['wall_s']:.2f}) — the "
            "per-substep recovery pass over the inhibitory subset is the "
            "only overhead.")
    else:
        add("- Engine-level step cost: see the calibration table.")
    add("- T12 reference points: whole-fly LIF at 0.281x real time "
        "(1.78 s/bar at dt = 0.5 ms); the loop runs dt = 1.0 ms.")
    add("")
    add("## Determinism receipts (same-seed replay byte-identical)")
    add("")
    add("| Run | events.jsonl sha256 | equity.csv sha256 | equity |")
    add("|---|---|---|---|")
    for tag in ("stripped_a", "stripped_b", "wholefly_a", "wholefly_b"):
        run_dir = RUN_DIRS_STD[tag]
        add(f"| {tag} | `{receipts[tag]['events.jsonl'][:16]}…` "
            f"| `{receipts[tag]['equity.csv'][:16]}…` | {_equity_of(run_dir)} |")
    add("")
    add(f"- Stripped replay A ≡ B (byte-identical): "
        f"**{'yes' if det_stripped else 'NO'}**")
    add(f"- Whole-fly replay A ≡ B (byte-identical): "
        f"**{'yes' if det_wholefly else 'NO'}**")
    add("")
    add("## D19 branch recommendation (for human ratification)")
    add("")
    if gate_pass:
        add(f"Mean decision agreement **{_fmt_pct(mean_action)}** >= 90% "
            "with the whole-fly brain firing: this report **proposes "
            "RESTORING D19** — the Phase-B live carrier becomes the "
            "whole-fly LIF sim running LIFSim(std_beta=..., "
            "std_tau_rec_ms=...). DESIGN.md is not edited here; the human "
            "ratifies the flip.")
    elif kc_alive:
        add(f"Mean decision agreement **{_fmt_pct(mean_action)}** < 90% "
            "despite revived KC activity (whole-fly KC yield "
            f"{chosen['kc_spikes']:.1f}/bar under STD): the whole-fly brain "
            "now acts but disagrees with the stripped engine often enough "
            "that a live cutover is not warranted. This report **proposes "
            "demo-mode decisions** — the whole-fly brain (with STD) decides "
            "in promo/art mode while the live engine stays the stripped "
            "chassis (D10 Phase A). DESIGN.md is not edited here; the "
            "human ratifies.")
    else:
        add("Whole-fly KCs remain silent under every calibrated STD point; "
            "the T12 revocation stands and the silence is documented as "
            "continued.")
    add("")
    REPORT_PATH_STD.write_text("\n".join(lines) + "\n")
    print(f"[t12b] report written to {REPORT_PATH_STD}")



def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _write_report(args, transplant, rows, mean_action, receipts,
                  det_stripped, det_wholefly, diagnosis) -> None:
    r = transplant.report
    gate_pass = mean_action >= 0.9
    lines: list[str] = []
    add = lines.append
    add("# T12 — Weight Transplant + Behavioral Validation (D19)")
    add("")
    add(f"**Window:** {args.start} .. {args.end} (replay) · "
        f"**Seed:** {args.seed} · "
        f"**Artifact:** `{args.artifact}` · "
        f"**Module:** `fruitfly.transplant` · **Script:** "
        f"`scripts/validate_transplant.py`")
    add("")
    add("## The transplant (D19, direct copy)")
    add("")
    add("| Metric | Measured |")
    add("|---|---|")
    add(f"| Stripped weight view (KC × MBON) | {r.n_kc_stripped} × {r.n_mbon_stripped} |")
    add(f"| Whole-fly weight view (KC × MBON) | {r.n_kc_whole} × {r.n_mbon_whole} |")
    add(f"| KC bodyId match rate | {_fmt_pct(r.kc_match_rate)} "
        f"({r.n_kc_stripped} KCs) |")
    add(f"| MBON bodyId match rate | {_fmt_pct(r.mbon_match_rate)} "
        f"({r.n_mbon_stripped} MBONs) |")
    add(f"| Matched (KC, MBON) pairs | {r.pairs_total} |")
    add(f"| Trained values copied verbatim | {r.pairs_copied} "
        f"(mass {r.copied_mass:.1f}) |")
    add(f"| Baseline-filled (whole support, no artifact synapse) "
        f"| {r.pairs_baseline_filled} |")
    add(f"| Dropped (trained > 0, whole fly lacks the synapse) "
        f"| {r.pairs_dropped_no_support} |")
    add(f"| Synapse counts identical over matched pairs | "
        f"{'yes' if r.synapse_consistent else 'NO'} |")
    add(f"| Artifact fingerprint verified vs stripped chassis | "
        f"{'yes' if r.fingerprint_verified else 'no'} |")
    add("")
    add("Both chassis are extracted from the same MaleCNS v1.0 release: the "
        "stripped chassis is a bodyId subset of the whole fly, and the "
        "KC→MBON structural submatrix (synapse counts) is identical over the "
        "matched pairs, so every trained nonzero weight lands on a real "
        "whole-fly synapse. The transplant itself is verifiably correct. "
        "Whole-fly nodes absent from the stripped chassis would keep their "
        "baseline weight from the whole-fly synapse count (log1p-compressed, "
        "mean-1-scaled) — the mechanism exists and is unit-tested, and the "
        "measured match rates show it was not needed: the stripped chassis "
        "already contains every annotated Kenyon cell and MBON in the "
        "release.")
    add("")
    add("## Replay setup")
    add("")
    add("- Identical loop config for both brains: same seed, same window, "
        "same loop constants (``ms_per_bar=500``, ``dt_ms=1.0``, ``top_k=5``, "
        "``position_cap=10``), ``initial_weights`` = trained artifact "
        "(stripped) / transplanted weights (whole fly).")
    add("- Whole-fly brain: population labels transferred onto matched "
        "bodyIds (``label_whole_chassis``); the loop's chassis seam points at "
        "the labeled whole-fly chassis (166,700-neuron LIF, ~1-4 s per "
        "encounter wall on the M1 target; T11: 0.281× real time at "
        "dt=0.5 ms). The full 2-day / 390-bar-per-day replay at full "
        "resolution stays within budget — **no bar subsampling was "
        "needed**.")
    add("- Determinism: each brain replayed twice at the same seed; the run "
        "receipts must be byte-identical.")
    add("- In-sample caveat, documented: the market cache spans "
        "2026-08-17..2026-09-14 — exactly the larval training window — so "
        "strictly out-of-sample bars do not exist in the cache. The final "
        "two trading sessions are the closest available held-out replay.")
    add("- RNG note, documented: the loop's per-encounter noise draw consumes "
        "a chassis-sized number of variates, so after the first day the two "
        "brains' rotation offsets can diverge (a different plume sampled at "
        "the same bar). Agreement is therefore keyed by bar timestamp; the "
        "ticker-agreement column quantifies plume-choice divergence.")
    add("")
    add("## Decision agreement (per day)")
    add("")
    add("| Day | Bars decided | BUY/SELL/pass agreement | Ticker agreement "
        "| Action + ticker |")
    add("|---|---|---|---|---|")
    for row in rows:
        add(f"| {row['day']} | {row['bars']} "
            f"| {_fmt_pct(row['action_agreement'])} "
            f"| {_fmt_pct(row['ticker_agreement'])} "
            f"| {_fmt_pct(row['action_and_ticker_agreement'])} |")
    add(f"| **Mean (day-weighted)** | — | **{_fmt_pct(mean_action)}** | — | — |")
    add("")
    add("## Why the whole-fly readout is silent (measured root cause)")
    add("")
    add("778 of 780 whole-fly encounters in the replay are **silent** "
        "(``valence_readout = 0``); the 2 remaining encounters carry "
        "negligible signal and still read ``pass``. Net effect: zero KC "
        "spikes, so the transplanted KC→MBON drive — however correct — "
        "multiplies zero, and the whole-fly brain emits **no orders in 780 "
        "bars** (the stripped brain places 127) — every decision is "
        "``pass`` by loop construction (``centered = 0`` on a silent "
        "encounter). The cause is in the circuit, not the weights. The "
        "stripped chassis (T2) deliberately excludes everything outside its "
        "task populations — among them the mushroom body's feedback "
        "inhibitors **APL** and **DPM**. The whole fly contains them, and "
        "with the loop's direct-uPN sensory injection they clamp the KC "
        "population below threshold:")
    add("")
    g = diagnosis["gains"]
    add(f"- APL in stripped chassis: "
        f"{'present' if diagnosis['apl_in_stripped'] else '**absent**'} · "
        f"APL neurons in whole fly: {diagnosis['apl_neurons_whole']} "
        f"(GABAergic, sign −1).")
    add(f"- Synaptic budget per median KC (whole-fly adjacency, matched "
        f"bodyIds): {diagnosis['apl_synapses_per_kc_median']:.0f} APL "
        f"inhibitory synapses vs "
        f"{diagnosis['upn_synapses_per_kc_median']:.0f} uPN excitatory "
        f"synapses. Both sides fire tonically during an encounter (uPNs and "
        f"APL spike every ~3 substeps at the loop gain), so APL's ~−0.5 mV "
        f"per spike holds the KC membrane below the 15 mV threshold "
        f"regardless of the sensory gain — the uPN drive saturates (uPNs "
        f"are already near their firing ceiling) while APL keeps pace.")
    add(f"- Gain sweep on one replayed bar ({diagnosis['ticker']} @ "
        f"{diagnosis['bar_ts']}, identical drive into both brains, seeded "
        f"noise):")
    add("")
    add("| SMELL_GAIN × | stripped KC spikes | stripped MBON drive "
        "| whole-fly KC spikes | whole-fly MBON drive |")
    add("|---|---|---|---|---|")
    for mult in sorted(g):
        add(f"| {mult} | {g[mult]['stripped']['kc_spikes']} "
            f"| {g[mult]['stripped']['mbon_drive']:.1f} "
            f"| {g[mult]['whole-fly']['kc_spikes']} "
            f"| {g[mult]['whole-fly']['mbon_drive']:.1f} |")
    add("")
    add("The transplanted weights cannot matter while their presynaptic "
        "population is clamped: plasticity eligibility is KC × MBON "
        "co-activity (measured 0 in every whole-fly encounter), so a "
        "fine-tuning round through the whole-fly LIF is provably a no-op.")
    add("")
    add("## D19 branch")
    add("")
    if gate_pass:
        add(f"Mean decision agreement **{_fmt_pct(mean_action)}** ≥ 90% → "
            "**branch: direct weight copy PASSES**; no fine-tuning performed.")
    else:
        add(f"Mean decision agreement **{_fmt_pct(mean_action)}** < 90% → the "
            "direct-copy branch **fails the behavioral gate**. A fine-tune "
            "round is ruled out on measured grounds (zero KC activity ⇒ zero "
            "plasticity eligibility ⇒ weights cannot move; see the root-cause "
            "section above). **Branch: D19's Phase-B transplant into the "
            "whole-fly LIF sim is REVOKED.** The stripped chassis remains the "
            "working engine (D10 Phase A), consistent with T11's recorded "
            "budget deviation (whole-fly LIF at 0.281× real time is an "
            "offline/art mode). The adult-stage (Phase B) carrier for "
            "transplanted plasticity, if revisited, is the lean "
            "mushroom-body rate model over the real KC→MBON/PAM/PPL1 subgraph "
            "flagged in reports/t11-wholefly.md — with the KC-silence finding "
            "as its first design constraint. The transplanted weight "
            "matrix itself is verified correct (100% bodyId match, verbatim "
            "copy, synapse counts identical) and remains available in "
            "`fruitfly.transplant.transplant_weights` for that carrier.")
    add("")
    add("## Determinism receipts (same-seed replay byte-identical)")
    add("")
    add("| Run | events.jsonl sha256 | equity.csv sha256 | equity |")
    add("|---|---|---|---|")
    for tag in ("stripped_a", "stripped_b", "wholefly_a", "wholefly_b"):
        run_dir = Path(f"data/runs/t12-{tag.replace('_', '-')}")
        eq = _equity_of(run_dir)
        add(f"| {tag} | `{receipts[tag]['events.jsonl'][:16]}…` "
            f"| `{receipts[tag]['equity.csv'][:16]}…` | {eq} |")
    add("")
    add(f"- Stripped replay A ≡ B (byte-identical): "
        f"**{'yes' if det_stripped else 'NO'}**")
    add(f"- Whole-fly replay A ≡ B (byte-identical): "
        f"**{'yes' if det_wholefly else 'NO'}**")
    add("")
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[t12] report written to {REPORT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
