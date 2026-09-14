#!/usr/bin/env python
"""T11 benchmark: LIF over the whole-fly connectome (all 166,700 neurons).

Loads the whole-fly chassis cache, runs LIFSim over a fixed 500 ms window x
10 bars with a deterministic tonic input to the sensory populations (same
drive pattern as scripts/bench_sim.py, but selected by cell type/class since
whole-fly nodes carry population="whole"), and prints the measured budget:
neuron count (must be 166,700), synapse count, load time, peak RSS, wall
time per bar and achievable sim rate in x real time.

Budget (DESIGN §13 target machine, 16 GB RAM): usable if >= 0.5x real time
and peak RSS <= 8 GB. A miss is reported as such — the recorded deviation
routes the whole-fly mode to the lean mushroom-body rate model fallback.

With ``--determinism``: two independent 100 ms same-input runs are compared
substep by substep; the spike logs must be byte-identical (the engine is
RNG-free, so any difference is a bug).

Usage::

    uv run python scripts/bench_wholefly.py [--bars 10] [--ms-per-bar 500]
        [--dt-ms 0.5] [--input-scale 1.0] [--determinism]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from fruitfly.connectome import load_whole_fly  # noqa: E402
from fruitfly.sim import LIFSim  # noqa: E402

#: The release neuron count (MaleCNS v1.0); DESIGN's 166,691 was the preprint.
EXPECTED_NEURONS = 166_700
_LC_RE = re.compile(r"^LC(?:4|21)(?:_.*)?$|^LC10")
_UPN_RE = re.compile(r"_(?:adPN|lPN|vPN|lvPN)\d*$")
#: Populations the sensory encoders (T4) drive; selected by type/class here
#: because whole-fly nodes all carry population="whole" (stripped-chassis
#: population labels are not assigned in this mode).
_T4_RE = re.compile(r"^T4")
_T5_RE = re.compile(r"^T5")
_LC_RE = re.compile(r"^LC(?:4|21)(?:_.*)?$|^LC10")
_GLOM_RE = re.compile(r"_(?:adPN|lPN|vPN|lvPN)\d*$")

#: Fraction of cells per driven population receiving encoder-like drive.
DRIVE_FRACTION = 0.2

#: Tonic drive (mV above rest) + sinusoidal modulation amplitude.
DRIVE_TONIC_MV = 18.0
DRIVE_SINE_MV = 6.0


def peak_rss_bytes() -> int:
    """Peak RSS of this process so far (bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) * (1 if sys.platform == "darwin" else 1024)


def build_input(chassis, scale: float) -> np.ndarray:
    """Deterministic per-bar input vector, indexed like chassis.nodes."""
    t = chassis.nodes["type"].fillna("").astype(str).to_numpy()
    inp = np.zeros(chassis.n_neurons, dtype=np.float64)
    for mask in (
        _T4_RE.match,  # T4 direction-selective (medulla)
        _T5_RE.match,  # T5 direction-selective (lobula)
        _LC_RE.match,  # LC4/LC21/LC10 looming-sensitive
    ):
        idx = np.flatnonzero([mask(x) is not None for x in t])
        inp[idx[idx % int(1.0 / DRIVE_FRACTION) == 0]] = scale * (
            DRIVE_TONIC_MV + DRIVE_SINE_MV
        )
    # uPN: uniglomerular antennal-lobe projection neurons. Whole-fly nodes
    # carry no class column; use the glomerulus-lineage suffix T2 derives
    # glomeruli from, excluding multiglomerular M_ types.
    idx = np.flatnonzero(
        [_UPN_RE.search(x) is not None and not x.startswith("M_") for x in t]
    )
    inp[idx[idx % int(1.0 / DRIVE_FRACTION) == 0]] = scale * (
        DRIVE_TONIC_MV + DRIVE_SINE_MV
    )
    return inp


def _spike_log(sim: LIFSim, inp: np.ndarray, steps: int) -> bytes:
    """Per-substep spike-count log for ``steps`` substeps of constant input."""
    chunks = bytearray()
    for _ in range(steps):
        out = sim.step(inp, duration_ms=sim.dt_ms)
        chunks += out["spikes"].tobytes()
    return bytes(chunks)


def run_determinism(dt_ms: float, ms: float = 100.0) -> None:
    """Two independent same-input 100 ms runs must produce byte-identical
    per-substep spike logs."""
    chassis = load_whole_fly()
    n_steps = int(round(ms / dt_ms))
    inp = build_input(chassis, 1.0)

    logs = []
    for run in (1, 2):
        sim = LIFSim(chassis, dt_ms=dt_ms, seed=run)  # seeds differ; engine ignores them
        t0 = time.perf_counter()
        log = _spike_log(sim, inp, n_steps)
        wall = time.perf_counter() - t0
        total = sim.spikes.sum()
        logs.append(hashlib.sha256(log).hexdigest())
        print(
            f"determinism run {run}: seed={run} {ms:g} ms, {n_steps} substeps, "
            f"{int(total):,} spikes, wall {wall:.2f} s, "
            f"sha256(log)={logs[-1]}"
        )
        del sim

    if logs[0] == logs[1]:
        print(f"DETERMINISM: PASS (byte-identical spike logs, sha256 {logs[0]})")
    else:
        print("DETERMINISM: FAIL — spike logs differ between identical runs")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=int, default=10)
    parser.add_argument("--ms-per-bar", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--determinism", action="store_true")
    args = parser.parse_args()

    if args.determinism:
        run_determinism(args.dt_ms)
        return

    t0 = time.perf_counter()
    chassis = load_whole_fly()
    t_load = time.perf_counter() - t0

    n = chassis.n_neurons
    print(f"neurons    : {n:,} ({'OK' if n == EXPECTED_NEURONS else 'MISMATCH'}, "
          f"expected {EXPECTED_NEURONS:,}; DESIGN preprint figure 166,691)")
    if n != EXPECTED_NEURONS:
        sys.exit(1)
    print(f"edges      : {chassis.adj.nnz:,} canonical")
    print(f"synapses   : {int(chassis.adj.data.sum()):,}")
    print(f"cache load : {t_load:.2f} s")

    sim = LIFSim(chassis, dt_ms=args.dt_ms, seed=0)
    print(f"sim init   : {time.perf_counter() - t0 - t_load:.2f} s after load, "
          f"peak RSS so far {peak_rss_bytes() / 2**30:.2f} GiB")

    inp = build_input(chassis, args.input_scale)
    driven = int((inp != 0).sum())
    print(f"input      : tonic {DRIVE_TONIC_MV:g} mV + sine {DRIVE_SINE_MV:g} mV "
          f"on {driven:,} sensory cells "
          f"(T4/T5/LC-looming/uPN, {DRIVE_FRACTION:.0%} of each)")

    bars = np.arange(args.bars, dtype=np.float64)
    mod = 1.0 + 0.5 * np.sin(2.0 * np.pi * bars / 39.0)

    t0 = time.perf_counter()
    total_spikes = 0
    for k in range(args.bars):
        bar_t0 = time.perf_counter()
        out = sim.step(inp * mod[k], duration_ms=args.ms_per_bar)
        total_spikes += out["total_spikes"]
        print(f"  bar {k + 1:>2}/{args.bars}: {time.perf_counter() - bar_t0:6.2f} s wall, "
              f"{out['total_spikes']:,} spikes")
    wall = time.perf_counter() - t0

    sim_ms = args.bars * args.ms_per_bar
    rate = sim_ms / 1000.0 / wall
    peak = peak_rss_bytes()
    print(f"simulated  : {sim_ms / 1000.0:.1f} s of biological time "
          f"({args.bars} bars x {args.ms_per_bar:g} ms, dt={args.dt_ms:g} ms)")
    print(f"wall time  : {wall:.1f} s ({wall / 60.0:.2f} min)")
    print(f"spikes     : {total_spikes:,} ({total_spikes / wall:,.0f}/s wall)")
    print(f"per bar    : {wall / max(args.bars, 1):.2f} s wall "
          f"({args.ms_per_bar / 1000.0:.1f} s sim)")
    print(f"sim rate   : {rate:.3f}x real time")
    print(f"peak RSS   : {peak / 2**30:.2f} GiB")
    ok_rate = rate >= 0.5
    ok_rss = peak <= 8 * 2**30
    verdict = "PASS" if (ok_rate and ok_rss) else "MISS"
    print(f"budget     : {verdict} "
          f"(need >= 0.5x real time [{'ok' if ok_rate else 'miss'}] "
          f"and <= 8 GiB RSS [{'ok' if ok_rss else 'miss'}])")
    if verdict == "MISS":
        print("deviation  : whole-fly LIF misses the usable budget -> lean "
              "mushroom-body rate-model fallback per DESIGN §13 (recorded, "
              "see reports/t11-wholefly.md)")


if __name__ == "__main__":
    main()
