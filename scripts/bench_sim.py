"""T3 benchmark: one full trading day of LIF simulation through the chassis.

Runs 390 trading bars (default 500 ms of simulated time per bar, i.e. 1000
substeps at dt=0.5 ms) through the stripped chassis (27,115 neurons, 3.9M
synapses) and prints measured wall time and throughput. Target: under
10 minutes of wall time for the default day.

Usage::

    uv run python scripts/bench_sim.py [--bars 390] [--ms-per-bar 500]
                                       [--dt-ms 0.5] [--input-scale 1.0]

The drive pattern is deterministic (no RNG): a tonic+sinusoidal current into
the sensory populations (T4, T5, LC-looming, uPN) at a fraction of cells,
representative of encoder output during an active session.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fruitfly.connectome import load_stripped_chassis
from fruitfly.sim import LIFSim

#: Populations the sensory encoders (T4) actually drive in stripped mode.
DRIVEN_POPULATIONS = ("T4", "T5", "LC-looming", "uPN")

#: Fraction of cells per driven population receiving encoder-like drive.
DRIVE_FRACTION = 0.2

#: Tonic drive (mV above rest) + sinusoidal modulation amplitude.
DRIVE_TONIC_MV = 18.0
DRIVE_SINE_MV = 6.0


def build_input(chassis, scale: float) -> np.ndarray:
    """Deterministic per-bar input vector, indexed like chassis.nodes."""
    pop = chassis.nodes["population"].to_numpy()
    inp = np.zeros(chassis.n_neurons, dtype=np.float64)
    for name in DRIVEN_POPULATIONS:
        idx = np.flatnonzero(pop == name)
        driven = idx[idx % int(1.0 / DRIVE_FRACTION) == 0]
        inp[driven] = scale * (DRIVE_TONIC_MV + DRIVE_SINE_MV)
    return inp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=int, default=390)
    parser.add_argument("--ms-per-bar", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--input-scale", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    chassis = load_stripped_chassis()
    t_load = time.perf_counter() - t0
    print(
        f"chassis: {chassis.n_neurons} neurons, {chassis.adj.nnz} edges "
        f"({chassis.meta.get('n_edges', '?')} canonical), "
        f"{chassis.adj.data.sum()} synapses — loaded in {t_load:.2f}s"
    )

    sim = LIFSim(chassis, dt_ms=args.dt_ms, seed=0)
    inp = build_input(chassis, args.input_scale)

    # Sinusoidal amplitude across bars: deterministic market-like modulation.
    bars = np.arange(args.bars, dtype=np.float64)
    mod = 1.0 + 0.5 * np.sin(2.0 * np.pi * bars / 39.0)

    t0 = time.perf_counter()
    total_spikes = 0
    for k in range(args.bars):
        out = sim.step(inp * mod[k], duration_ms=args.ms_per_bar)
        total_spikes += out["total_spikes"]
    wall = time.perf_counter() - t0

    sim_ms = args.bars * args.ms_per_bar
    print(f"simulated : {sim_ms / 1000.0:.1f} s of biological time "
          f"({args.bars} bars x {args.ms_per_bar:g} ms, dt={args.dt_ms:g} ms)")
    print(f"wall time : {wall:.1f} s ({wall / 60.0:.2f} min)")
    print(f"spikes    : {total_spikes}")
    print(f"throughput: {total_spikes / wall:,.0f} spikes/sec (wall) | "
          f"{sim_ms / 1000.0 / wall:,.1f}x real time")
    print(f"per bar   : {wall / max(args.bars, 1) * 1000.0:.1f} ms wall")
    print(f"result    : {'PASS' if wall <= 600.0 else 'OVER TARGET'} "
          f"(target <= 600 s)")


if __name__ == "__main__":
    main()
