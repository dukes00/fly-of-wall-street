"""T9 calibration driver: parameter sweep, larval training, sanity probe.

Usage (from the repo root, via uv so the project env is used):

    uv run python scripts/calibrate.py sweep [--jobs N] [--only SUBSTR]
    uv run python scripts/calibrate.py train --start DATE --end DATE [--seed S]
        [--out PATH]
    uv run python scripts/calibrate.py probe --day DATE [--seed S]
        [--weights PATH]

``sweep`` grids {position_cap × death_threshold × bar_granularity} (3 values
each) over a reduced representative window (2026-08-24..2026-08-28, 5 trading
days) at a fixed seed. Coarser granularities are resampled in-memory from the
1-minute cache (OHLCV aggregation) — the cache is never re-fetched. Cells run
in parallel worker processes (same-seed runs are byte-identical regardless of
load, per reports/t7-loop.md); each cell writes its run artifacts under
``data/runs/t9-calibration/cells/`` and one JSON metrics row, and the parent
collects them into ``data/runs/t9-calibration/sweep.csv`` + ``sweep.json``.

``train`` runs one full-window training run at the default config and
persists the trained KC→MBON weights (fruitfly.train.train_larval).

``probe`` loads a larval artifact, replays one day against a fresh fly, and
prints the two decision distributions plus the per-encounter divergence —
the T9 acceptance sanity probe.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

GRANULARITIES: dict[str, int] = {"1min": 1, "2min": 2, "5min": 5}
POSITION_CAPS: tuple[int, ...] = (6, 10, 14)
DEATH_THRESHOLDS: tuple[float, ...] = (-0.35, -0.50, -0.65)
SWEEP_START, SWEEP_END = "2026-08-24", "2026-08-28"
SWEEP_SEED = 7
OUT_DIR = Path("data/runs/t9-calibration")


# ---------------------------------------------------------------------------
# Bar resampling (D16 sweep axis: in-memory, no re-fetch)
# ---------------------------------------------------------------------------


def resample_bars(
    frames: dict[str, pd.DataFrame], minutes: int
) -> dict[str, pd.DataFrame]:
    """Aggregate session-filtered 1-minute OHLCV frames to ``minutes`` bars.

    Left-labeled bins on the 13:30 session anchor; inter-session and lunch
    gaps produce empty bins, dropped by ``dropna``. Volume sums, prices
    follow standard OHLC aggregation. Empty input frames pass through.
    """
    if minutes <= 1:
        return frames
    out: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        if df.empty:
            out[sym] = df
            continue
        agg = (
            df.resample(f"{minutes}min", label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        out[sym] = agg
    return out


# ---------------------------------------------------------------------------
# One sweep cell (runs in a worker process; patches the loop's data seam)
# ---------------------------------------------------------------------------


def run_cell(
    position_cap: int,
    death_threshold: float,
    granularity: str,
    chassis: str = "stripped",
    std_beta: float | None = None,
    std_tau_rec_ms: float | None = None,
    *,
    seed: int = SWEEP_SEED,
    start: str = SWEEP_START,
    end: str = SWEEP_END,
    out_root: Path = OUT_DIR,
) -> dict:
    """Run one grid cell and return its metrics row (also written to JSON)."""
    import fruitfly.loop as loop
    from fruitfly.data import BASKET, load_bars

    minutes = GRANULARITIES[granularity]
    frames = load_bars(list(BASKET), None, None)
    if minutes > 1:
        frames = resample_bars(frames, minutes)
    loop.load_bars = lambda symbols, start=None, end=None: frames

    name = f"cap{position_cap}_death{death_threshold:+.2f}_{granularity}"
    run_dir = out_root / "cells" / name
    config = loop.BacktestConfig(
        seed=seed,
        start=start,
        end=end,
        position_cap=position_cap,
        death_threshold=death_threshold,
        chassis=chassis,
        std_beta=std_beta,
        std_tau_rec_ms=std_tau_rec_ms,
        out_dir=run_dir,
    )
    t0 = time.perf_counter()
    result = loop.run_backtest(config)
    wall_s = time.perf_counter() - t0

    equity = pd.read_csv(result.run_dir / "equity.csv")["equity"]
    peak = equity.cummax()
    max_drawdown_pct = float((1.0 - equity / peak).max() * 100.0)
    row = {
        "cell": name,
        "position_cap": position_cap,
        "death_threshold": death_threshold,
        "granularity": granularity,
        "seed": seed,
        "start": start,
        "end": end,
        "chassis": chassis,
        "final_return_pct": (result.final_equity / config.initial_cash - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown_pct,
        "n_deaths": result.n_deaths,
        "trades": result.n_orders,
        "n_bars": result.n_bars,
        "wall_s": round(wall_s, 1),
    }
    cell_path = out_root / "cells" / f"{name}.json"
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    cell_path.write_text(json.dumps(row, sort_keys=True) + "\n")
    return row


def _cell_worker(args: tuple) -> dict:
    """ProcessPoolExecutor entry: unpack and run one cell."""
    return run_cell(*args)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def cmd_sweep(args: argparse.Namespace) -> int:
    chassis_args = (args.chassis, args.std_beta, args.std_tau_rec_ms)
    cells = [
        (cap, dth, gran) + chassis_args
        for gran in ("1min", "2min", "5min")
        for dth in DEATH_THRESHOLDS
        for cap in POSITION_CAPS
    ]
    if args.only:
        cells = [c for c in cells if args.only in f"cap{c[0]}_death{c[1]:+.2f}_{c[2]}"]
    print(f"T9 sweep: {len(cells)} cells, window {SWEEP_START}..{SWEEP_END}, "
          f"seed {SWEEP_SEED}, jobs {args.jobs}")
    print(f"  chassis {args.chassis}"
          + (f" std_beta={args.std_beta}" if args.std_beta is not None else "")
          + (f" std_tau_rec_ms={args.std_tau_rec_ms}"
             if args.std_tau_rec_ms is not None else ""))
    t0 = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_cell_worker, cells))
    else:
        rows = [run_cell(*c) for c in cells]
    rows.sort(key=lambda r: (r["granularity"], r["death_threshold"], r["position_cap"]))

    out = OUT_DIR / "sweep.json"
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    cols = ["granularity", "position_cap", "death_threshold", "final_return_pct",
            "max_drawdown_pct", "n_deaths", "trades", "n_bars", "wall_s", "cell"]
    pd.DataFrame(rows)[cols].to_csv(OUT_DIR / "sweep.csv", index=False)

    print(f"\n{'gran':>5} {'cap':>4} {'death':>6} {'ret%':>9} {'mdd%':>7} "
          f"{'deaths':>6} {'trades':>6} {'wall_s':>8}")
    for r in rows:
        print(f"{r['granularity']:>5} {r['position_cap']:>4} "
              f"{r['death_threshold']:>6.2f} {r['final_return_pct']:>9.3f} "
              f"{r['max_drawdown_pct']:>7.3f} {r['n_deaths']:>6} "
              f"{r['trades']:>6} {r['wall_s']:>8.1f}")
    print(f"\ntotal wall {time.perf_counter() - t0:.1f}s -> {out}")
    return 0


# ---------------------------------------------------------------------------
# Full-window larval training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    from fruitfly.loop import BacktestConfig
    from fruitfly.train import train_larval

    config = BacktestConfig(seed=args.seed, start=args.start, end=args.end,
                            chassis=args.chassis, std_beta=args.std_beta,
                            std_tau_rec_ms=args.std_tau_rec_ms)
    print(f"T9 larval training: seed {args.seed} {args.start}..{args.end} -> {args.out}")
    train = train_larval(config, out_path=Path(args.out))
    r = train.run
    print(
        f"  bars {r.n_bars}  orders {r.n_orders}  deaths {r.n_deaths}\n"
        f"  final equity {r.final_equity:.2f} ({r.wall_s:.1f}s wall)\n"
        f"  artifact {train.artifact} (weights {train.weights.shape}, "
        f"fingerprint {train.fingerprint[:12]}…)\n"
        f"  meta {train.meta}"
    )
    return 0


# ---------------------------------------------------------------------------
# Sanity probe: fresh fly vs larval artifact on one day
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    from fruitfly.connectome import load_stripped_chassis, load_whole_fly
    from fruitfly.loop import BacktestConfig, run_backtest
    from fruitfly.train import decision_map, load_larval_weights

    if args.chassis == "whole":
        chassis = load_whole_fly()
    else:
        chassis = load_stripped_chassis()
    lw = load_larval_weights(args.weights, chassis)
    meta = {k: lw.meta.get(k, "?") for k in ("seed", "start", "end")}
    print(f"probe day {args.day}, seed {args.seed}")
    print(f"  artifact {args.weights} trained seed={meta['seed']} "
          f"window {meta['start']}..{meta['end']}")

    tag = Path(args.weights).stem
    fresh_cfg = BacktestConfig(seed=args.seed, start=args.day, end=args.day,
                               chassis=args.chassis, std_beta=args.std_beta,
                               std_tau_rec_ms=args.std_tau_rec_ms,
                               out_dir=OUT_DIR / "probe" / tag / "fresh")
    trained_cfg = BacktestConfig(seed=args.seed, start=args.day, end=args.day,
                                 initial_weights=lw.weights,
                                 chassis=args.chassis, std_beta=args.std_beta,
                                 std_tau_rec_ms=args.std_tau_rec_ms,
                                 out_dir=OUT_DIR / "probe" / tag / "larval")
    fresh = run_backtest(fresh_cfg)
    trained = run_backtest(trained_cfg)

    fresh_map = decision_map(fresh.run_dir / "events.jsonl")
    trained_map = decision_map(trained.run_dir / "events.jsonl")
    common = sorted(set(fresh_map) & set(trained_map))
    divergent = [(k, fresh_map[k], trained_map[k]) for k in common
                 if fresh_map[k][0] != trained_map[k][0]]

    def dist(d):
        out: dict[str, int] = {}
        for action, _reason in d.values():
            out[action] = out.get(action, 0) + 1
        return out

    fresh_dist, trained_dist = dist(fresh_map), dist(trained_map)
    actions = sorted(set(fresh_dist) | set(trained_dist))
    print(f"\n  encounters {len(fresh_map)} (fresh) / {len(trained_map)} (larval); "
          f"common {len(common)}")
    print(f"  {'action':<6} {'fresh':>6} {'larval':>6}")
    for a in actions:
        print(f"  {a:<6} {fresh_dist.get(a, 0):>6} {trained_dist.get(a, 0):>6}")
    frac = (len(divergent) / len(common) * 100.0) if common else 0.0
    print(f"\n  divergent actions on common encounters: {len(divergent)}/{len(common)} "
          f"({frac:.1f}%)")
    for (ts, ticker), a, b in divergent[:12]:
        print(f"    {ts} {ticker:<6} {a[0]:<5} -> {b[0]:<5} ({a[1]} -> {b[1]})")

    summary = {
        "day": args.day,
        "seed": args.seed,
        "artifact": str(args.weights),
        "artifact_meta": lw.meta,
        "fresh_equity": round(fresh.final_equity, 2),
        "larval_equity": round(trained.final_equity, 2),
        "n_encounters": len(common),
        "divergent": len(divergent),
        "divergence_pct": round(frac, 2),
        "fresh_dist": fresh_dist,
        "larval_dist": trained_dist,
    }
    path = OUT_DIR / "probe" / f"probe_{args.day}_{Path(args.weights).stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\n  summary -> {path}")
    print(f"  final equity: fresh {fresh.final_equity:.2f} vs larval "
          f"{trained.final_equity:.2f}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def _add_chassis_args(p: argparse.ArgumentParser) -> None:
    """Shared --chassis / --std-* flags (sweep, train, probe)."""
    p.add_argument(
        "--chassis", choices=("stripped", "whole"), default="stripped",
        help="Brain chassis; 'whole' implies the calibrated T12b STD "
        "(beta=0.1, tau_rec=500 ms) unless overridden.",
    )
    p.add_argument(
        "--std-beta", type=float, default=None,
        help="STD depletion fraction (whole-fly default 0.1).",
    )
    p.add_argument(
        "--std-tau-rec-ms", type=float, default=None,
        help="STD recovery time constant in ms (whole-fly default 500).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep", help="Run the calibration grid.")
    p.add_argument("--jobs", type=int, default=1, help="Parallel worker processes.")
    p.add_argument("--only", default=None,
                   help="Substring filter on cell names (e.g. 'cap6', '5min').")
    p.set_defaults(func=cmd_sweep)
    _add_chassis_args(p)

    p = sub.add_parser("train", help="Full-window larval training run.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/fly-larval-weights.npz")
    p.set_defaults(func=cmd_train)
    _add_chassis_args(p)

    p = sub.add_parser("probe", help="Fresh vs larval-weights decision comparison.")
    p.add_argument("--day", required=True, help="Probe day (YYYY-MM-DD).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--weights", default="data/fly-larval-weights.npz")
    p.set_defaults(func=cmd_probe)
    _add_chassis_args(p)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
