#!/usr/bin/env python
"""T15 acceptance driver: multi-day adult replay over the recorded cache.

Runs the persistent adult fly (``fruitfly.adult``) in replay mode — recorded
1-minute bars, simulated fills at bar close — over one or more NYSE sessions
within the market cache window (2026-08-17 .. 2026-09-14). Receipts land in
the run directory in the shared backtest shape (``equity.csv`` +
``events.jsonl``), so the dashboard and post-mortem read them unchanged.

The state file (``<run-dir>/state.npz``) is persisted every bar. Re-running
the SAME command resumes from it: the process can be SIGKILLed at any point
and the next invocation continues the equity series and event log
byte-identically. Pass ``--fresh`` to discard a previous run's state and
receipts and start a new fly instead.

Run from the repo root:  uv run python scripts/replay_adult.py \
    --seed 7 --start 2026-08-17 --end 2026-08-19 [--larval-weights data/fly-larval-weights.npz]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fruitfly.adult import STATE_FILE, AdultConfig, AdultRun  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", required=True, help="e.g. 2026-08-17")
    parser.add_argument("--end", required=True, help="e.g. 2026-08-19")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--persist-every", type=int, default=1)
    parser.add_argument(
        "--larval-weights",
        default=None,
        help="T9 .npz artifact (default: none — fresh structural plasticity)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete the run directory's state + receipts first (new fly)",
    )
    args = parser.parse_args()

    config = AdultConfig(
        seed=args.seed,
        start=args.start,
        end=args.end,
        mode="replay",
        run_dir=args.run_dir,
        persist_every=args.persist_every,
        larval_weights=args.larval_weights,
    )
    run_dir = config.run_directory()
    if args.fresh:
        for name in (STATE_FILE, "equity.csv", "events.jsonl"):
            (run_dir / name).unlink(missing_ok=True)

    result = AdultRun(config).run()
    tag = "resumed" if result.resumed else "fresh"
    print(
        f"adult[{tag}]: run_dir={result.run_dir} bars={result.n_bars} "
        f"events={result.n_events} orders={result.n_orders} deaths={result.n_deaths} "
        f"final_equity={result.final_equity:.2f} wall={result.wall_s:.1f}s"
    )
    print("dashboard:  uv run python -m fruitfly dashboard --run-dir", result.run_dir)
    print("postmortem: uv run python -m fruitfly postmortem --run-dir", result.run_dir)


if __name__ == "__main__":
    main()
