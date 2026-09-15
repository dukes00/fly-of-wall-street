"""Brain-shootout evaluation: held-out comparison of fly brains.

Usage (from the repo root, via uv so the project env is used; point
``FRUITFLY_MARKET_DIR`` at the extended IEX cache):

    FRUITFLY_MARKET_DIR=data/market/history uv run python scripts/eval_brains.py \
        --chassis stripped --artifact data/fly-stripped-hist-weights.npz
    FRUITFLY_MARKET_DIR=data/market/history uv run python scripts/eval_brains.py \
        --chassis whole --artifact data/fly-whole-weights.npz

For the given artifact + chassis the script replays the held-out days (the
most recent trading days of the cache; default 10, evenly spaced if fewer
are requested) — one seeded per-day backtest each, with the artifact's
trained KC→MBON weights injected, the chassis/STD config passed exactly
like ``scripts/calibrate.py`` does — and aggregates per day: final return
%, max drawdown %, trade count, deaths. Results land in a deterministic
JSON receipt per arm (``<results-dir>/results_<chassis>.json``) and the
markdown comparison table (stripped arm vs whole-fly arm vs SPX buy-hold
over the same days, via ``fruitfly.scoreboard.spx_buyhold``) is rendered
into ``reports/brain-shootout.md``. The whole-fly arm renders as
``PENDING`` until its eval has been run.

Determinism hard gate: fixed seed everywhere, fixed member order and float
formatting in every artifact — the same inputs produce a byte-identical
report (the per-day runs themselves are seed-locked and byte-identical per
reports/t7-loop.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

#: Where per-arm JSON receipts and per-day run artifacts are written.
RESULTS_DIR = Path("data/runs/brain-shootout")

#: Default report target (the orchestrator fills the whole-fly arm later).
REPORT_PATH = Path("reports/brain-shootout.md")

#: Size of the held-out window: the most recent cache trading days the
#: eval replays (the spec'd default: 10 days, 2026-08-28..2026-09-11).
RECENT_POOL = 10

#: The whole-fly arm is marked PENDING until its eval runs; this descriptor
#: is rendered into the arms table in the meantime.
WHOLE_PENDING = {
    "artifact": "data/fly-whole-weights.npz",
    "window": "2026-03-02..2026-06-30 (whole-fly training in flight)",
}

#: Percent formatting precision used everywhere (deterministic rendering).
PCT = "{:.3f}"


# ---------------------------------------------------------------------------
# Held-out day selection
# ---------------------------------------------------------------------------


def cache_trading_days() -> list[str]:
    """Most-recent-last list of ISO trading days present in the bar cache.

    Union over the basket symbols of the per-day timestamps in the parquet
    cache (respects ``FRUITFLY_MARKET_DIR`` via ``fruitfly.data``), so the
    held-out window is whatever the cache actually holds.
    """
    from fruitfly.data import BASKET, cache_path

    days: set[str] = set()
    for symbol in BASKET:
        path = cache_path(symbol)
        if not path.exists():
            continue
        ts = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        days.update(pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).strftime("%Y-%m-%d"))
    if not days:
        raise ValueError("bar cache holds no trading days")
    return sorted(days)


def select_days(available: list[str], n: int, pool: int = RECENT_POOL) -> list[str]:
    """Held-out days: evenly spaced days out of the ``pool`` most recent.

    The held-out window is the ``pool`` most recent cache trading days
    (default 10); ``n`` evenly spaced days within it (endpoints included).
    ``n == pool`` returns the whole window.
    """
    if n < 1:
        raise ValueError(f"--days must be >= 1, got {n}")
    recent = available[-pool:] if pool < len(available) else list(available)
    if n > len(recent):
        raise ValueError(
            f"--days {n} exceeds the {len(recent)} held-out trading days "
            f"(the {pool} most recent cache days)"
        )
    if n == 1:
        return [recent[-1]]
    if n == len(recent):
        return list(recent)
    idx = sorted({round(i * (len(recent) - 1) / (n - 1)) for i in range(n)})
    return [recent[i] for i in idx]

# ---------------------------------------------------------------------------
# Per-day metrics
# ---------------------------------------------------------------------------


def day_metrics(
    equity_csv: Path, final_equity: float, n_orders: int, n_deaths: int,
    initial_cash: float,
) -> dict:
    """Aggregate one held-out day: return %, drawdown %, trades, deaths.

    ``equity_csv`` is the run's per-bar ``timestamp,equity,...`` receipt;
    the drawdown is measured bar-by-bar against the running peak.
    """
    eq = pd.read_csv(equity_csv)["equity"].to_numpy(dtype=float)
    running_peak = pd.Series(eq).cummax().to_numpy()
    max_dd_pct = 100.0 * float(max(1.0 - eq / running_peak))
    return {
        "final_return_pct": round(100.0 * (final_equity / initial_cash - 1.0), 6),
        "max_drawdown_pct": round(max_dd_pct, 6),
        "trades": n_orders,
        "deaths": n_deaths,
    }


def spx_daily_returns(days: list[str]) -> dict:
    """SPX buy-hold per-day return % + window totals over the same days.

    Per-day return is close-to-close (the base is the business day before
    the first held-out day); the window total is the compounded return and
    the drawdown is measured across the held-out days' equity points.
    """
    scoreboard = import_module("fruitfly.scoreboard")
    first, last = pd.Timestamp(days[0], tz="UTC"), pd.Timestamp(days[-1], tz="UTC")
    prior = pd.bdate_range(end=first, periods=2)[0]
    equity = scoreboard.spx_buyhold(str(prior.date()), str(last.date()))
    pts = equity.loc[  # prior close (base) + one point per held-out day
        [str(prior.date()), *days]
    ]
    rets = (pts.pct_change().iloc[1:] * 100.0).to_numpy(dtype=float)
    window = pts.to_numpy(dtype=float)
    peak = pd.Series(window).cummax().to_numpy()
    return {
        "daily_return_pct": [round(float(r), 6) for r in rets],
        "window_return_pct": round(
            100.0 * float(window[-1] / window[0] - 1.0), 6
        ),
        "max_drawdown_pct": round(
            100.0 * float(max(1.0 - window / peak)), 6
        ),
    }


# ---------------------------------------------------------------------------
# One arm: replay the held-out days
# ---------------------------------------------------------------------------


def evaluate_arm(
    chassis: str,
    artifact: str | Path,
    days: list[str],
    seed: int,
    std_beta: float | None,
    std_tau_rec_ms: float | None,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Replay every held-out day with the artifact's trained weights.

    The chassis/STD triple is passed exactly like ``calibrate.py`` does
    (``BacktestConfig`` normalizes the whole-fly STD defaults); the artifact
    is fingerprint-verified against the runtime chassis, so a brain trained
    on a different one is rejected. Returns the arm's deterministic receipt
    dict and also writes it to ``results_<chassis>.json``.
    """
    from fruitfly.connectome import load_stripped_chassis, load_whole_fly
    from fruitfly.loop import INITIAL_CASH, BacktestConfig, run_backtest
    from fruitfly.train import load_larval_weights

    brain = load_whole_fly() if chassis == "whole" else load_stripped_chassis()
    lw = load_larval_weights(artifact, brain)

    runs_root = results_dir / "runs" / chassis
    per_day: list[dict] = []
    for day in days:
        config = BacktestConfig(
            seed=seed, start=day, end=day, chassis=chassis,
            std_beta=std_beta, std_tau_rec_ms=std_tau_rec_ms,
            initial_weights=lw.weights, out_dir=runs_root / day,
        )
        result = run_backtest(config)
        per_day.append(
            {"day": day, **day_metrics(
                result.run_dir / "equity.csv", result.final_equity,
                result.n_orders, result.n_deaths, INITIAL_CASH,
            )}
        )

    artifact_path = Path(artifact)
    receipt = {
        "arm": chassis,
        "chassis": chassis,
        "artifact": str(artifact_path),
        "artifact_md5": hashlib.md5(artifact_path.read_bytes()).hexdigest(),
        "artifact_meta": dict(lw.meta),
        "seed": seed,
        "std_beta": std_beta,
        "std_tau_rec_ms": std_tau_rec_ms,
        "days": per_day,
        "totals": {
            "mean_return_pct": round(
                sum(d["final_return_pct"] for d in per_day) / len(per_day), 6
            ),
            "max_drawdown_pct": round(
                max(d["max_drawdown_pct"] for d in per_day), 6
            ),
            "trades": sum(d["trades"] for d in per_day),
            "deaths": sum(d["deaths"] for d in per_day),
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"results_{chassis}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def load_receipts(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Load every arm receipt present in ``results_dir`` (sorted keys)."""
    receipts: dict[str, dict] = {}
    for path in sorted(results_dir.glob("results_*.json")):
        receipt = json.loads(path.read_text())
        receipts[receipt["chassis"]] = receipt
    return receipts


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _cell(value: float | int | None, fmt=PCT.format) -> str:
    return "—" if value is None else fmt(value)


def render_report(
    receipts: dict[str, dict],
    report_path: Path = REPORT_PATH,
) -> str:
    """Render the markdown comparison table from the arm receipts.

    The whole-fly arm without a receipt renders as PENDING. Deterministic:
    fixed column order, fixed float formatting, same inputs → same bytes.
    """
    stripped = receipts.get("stripped")
    whole = receipts.get("whole")
    if stripped is None:
        raise ValueError("render_report needs the stripped arm receipt")

    days = [d["day"] for d in stripped["days"]]
    for name, rc in (("whole", whole),):
        if rc is not None and [d["day"] for d in rc["days"]] != days:
            raise ValueError(f"{name} receipt covers different days than stripped")
    spx = spx_daily_returns(days)

    def row(day: str | None, i: int | None) -> list[str]:
        cells = [day if day is not None else "TOTAL"]
        if i is None:
            cells += [_cell(spx["window_return_pct"]), _cell(spx["max_drawdown_pct"])]
        else:
            cells += [_cell(spx["daily_return_pct"][i]), "—"]
        for rc in (stripped, whole):
            if rc is None:
                cells += ["PENDING"] * 4
                continue
            if i is None:
                t = rc["totals"]
                cells += [
                    _cell(t["mean_return_pct"]),
                    _cell(t["max_drawdown_pct"]),
                    str(t["trades"]),
                    str(t["deaths"]),
                ]
            else:
                d = rc["days"][i]
                cells += [
                    _cell(d["final_return_pct"]),
                    _cell(d["max_drawdown_pct"]),
                    str(d["trades"]),
                    str(d["deaths"]),
                ]
        return cells

    header = (
        "| day | SPX ret% | SPX dd% "
        "| stripped ret% | stripped dd% | stripped trades | stripped deaths "
        "| whole ret% | whole dd% | whole trades | whole deaths |"
    )
    sep = "|---" + "|---:" * 10 + "|"
    lines = [
        "# Brain shootout — stripped vs whole fly (held-out)",
        "",
        "Held-out comparison of the two trained brains (same training window",
        "2026-03-02..2026-06-30, seed 7) over the most recent trading days of",
        "the extended IEX cache, against SPX buy-hold over the same days",
        "(`fruitfly.scoreboard.spx_buyhold`). Generated by",
        "`scripts/eval_brains.py`; identical inputs → byte-identical report.",
        "",
        "## Arms",
        "",
        "| arm | chassis | artifact | md5 | trained window | meta |",
        "|---|---|---|---|---|---|",
    ]
    for rc in (stripped, whole):
        if rc is None:
            lines.append(
                f"| whole | whole | {WHOLE_PENDING['artifact']} | — "
                f"| {WHOLE_PENDING['window']} | **PENDING** |"
            )
            continue
        meta = rc["artifact_meta"]
        lines.append(
            f"| {rc['arm']} | {rc['chassis']} | `{rc['artifact']}` "
            f"| {rc['artifact_md5'][:12]}… | {meta.get('start', '?')}"
            f"..{meta.get('end', '?')} | seed={meta.get('seed', '?')} "
            f"bars={meta.get('n_bars', '?')} deaths={meta.get('n_deaths', '?')} "
            f"std=({rc['std_beta']},{rc['std_tau_rec_ms']}) |"
        )

    notes = [
        "",
        "## Held-out comparison",
        "",
        "Per-day: final return % and max drawdown % of the day's backtest",
        "(each day a fresh fly at $100,000, artifact weights injected, fixed",
        f"seed {stripped['seed']}); trades = orders, deaths = fly deaths. TOTAL row:",
        "arms aggregate mean daily return %, worst-day drawdown %, summed",
        "trades/deaths; SPX shows the compounded window return % and the",
        "drawdown % across the held-out days.",
    ]
    if whole is None:
        notes.append("Missing whole-fly arm = PENDING.")
    lines += [*notes, "", header, sep]
    for i, day in enumerate(days):
        lines.append("| " + " | ".join(row(day, i)) + " |")
    lines.append("| " + " | ".join(row(None, None)) + " |")
    lines.append("")

    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace) -> None:
    if args.days < 1:
        raise ValueError(f"--days must be >= 1, got {args.days}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chassis", choices=("stripped", "whole"), required=True,
                        help="Brain chassis of the artifact being evaluated.")
    parser.add_argument("--artifact", required=True,
                        help="Larval .npz artifact with the trained weights.")
    parser.add_argument("--days", type=int, default=10,
                        help="Held-out days (most recent cache days, evenly "
                             "spaced if fewer; default 10 = all recent).")
    parser.add_argument("--seed", type=int, default=7,
                        help="Fixed eval seed (default 7, matches training).")
    parser.add_argument("--std-beta", type=float, default=None,
                        help="STD depletion fraction (whole-fly default 0.1).")
    parser.add_argument("--std-tau-rec-ms", type=float, default=None,
                        help="STD recovery time constant in ms (whole default 500).")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Receipt + run-artifact directory.")
    parser.add_argument("--report", default=str(REPORT_PATH),
                        help="Markdown report to (re)render.")
    args = parser.parse_args(argv)
    _validate_args(args)

    days = select_days(cache_trading_days(), args.days)
    print(f"eval {args.chassis}: {args.artifact} seed {args.seed}, "
          f"held-out {days[0]}..{days[-1]} ({len(days)} days)")
    evaluate_arm(args.chassis, args.artifact, days, args.seed,
                 args.std_beta, args.std_tau_rec_ms, Path(args.results_dir))
    receipts = load_receipts(Path(args.results_dir))
    render_report(receipts, Path(args.report))
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
