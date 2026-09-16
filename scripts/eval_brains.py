"""Brain-shootout evaluation: held-out comparison of fly brains.

Usage (from the repo root, via uv so the project env is used; point
``FRUITFLY_MARKET_DIR`` at the extended IEX cache):

    FRUITFLY_MARKET_DIR=data/market/history uv run python scripts/eval_brains.py \
        --chassis stripped --artifact data/fly-stripped-hist-weights.npz
    FRUITFLY_MARKET_DIR=data/market/history uv run python scripts/eval_brains.py \
        --chassis whole --artifact data/fly-whole-weights.npz

For the given artifact + chassis the script replays the held-out days (10
days drawn evenly spaced from the 40-day recent pool of cache trading days,
TRAINING2 §7) — one seeded per-day backtest each, with the artifact's
trained KC→MBON weights injected, the chassis/STD config passed exactly
like ``scripts/calibrate.py`` does — and aggregates per day: final return
%, max drawdown %, trade count, deaths. Results land in a deterministic
JSON receipt per arm (``<results-dir>/results_<arm>.json``) and the
markdown comparison tables (stripped arm vs whole-fly arm vs benchmark —
the ^GSPC daily close, or the SPY proxy from the fetched cache when the
SPX daily cache does not cover the window) are rendered
into ``reports/brain-shootout.md``. The whole-fly arm renders as
``PENDING`` until its eval has been run. Before any arm runs, the A9
artifact meta contract is asserted: the artifact's recorded credit knobs +
trained basket must match the eval runtime config (mismatch = exit 2);
legacy artifacts without the A9 meta keys evaluate only at all-default
configs (warn-and-continue), and the never-seen basket guard applies.

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
import warnings
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import date
from importlib import import_module
from pathlib import Path

import pandas as pd

#: Where per-arm JSON receipts and per-day run artifacts are written.
RESULTS_DIR = Path("data/runs/brain-shootout")

#: Default report target (the orchestrator fills the whole-fly arm later).
REPORT_PATH = Path("reports/brain-shootout.md")

#: Size of the held-out window: the most recent cache trading days the
#: eval replays (TRAINING2 §7: the sampler must draw from >= 40 days).
RECENT_POOL = 40

#: The never-seen eval basket (TRAINING2 §3.4): symbols that must appear in
#: no training run, ever. One symbol per line, ``#`` comments. The guard is
#: inert until the file is committed.
NEVER_SEEN_PATH = Path("baskets/eval20-neverseen.txt")

#: The whole-fly arm is marked PENDING until its eval runs; this descriptor
#: is rendered into the arms table in the meantime.
WHOLE_PENDING = {
    "artifact": "data/fly-whole-weights.npz",
    "window": "2026-03-02..2026-06-30 (whole-fly training in flight)",
}

#: Ratio/t-statistic formatting precision (paired-t, capture ratios).
TWO = "{:.2f}"

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
    (default 40, TRAINING2 §7); ``n`` evenly spaced days within it
    (endpoints included). ``n == pool`` returns the whole window.
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
# Benchmark: ^GSPC daily cache, else the SPY proxy from the 1m parquet
# ---------------------------------------------------------------------------


def spy_daily_returns(days: list[str], symbol: str = "SPY") -> dict:
    """Benchmark per-day return % from the fetched 1m parquet cache.

    The OOS-sweep SPY-proxy pattern: last close of the day for ``symbol``,
    first->last close per period, close-to-close per-day returns with the
    last close strictly before the window as the base. Same shape as
    :func:`spx_daily_returns`.
    """
    from fruitfly.data import cache_path

    path = cache_path(symbol)
    if not path.exists():
        raise ValueError(f"no {symbol} bars in the cache; cannot benchmark")
    df = pd.read_parquet(path, columns=["timestamp", "close"])
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    close = pd.Series(df["close"].to_numpy(dtype=float), index=idx)
    day_close = close.groupby(idx.strftime("%Y-%m-%d")).last().sort_index()
    missing = [day for day in days if day not in day_close.index]
    if missing:
        raise ValueError(f"{symbol} cache misses {len(missing)} of the "
                         f"held-out days (first: {missing[0]})")
    prior = day_close.index[day_close.index < days[0]]
    if len(prior) == 0:
        raise ValueError(f"no {symbol} close strictly before {days[0]} "
                         "for the benchmark base")
    pts = day_close.loc[[prior[-1], *days]].to_numpy(dtype=float)
    rets = (pts[1:] / pts[:-1] - 1.0) * 100.0
    peak = pd.Series(pts).cummax().to_numpy()
    return {
        "source": "spy",
        "daily_return_pct": [round(float(r), 6) for r in rets],
        "window_return_pct": round(100.0 * float(pts[-1] / pts[0] - 1.0), 6),
        "max_drawdown_pct": round(100.0 * float(max(1.0 - pts / peak)), 6),
    }


def benchmark_daily_returns(days: list[str]) -> dict:
    """Benchmark per-day returns over ``days``: ^GSPC when covered, else SPY.

    The ^GSPC daily cache is 2026-only, and ``spx_buyhold``'s
    reindex+interpolate silently flattens uncovered windows — so coverage
    is checked against the raw daily index first. Uncovered windows fall
    back to the SPY proxy from the same fetched cache.
    """
    try:
        from fruitfly.scoreboard import _load_spx_daily

        daily = _load_spx_daily()
        first, last = pd.Timestamp(days[0], tz="UTC"), pd.Timestamp(days[-1], tz="UTC")
        if len(daily) and daily.index.min() <= first and daily.index.max() >= last:
            out = spx_daily_returns(days)
            return {"source": "spx", **out}
    except (FileNotFoundError, OSError):
        pass
    return spy_daily_returns(days)


# ---------------------------------------------------------------------------
# A9 artifact-meta contract: meta <-> eval runtime config, never-seen guard
# ---------------------------------------------------------------------------


class MetaMismatch(ValueError):
    """Artifact meta does not match the eval runtime config (exit 2)."""


class NeverSeenGuardError(ValueError):
    """The eval basket intersects never-seen names the artifact was trained on."""


def _knob_defaults() -> dict[str, object]:
    """``BacktestConfig``'s declared defaults for the A9 knobs."""
    from fruitfly.loop import BacktestConfig

    return {f.name: f.default for f in fields(BacktestConfig)}


def _fmt_knob(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def parse_meta_basket(meta: dict[str, str]) -> list[str] | None:
    """The trained basket from the meta echo; ``None`` on legacy artifacts.

    A present-but-malformed entry fails closed.
    """
    raw = meta.get("basket")
    if raw is None:
        return None
    try:
        basket = json.loads(raw)
        if not isinstance(basket, list) or any(
            not isinstance(s, str) or not s for s in basket
        ):
            raise ValueError("not a JSON string list")
    except ValueError as exc:
        raise MetaMismatch(
            f"artifact meta basket is malformed: {raw!r} ({exc})"
        ) from exc
    return basket


def assert_meta_matches(
    meta: dict[str, str],
    config,
    eval_basket: list[str],
    artifact: str | Path = "",
) -> None:
    """A9: the artifact's recorded meta must match the eval runtime config.

    Knob-by-knob comparison against the config the eval will actually run;
    any mismatch raises :class:`MetaMismatch` (the CLI turns that into
    exit 2) with a line-per-knob diff. Legacy artifacts without the A9 meta
    keys fail closed on a non-default eval config but warn-and-continue
    when the config is entirely at defaults (so v0.6 artifacts still
    evaluate). The same legacy rule covers the trained basket.
    """
    defaults = _knob_defaults()
    diffs: list[str] = []
    legacy: list[str] = []
    for name in _meta_knob_names():
        if not hasattr(config, name):
            continue  # knob not on the installed BacktestConfig: unassertable
        actual = _fmt_knob(getattr(config, name))
        if name in meta:
            if str(meta[name]) != actual:
                diffs.append(
                    f"  {name}: artifact {meta[name]!r} != eval config {actual!r}"
                )
        elif actual != _fmt_knob(defaults.get(name)):
            diffs.append(
                f"  {name}: legacy artifact has no recorded {name}, but the "
                f"eval config is non-default ({actual!r} != default "
                f"{_fmt_knob(defaults.get(name))!r})"
            )
        else:
            legacy.append(name)

    trained = parse_meta_basket(meta)
    basket_legacy = False
    if trained is not None:
        if trained != list(eval_basket):
            diffs.append(
                f"  basket: artifact trained on [{','.join(trained)}] != "
                f"eval basket [{','.join(eval_basket)}]"
            )
    elif list(eval_basket) != list(_default_basket()):
        diffs.append(
            "  basket: legacy artifact has no recorded basket, but the eval "
            f"basket is non-default ([{','.join(eval_basket)}])"
        )
    else:
        basket_legacy = True

    if diffs:
        header = (
            f"artifact meta does not match the eval runtime config "
            f"({artifact}):"
        )
        raise MetaMismatch("\n".join([header, *diffs]))
    if legacy or basket_legacy:
        warnings.warn(
            "legacy artifact without A9 meta echo (missing: "
            f"{', '.join(legacy + (['basket'] if basket_legacy else []))}); "
            "eval config is all-default — continuing",
            stacklevel=2,
        )


def _meta_knob_names() -> tuple[str, ...]:
    from fruitfly.train import META_KNOBS

    return META_KNOBS


def _default_basket() -> list[str]:
    import fruitfly.data

    return list(fruitfly.data.BASKET)


def load_never_seen(path: Path = NEVER_SEEN_PATH) -> frozenset[str]:
    """Symbols in the never-seen basket file; empty when not yet committed."""
    if not path.exists():
        return frozenset()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        symbol = line.split("#", 1)[0].strip()
        if symbol:
            out.add(symbol)
    return frozenset(out)


def assert_never_seen(
    trained_basket: list[str] | None,
    eval_basket: list[str],
    path: Path = NEVER_SEEN_PATH,
) -> None:
    """Refuse to eval a never-seen basket on a fly trained on those names.

    Mirrors the loop-side guard: evaluating ON the never-seen basket is the
    intended use — but only for artifacts whose meta basket proves they
    never saw those symbols. Inert while the basket file is uncommitted or
    the eval basket does not intersect it; legacy artifacts without a meta
    basket cannot be checked (the A9 assertion already failed them closed
    on non-default baskets).
    """
    never_seen = load_never_seen(path)
    if not never_seen or not (never_seen & set(eval_basket)):
        return
    overlap = never_seen & set(trained_basket or ())
    if overlap:
        raise NeverSeenGuardError(
            "never-seen guard: artifact was TRAINED on "
            f"[{','.join(sorted(overlap))}] which the eval basket also "
            "contains — those names must appear in no training run, ever"
        )


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
    period: dict[str, str] | None,
    basket: list[str] | None,
    results_dir: Path = RESULTS_DIR,
    arm: str | None = None,
    knobs: dict | None = None,
) -> dict:
    """Replay every held-out day with the artifact's trained weights.

    The chassis/STD triple is passed exactly like ``calibrate.py`` does
    (``BacktestConfig`` normalizes the whole-fly STD defaults); the artifact
    is fingerprint-verified against the runtime chassis, so a brain trained
    on a different one is rejected. Before any day runs, the A9 artifact
    meta contract is enforced (meta knobs + basket vs this arm's runtime
    config; never-seen guard). Returns the arm's deterministic receipt dict
    and also writes it to ``results_<arm>.json`` (``arm`` defaults to the
    chassis name; pass a distinct name per config cell so sibling cells do
    not overwrite each other's receipts).
    """
    import fruitfly.data
    from fruitfly.connectome import load_stripped_chassis, load_whole_fly
    from fruitfly.loop import INITIAL_CASH, BacktestConfig, run_backtest
    from fruitfly.train import decode_baselines, load_larval_weights

    brain = load_whole_fly() if chassis == "whole" else load_stripped_chassis()
    lw = load_larval_weights(artifact, brain)
    train_start = str(lw.meta.get("start", ""))
    train_end = str(lw.meta.get("end", ""))
    # Overlap guard (audit 2026-09-15, amended 2026-09-16): days BEFORE the
    # training window are genuine out-of-sample (no leakage possible from a
    # later-trained artifact); only days INSIDE [train_start, train_end] are
    # refused.
    if days[0] <= train_end and days[-1] >= train_start:
        raise ValueError(
            f"held-out window {days[0]}..{days[-1]} overlaps the artifact's "
            f"training window {train_start}..{train_end}; refusing to "
            f"evaluate on training days"
        )

    # One runtime config for the whole arm; the assertion and every day's
    # run see exactly the same knob values (the A9 compare is meaningful
    # only against the config the eval actually executes).
    config = BacktestConfig(
        seed=seed, start=days[0], end=days[-1], chassis=chassis,
        std_beta=std_beta, std_tau_rec_ms=std_tau_rec_ms,
        **(knobs or {}),
    )
    eval_basket = list(basket) if basket is not None else list(fruitfly.data.BASKET)
    assert_meta_matches(lw.meta, config, eval_basket, artifact=artifact)
    trained_basket = parse_meta_basket(lw.meta)
    assert_never_seen(trained_basket, eval_basket)
    # Baseline warm-start (TRAINING2 §2B / risk 5): an advantage-mode
    # replay must credit against the trained bucket EMA, not a cold one.
    if ("initial_baselines" in lw.meta
            and getattr(config, "entry_credit", None) == "advantage"):
        config = replace(
            config,
            initial_baselines=decode_baselines(lw.meta["initial_baselines"]),
        )

    runs_root = results_dir / "runs" / (arm or chassis)
    per_day: list[dict] = []
    for day in days:
        result = run_backtest(replace(config, start=day, end=day,
                                      out_dir=runs_root / day))
        per_day.append(
            {"day": day, **day_metrics(
                result.run_dir / "equity.csv", result.final_equity,
                result.n_orders, result.n_deaths, INITIAL_CASH,
            )}
        )

    arm_name = arm or chassis
    artifact_path = Path(artifact)
    receipt = {
        "arm": arm_name,
        "chassis": chassis,
        "artifact": str(artifact_path),
        "artifact_md5": hashlib.md5(artifact_path.read_bytes()).hexdigest(),
        "artifact_meta": dict(lw.meta),
        "meta_knobs": {
            name: lw.meta[name] for name in _meta_knob_names() if name in lw.meta
        },
        **({"meta_basket": trained_basket} if trained_basket is not None else {}),
        "seed": seed,
        "std_beta": std_beta,
        "std_tau_rec_ms": std_tau_rec_ms,
        **({"period": period} if period is not None else {}),
        **({"basket": basket} if basket is not None else {}),
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
    (results_dir / f"results_{arm_name}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def load_receipts(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Load every arm receipt present in ``results_dir`` (sorted keys).

    Receipts are keyed by ``arm`` (falling back to ``chassis`` for
    pre-A9 receipts), so sibling config cells with distinct arm names
    coexist instead of overwriting each other.
    """
    receipts: dict[str, dict] = {}
    for path in sorted(results_dir.glob("results_*.json")):
        receipt = json.loads(path.read_text())
        receipts[receipt.get("arm", receipt["chassis"])] = receipt
    return receipts


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _cell(value: float | int | None, fmt=PCT.format) -> str:
    return "—" if value is None else fmt(value)


def paired_t(a: list[float], b: list[float]) -> float | None:
    """Paired t of two equal-length daily return series (None if degenerate).

    The §7 primary portfolio-level statistic: both arms replay the same
    tape, so the day-level difference series is the paired unit. Undefined
    (None) for fewer than 2 shared days or a zero-variance difference
    series — a constant difference is not a t-statistic.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0.0:
        return None
    return mean / (var ** 0.5) * (n ** 0.5)


def capture_ratios(
    strategy: list[float], benchmark: list[float]
) -> dict[str, float | None]:
    """Up/down-capture: strategy mean over benchmark up/down days relative
    to the benchmark mean over those same days (None when the benchmark
    has no up or no down day in the cell)."""
    if not benchmark or len(strategy) != len(benchmark):
        return {"up_capture": None, "down_capture": None}
    up_s = [s for s, b in zip(strategy, benchmark, strict=True) if b > 0]
    up_b = [b for b in benchmark if b > 0]
    down_s = [s for s, b in zip(strategy, benchmark, strict=True) if b < 0]
    down_b = [b for b in benchmark if b < 0]
    up = sum(up_s) / sum(up_b) if up_s and sum(up_b) > 0 else None
    down = sum(down_s) / sum(down_b) if down_s and sum(down_b) < 0 else None
    return {"up_capture": up, "down_capture": down}


def render_report(
    receipts: dict[str, dict],
    report_path: Path = REPORT_PATH,
    incumbent: str = "stripped",
) -> str:
    """Render the markdown comparison tables from the arm receipts.

    The whole-fly arm without a receipt renders as PENDING. Per cell (an
    arm receipt), the statistics table renders the paired t of daily
    final-return % against the incumbent arm on shared days plus the
    up/down-capture against the benchmark (TRAINING2 §7). Deterministic:
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
    bench = benchmark_daily_returns(days)

    def row(day: str | None, i: int | None) -> list[str]:
        cells = [day if day is not None else "TOTAL"]
        if i is None:
            cells += [_cell(bench["window_return_pct"]), _cell(bench["max_drawdown_pct"])]
        else:
            cells += [_cell(bench["daily_return_pct"][i]), "—"]
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
        "| day | bench ret% | bench dd% "
        "| stripped ret% | stripped dd% | stripped trades | stripped deaths "
        "| whole ret% | whole dd% | whole trades | whole deaths |"
    )
    sep = "|---" + "|---:" * 10 + "|"
    bench_name = bench.get("source", "spx")
    bench_desc = (
        "the ^GSPC daily close (`fruitfly.scoreboard.spx_buyhold`)"
        if bench_name == "spx"
        else "the SPY proxy from the fetched 1m cache (the ^GSPC daily "
             "cache does not cover the window)"
    )
    lines = [
        "# Brain shootout — stripped vs whole fly (held-out)",
        "",
        "Held-out comparison of the two trained brains (same training window",
        "2026-03-02..2026-06-30, seed 7) over the most recent trading days of",
        "the extended IEX cache, against",
        f"{bench_desc} over the same days. Generated by",
        "`scripts/eval_brains.py`; identical inputs → byte-identical report.",
        "",
        "## Arms",
        "",
        "| arm | chassis | artifact | md5 | trained window | meta |",
        "|---|---|---|---|---|---|",
    ]
    extras = [receipts[name] for name in sorted(receipts)
              if name not in ("stripped", "whole")]
    for rc in (stripped, whole, *extras):
        if rc is None:
            lines.append(
                f"| whole | whole | {WHOLE_PENDING['artifact']} | — "
                f"| {WHOLE_PENDING['window']} | **PENDING** |"
            )
            continue
        meta_extra = ""
        if "period" in rc:
            meta_extra += (f" period={rc['period']['start']}"
                           f"..{rc['period']['end']}")
        if "basket" in rc:
            meta_extra += f" basket={','.join(rc['basket'])}"
        knobs_rc = rc.get("meta_knobs") or {}
        if knobs_rc:
            meta_extra += " cfg " + " ".join(
                f"{name}={knobs_rc[name]}" for name in _meta_knob_names()
                if name in knobs_rc
            )
        if rc.get("meta_basket"):
            meta_extra += f" trained_basket={','.join(rc['meta_basket'])}"
        meta = rc["artifact_meta"]
        lines.append(
            f"| {rc['arm']} | {rc['chassis']} | `{rc['artifact']}` "
            f"| {rc['artifact_md5'][:12]}… | {meta.get('start', '?')}"
            f"..{meta.get('end', '?')} | seed={meta.get('seed', '?')} "
            f"bars={meta.get('n_bars', '?')} deaths={meta.get('n_deaths', '?')} "
            f"std=({rc['std_beta']},{rc['std_tau_rec_ms']}){meta_extra} |"
        )

    notes = [
        "",
        "## Held-out comparison",
        "",
        "Per-day: final return % and max drawdown % of the day's backtest",
        "(each day a fresh fly at $100,000, artifact weights injected, fixed",
        f"seed {stripped['seed']}); trades = orders, deaths = fly deaths. TOTAL row:",
        "arms aggregate mean daily return %, worst-day drawdown %, summed",
        "trades/deaths; the benchmark shows the compounded window return %",
        "and the drawdown % across the held-out days.",
    ]
    if whole is None:
        notes.append("Missing whole-fly arm = PENDING.")
    lines += [*notes, "", header, sep]
    for i, day in enumerate(days):
        lines.append("| " + " | ".join(row(day, i)) + " |")
    lines.append("| " + " | ".join(row(None, None)) + " |")
    lines.append("")

    # Per-cell statistics (TRAINING2 §7): paired t vs the incumbent arm on
    # shared days (primary portfolio-level statistic) and up/down-capture
    # against the benchmark over the cell's own days.
    inc_rc = receipts.get(incumbent)
    cell_names = [n for n in (incumbent, *sorted(receipts)) if n in receipts]
    seen: set[str] = set()
    cell_names = [n for n in cell_names if not (n in seen or seen.add(n))]
    inc_by_day = (
        {d["day"]: d["final_return_pct"] for d in inc_rc["days"]}
        if inc_rc is not None else {}
    )
    lines += [
        "## Per-cell statistics",
        "",
        "Paired t of daily final-return % against the incumbent arm "
        f"({incumbent}) on shared days; up/down-capture against the "
        f"benchmark ({bench_name}) over the cell's own days.",
        "",
        "| arm | shared days | paired-t | up-capture | down-capture |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in cell_names:
        rc = receipts[name]
        by_day = {d["day"]: d["final_return_pct"] for d in rc["days"]}
        shared = [day for day in inc_by_day if day in by_day]
        t = (
            paired_t([by_day[d] for d in shared], [inc_by_day[d] for d in shared])
            if name != incumbent and inc_rc is not None else None
        )
        ordered = list(by_day)
        cap = capture_ratios(
            [by_day[d] for d in ordered],
            benchmark_daily_returns(ordered)["daily_return_pct"],
        )
        lines.append(
            f"| {name} | {len(shared)} | {_cell(t, TWO.format)} "
            f"| {_cell(cap['up_capture'], TWO.format)} "
            f"| {_cell(cap['down_capture'], TWO.format)} |"
        )
    lines.append("")

    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_iso_day(raw: str | None, flag: str) -> str | None:
    if raw is None:
        return None
    try:
        date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{flag} must be YYYY-MM-DD, got {raw!r}") from exc
    return raw


def _parse_basket(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    basket = [symbol.strip() for symbol in raw.split(",")]
    if len(basket) < 2 or any(not symbol for symbol in basket):
        raise ValueError(
            "--basket needs at least 2 non-empty comma-separated symbols, "
            f"got {raw!r}"
        )
    return basket


@contextmanager
def _basket_override(basket: list[str] | None):
    """Set the module-level ``BASKET`` seams for the duration of the run.

    ``fruitfly.loop`` binds ``BASKET`` at import time (``from fruitfly.data
    import BASKET``), so BOTH module attributes must be patched; deferred
    importers (``fruitfly.data`` consumers like ``cache_trading_days`` and
    ``fruitfly.scoreboard``) read ``fruitfly.data.BASKET`` at call time.
    """
    if basket is None:
        yield
        return
    import fruitfly.data
    import fruitfly.loop

    modules = (fruitfly.data, fruitfly.loop)
    saved = {id(m): m.BASKET for m in modules}
    try:
        for module in modules:
            module.BASKET = list(basket)
        yield
    finally:
        for module in modules:
            module.BASKET = saved[id(module)]


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
                        help="Held-out days (drawn from the RECENT_POOL most "
                             "recent cache days, evenly spaced if fewer; "
                             "default 10 of a 40-day pool).")
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
    parser.add_argument(
        "--start", type=str, default=None, metavar="YYYY-MM-DD",
        help="Bound the held-out day pool to days >= this date (inclusive).")
    parser.add_argument(
        "--end", type=str, default=None, metavar="YYYY-MM-DD",
        help="Bound the held-out day pool to days <= this date (inclusive).")
    parser.add_argument(
        "--basket", type=str, default=None,
        help="Comma-separated ticker basket overriding the trained basket "
             "for the cache day pool and the backtest (>= 2 symbols).")
    parser.add_argument(
        "--arm", type=str, default=None,
        help="Receipt/report arm name (default: the chassis; use a distinct "
             "name per config cell so sibling cells keep their receipts).")
    parser.add_argument(
        "--incumbent", type=str, default="stripped",
        help="Arm the per-cell paired-t statistics are computed against.")
    from fruitfly.train import add_knob_args, knob_kwargs

    add_knob_args(parser)
    args = parser.parse_args(argv)
    _validate_args(args)
    if args.start and args.end and args.start > args.end:
        raise ValueError(f"--start {args.start} must be <= --end {args.end}")
    start = _parse_iso_day(args.start, "--start")
    end = _parse_iso_day(args.end, "--end")
    basket = _parse_basket(args.basket)

    available = cache_trading_days()
    if start is not None or end is not None:
        available = [
            day for day in available
            if (start is None or day >= start) and (end is None or day <= end)
        ]
        if not available:
            raise ValueError(
                f"--start/--end bound [{start or '*'}, {end or '*'}] leaves no "
                f"trading days in the cache")
    period = (
        {"start": start, "end": end}
        if start is not None or end is not None else None
    )

    with _basket_override(basket):
        days = select_days(available, args.days)
        basket_note = f", basket {','.join(basket)}" if basket else ""
        print(f"eval {args.chassis}: {args.artifact} seed {args.seed}"
              f"{basket_note}, "
              f"held-out {days[0]}..{days[-1]} ({len(days)} days)")
        evaluate_arm(args.chassis, args.artifact, days, args.seed,
                     args.std_beta, args.std_tau_rec_ms, period, basket,
                     Path(args.results_dir), arm=args.arm,
                     knobs=knob_kwargs(args))
    receipts = load_receipts(Path(args.results_dir))
    render_report(receipts, Path(args.report), incumbent=args.incumbent)
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
