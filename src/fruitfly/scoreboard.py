"""T8: scoreboard — the four DESIGN §11 benchmarks and the run report writer.

Computable benchmarks over the SAME universe and window as a run:

1. :func:`spx_buyhold` — S&P 500 buy-and-hold from ``GSPC_1d.parquet``,
   interpolated onto the comparison grid.
2. :data:`SPIVA_ROW` — static cited reference row (percent of active
   large-cap funds that underperformed the S&P 500 over 1y/3y/5y/15y).
3. :func:`monkey_darts` — seeded random portfolio, same universe, same
   rebalance cadence as the fly (one rebalance per daily session).
4. :func:`logistic_control` — logistic regression on the identical state
   features (returns, rsi, volatility, volume_delta) built by
   :func:`fruitfly.senses.smell.build_features`, the shared builder the
   smell channel uses; trained on the first 70% of the window, traded on
   the rest.

:func:`compare_run` renders ``reports/t8-scoreboard.md`` comparing a run
directory (T7 run-format contract) against all four.

Determinism: :func:`monkey_darts` and :func:`logistic_control` are pure
functions of their inputs (seeded numpy Generator; deterministic sklearn
lbfgs) — fixed seed reproduces byte-identical equity curves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fruitfly.senses.smell import FEATURES, VOL_WINDOW, build_features

#: Parquet cache root (T6 contract).
CACHE_DIR = Path("data/market")

#: Default report output path (reused by T10's post-mortem).
REPORT_PATH = Path("reports/t8-scoreboard.md")

#: Static SPIVA reference row: percent of active U.S. large-cap funds that
#: underperformed the S&P 500, from the SPIVA U.S. Scorecard Year-End 2025
#: (data as of Dec. 31, 2025). "All Large-Cap" category vs S&P 500. The
#: year-end 2025 scorecard IS the most recent verifiable one as of
#: 2026-09-15 (published March 2026); figures cross-checked against
#: secondary coverage (79% headline matches the widely reported 1-year
#: figure). SPIVA includes merged/liquidated funds (no survivorship bias).
SPIVA_ROW = {
    "name": "SPIVA: active large-cap funds underperforming S&P 500",
    "category": "All Large-Cap vs S&P 500",
    "underperforming_pct": {"1y": 78.78, "3y": 66.84, "5y": 88.96, "15y": 89.93},
    "source": "S&P Dow Jones Indices, SPIVA U.S. Scorecard Year-End 2025 (data as of Dec. 31, 2025)",
    "url": "https://www.spglobal.com/spdji/en/research-insights/spiva/",
    "accessed": "2026-09-15",
    "note": (
        "Static reference row, not computed from market data. Percentages are "
        "net-of-fees returns of ALL large-cap active funds (incl. merged/liquidated) "
        "vs the S&P 500."
    ),
}

#: Rebalance cadence for :func:`monkey_darts`: one rebalance at the first bar
#: of each daily session — the fly's wake/sleep cycle (DESIGN D7/D16), applied
#: to the same 1-minute bar universe.
MONKEY_CADENCE = "daily (first bar of each session)"


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _load_spx_daily() -> pd.Series:
    """S&P 500 daily closes (^GSPC proxy) as a UTC-datetime-indexed Series."""
    df = pd.read_parquet(CACHE_DIR / "GSPC_1d.parquet")
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).normalize()
    close = pd.Series(df["close"].to_numpy(dtype=float), index=idx)
    return close[~close.index.duplicated(keep="first")].sort_index()


def spx_buyhold(start: str, end: str, capital: float = 100_000.0) -> pd.Series:
    """S&P 500 buy-and-hold equity curve on a daily business-day grid.

    Reads ``GSPC_1d.parquet`` (2y of daily ^GSPC), filters to
    ``[start, end]``, interpolates the daily close onto the business-day
    comparison grid (holidays get linearly interpolated values), and scales
    to ``capital`` at the first grid day. Pure and deterministic.
    """
    daily = _load_spx_daily()
    grid = pd.bdate_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
    close = daily.reindex(grid).interpolate(method="time", limit_direction="both")
    equity = capital * close / close.iloc[0]
    equity.name = "spx_buyhold"
    return equity


def _union_grid(bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Sorted union of all symbols' bar timestamps."""
    grids = [df.index for df in bars_by_symbol.values() if len(df)]
    if not grids:
        raise ValueError("bars_by_symbol contains no bars")
    grid = grids[0]
    for g in grids[1:]:
        grid = grid.union(g)
    return grid


def monkey_darts(
    bars_by_symbol: dict[str, pd.DataFrame],
    seed: int,
    capital: float = 100_000.0,
    n_positions: int = 10,
) -> pd.Series:
    """Monkey-with-darts benchmark: seeded random holdings, fly cadence.

    Same universe (``bars_by_symbol`` keys) and same 1-minute bar grid as
    the fly. Rebalance cadence: the FIRST BAR OF EACH DAILY SESSION
    (matching the fly's wake/sleep cycle, D7) — each session the monkey
    throws darts: ``n_positions`` symbols drawn uniformly without
    replacement from the universe via ``numpy.random.default_rng(seed)``,
    held equal-weight (bought at the session's first bar close) until the
    next session boundary. Deterministic: fixed seed reproduces the curve
    byte-identically.
    """
    symbols = sorted(s for s, df in bars_by_symbol.items() if len(df))
    if not symbols:
        raise ValueError("monkey_darts needs at least one symbol with bars")
    grid = _union_grid(bars_by_symbol)
    rng = np.random.default_rng(seed)

    days = grid.normalize().unique()
    values = np.empty(len(grid), dtype=float)
    pos = 0
    level = float(capital)
    for day in days:
        mask = grid.normalize() == day
        sub = grid[mask]
        k = min(n_positions, len(symbols))
        picks = rng.choice(len(symbols), size=k, replace=False)
        held = [symbols[i] for i in picks]
        norms = []
        for sym in held:
            close = bars_by_symbol[sym]["close"].reindex(sub)
            if close.notna().sum() == 0:
                norms.append(pd.Series(1.0, index=sub))  # no bars today: slot in cash
                continue
            base = close.dropna().iloc[0]
            norms.append((close / base).ffill().bfill())
        levels = level * pd.concat(norms, axis=1).mean(axis=1).to_numpy()
        values[pos : pos + len(sub)] = levels
        pos += len(sub)
        level = levels[-1]
    return pd.Series(values, index=grid, name="monkey_darts")


def feature_matrix(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar rows of the shared smell feature builder.

    Row ``t`` is exactly ``build_features(bars.iloc[: t + 1])`` (a trailing
    window of ``VOL_WINDOW + 1`` rows suffices for every feature), with
    columns in ``FEATURES`` order. Used by :func:`logistic_control`;
    identity with :func:`build_features` is asserted in the tests.
    """
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    rows = []
    for t in range(len(bars)):
        lo = max(0, t - VOL_WINDOW)
        window = pd.DataFrame(
            {"close": close[lo : t + 1], "volume": volume[lo : t + 1]}
        )
        rows.append(build_features(window))
    return pd.DataFrame(rows, columns=list(FEATURES), index=bars.index)


def logistic_control(
    bars_by_symbol: dict[str, pd.DataFrame],
    seed: int,
    capital: float = 100_000.0,
    train_fraction: float = 0.7,
    threshold: float = 0.5,
) -> tuple[pd.Series, dict]:
    """Logistic-regression control on the identical smell state features.

    The scientific control: features are EXACTLY ``(returns, rsi,
    volatility, volume_delta)`` from the shared builder
    :func:`fruitfly.senses.smell.build_features` — the same values the
    smell channel mixes into its modulator. Pooled across all symbols:

    - Label: 1 when the symbol's NEXT bar return is positive.
    - Train on the first ``train_fraction`` (70%) of the window's bars;
      trade the rest.
    - Model: sklearn ``LogisticRegression(solver="lbfgs", C=1.0,
      max_iter=1000)`` on standardized features (train-set mean/std).
      lbfgs is fully deterministic — ``seed`` is recorded but unused by
      the solver.
    - Trade rule: at each bar, long (equal split of full capital across
      signals) every symbol with P(next-bar up) > ``threshold`` (0.5);
      otherwise that slot stays in cash. Position returns are realized on
      the following bar's close.

    Returns ``(equity_curve, summary)``. Deterministic: fixed inputs give
    byte-identical outputs.
    """
    symbols = sorted(s for s, df in bars_by_symbol.items() if len(df))
    if not symbols:
        raise ValueError("logistic_control needs at least one symbol with bars")
    grid = _union_grid(bars_by_symbol)
    split = int(len(grid) * train_fraction)
    train_ts = grid[:split]
    test_ts = grid[split:]
    if len(test_ts) < 2:
        raise ValueError("window too short to trade after the 70% train split")

    # Per-symbol per-bar feature rows and next-bar returns, aligned to the
    # union grid (NaN where the symbol has no bar).
    X_parts: dict[str, pd.DataFrame] = {}
    R_parts: dict[str, pd.Series] = {}
    for sym in symbols:
        df = bars_by_symbol[sym]
        feats = feature_matrix(df)
        nxt = df["close"].shift(-1) / df["close"] - 1.0
        X_parts[sym] = feats
        R_parts[sym] = nxt

    # Training matrix: pooled rows in the train window with a valid label
    # and enough history (skip each symbol's first VOL_WINDOW bars).
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    train_end = train_ts[-1]
    for sym in symbols:
        feats = X_parts[sym]
        nxt = R_parts[sym]
        hist_ok = np.arange(len(feats)) >= VOL_WINDOW
        label_ok = nxt.notna().to_numpy()
        in_train = feats.index <= train_end
        keep = hist_ok & label_ok & in_train
        if keep.any():
            Xs.append(feats.to_numpy(dtype=float)[keep])
            ys.append((nxt.to_numpy()[keep] > 0.0).astype(int))
    if not Xs:
        raise ValueError("no training rows with sufficient history")
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0  # constant feature: leave as-is after centering
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=0)
    model.fit((X_train - mean) / std, y_train)

    # Probability of next-bar-up per symbol per test bar.
    P = {}
    R = {}
    for sym in symbols:
        feats = X_parts[sym]
        nxt = R_parts[sym]
        test_rows = feats.loc[feats.index.isin(test_ts)]
        if len(test_rows):
            scaled = (test_rows.to_numpy(dtype=float) - mean) / std
            P[sym] = pd.Series(
                model.predict_proba(scaled)[:, 1], index=test_rows.index
            ).reindex(test_ts)
        else:
            P[sym] = pd.Series(np.nan, index=test_ts)
        R[sym] = nxt.reindex(test_ts)
    P = pd.DataFrame(P)
    R = pd.DataFrame(R)

    # Trade: at bar t, hold symbols with P > threshold and a known next-bar
    # return; equal split, full investment; realized on the next bar.
    held = (P > threshold) & R.notna()
    port_ret = R.where(held).mean(axis=1).fillna(0.0)
    step = port_ret.shift(1)  # decision made at the previous bar
    step.iloc[0] = 0.0
    equity = pd.Series(
        capital * np.cumprod(1.0 + step.to_numpy()), index=test_ts, name="logistic_control"
    )

    summary = {
        "model": "sklearn LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000), "
        "features standardized with train-set mean/std",
        "feature_builder": "fruitfly.senses.smell.build_features",
        "features": list(FEATURES),
        "coefficients": {f: float(c) for f, c in zip(FEATURES, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "threshold": float(threshold),
        "train_fraction": float(train_fraction),
        "train_rows": int(len(y_train)),
        "train_up_rate": float(y_train.mean()),
        "test_bars": int(len(test_ts)),
        "seed": int(seed),
        "seed_note": "lbfgs is deterministic; seed recorded for run labeling only",
    }
    if len(test_ts) > 2:
        # Accuracy on the pooled test rows (same valid-label/history filter).
        Xt: list[np.ndarray] = []
        yt: list[np.ndarray] = []
        for sym in symbols:
            feats = X_parts[sym]
            nxt = R_parts[sym]
            hist_ok = np.arange(len(feats)) >= VOL_WINDOW
            label_ok = nxt.notna().to_numpy()
            in_test = feats.index.isin(test_ts)
            keep = hist_ok & label_ok & in_test
            if keep.any():
                Xt.append(feats.to_numpy(dtype=float)[keep])
                yt.append((nxt.to_numpy()[keep] > 0.0).astype(int))
        if Xt:
            X_test = np.concatenate(Xt)
            y_test = np.concatenate(yt)
            pred = model.predict((X_test - mean) / std)
            summary["test_rows"] = int(len(y_test))
            summary["test_accuracy"] = float((pred == y_test).mean())
    return equity, summary


# ---------------------------------------------------------------------------
# Run comparison report
# ---------------------------------------------------------------------------


def _metrics(series: pd.Series) -> dict[str, float]:
    """Final return % and max drawdown % of an equity curve."""
    vals = series.to_numpy(dtype=float)
    final_return = vals[-1] / vals[0] - 1.0
    drawdown = (vals / np.maximum.accumulate(vals) - 1.0).min()
    return {"final_return_pct": 100.0 * final_return, "max_drawdown_pct": 100.0 * drawdown}


def _align(series: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Reindex an equity curve onto the comparison grid (as-of semantics).

    Works across grids of different granularity (e.g. daily SPX closes
    onto an intraday bar grid): values carry forward from the most recent
    known stamp, and the head is back-filled so the aligned curve starts
    at its first known value.
    """
    combined = series.reindex(series.index.union(grid)).ffill().bfill()
    return combined.reindex(grid)


def _load_events(run_dir: Path) -> dict:
    """Parse events.jsonl into per-type counts plus death/hatch tallies."""
    path = run_dir / "events.jsonl"
    counts: dict[str, int] = {}
    deaths = 0
    hatches = 0
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                counts[event.get("type", "?")] = counts.get(event.get("type", "?"), 0) + 1
                if event.get("type") == "death":
                    deaths += 1
                elif event.get("type") == "hatch":
                    hatches += 1
    return {"counts": counts, "deaths": deaths, "hatches": hatches}


def compare_run(
    run_dir: str | Path,
    start: str,
    end: str,
    capital: float = 100_000.0,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    seed: int | None = None,
    out_path: str | Path = REPORT_PATH,
) -> str:
    """Render the T8 scoreboard report comparing a run against all benchmarks.

    Reads the T7 run-format contract artifacts (``equity.csv`` with header
    ``timestamp,equity,cash,n_positions``; optional ``events.jsonl``) from
    ``run_dir``, evaluates all benchmarks over the SAME window on the fly's
    bar grid, and renders ``out_path`` (default
    ``reports/t8-scoreboard.md``). ``bars_by_symbol`` defaults to the cached
    ``BASKET`` 1-minute bars via :func:`fruitfly.data.load_bars`. ``seed``
    defaults to the one parsed from the ``backtest_{seed}_{start}_{end}``
    run-dir name. Returns the rendered markdown text.
    """
    from fruitfly.data import BASKET, load_bars  # deferred: keeps import light

    run_dir = Path(run_dir)
    equity_csv = pd.read_csv(run_dir / "equity.csv")
    fly = pd.Series(
        equity_csv["equity"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(equity_csv["timestamp"], utc=True)),
        name="fly",
    )
    grid = fly.index
    if seed is None:
        match = re.search(r"backtest_(\d+)_", run_dir.name)
        seed = int(match.group(1)) if match else 0

    if bars_by_symbol is None:
        bars_by_symbol = load_bars(BASKET, start, end)

    spx = _align(spx_buyhold(start, end, capital), grid)
    monkey = _align(monkey_darts(bars_by_symbol, seed=seed, capital=capital), grid)
    control, control_summary = logistic_control(bars_by_symbol, seed=seed, capital=capital)
    rows = [
        ("Fly (this run)", _metrics(fly)),
        ("S&P 500 buy-and-hold", _metrics(spx)),
        (f"Monkey-with-darts (seed {seed}, rebalance: {MONKEY_CADENCE})", _metrics(monkey)),
        (
            f"Logistic control (smell features, train {control_summary['train_fraction']:.0%} "
            f"/ trade {1 - control_summary['train_fraction']:.0%}, threshold "
            f"{control_summary['threshold']})",
            _metrics(control),
        ),
    ]

    lines = [
        f"# T8 Scoreboard — {run_dir.name}",
        "",
        f"- Window: {start} .. {end} (inclusive), capital {capital:,.0f}",
        f"- Grid: fly equity curve, {len(grid)} bars "
        f"({grid[0]} .. {grid[-1]})",
        f"- Benchmarks computed on the same universe and bar grid; S&P 500 "
        f"interpolated from daily ^GSPC closes.",
        "",
        "## Scoreboard",
        "",
        "| Benchmark | Final return % | Max drawdown % |",
        "|---|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['final_return_pct']:+.2f}% | {metrics['max_drawdown_pct']:.2f}% |"
        )
    hz = SPIVA_ROW["underperforming_pct"]
    lines.append(
        f"| {SPIVA_ROW['name']} (static reference) | 1y {hz['1y']:.2f}% / 3y {hz['3y']:.2f}% "
        f"/ 5y {hz['5y']:.2f}% / 15y {hz['15y']:.2f}% underperforming | — |"
    )

    events = _load_events(run_dir)
    lines += [
        "",
        "## Run events",
        "",
    ]
    if events["counts"]:
        lines.append(", ".join(f"{k}: {v}" for k, v in sorted(events["counts"].items())))
        lines.append(f"- deaths: {events['deaths']}, hatches: {events['hatches']}")
    else:
        lines.append("No events.jsonl found in run directory.")

    lines += [
        "",
        "## Logistic control model",
        "",
        f"- Features: {', '.join(control_summary['features'])} — built by "
        f"`{control_summary['feature_builder']}` (the smell channel's shared builder).",
        f"- {control_summary['model']}",
        f"- Coefficients: "
        + ", ".join(f"{f}={c:+.4f}" for f, c in control_summary["coefficients"].items())
        + f", intercept={control_summary['intercept']:+.4f}",
        f"- Train rows: {control_summary['train_rows']} (up-rate "
        f"{control_summary['train_up_rate']:.3f}); test bars: {control_summary['test_bars']}",
    ]
    if "test_accuracy" in control_summary:
        lines.append(
            f"- Test accuracy (direction): {control_summary['test_accuracy']:.3f} "
            f"over {control_summary['test_rows']} rows."
        )
    lines += [
        "",
        "## SPIVA reference",
        "",
        f"- {SPIVA_ROW['source']}",
        f"- Source URL: {SPIVA_ROW['url']} (accessed {SPIVA_ROW['accessed']})",
        f"- {SPIVA_ROW['note']}",
        "",
    ]
    report = "\n".join(lines)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report