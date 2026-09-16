"""G-A acceptance-gate calculator (§6 G-A, A8): empirical A-vs-B referee.

Usage:
    python scripts/gates.py --run-dir runs/candidate --reference runs/incumbent \
        [--market-dir data/market] [--horizon-bars 30] [--json out.json]

Gates:
  G-A.1 entry-signal gate: HAC(Bartlett, lag=horizon) Spearman t of
        balance_used vs forward-horizon-bar return of the encountered ticker,
        plus a non-overlapping subsample (sign agreement required).
  G-A.2 behavior gate: executed-trade win/loss ratio > 1.0.
  G-A.3 turnover guard: trades/day and buys/encounter within [0.5x, 2.0x]
        of the reference run. Outside band -> fail regardless of others.
  G-A.4 learning gate: fresh-vs-trained decision-change rate > 0 and
        n_deaths(run) <= n_deaths(reference).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- data loading

def load_events(run_dir: Path) -> list[dict]:
    events = []
    with (run_dir / "events.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _market_index(market_dir: Path) -> dict[str, pd.DataFrame]:
    idx: dict[str, pd.DataFrame] = {}
    for path in sorted(market_dir.glob("*.parquet")):
        ticker = path.stem.split("_")[0]
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"])
        df = df.assign(_ts=ts).drop_duplicates("_ts", keep="last").sort_values("_ts")
        idx[ticker] = df.reset_index(drop=True)
    return idx


def _bar_pos(df: pd.DataFrame, ts) -> int:
    """Index of the last bar whose timestamp is <= ts (the encounter bar)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    pos = int(df["_ts"].searchsorted(ts, side="right") - 1)
    return max(pos, 0)


def forward_return(market: dict[str, pd.DataFrame], ticker: str, ts,
                   horizon: int) -> float | None:
    df = market.get(ticker)
    if df is None:
        return None
    pos = _bar_pos(df, ts)
    if pos + horizon >= len(df):
        return None
    c0 = float(df["close"].iloc[pos])
    c1 = float(df["close"].iloc[pos + horizon])
    if c0 <= 0:
        return None
    return c1 / c0 - 1.0


def bar_index(market: dict[str, pd.DataFrame], ticker: str, ts_ns: int) -> int:
    return _bar_pos(market[ticker], ts_ns)


# ------------------------------------------------------- HAC (Newey-West) OLS

def hac_tstat(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
    """OLS y ~ x; Newey-West (Bartlett kernel) t-stat of the slope."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return 0.0, 0.0
    b = float(xc @ yc) / sxx
    e = yc - b * xc
    # S = sum x_i^2 e_i^2 + 2 * sum_l w_l * sum x_i x_{i-l} e_i e_{i-l}
    xe = xc * e
    s = float(xe @ xe)
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * float(xe[k:] @ xe[:-k])
    var = s / (sxx * sxx)
    if var <= 0 or not math.isfinite(var):
        # Degenerate residuals (near-perfect fit): classical OLS fallback.
        sse = float(e @ e)
        if sse <= 0:
            return b, math.copysign(1e9, b)
        var = sse / (n - 2) / sxx
    return b, b / math.sqrt(var)


# ----------------------------------------------------------------- G-A.1

def collect_samples(events: list[dict], market: dict[str, pd.DataFrame],
                    horizon: int) -> list[dict]:
    """Signal-bearing encounters: balance_used at decision + fwd return."""
    samples = []
    for ev in events:
        if ev.get("type") != "encounter":
            continue
        if ev.get("balance_used") is None:
            continue
        r = forward_return(market, ev["ticker"], ev["ts"], horizon)
        if r is None:
            continue
        samples.append({"ts": ev["ts"], "ticker": ev["ticker"],
                        "balance": float(ev["balance_used"]), "fwd": r})
    return samples


def spearman_gate(samples: list[dict], horizon: int,
                  market: dict[str, pd.DataFrame]) -> dict:
    def stats(sub: list[dict]) -> dict:
        x = np.array([s["balance"] for s in sub], dtype=float)
        y = np.array([s["fwd"] for s in sub], dtype=float)
        if len(sub) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return {"n": len(sub), "rho": 0.0, "t": 0.0}
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rho = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else 0.0
        _, t = hac_tstat(rx, ry, lag=horizon)
        return {"n": len(sub), "rho": round(rho, 6), "t": round(t, 4)}

    full = stats(samples)

    # Non-overlapping subsample: one encounter per ticker per horizon-bar window,
    # deterministic first-by-ts.
    keep: dict[str, dict[int, dict]] = {}
    for s in samples:
        b = bar_index(market, s["ticker"], s["ts"]) // horizon
        per_ticker = keep.setdefault(s["ticker"], {})
        cur = per_ticker.get(b)
        if cur is None or s["ts"] < cur["ts"]:
            per_ticker[b] = s
    nonoverlap = [s for tk in keep.values() for s in tk.values()]
    nonoverlap.sort(key=lambda s: s["ts"])
    sub = stats(nonoverlap)

    sign_ok = (full["t"] > 0) == (sub["t"] > 0)
    passed = abs(full["t"]) > 2.0 and sign_ok
    return {"full": full, "non_overlapping": sub, "sign_agreement": bool(sign_ok),
            "pass": passed}


# ----------------------------------------------------------------- G-A.2

def closed_trades(events: list[dict]) -> list[dict]:
    """Pair orders FIFO by ticker: buy then sell closes a trade."""
    orders: dict[str, list[dict]] = {}
    for ev in events:
        if ev.get("type") == "order":
            orders.setdefault(ev["ticker"], []).append(ev)
    trades = []
    for ticker, obs in orders.items():
        open_lots: list[dict] = []
        for o in sorted(obs, key=lambda o: o["ts"]):
            side, shares, price = o["side"], int(o["shares"]), float(o["price"])
            if side == "buy":
                open_lots.append({"ts": o["ts"], "shares": shares, "price": price})
            elif side == "sell" and open_lots:
                lot = open_lots.pop(0)
                shares_matched = min(shares, lot["shares"])
                trades.append({"ticker": ticker, "entry_ts": lot["ts"], "exit_ts": o["ts"],
                               "shares": shares_matched,
                               "pnl": (price - lot["price"]) * shares_matched})
    return trades


def behavior_gate(trades: list[dict]) -> dict:
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    if len(trades) == 0:
        ratio, passed = 0.0, False
    elif losses == 0:
        ratio, passed = float("inf"), wins > 0
    else:
        ratio, passed = wins / losses, wins / losses > 1.0
    return {"n_trades": len(trades), "wins": wins, "losses": losses,
            "win_loss_ratio": ratio if math.isfinite(ratio) else "inf",
            "pass": passed}


# ----------------------------------------------------------------- G-A.3

def turnover_metrics(events: list[dict], trades: list[dict]) -> dict:
    ts_list = [pd.Timestamp(ev["ts"]) for ev in events if "ts" in ev]
    # count calendar days spanned for a fair trades/day
    days = len({ts.date() for ts in ts_list}) or 1
    n_trades = len(trades)
    n_encounters = sum(1 for ev in events if ev.get("type") == "encounter")
    n_buys = sum(1 for ev in events if ev.get("type") == "order" and ev.get("side") == "buy")
    return {"trades_per_day": round(n_trades / days, 6),
            "buys_per_encounter": round(n_buys / n_encounters, 6) if n_encounters else 0.0}


def turnover_gate(run_m: dict, ref_m: dict, lo: float = 0.5, hi: float = 2.0) -> dict:
    def in_band(a: float, b: float) -> bool:
        if b == 0:
            return a == 0
        return lo <= a / b <= hi

    checks = {k: {"run": run_m[k], "reference": ref_m[k],
                  "ratio": (run_m[k] / ref_m[k]) if ref_m[k] else None,
                  "in_band": in_band(run_m[k], ref_m[k])}
              for k in run_m}
    passed = all(c["in_band"] for c in checks.values())
    return {"metrics": checks, "band": [lo, hi], "pass": passed}


# ----------------------------------------------------------------- G-A.4

def learning_gate(run_events: list[dict], ref_events: list[dict]) -> dict:
    def decisions(events: list[dict]) -> dict[tuple[int, str], str]:
        out = {}
        for ev in events:
            if ev.get("type") == "decision":
                out[(ev["ts"], ev["ticker"])] = ev.get("action")
        return out

    rd, fd = decisions(run_events), decisions(ref_events)
    matched = [(rd[k], fd[k]) for k in rd if k in fd]
    changed = sum(1 for a, b in matched if a != b)
    rate = changed / len(matched) if matched else 0.0
    n_deaths_run = sum(1 for ev in run_events if ev.get("type") == "death")
    n_deaths_ref = sum(1 for ev in ref_events if ev.get("type") == "death")
    return {"n_matched_decisions": len(matched), "n_changed": changed,
            "decision_change_rate": round(rate, 6),
            "n_deaths_run": n_deaths_run, "n_deaths_reference": n_deaths_ref,
            "pass": rate > 0.0 and n_deaths_run <= n_deaths_ref}


# ----------------------------------------------------------------- output

def render_table(result: dict) -> str:
    lines = []

    def row(label: str, value: str, verdict: bool | None) -> None:
        tag = ("PASS" if verdict else "FAIL") if verdict is not None else "  -"
        lines.append(f"  {label:<44} {value:<28} {tag}")

    g1 = result["GA1"]
    f, s = g1["full"], g1["non_overlapping"]
    lines.append(f"G-A.1 entry-signal (Spearman, HAC lag={result['horizon']}):")
    row("full: rho / t / n", f"{f['rho']:+.4f} / {f['t']:+.3f} / n={f['n']}", None)
    row("non-overlapping: rho / t / n",
        f"{s['rho']:+.4f} / {s['t']:+.3f} / n={s['n']}", None)
    row("sign agreement", str(g1["sign_agreement"]), None)
    lines.append(f"  G-A.1 verdict: {'PASS' if g1['pass'] else 'FAIL'}")

    g2 = result["GA2"]
    row("G-A.2 win/loss ratio",
        f"{g2['win_loss_ratio']} (w={g2['wins']} l={g2['losses']} n={g2['n_trades']})", g2["pass"])

    g3 = result["GA3"]
    lines.append(f"G-A.3 turnover (band {g3['band'][0]}x-{g3['band'][1]}x of reference):")
    for k, c in g3["metrics"].items():
        r = "n/a" if c["ratio"] is None else f"{c['ratio']:.3f}x"
        row(f"  {k}", f"run={c['run']} ref={c['reference']} ({r})", c["in_band"])
    lines.append(f"  G-A.3 verdict: {'PASS' if g3['pass'] else 'FAIL'}")

    g4 = result["GA4"]
    row("G-A.4 decision-change rate",
        f"{g4['decision_change_rate']} ({g4['n_changed']}/{g4['n_matched_decisions']})", None)
    row("G-A.4 deaths", f"run={g4['n_deaths_run']} ref={g4['n_deaths_reference']}", None)
    lines.append(f"  G-A.4 verdict: {'PASS' if g4['pass'] else 'FAIL'}")

    lines.append(f"OVERALL: {'PASS' if result['overall'] else 'FAIL'}")
    return "\n".join(lines)


def compute_gates(run_dir: Path, ref_dir: Path, market_dir: Path,
                  horizon: int) -> dict:
    market = _market_index(market_dir)
    run_events = load_events(run_dir)
    ref_events = load_events(ref_dir)

    samples = collect_samples(run_events, market, horizon)
    ga1 = spearman_gate(samples, horizon, market)
    trades = closed_trades(run_events)
    ga2 = behavior_gate(trades)
    ga3 = turnover_gate(turnover_metrics(run_events, trades),
                        turnover_metrics(ref_events, closed_trades(ref_events)))
    ga4 = learning_gate(run_events, ref_events)
    overall = all(g["pass"] for g in (ga1, ga2, ga3, ga4))
    return {"GA1": ga1, "GA2": ga2, "GA3": ga3, "GA4": ga4,
            "horizon": horizon, "overall": overall}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G-A acceptance-gate calculator")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--market-dir", default=None)
    ap.add_argument("--horizon-bars", type=int, default=30)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    market_dir = Path(args.market_dir or __import__("os").environ.get(
        "FRUITFLY_MARKET_DIR", "data/market/history"))
    result = compute_gates(Path(args.run_dir), Path(args.reference),
                           market_dir, args.horizon_bars)

    print(render_table(result))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str))
    return 0 if result["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
