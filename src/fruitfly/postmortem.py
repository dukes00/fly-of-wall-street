"""T10: post-mortem generator — the fly's obituary (DESIGN §12).

Reads a T7 run's receipts (``events.jsonl`` + ``equity.csv``, the contract
written by :func:`fruitfly.loop.run_backtest`) and renders a markdown
post-mortem:

- **Lifespan**: hatch -> death life segments (or "alive at end of log").
  ``hatch`` events carry no timestamp, so a life's birth is the first
  timestamped event (in practice the session ``wake``) after its hatch.
- **Cause of death**: the death event's recorded equity vs hatch equity plus
  the equity trajectory into the death print (intraday peak -> death print
  fall, realized losses, death liquidations). Every factual claim carries an
  events.jsonl LINE-RANGE citation of the form ``events.jsonl:A-B``
  (1-based, inclusive) computed from the log itself, so a cited range always
  contains the event it cites. A fly that never dies gets a "None — alive"
  verdict with the same citation discipline.
- **P&L vs scoreboard**: :func:`fruitfly.scoreboard.compare_run` is called on
  the same run over the same window and its report body (the four-benchmark
  table: fly vs S&P 500 buy-and-hold vs monkey-with-darts vs logistic
  control, plus the SPIVA reference) is embedded verbatim.
- **One-trial grudges / favorites**: a fully documented, deterministic
  heuristic over encounter+decision pairs — see :func:`_behavior_autopsy`.
- **Event tallies**: per-type counts, deaths, hatches.

The report is a pure function of the run directory — no wall-clock, no RNG —
so rendering the same run twice is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from fruitfly.__main__ import register_command
from fruitfly.loop import APPROACH_THR, DEATH_THRESHOLD
from fruitfly.scoreboard import compare_run

__all__ = ["generate_postmortem", "REPORT_PATH"]

#: Default output path for the post-mortem report.
REPORT_PATH = Path("reports/t10-postmortem.md")

#: Innate-readout threshold for the grudge probe: the loop's own approach
#: threshold. A re-encounter with raw_balance at or above this (innate
#: reflex says approach) but balance at or below its negation (learned
#: readout says avoid) after a pending shock is a one-trial grudge.
GRUDGE_APPROACH_THR = APPROACH_THR

#: Approach trials (positive-balance encounter + ``buy`` decision) a ticker
#: needs before it is reported as a favorite.
FAVORITE_MIN_BUYS = 3

#: Citation format: ``events.jsonl:A`` or ``events.jsonl:A-B`` (1-based,
#: inclusive). Parsed by tests to verify cited ranges contain their events.
_CITE = "events.jsonl:{}"

_RUN_NAME_RE = re.compile(
    r"backtest_\d+_(\d{4}-\d{2}-\d{2}(?:T[\d:]+)?)_(\d{4}-\d{2}-\d{2}(?:T[\d:]+)?).*"
)


# ---------------------------------------------------------------------------
# Parsed run state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """An encounter event immediately followed by its decision event."""

    enc_line: int
    dec_line: int
    ts: str
    ticker: str
    balance: float
    raw_balance: float
    action: str
    reason: str


@dataclass(frozen=True)
class Grudge:
    """One one-trial post-shock avoidance trial."""

    ticker: str
    shock_line: int
    enc_line: int
    dec_line: int
    balance: float
    raw_balance: float
    action: str




@dataclass(frozen=True)
class Life:
    """One hatch -> (death | end-of-log) segment of the run."""

    n: int
    hatch_line: int
    hatch_equity: float | None
    born_ts: str | None
    death: dict | None = None
    death_line: int | None = None


def _read_events(run_dir: Path) -> list[tuple[int, dict]]:
    """Parse events.jsonl into (1-based line number, event) pairs."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no events.jsonl in run dir {run_dir}")
    out: list[tuple[int, dict]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                out.append((i, json.loads(line)))
    return out


def _read_equity(run_dir: Path) -> pd.DataFrame:
    """Load equity.csv with a UTC DatetimeIndex column."""
    path = run_dir / "equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"no equity.csv in run dir {run_dir}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _pairs(events: list[tuple[int, dict]]) -> list[Pair]:
    """Pair each encounter with the decision event immediately after it."""
    pairs: list[Pair] = []
    for idx, (line, ev) in enumerate(events):
        if ev.get("type") != "encounter":
            continue
        nxt = events[idx + 1][1] if idx + 1 < len(events) else None
        if (
            nxt is None
            or nxt.get("type") != "decision"
            or nxt.get("ticker") != ev.get("ticker")
        ):
            continue
        pairs.append(
            Pair(
                enc_line=line,
                dec_line=events[idx + 1][0],
                ts=str(ev.get("ts", "")),
                ticker=str(ev.get("ticker", "?")),
                balance=float(ev.get("balance", 0.0)),
                raw_balance=float(ev.get("raw_balance", 0.0)),
                action=str(nxt.get("action", "?")),
                reason=str(nxt.get("reason", "?")),
            )
        )
    return pairs


def _lives(events: list[tuple[int, dict]]) -> list[Life]:
    """Split the log into hatch -> death (or end-of-log) life segments.

    The loop emits a duplicate ``hatch`` after a death (brain reset + innate
    re-anchor); consecutive hatch events are merged into one life birth.
    """
    lives: list[Life] = []
    cur: Life | None = None
    for line, ev in events:
        etype = ev.get("type")
        if etype == "hatch":
            if cur is None:
                cur = Life(len(lives) + 1, line, ev.get("hatch_equity"), None)
            elif cur.death is None:
                # Duplicate hatch (e.g. the post-death reset pair): keep the
                # first line, adopt the latest recorded hatch equity.
                cur = replace(cur, hatch_equity=ev.get("hatch_equity"))
            else:
                lives.append(cur)
                cur = Life(len(lives) + 1, line, ev.get("hatch_equity"), None)
        elif etype == "death":
            if cur is None:
                cur = Life(len(lives) + 1, line, ev.get("hatch_equity"), None)
            lives.append(replace(cur, death=ev, death_line=line))
            cur = None
        elif cur is not None and cur.born_ts is None and ev.get("ts"):
            cur = replace(cur, born_ts=str(ev["ts"]))
    if cur is not None:
        lives.append(cur)
    return lives


# ---------------------------------------------------------------------------
# Behavioral autopsy: one-trial grudges and favorites
# ---------------------------------------------------------------------------


def _behavior_autopsy(
    pairs: list[Pair],
) -> tuple[list[Grudge], dict[str, int], dict[str, int]]:
    """Score one-trial grudges and repeated-approach favorites.

    Heuristic (deterministic, documented verbatim in the report). The
    loop's decision is a deterministic function of the post-plasticity
    balance, so "avoid despite positive valence" can only be observed by
    comparing the INNATE readout (``raw_balance``, the hatch-anchored
    approach/avoid balance before plasticity) with the LEARNED one
    (``balance``, after KC->MBON plasticity):

    - An *encounter shock* for ticker X is a pair whose learned balance is
      negative (``balance < 0`` — the avoid side; the paired decision is
      typically ``avoid``).
      - **one-trial grudge** — the probe pair is itself an avoid trial
        (``balance < 0``, decision ``avoid``) but its ``raw_balance >=
        GRUDGE_APPROACH_THR`` (the loop's +0.005 approach threshold): the
        innate reflex says approach while the learned readout says avoid.
        Plasticity flipped an innately positive odor into avoidance after
        a single aversive trial. Scored once per pending shock; the
        grudge trial itself then becomes the new pending shock.
      - Any other outcome (renewed avoid trial with innately negative
        raw_balance, a buy, a cap-blocked pass, a neutral-zone readout)
        keeps or resets the pending shock without a grudge.
    - An *approach trial* is a pair with balance >= ``GRUDGE_APPROACH_THR``
      and a ``buy`` decision. A ticker with >= ``FAVORITE_MIN_BUYS``
      approach trials is a **favorite** (repeated strong approach).

    Returns ``(grudges, approach_trials, total_buys)``.
    """
    pending: dict[str, Pair] = {}
    grudges: list[Grudge] = []
    approaches: Counter[str] = Counter()
    buys: Counter[str] = Counter()
    for p in pairs:
        buys[p.ticker] += p.action == "buy"
        shock = pending.get(p.ticker)
        if p.balance < 0.0:
            if (
                shock is not None
                and p.raw_balance >= GRUDGE_APPROACH_THR
                and p.action == "avoid"
            ):
                # learned avoid despite an innately positive readout,
                # immediately after the pending shock: one-trial grudge
                grudges.append(
                    Grudge(
                        p.ticker,
                        shock.enc_line,
                        p.enc_line,
                        p.dec_line,
                        p.balance,
                        p.raw_balance,
                        p.action,
                    )
                )
                pending.pop(p.ticker, None)
            else:
                pending[p.ticker] = p
            continue
        pending.pop(p.ticker, None)
        if p.balance >= GRUDGE_APPROACH_THR and p.action == "buy":
            approaches[p.ticker] += 1
    return grudges, dict(approaches), dict(buys)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _cite(a: int, b: int | None = None) -> str:
    return _CITE.format(a if b is None or b == a else f"{a}-{b}")


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def _day_cite_range(events: list[tuple[int, dict]], death_idx: int) -> tuple[int, int]:
    """Log range for a death: from the last session/life boundary before it
    (wake/sleep/hatch/death) through the death line. Always contains the
    death event."""
    end = events[death_idx][0]
    start = 1
    for i in range(death_idx - 1, -1, -1):
        if events[i][1].get("type") in {"wake", "sleep", "hatch", "death"}:
            start = events[i][0]
            break
    return start, end


def _window(run_dir: Path, equity: pd.DataFrame) -> tuple[str, str]:
    """Backtest window as (start, end) strings for scoreboard.compare_run:
    parsed from the ``backtest_{seed}_{start}_{end}`` dir name when possible,
    else the equity grid's first/last dates."""
    m = _RUN_NAME_RE.match(run_dir.name)
    if m:
        start, end = m.group(1), m.group(2)
    else:
        ts = equity["timestamp"]
        start = ts.iloc[0].strftime("%Y-%m-%d")
        end = ts.iloc[-1].strftime("%Y-%m-%d")
    if len(end) <= 10:  # date-only end: the whole session of that day,
        # matching loop._window semantics (load_bars' end bound is inclusive)
        end = f"{end}T23:59:59.999999"
    return start, end


def _lifespan_section(
    lives: list[Life], equity: pd.DataFrame, n_events: int
) -> list[str]:
    ts = equity["timestamp"]
    rows: list[str] = []
    for life in lives:
        born = life.born_ts if life.born_ts is not None else "(no timestamped event)"
        if life.death is not None:
            died = str(life.death.get("ts", "?"))
            death_at = _cite(life.death_line) if life.death_line else "?"
        else:
            died = "alive at end of log"
            death_at = "—"
        lo = pd.Timestamp(born) if life.born_ts else ts.iloc[0]
        hi = pd.Timestamp(died) if life.death is not None else ts.iloc[-1]
        bars = int(((ts >= lo) & (ts <= hi)).sum())
        hatch_eq = (
            _fmt(float(life.hatch_equity)) if life.hatch_equity is not None else "?"
        )
        rows.append(
            f"| {life.n} | {_cite(life.hatch_line)} | {hatch_eq} | {born} "
            f"| {died} | {death_at} | {bars} |"
        )
    return [
        "## Lifespan",
        "",
        "One row per hatch -> death (or end-of-log) segment. ``hatch`` events "
        "carry no timestamp, so a life's birth is its first timestamped event "
        "(the session ``wake``). *Bars alive* counts equity.csv rows in the "
        "segment.",
        "",
        "| Life | Hatched (log) | Hatch equity | Born (first ts) | Died "
        "| Death (log) | Bars alive |",
        "|---|---|---:|---|---|---|---:|",
        *rows,
        "",
        f"Equity grid: {len(ts)} bars, {ts.iloc[0]} .. {ts.iloc[-1]}; "
        f"event log: {n_events} lines.",
        "",
    ]


def _cause_section(
    events: list[tuple[int, dict]], lives: list[Life], equity: pd.DataFrame
) -> list[str]:
    deaths = [life for life in lives if life.death is not None]
    lines = ["## Cause of death", ""]
    ts = equity["timestamp"]

    if not deaths:
        hatch_eq = float(lives[0].hatch_equity) if lives else float("nan")
        final = float(equity["equity"].iloc[-1])
        mn = float(equity["equity"].min())
        lines += [
            "**None — the fly was alive at the end of the log.**",
            "",
            f"- 0 death events; final equity {_fmt(final)} at {ts.iloc[-1]} "
            f"({100.0 * (final / hatch_eq - 1.0):+.2f}% vs hatch equity "
            f"{_fmt(hatch_eq)}).",
            f"- Minimum equity over the run: {_fmt(mn)} "
            f"({100.0 * (1.0 - mn / hatch_eq):.2f}% below hatch) — versus the "
            f"default D14 death rule (equity <= hatch equity x "
            f"{1.0 + DEATH_THRESHOLD:.2f}).",
        ]
        # Cite the final session: last wake through the end of the log.
        last_wake = 1
        for line, ev in events:
            if ev.get("type") == "wake":
                last_wake = line
        lines.append(
            f"- The final session closed with the usual sugar_shock and sleep "
            f"at {events[-1][1].get('ts', '?')} ({_cite(last_wake, events[-1][0])})."
        )
        lines.append("")
        return lines

    for life in deaths:
        ev = life.death or {}
        assert life.death_line is not None
        death_idx = next(
            i for i, (_, e) in enumerate(events) if e is ev
        )
        start, end = _day_cite_range(events, death_idx)
        death_eq = float(ev.get("equity", float("nan")))
        hatch_eq = float(ev.get("hatch_equity", float("nan")))
        death_ts = pd.Timestamp(str(ev.get("ts", ts.iloc[0])))
        lo = (
            pd.Timestamp(life.born_ts)
            if life.born_ts
            else death_ts.normalize()
        )
        window = equity[(ts >= lo) & (ts <= death_ts)]
        lines.append(
            f"Death {life.n} at {ev.get('ts', '?')} "
            f"({_cite(start, end)}):"
        )
        lines.append("")
        lines.append(
            f"- Recorded payload: {json.dumps(ev, sort_keys=True)} — equity "
            f"{_fmt(death_eq)} vs hatch equity {_fmt(hatch_eq)}, i.e. "
            f"{100.0 * (1.0 - death_eq / hatch_eq):.2f}% below hatch when the "
            f"death rule (D14) fired."
        )
        if len(window):
            up_to_death = window[window["timestamp"] <= death_ts]
            peak_i = up_to_death["equity"].idxmax()
            peak = float(up_to_death.loc[peak_i, "equity"])
            peak_ts = up_to_death.loc[peak_i, "timestamp"]
            lines.append(
                f"- Equity trajectory into the death print: peak {_fmt(peak)} "
                f"at {peak_ts}, then {100.0 * (1.0 - death_eq / peak):.2f}% "
                f"down to the death equity (equity.csv, {lo.date()} "
                f".. {death_ts.date()})."
            )
        day_orders = [
            (line, e)
            for line, e in events
            if e.get("type") == "order"
            and str(e.get("ts", "")).startswith(death_ts.strftime("%Y-%m-%d"))
        ]
        reasons = Counter(str(e.get("reason", "?")) for _, e in day_orders)
        if reasons:
            lines.append(
                "- Order reasons on the death day: "
                + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
                + f" ({_cite(start, end)})."
            )
        losses = sorted(
            (
                (float(e["realized_pnl"]), line, str(e.get("ticker", "?")))
                for line, e in day_orders
                if e.get("realized_pnl") is not None
                and float(e["realized_pnl"]) < 0.0
                and str(e.get("ts", "")) <= str(ev.get("ts", ""))
            ),
        )
        if losses:
            top = ", ".join(
                f"{tkr} {_fmt(pnl)} ({_cite(line)})" for pnl, line, tkr in losses[:3]
            )
            lines.append(
                f"- Largest realized losses up to the death print: {top}."
            )
        liquidations = [
            line
            for line, e in events
            if e.get("type") == "order"
            and e.get("reason") == "death_liquidation"
            and str(e.get("ts", "")) == str(ev.get("ts", ""))
        ]
        if liquidations:
            lines.append(
                f"- Death liquidations: {len(liquidations)} position(s) closed "
                f'with reason "death_liquidation" '
                f"({_cite(liquidations[0], liquidations[-1])})."
            )
        else:
            lines.append("- No open positions were liquidated at death.")
        lines.append("")

    if lives and lives[-1].death is None:
        tail = [
            str(e.get("ts", ""))
            for _, e in events[-2:]
            if e.get("type") in {"sugar_shock", "sleep"}
        ]
        lines.append(
            f"After the last death the fly re-hatched "
            f"({_cite(lives[-1].hatch_line)}) and was alive at the end of the "
            f"log (final events: {', '.join(t for t in tail if t)}; "
            f"{_cite(events[-2][0], events[-1][0])})."
        )
        lines.append("")
    return lines


def _scoreboard_section(run_dir: Path, bars_by_symbol: dict | None) -> str:
    """Render the T8 four-benchmark comparison for this run and return its
    body (the report's own H1 is dropped; the section heading is ours)."""
    try:
        equity = _read_equity(run_dir)
        start, end = _window(run_dir, equity)
        with tempfile.TemporaryDirectory() as td:
            body = compare_run(
                run_dir,
                start,
                end,
                bars_by_symbol=bars_by_symbol,
                out_path=Path(td) / "scoreboard.md",
            )
    except Exception as exc:  # missing market cache, malformed receipts, ...
        return (
            f"The four-benchmark scoreboard could not be rendered "
            f"(`fruitfly.scoreboard.compare_run` failed: "
            f"{type(exc).__name__}: {exc})."
        )
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _behavior_section(
    grudges: list[Grudge], approaches: dict[str, int], buys: dict[str, int]
) -> list[str]:
    thr = f"{GRUDGE_APPROACH_THR:g}"
    lines = [
        "## One-trial grudges and favorites",
        "",
        "Heuristic (deterministic; exact definitions):",
        "",
        "```",
        "encounter shock : pair with balance < 0 (avoid side of the MBON",
        "                  approach/avoid readout; decision typically `avoid`).",
        "one-trial grudge: the ticker's FIRST encounter+decision pair after a",
        "                  pending shock, when the INNATE readout (raw_balance,",
        f"                  hatch-anchored, pre-plasticity) is >= +{thr} — the",
        "                  loop's APPROACH_THR, the innate reflex says approach —",
        "                  yet the LEARNED readout (balance, post KC->MBON",
        f"                  plasticity) is <= -{thr} and the decision is `avoid`.",
        "                  Plasticity flipped an innately positive odor into",
        "                  avoidance after a single aversive trial. Scored once",
        "                  per pending shock; a renewed shock updates it, any",
        "                  other outcome (buy, cap-blocked pass, neutral",
        "                  readout) resolves it.",
        f"approach trial  : pair with balance >= +{thr} and decision `buy`.",
        f"favorite        : ticker with >= {FAVORITE_MIN_BUYS} approach trials",
        "                  (repeated strong approach).",
        "```",
        "",
    ]
    lines += ["### Grudges (one-trial post-shock avoidance)", ""]
    if grudges:
        by_ticker: dict[str, list[Grudge]] = {}
        for g in grudges:
            by_ticker.setdefault(g.ticker, []).append(g)
        lines += [
            "| Ticker | One-trial grudges | First example |",
            "|---|---:|---|",
        ]
        for tkr in sorted(by_ticker, key=lambda t: (-len(by_ticker[t]), t)):
            first = by_ticker[tkr][0]
            lines.append(
                f"| {tkr} | {len(by_ticker[tkr])} "
                f"| shock at {_CITE.format(first.shock_line)}, grudge at "
                f"{_cite(first.enc_line, first.dec_line)} "
                f"(balance {first.balance:+.3f}, innate {first.raw_balance:+.3f}, "
                f"{first.action}) |"
            )
    else:
        lines.append("None detected.")
    lines += ["", "### Favorites (repeated strong approach)", ""]
    favorites = {t: c for t, c in approaches.items() if c >= FAVORITE_MIN_BUYS}
    if favorites:
        lines += ["| Ticker | Approach trials | Total buys |", "|---|---:|---:|"]
        for tkr in sorted(favorites, key=lambda t: (-favorites[t], t)):
            lines.append(f"| {tkr} | {favorites[tkr]} | {buys.get(tkr, 0)} |")
    else:
        lines.append("None detected.")
    lines.append("")
    return lines


def _tally_section(counts: Counter[str]) -> list[str]:
    lines = [
        "## Event tallies",
        "",
        ", ".join(f"{k}: {v:,}" for k, v in sorted(counts.items())) + ".",
        "",
        f"- deaths: {counts.get('death', 0):,}, hatches: {counts.get('hatch', 0):,}",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate_postmortem(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Render the T10 post-mortem (DESIGN §12) for a run directory.

    Reads ``events.jsonl`` + ``equity.csv`` from ``run_dir``, embeds the T8
    scoreboard comparison (``fruitfly.scoreboard.compare_run`` over the same
    window; ``bars_by_symbol`` overrides the market cache for offline use),
    and returns the markdown. When ``out_path`` is given the report is also
    written there (parents created). Pure function of the run receipts:
    rendering twice is byte-identical.
    """
    run_dir = Path(run_dir)
    events = _read_events(run_dir)
    equity = _read_equity(run_dir)
    counts = Counter(str(ev.get("type", "?")) for _, ev in events)
    lives = _lives(events)
    pairs = _pairs(events)
    grudges, approaches, buys = _behavior_autopsy(pairs)

    final = float(equity["equity"].iloc[-1])
    hatch_eq = float(lives[0].hatch_equity) if lives else float("nan")

    lines = [
        f"# T10 Post-mortem — {run_dir.name}",
        "",
        f"Final equity {_fmt(final)} ({100.0 * (final / hatch_eq - 1.0):+.2f}% "
        f"on {_fmt(hatch_eq)} hatch capital). The four-benchmark table below is "
        f"rendered by `fruitfly.scoreboard.compare_run` over the identical "
        f"window and bar grid.",
        "",
        *_lifespan_section(lives, equity, len(events)),
        *_cause_section(events, lives, equity),
        "## P&L vs scoreboard (T8 benchmarks)",
        "",
        _scoreboard_section(run_dir, bars_by_symbol),
        "",
        *_behavior_section(grudges, approaches, buys),
        *_tally_section(counts),
    ]
    report = "\n".join(lines).rstrip() + "\n"

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    return report


def _register() -> None:
    def builder(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "postmortem",
            help="Render the T10 post-mortem report for a run dir (DESIGN §12).",
        )
        p.add_argument(
            "--run-dir",
            required=True,
            help="Run directory with events.jsonl + equity.csv.",
        )
        p.add_argument(
            "--out",
            default=str(REPORT_PATH),
            help=f"Output markdown path (default: {REPORT_PATH}).",
        )
        p.set_defaults(func=_cmd_postmortem)

    register_command("postmortem", builder)


def _cmd_postmortem(args: argparse.Namespace) -> int:
    report = generate_postmortem(args.run_dir, args.out)
    print(
        f"post-mortem: {args.run_dir}\n"
        f"  report : {args.out} ({len(report.encode('utf-8')):,} bytes)"
    )
    return 0


_register()
