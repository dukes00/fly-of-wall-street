"""T14: live local dashboard v3 (DESIGN §12, D18/D21).

FastAPI + uvicorn + Server-Sent Events push, one embedded static page
(vanilla JS + ``<canvas>``, no build step, no npm, no JS libraries, nothing
loaded off-machine). Read-only file interface to a run's receipts —
``equity.csv`` plus ``events.jsonl`` under ``data/runs/`` — never attached to
the sim process: the server polls the receipt files on a fixed interval and
serves exactly what is on disk (D21, thin by design).

The UI is organized around the four glance questions, all reconstructed from
receipts by ``build_state`` (unknown event types are tolerated and render
generically, missing fields render as ``—``):

* **Is it doing something?** — sticky glance strip: status dot (green =
  events flowing, amber = quiet, red = stalled, computed from the latest
  event timestamp vs wall clock), last-event age in seconds, events per
  minute over the last 10 sim-minutes, latest bar timestamp.
* **When did it do something?** — activity timeline: newest-first feed of
  notable events (orders, decisions, credits, deaths/hatches, anchors,
  sugar shocks, sleep/wake, cancels, unknown types rendered generically)
  with filter chips, capped server-side.
* **How is it doing financially?** — equity, day P&L (vs prior day's close),
  cumulative P&L (vs hatch equity), realized/unrealized split (FIFO logic),
  plus sortable tables: positions, per-ticker league, trades.
* **How are the signals and stimuli?** — encounter-intensity histogram,
  centered-balance series with inferred threshold lines, per-day decision
  mix, per-day activity bars (orders/day, structural-score activity),
  last-sniff card, daily sugar_shock channels.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from fruitfly.__main__ import register_command

#: SSE poll interval (s): deterministic, file mtime+size change triggers push.
POLL_SECONDS = 1.0

#: Default/max event-log tail length for ``/api/events``.
EVENTS_TAIL_DEFAULT = 50
EVENTS_TAIL_MAX = 1000

#: Max points in downsampled time series (equity, drawdown, balance, P&L).
SERIES_MAX_POINTS = 240

#: Timeline rows sent to the client; the most recent rows are kept.
TIMELINE_CAP = 200

#: Sim-minute window for the events-per-minute glance rate.
RATE_WINDOW_MIN = 10.0

#: Glance-strip status thresholds, in wall-clock seconds since the latest
#: event timestamp: green (live) below ``LIVE_GRACE_S``, amber (quiet)
#: below ``QUIET_GRACE_S``, red (stalled) beyond. A run dir with neither
#: receipt file reports ``waiting``.
LIVE_GRACE_S = 120.0
QUIET_GRACE_S = 900.0

#: Receipt file names inside a run directory.
_EQUITY_CSV = "equity.csv"
_EVENTS_JSONL = "events.jsonl"

#: Embedded static page: the whole UI lives in ``_INDEX_HTML`` below.

# ---------------------------------------------------------------------------
# Receipt readers (pure functions of the files on disk; torn final lines are
# skipped so a run mid-write never crashes the dashboard)
# ---------------------------------------------------------------------------


def _fingerprint(run_dir: Path) -> tuple:
    """Change detector for the SSE poll: mtime+size of both receipt files."""
    marks: list[tuple[int, int] | None] = []
    for name in (_EQUITY_CSV, _EVENTS_JSONL):
        try:
            st = (run_dir / name).stat()
        except OSError:
            marks.append(None)
        else:
            marks.append((st.st_mtime_ns, st.st_size))
    return tuple(marks)


def _read_equity(run_dir: Path) -> list[list[str]]:
    """Parse equity.csv rows (partial trailing row tolerated)."""
    rows: list[list[str]] = []
    try:
        with (run_dir / _EQUITY_CSV).open(newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) == 4:
                    rows.append(row)
    except OSError:
        pass
    return rows


def _read_events(run_dir: Path) -> list[dict]:
    """Parse events.jsonl (a torn final line mid-write is skipped)."""
    events: list[dict] = []
    try:
        with (run_dir / _EVENTS_JSONL).open() as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return events



def _num(x) -> float | None:
    """Tolerant float coercion: missing/corrupt receipt fields read as None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _shares(x) -> int:
    """Tolerant int coercion for order share counts (bad field reads as 0)."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _thin(rows: list, max_points: int) -> list:
    """Evenly thin a series to <= max_points, always keeping the last row."""
    if len(rows) <= max_points:
        return list(rows)
    step = -(-len(rows) // max_points)
    picked = rows[::step]
    if rows[-1] is not picked[-1]:
        picked.append(rows[-1])
    return picked


def _epoch(ts) -> float | None:
    """Parse a receipt timestamp to a wall-clock epoch (``None`` if absent
    or not an ISO-8601 string — sim-side placeholder stamps like ``t1``
    read as ``None`` and are excluded from glance math."""
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _fmt(x, spec: str) -> str:
    """Fixed-decimal number for a timeline description (``None`` → ``—``)."""
    return "—" if x is None else format(x, spec)


def _timeline_row(ev: dict) -> dict | None:
    """One activity-timeline row for a notable event (``None`` = skip).

    Encounters are excluded (they feed the stimuli panels; the latest one
    gets the last-sniff card). Unknown types — including upcoming v0.7
    events — render generically instead of being dropped: a ``k=v``
    summary of their extra fields, bucketed under ``system`` (types whose
    name contains ``credit`` bucket under ``credits``).
    """
    kind = ev.get("type")
    ts = ev.get("ts")
    if kind == "encounter" or not isinstance(ts, str):
        return None
    row: dict = {"ts": ts, "type": kind}
    if kind == "order":
        row["cat"] = "orders"
        row["icon"] = "⇄"
        pnl = _num(ev.get("realized_pnl"))
        row["desc"] = "{} {} {} @ {} — {}".format(
            ev.get("ticker") or "?", ev.get("side") or "?",
            _fmt(_shares(ev.get("shares")) or None, "d"), _fmt(_num(ev.get("price")), ".2f"),
            ev.get("reason") or "?",
        ) + ("" if pnl is None else f" · pnl {_fmt(pnl, '+.2f')}")
        row["ticker"] = ev.get("ticker")
        row["side"] = ev.get("side")
        row["realized_pnl"] = _num(ev.get("realized_pnl"))
    elif kind == "death":
        row["cat"] = "system"
        row["icon"] = "☠"
        row["desc"] = "fly died · equity " + _fmt(_num(ev.get("equity")), ".2f")
    elif kind == "hatch":
        row["cat"] = "system"
        row["icon"] = "✦"
        row["desc"] = "hatch · equity " + _fmt(_num(ev.get("hatch_equity")), ".2f")
    elif kind == "anchor":
        row["cat"] = "system"
        row["icon"] = "⊕"
        row["desc"] = "innate-balance anchor " + _fmt(_num(ev.get("balance")), "+.4f")
    elif kind == "sugar_shock":
        row["cat"] = "system"
        row["icon"] = "🍯"
        row["desc"] = "sugar shock · reward {} punishment {} hunger {}".format(
            _fmt(_num(ev.get("reward")), "+.4f"),
            _fmt(_num(ev.get("punishment")), "+.4f"),
            _fmt(_num(ev.get("hunger")), "+.4f"),
        )
    elif kind == "sleep":
        row["cat"] = "system"
        row["icon"] = "☾"
        row["desc"] = "sleep"
    elif kind == "wake":
        row["cat"] = "system"
        row["icon"] = "☀"
        row["desc"] = "wake"
    elif kind == "cancel":
        row["cat"] = "system"
        row["icon"] = "⊘"
        row["desc"] = "cancel " + (ev.get("ticker") or "")
    elif "credit" in kind:
        row["cat"] = "credits"
        row["icon"] = "¢"
        row["desc"] = "{} credit · pnl {} reward {} punishment {}".format(
            ev.get("ticker") or "?",
            _fmt(_num(ev.get("realized_pnl")), "+.2f"),
            _fmt(_num(ev.get("reward")), "+.4f"),
            _fmt(_num(ev.get("punishment")), "+.4f"),
        )
    elif kind == "decision":
        row["cat"] = "decisions"
        row["icon"] = "⚑"
        row["action"] = ev.get("action")
        row["desc"] = "{} {} — {}".format(
            ev.get("ticker") or "?", ev.get("action") or "?",
            ev.get("reason") or "?",
        )
    else:  # unknown / upcoming schema: generic one-line summary
        row["cat"] = "system"
        row["icon"] = "•"
        rest = " ".join(f"{k}={v}" for k, v in ev.items()
                        if k not in ("type", "ts") and v is not None)
        row["desc"] = kind + (" · " + rest if rest else "")
    return row


#: Decision ``reason`` → decision-mix bucket; anything else counts neutral.
_DECISION_BUCKETS = {
    "approach": "approach",
    "avoid": "avoid",
    "valence_flip": "avoid",
    "cap": "cap",
    "dust": "dust",
}
_DECISION_MIX_ORDER = ("approach", "avoid", "neutral", "cap", "dust")


def build_state(run_dir: Path, now: float | None = None) -> dict:
    """Snapshot of one run, reconstructed entirely from its receipt files.

    ``now`` is the wall-clock epoch for the glance-strip recency math
    (defaults to the actual time; tests pin it for determinism).
    """
    equity_rows = _read_equity(run_dir)
    events = _read_events(run_dir)
    now = time.time() if now is None else now
    state: dict = {
        "run_dir": str(run_dir),
        "status": "waiting",  # recomputed below from event recency
        "hatch_equity": None,
        "equity": None,
        "cash": None,
        "n_positions": 0,
        "n_bars": len(equity_rows),
        "latest_ts": equity_rows[-1][0] if equity_rows else None,
        "latest_bar_ts": equity_rows[-1][0] if equity_rows else None,
        "last_event_age_s": None,
        "events_per_min": 0.0,
        "day_pnl": None,
        "day_pnl_prior_close": None,
        "cum_pnl": None,
        "deaths": 0,
        "n_events": len(events),
        "n_encounters": 0,
        "mbon": None,
        "neuromod": None,
        "equity_series": [],
        "drawdown_series": [],
        "death_marks": [],
        "neural": {
            "balance_series": [],
            "anchor_series": [],
            "thresholds": None,
            "decision_mix": {},
            "sugar_daily": {},
            "intensity_hist": {
                "edges": [round(i / 10, 1) for i in range(11)],
                "counts": [0] * 10,
            },
        },
        "trades": [],
        "pnl_by_ticker": {},
        "open_positions": [],
        "trade_credits": [],
        "unrealized_total": 0.0,
        "realized_total": 0.0,
        "realized_series": [],
        "realized_daily": {},
        "league": {},
        "timeline": [],
        "last_sniff": None,
        "daily_activity": {},
    }

    decisions: dict[tuple, str] = {}  # (ts, ticker) -> reason
    actions: dict[tuple, str] = {}  # (ts, ticker) -> decision action
    encounters: list[tuple[str, str, float]] = []  # (ts, ticker, centered bal)
    mix: dict[str, dict[str, int]] = {}
    sugar: dict[str, dict] = {}
    activity: dict[str, dict[str, int]] = {}  # day -> orders/encounters/structural
    timeline: list[dict] = []
    last_sniff: dict | None = None
    league: dict[str, dict] = {}  # ticker -> trades/wins/losses/realized
    cum_realized = 0.0
    realized_pts: list[list] = []
    lots: dict[str, list[list]] = {}  # ticker -> FIFO [[shares, price], ...]
    last_fill: dict[str, float] = {}  # ticker -> last known fill price

    for ev in events:
        kind = ev.get("type")
        ts = ev.get("ts")
        if kind == "hatch":
            h = _num(ev.get("hatch_equity"))
            if h is not None:
                state["hatch_equity"] = h
        elif kind == "death":
            state["deaths"] += 1
            state["death_marks"].append({"ts": ts, "equity": _num(ev.get("equity"))})
        elif kind == "encounter":
            state["n_encounters"] += 1
            bal = _num(ev.get("balance"))
            struct = _num(ev.get("structural_score"))
            used = _num(ev.get("balance_used"))
            intensity = _num(ev.get("intensity"))
            state["mbon"] = {
                "ts": ts,
                "ticker": ev.get("ticker"),
                "balance": bal,
                "raw_balance": _num(ev.get("raw_balance")),
                "structural_score": struct,
                "balance_used": used,
                "valence_readout": _num(ev.get("valence_readout")),
                "mbon_rate_hz": _num(ev.get("mbon_rate_hz")),
            }
            last_sniff = {
                "ts": ts,
                "ticker": ev.get("ticker"),
                "intensity": intensity,
                "balance": bal,
                "balance_used": used,
                "structural_score": struct,
                "raw_balance": _num(ev.get("raw_balance")),
            }
            if ts is not None and bal is not None:
                encounters.append((ts, ev.get("ticker"), bal))
                state["neural"]["balance_series"].append(
                    [ts, bal, _num(ev.get("raw_balance"))]
                )
            if intensity is not None:
                counts = state["neural"]["intensity_hist"]["counts"]
                counts[min(max(int(intensity * 10), 0), 9)] += 1
            if isinstance(ts, str):
                a = activity.setdefault(
                    ts[:10], {"orders": 0, "encounters": 0, "structural": 0}
                )
                a["encounters"] += 1
                if struct:
                    a["structural"] += 1
        elif kind == "decision":
            reason = ev.get("reason")
            if isinstance(ts, str):
                decisions[(ts, ev.get("ticker"))] = reason
                actions[(ts, ev.get("ticker"))] = ev.get("action")
                bucket = _DECISION_BUCKETS.get(reason, "neutral")
                day = ts[:10]
                mix.setdefault(day, dict.fromkeys(_DECISION_MIX_ORDER, 0))
                mix[day][bucket] += 1
        elif kind == "sugar_shock":
            day = ts[:10] if isinstance(ts, str) else "?"
            sugar[day] = {
                "reward": _num(ev.get("reward")),
                "punishment": _num(ev.get("punishment")),
                "hunger": _num(ev.get("hunger")),
                "arousal": _num(ev.get("arousal")),
            }
            state["neuromod"] = {"ts": ts, **sugar[day]}
        elif kind == "trade_credit":
            # Upcoming schema, rendered when present: a per-trade credit
            # with neuromod consequences. Rendered only — the realized P&L
            # aggregation stays sourced from ``order`` events.
            state["trade_credits"].append(
                {
                    "ts": ts,
                    "ticker": ev.get("ticker"),
                    "realized_pnl": _num(ev.get("realized_pnl")),
                    "reward": _num(ev.get("reward")),
                    "punishment": _num(ev.get("punishment")),
                }
            )
        elif kind == "anchor":
            bal = _num(ev.get("balance"))
            if ts is not None and bal is not None:
                state["neural"]["anchor_series"].append([ts, bal])
        elif kind == "order":
            ticker = ev.get("ticker")
            side = ev.get("side")
            n = _shares(ev.get("shares"))
            price = _num(ev.get("price"))
            realized = _num(ev.get("realized_pnl"))
            state["trades"].append(
                {
                    "ts": ts,
                    "ticker": ticker,
                    "side": side,
                    "shares": n if n else None,
                    "price": price,
                    "reason": ev.get("reason"),
                    "realized_pnl": realized,
                }
            )
            # FIFO book: buys append lots, sells consume from the front.
            if side == "buy" and n > 0 and price is not None:
                lots.setdefault(ticker, []).append([n, price])
                last_fill[ticker] = price
            elif side == "sell" and n > 0:
                book = lots.setdefault(ticker, [])
                remaining = n
                while remaining > 0 and book:
                    take = min(book[0][0], remaining)
                    book[0][0] -= take
                    remaining -= take
                    if book[0][0] == 0:
                        book.pop(0)
                if not book:
                    lots.pop(ticker, None)
                if price is not None:
                    last_fill[ticker] = price
            if realized is not None:
                cum_realized += realized
                realized_pts.append([ts, round(cum_realized, 6)])
                if ticker is not None:
                    state["pnl_by_ticker"][ticker] = round(
                        state["pnl_by_ticker"].get(ticker, 0.0) + realized, 6
                    )
                day = ts[:10] if isinstance(ts, str) else "?"
                state["realized_daily"][day] = round(
                    state["realized_daily"].get(day, 0.0) + realized, 6
                )
            lg = league.setdefault(
                ticker, {"trades": 0, "wins": 0, "losses": 0, "realized": 0.0}
            )
            lg["trades"] += 1
            if realized is not None and realized > 0:
                lg["wins"] += 1
            elif realized is not None and realized < 0:
                lg["losses"] += 1
            if realized is not None:
                lg["realized"] = round(lg["realized"] + realized, 6)
            if isinstance(ts, str):
                activity.setdefault(
                    ts[:10], {"orders": 0, "encounters": 0, "structural": 0}
                )["orders"] += 1
        row = _timeline_row(ev)
        if row is not None:
            timeline.append(row)
        # Unknown event types are tolerated: they stay queryable via
        # /api/events and render in the event log.

    # Approach/avoid thresholds inferred from the decisions: an encounter
    # that triggered "approach" sat above the approach threshold, one that
    # triggered "avoid"/"valence_flip" sat below -avoid. The tightest
    # observed signal bounds each threshold from above (no config is read).
    thr: dict[str, float] = {}
    appr = [b for ts_, t_, b in encounters if decisions.get((ts_, t_)) == "approach"]
    avoid = [
        abs(b)
        for ts_, t_, b in encounters
        if decisions.get((ts_, t_)) in ("avoid", "valence_flip")
    ]
    if appr:
        thr["approach"] = round(min(appr), 9)
    if avoid:
        thr["avoid"] = round(min(avoid), 9)
    state["neural"]["thresholds"] = {"inferred": True, **thr} if thr else None

    # Glance strip liveness: the events.jsonl wall-clock mtime is the only
    # honest "is the writer alive" signal — sim-time age vs wall clock is
    # meaningless for fast lanes (a stripped run emits ~156 sim-min per wall
    # min, so a live run looks "stalled" on sim age). ``last_event_age_s``
    # stays sim-time (informative for slow whole-fly runs); the status dot
    # uses mtime. A run dir with neither receipt file reports ``waiting``.
    if not (run_dir / _EQUITY_CSV).exists() and not (run_dir / _EVENTS_JSONL).exists():
        state["status"] = "waiting"
    else:
        epochs = [e for e in (_epoch(ev.get("ts")) for ev in events) if e is not None]
        if epochs:
            newest = max(epochs)
            age = max(0.0, now - newest)
            state["last_event_age_s"] = round(age, 3)
            window = RATE_WINDOW_MIN * 60.0
            state["events_per_min"] = round(
                sum(1 for e in epochs if e >= newest - window) / RATE_WINDOW_MIN, 4
            )
            events_path = run_dir / _EVENTS_JSONL
            mtime_age = max(0.0, now - events_path.stat().st_mtime)
            state["status"] = (
                "live" if mtime_age < LIVE_GRACE_S
                else "quiet" if mtime_age < QUIET_GRACE_S
                else "stalled"
            )
        else:
            state["status"] = "quiet"

    # Per-ticker league: win-rate over trades that booked a nonzero verdict.
    for lg in league.values():
        decided = lg["wins"] + lg["losses"]
        lg["win_rate"] = round(lg["wins"] / decided, 4) if decided else None
    state["league"] = league

    # Last-sniff card: latest encounter plus the action it triggered.
    if last_sniff is not None:
        last_sniff["action"] = actions.get(
            (last_sniff["ts"], last_sniff["ticker"])
        ) or decisions.get((last_sniff["ts"], last_sniff["ticker"]))
        state["last_sniff"] = last_sniff

    # Activity timeline, newest first, capped server-side.
    state["timeline"] = timeline[-TIMELINE_CAP:][::-1]
    state["daily_activity"] = dict(sorted(activity.items()))

    # Open positions: whatever survived the FIFO replay, marked at the last
    # known fill (the receipts carry no live per-ticker marks).
    unrealized_total = 0.0
    for t, book in sorted(lots.items()):
        shares = sum(lot[0] for lot in book)
        if shares <= 0:
            continue
        avg_cost = sum(lot[0] * lot[1] for lot in book) / shares
        mark = last_fill.get(t)
        unrealized = round(shares * (mark - avg_cost), 6) if mark is not None else None
        if unrealized is not None:
            unrealized_total += unrealized
        state["open_positions"].append(
            {
                "ticker": t,
                "shares": shares,
                "avg_cost": round(avg_cost, 6),
                "mark": mark,
                "unrealized": unrealized,
            }
        )
    state["unrealized_total"] = round(unrealized_total, 6)
    state["realized_total"] = round(cum_realized, 6)

    # Time series (downsampled for the wire; each row self-describes with ts).
    eq = [[r[0], float(r[1]), float(r[2]), int(r[3])] for r in equity_rows]
    state["equity_series"] = _thin(eq, SERIES_MAX_POINTS)
    peak = float("-inf")
    dd: list[list] = []
    for row in eq:
        peak = max(peak, row[1])
        dd.append([row[0], round(1.0 - row[1] / peak, 9) if peak > 0 else 0.0])
    state["drawdown_series"] = _thin(dd, SERIES_MAX_POINTS)
    state["neural"]["balance_series"] = _thin(
        state["neural"]["balance_series"], SERIES_MAX_POINTS
    )
    state["neural"]["decision_mix"] = dict(sorted(mix.items()))
    state["neural"]["sugar_daily"] = dict(sorted(sugar.items()))
    state["realized_series"] = _thin(realized_pts, SERIES_MAX_POINTS)

    if equity_rows:
        last = equity_rows[-1]
        state["equity"] = float(last[1])
        state["cash"] = float(last[2])
        state["n_positions"] = int(last[3])
        state["market_value"] = round(state["equity"] - state["cash"], 2)
        # Day P&L vs the prior day's close (last bar of the previous day);
        # None until the run spans more than one day.
        closes: dict[str, float] = {}
        for r in equity_rows:
            closes[r[0][:10]] = float(r[1])  # later rows overwrite: day close
        day_list = sorted(closes)
        if len(day_list) > 1:
            prior = closes[day_list[-2]]
            state["day_pnl"] = round(state["equity"] - prior, 6)
            state["day_pnl_prior_close"] = prior
        base = (
            state["hatch_equity"]
            if state["hatch_equity"] is not None
            else float(equity_rows[0][1])
        )
        state["cum_pnl"] = round(state["equity"] - base, 6)
    return state


#: Embedded static page (vanilla JS + canvas; nothing loaded off-machine).
#: Everything renders client-side from /api/state snapshots; the server
#: only aggregates. The four glance questions drive the layout: the sticky
#: glance strip answers "is it doing something", the timeline answers
#: "when", the header numbers + sortable tables answer "how financially",
#: and the stimulus panels answer "how are the signals".
_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fruit Fly of Wall Street — live dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font: 13px/1.45 ui-monospace, Menlo, monospace; margin: 0;
         background: #101418; color: #d8e0e8; }
  #glance { position: sticky; top: 0; z-index: 10; padding: 8px 16px;
            border-bottom: 1px solid #2a3440; background: rgba(16,20,24,.96);
            display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
  #glance h1 { font-size: 14px; margin: 0 10px 0 0; letter-spacing: .04em; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 6px; }
  .dot.live { background: #7ad48a; box-shadow: 0 0 7px #7ad48a; }
  .dot.quiet { background: #e0a040; box-shadow: 0 0 7px #e0a040; }
  .dot.stalled, .dot.waiting { background: #e55; }
  .metric { display: flex; flex-direction: column; line-height: 1.15; }
  .metric .label { color: #8fa3b5; font-size: 10px; text-transform: uppercase;
                   letter-spacing: .06em; }
  .metric b { color: #e8f0f8; font-variant-numeric: tabular-nums; }
  .metric.big b { font-size: 26px; color: #fff; }
  .kv { color: #8fa3b5; } .kv b { color: #e8f0f8; }
  main { display: grid; gap: 14px; padding: 14px;
         grid-template-columns: repeat(auto-fit, minmax(min(430px, 100%), 1fr)); }
  section { border: 1px solid #2a3440; border-radius: 8px; padding: 10px 12px;
            background: #151b22; min-width: 0; }
  section.wide { grid-column: 1 / -1; }
  h2 { font-size: 12px; margin: 0 0 8px; color: #7ec8e3; letter-spacing: .06em;
       text-transform: uppercase; }
  canvas { width: 100%; height: 190px; display: block; background: #0c1014;
           border-radius: 4px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid #222c36; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8fa3b5; font-weight: normal; cursor: pointer; user-select: none;
       white-space: nowrap; }
  th[aria-sort] { color: #d8ecf8; }
  th[aria-sort="ascending"]::after { content: " ▲"; color: #7ec8e3; }
  th[aria-sort="descending"]::after { content: " ▼"; color: #7ec8e3; }
  tbody tr:nth-child(even) { background: #131a21; }
  .pos { color: #7ad48a; } .neg { color: #e88; } .sys { color: #e0a040; }
  .dash { color: #5a6a78; }
  .chip { font: inherit; font-size: 11px; background: #0c1014; color: #8fa3b5;
          border: 1px solid #2a3440; border-radius: 999px; padding: 1px 10px;
          margin-right: 4px; cursor: pointer; }
  .chip.active { color: #0c1014; font-weight: bold; }
  .chip.active[data-filter="all"], .chip.active[data-filter="decisions"] {
    background: #7ec8e3; border-color: #7ec8e3; }
  .chip.active[data-filter="orders"] { background: #7ad48a; border-color: #7ad48a; }
  .chip.active[data-filter="credits"] { background: #b48ce0; border-color: #b48ce0; }
  .chip.active[data-filter="system"] { background: #e0a040; border-color: #e0a040; }
  #timeline { max-height: 420px; overflow-y: auto; }
  .t-row { display: flex; gap: 8px; align-items: baseline; padding: 2px 0;
           border-bottom: 1px dotted #222c36; }
  .t-age { flex: none; min-width: 58px; text-align: right; color: #5a6a78;
           font-variant-numeric: tabular-nums; }
  .t-icon { flex: none; width: 18px; text-align: center; }
  .t-row.orders .t-icon { color: #7ad48a; }
  .t-row.decisions .t-icon { color: #7ec8e3; }
  .t-row.credits .t-icon { color: #b48ce0; }
  .t-row.system .t-icon { color: #e0a040; }
  .t-row.neg .t-icon, .t-row.neg .t-desc { color: #e88; }
  .t-desc { word-break: break-word; }
  .sniff-row { display: flex; justify-content: space-between; gap: 12px;
               padding: 3px 0; border-bottom: 1px dotted #222c36; }
  .sniff-row .k { color: #8fa3b5; }
  .sniff-row .v { font-variant-numeric: tabular-nums; }
  #events { max-height: 220px; overflow-y: auto; font-size: 11px; }
  .ev { padding: 1px 0; border-bottom: 1px dotted #222c36; white-space: pre-wrap; }
  .ev .type { color: #7ec8e3; }
</style>
</head>
<body>
<header id="glance">
  <h1>fruit fly of wall street</h1>
  <span class="metric"><span class="label"><span
  class="dot waiting" id="g-dot"></span><span id="g-status">connecting</span></span></span>
  <span class="metric big"><span class="label">last event</span><b id="g-age">—</b></span>
  <span class="metric"><span class="label">events/min · 10 sim-min</span><b id="g-rate">—</b></span>
  <span class="metric"><span class="label">latest bar</span><b id="g-bar">—</b></span>
  <span class="metric"><span class="label">equity</span><b id="h-equity">—</b></span>
  <span class="metric"><span class="label">day pnl</span><b id="h-day">—</b></span>
  <span class="metric"><span class="label">cum pnl</span><b id="h-cum">—</b></span>
  <span class="metric"><span class="label">realized</span><b id="h-real">—</b></span>
  <span class="metric"><span class="label">unrealized</span><b id="h-unreal">—</b></span>
  <span class="kv">cash <b id="h-cash">—</b></span>
  <span class="kv">hatch <b id="h-hatch">—</b></span>
  <span class="kv">deaths <b id="h-deaths">—</b></span>
  <span class="kv">bars <b id="h-bars">—</b></span>
</header>
<main>
  <section class="wide" id="panel-equity">
    <h2>Equity &amp; drawdown</h2>
    <canvas id="c-equity"></canvas>
  </section>
  <section id="panel-timeline">
    <h2>Activity timeline</h2>
    <div id="chips">
      <button class="chip active" data-filter="all">all</button>
      <button class="chip" data-filter="orders">orders</button>
      <button class="chip" data-filter="decisions">decisions</button>
      <button class="chip" data-filter="credits">credits</button>
      <button class="chip" data-filter="system">system</button>
    </div>
    <div id="timeline"></div>
  </section>
  <section id="panel-neural">
    <h2>KC→MBON balance (centered vs raw) &amp; inferred thresholds</h2>
    <canvas id="c-balance"></canvas>
    <p class="kv" id="p-thresholds"></p>
  </section>
  <section id="panel-intensity">
    <h2>Encounter intensity histogram</h2>
    <canvas id="c-intensity"></canvas>
  </section>
  <section id="panel-decisions">
    <h2>Per-day decision mix &amp; sugar channels</h2>
    <canvas id="c-mix"></canvas>
    <p class="kv" id="p-sugar"></p>
  </section>
  <section id="panel-activity">
    <h2>Per-day activity — orders &amp; structural sniffs</h2>
    <canvas id="c-activity"></canvas>
  </section>
  <section id="panel-pnl">
    <h2>Aggregate P&amp;L — cumulative realized</h2>
    <canvas id="c-realized"></canvas>
    <p class="kv" id="p-split"></p>
  </section>
  <section id="panel-sniff">
    <h2>Last sniff</h2>
    <div id="p-sniff"></div>
  </section>
  <section class="wide" id="panel-trading">
    <h2>Trades</h2>
    <table id="t-trades" class="sortable"><thead><tr>
      <th data-key="ts" data-type="str">ts</th>
      <th data-key="ticker" data-type="str">ticker</th>
      <th data-key="side" data-type="str">side</th>
      <th data-key="shares" data-type="num">shares</th>
      <th data-key="price" data-type="num">price</th>
      <th data-key="realized_pnl" data-type="num">realized pnl</th>
      <th data-key="reason" data-type="str">reason</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <section id="panel-positions">
    <h2>Open positions (FIFO, mark = last fill — approximation)</h2>
    <table id="t-positions" class="sortable"><thead><tr>
      <th data-key="ticker" data-type="str">ticker</th>
      <th data-key="shares" data-type="num">shares</th>
      <th data-key="avg_cost" data-type="num">cost</th>
      <th data-key="mark" data-type="num">mark</th>
      <th data-key="unrealized" data-type="num">unrealized</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <section id="panel-league">
    <h2>Per-ticker league</h2>
    <table id="t-league" class="sortable"><thead><tr>
      <th data-key="ticker" data-type="str">ticker</th>
      <th data-key="trades" data-type="num">trades</th>
      <th data-key="realized" data-type="num">realized pnl</th>
      <th data-key="win_rate" data-type="num">win-rate</th>
    </tr></thead><tbody></tbody></table>
    <h2>Per-day realized</h2>
    <table id="t-daily" class="sortable"><thead><tr>
      <th data-key="day" data-type="str">day</th>
      <th data-key="realized" data-type="num">realized pnl</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <section class="wide" id="panel-events">
    <h2>Event log tail</h2>
    <div id="events"></div>
  </section>
</main>
<script>
"use strict";
const DASH = "—";
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x))
      ? DASH
      : Number(x).toLocaleString("en-US", { minimumFractionDigits: d,
        maximumFractionDigits: d });
const el = (id) => document.getElementById(id);
const cls = (x) => x > 0 ? "pos" : x < 0 ? "neg" : "";
const fmtAge = (s) => s === null || s === undefined ? DASH
  : s < 60 ? s.toFixed(1) + "s"
  : s < 3600 ? Math.floor(s / 60) + "m " + String(Math.round(s % 60)).padStart(2, "0") + "s"
  : Math.floor(s / 3600) + "h " + String(Math.round((s % 3600) / 60)).padStart(2, "0") + "m";
const relAge = (sec) => sec < 90 ? sec.toFixed(0) + "s"
  : sec < 5400 ? (sec / 60).toFixed(1) + "m"
  : (sec / 3600).toFixed(1) + "h";

let snap = null;        // last /api/state snapshot
let snapAt = 0;         // Date.now()/1000 when it arrived
let filter = "all";
const sorts = {};       // table id -> {key, dir}

function line(ctx, pts, color, y0, y1, w, h) {
  if (pts.length < 2) return;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = (i / (pts.length - 1)) * w;
    const y = h - ((p - y0) / (y1 - y0 || 1)) * h;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
}

function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#2a3440"; ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  return [ctx, w, h];
}

function draw(canvas, series) {
  const [ctx, w, h] = prep(canvas);
  const flat = series.map(([pts]) => pts);
  const vals = flat.flat().filter((v) => v !== null && v !== undefined);
  if (!vals.length) { ctx.fillStyle = "#5a6a78"; ctx.fillText("no data yet", 8, 16); return; }
  let y0 = Math.min(...vals), y1 = Math.max(...vals);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const colors = ["#7ec8e3", "#e0a040", "#7ad48a", "#e88", "#b48ce0"];
  series.forEach(([pts, color, dashed], i) => {
    ctx.setLineDash(dashed ? [4, 4] : []);
    line(ctx, pts, color || colors[i % colors.length], y0, y1, w, h);
    ctx.setLineDash([]);
  });
  ctx.fillStyle = "#5a6a78"; ctx.font = "11px monospace";
  ctx.fillText(fmt(y1), 4, 12); ctx.fillText(fmt(y0), 4, h - 4);
}

function drawBars(canvas, labels, seriesList) {
  const [ctx, w, h] = prep(canvas);
  const all = seriesList.flatMap((s) => s.vals).filter((v) => v !== null);
  if (!all.length || !labels.length) {
    ctx.fillStyle = "#5a6a78"; ctx.fillText("no data yet", 8, 16); return;
  }
  const max = Math.max(...all, 1);
  const gw = w / labels.length;
  const bw = gw / (seriesList.length + 0.4);
  seriesList.forEach((s, si) => {
    ctx.fillStyle = s.color;
    s.vals.forEach((v, i) => {
      if (v === null) return;
      const bh = (v / max) * (h - 26);
      ctx.fillRect(i * gw + si * bw + 1, h - bh - 14, Math.max(bw - 2, 1), bh);
    });
  });
  labels.forEach((d, i) => {
    if (labels.length <= 20 || i % 5 === 0) ctx.fillText(d, i * gw + 2, h - 3);
  });
  ctx.fillText(String(Math.round(max * 100) / 100), 4, 12);
}

function hist(canvas, counts, edges) {
  const [ctx, w, h] = prep(canvas);
  const max = Math.max(...counts, 1);
  const bw = w / counts.length;
  counts.forEach((n, i) => {
    const bh = (n / max) * (h - 20);
    ctx.fillStyle = n ? "#7ec8e3" : "#1a222b";
    ctx.fillRect(i * bw + 1, h - bh - 12, bw - 2, bh);
    ctx.fillStyle = "#5a6a78"; ctx.font = "10px monospace";
    ctx.fillText(edges[i].toFixed(1), i * bw + 1, h - 2);
  });
}

const rowTone = (r) => {
  if (r.type === "order") return r.side === "sell" ? "neg" : "";
  if (r.type === "decision") return (r.action === "avoid" || r.action === "sell") ? "neg" : "";
  return r.type === "death" ? "neg" : "";
};

function renderTimeline(s) {
  const box = el("timeline");
  box.innerHTML = "";
  const rows = (s.timeline || []).filter((r) => filter === "all" || r.cat === filter);
  const newest = rows.length ? Date.parse(rows[0].ts) : NaN;
  const frag = document.createDocumentFragment();
  rows.forEach((r) => {
    const div = document.createElement("div");
    div.className = "t-row " + (r.cat || "system") + (rowTone(r) ? " " + rowTone(r) : "");
    const age = document.createElement("span"); age.className = "t-age";
    const t = Date.parse(r.ts);
    age.textContent = isNaN(t) || isNaN(newest) ? DASH : relAge((newest - t) / 1000);
    const icon = document.createElement("span"); icon.className = "t-icon";
    icon.textContent = r.icon || "•";
    const desc = document.createElement("span"); desc.className = "t-desc";
    desc.textContent = (r.ts.length > 19 ? r.ts.slice(11, 19) + " " : r.ts + " ") + (r.desc || "");
    div.appendChild(age); div.appendChild(icon); div.appendChild(desc);
    frag.appendChild(div);
  });
  if (!rows.length) {
    const d = document.createElement("div"); d.className = "dash";
    d.textContent = "no events in this filter"; frag.appendChild(d);
  }
  box.appendChild(frag);
}

function sniffRow(k, v, tone) {
  const div = document.createElement("div"); div.className = "sniff-row";
  const key = document.createElement("span"); key.className = "k"; key.textContent = k;
  const val = document.createElement("span"); val.className = "v " + (tone || "");
  val.textContent = v;
  div.appendChild(key); div.appendChild(val);
  return div;
}

function renderSniff(s) {
  const box = el("p-sniff");
  box.innerHTML = "";
  const sn = s.last_sniff;
  if (!sn) {
    const d = document.createElement("span"); d.className = "dash";
    d.textContent = "no encounters yet"; box.appendChild(d); return;
  }
  const tone = sn.action === "buy" || sn.action === "approach" ? "pos"
    : sn.action === "sell" || sn.action === "avoid" ? "neg" : "sys";
  box.appendChild(sniffRow("ticker", sn.ticker || DASH));
  box.appendChild(sniffRow("intensity", fmt(sn.intensity, 3)));
  box.appendChild(sniffRow("balance (centered)", fmt(sn.balance, 4)));
  box.appendChild(sniffRow("balance_used", fmt(sn.balance_used, 4)));
  box.appendChild(sniffRow("structural_score", fmt(sn.structural_score, 3)));
  box.appendChild(sniffRow("action", sn.action || DASH, tone));
  box.appendChild(sniffRow("at", sn.ts || DASH));
}

const TABLES = {
  "t-trades": (s) => [s.trades || [], [
    { key: "ts", val: (t) => t.ts, txt: (t) => t.ts || DASH },
    { key: "ticker", val: (t) => t.ticker, txt: (t) => t.ticker || DASH },
    { key: "side", val: (t) => t.side, txt: (t) => t.side || DASH,
      cls: (t) => t.side === "buy" ? "pos" : t.side === "sell" ? "neg" : "" },
    { key: "shares", val: (t) => t.shares, txt: (t) => fmt(t.shares, 0) },
    { key: "price", val: (t) => t.price, txt: (t) => fmt(t.price) },
    { key: "realized_pnl", val: (t) => t.realized_pnl, txt: (t) => fmt(t.realized_pnl),
      cls: (t) => cls(t.realized_pnl) },
    { key: "reason", val: (t) => t.reason, txt: (t) => t.reason || DASH },
  ]],
  "t-positions": (s) => [s.open_positions || [], [
    { key: "ticker", val: (p) => p.ticker, txt: (p) => p.ticker || DASH },
    { key: "shares", val: (p) => p.shares, txt: (p) => fmt(p.shares, 0) },
    { key: "avg_cost", val: (p) => p.avg_cost, txt: (p) => fmt(p.avg_cost, 4) },
    { key: "mark", val: (p) => p.mark, txt: (p) => fmt(p.mark, 4) },
    { key: "unrealized", val: (p) => p.unrealized, txt: (p) => fmt(p.unrealized),
      cls: (p) => cls(p.unrealized) },
  ]],
  "t-league": (s) => [Object.entries(s.league || {}).map(([t, v]) => ({ ticker: t, ...v })), [
    { key: "ticker", val: (r) => r.ticker, txt: (r) => r.ticker || DASH },
    { key: "trades", val: (r) => r.trades, txt: (r) => String(r.trades ?? 0) },
    { key: "realized", val: (r) => r.realized, txt: (r) => fmt(r.realized),
      cls: (r) => cls(r.realized) },
    { key: "win_rate", val: (r) => r.win_rate,
      txt: (r) => (r.win_rate === null || r.win_rate === undefined)
        ? DASH : (r.win_rate * 100).toFixed(0) + "%" },
  ]],
  "t-daily": (s) => [Object.entries(s.realized_daily || {})
    .map(([d, v]) => ({ day: d, realized: v })), [
    { key: "day", val: (r) => r.day, txt: (r) => r.day || DASH },
    { key: "realized", val: (r) => r.realized, txt: (r) => fmt(r.realized),
      cls: (r) => cls(r.realized) },
  ]],
};

function renderTable(id, rows, cols) {
  const t = el(id);
  const s = sorts[id];
  let data = rows.slice();
  if (s) {
    const col = cols.find((c) => c.key === s.key);
    if (col) data.sort((a, b) => {
      const va = col.val(a), vb = col.val(b);
      const na = va === null || va === undefined, nb = vb === null || vb === undefined;
      if (na || nb) return na && nb ? 0 : na ? 1 : -1;
      if (typeof va === "string" || typeof vb === "string")
        return String(va).localeCompare(String(vb)) * s.dir;
      return (va - vb) * s.dir;
    });
  }
  const tb = t.tBodies[0];
  tb.innerHTML = "";
  data.forEach((r) => {
    const tr = tb.insertRow();
    cols.forEach((c) => {
      const td = tr.insertCell();
      td.textContent = c.txt(r);
      const tone = c.cls && c.cls(r);
      if (tone) td.className = tone;
    });
  });
  if (!tb.rows.length)
    tb.innerHTML = '<tr><td colspan="' + cols.length + '" class="dash">—</td></tr>';
  t.querySelectorAll("th[data-key]").forEach((th) => {
    if (s && th.dataset.key === s.key)
      th.setAttribute("aria-sort", s.dir === 1 ? "ascending" : "descending");
    else th.removeAttribute("aria-sort");
  });
}

function setNum(id, x, d = 2) {
  const node = el(id);
  node.textContent = fmt(x, d);
  node.className = cls(x);
}

function render(s) {
  snap = s; snapAt = Date.now() / 1000;
  // --- glance strip --------------------------------------------------------
  el("g-status").textContent = s.status;
  el("g-dot").className = "dot " + s.status;
  el("g-age").textContent = fmtAge(s.last_event_age_s);
  el("g-rate").textContent = fmt(s.events_per_min, 2);
  el("g-bar").textContent = (s.latest_bar_ts || DASH).replace("T", " ").slice(0, 19);
  setNum("h-equity", s.equity);
  setNum("h-day", s.day_pnl);
  setNum("h-cum", s.cum_pnl);
  setNum("h-real", s.realized_total);
  setNum("h-unreal", s.unrealized_total);
  el("h-cash").textContent = fmt(s.cash);
  el("h-hatch").textContent = fmt(s.hatch_equity);
  el("h-deaths").textContent = s.deaths;
  el("h-bars").textContent = s.n_bars;
  // --- equity + drawdown ---------------------------------------------------
  const eq = (s.equity_series || []).map((r) => r[1]);
  const hatch = s.hatch_equity;
  draw(el("c-equity"), [
    [eq, "#7ec8e3"],
    [hatch !== null && hatch !== undefined ? eq.map(() => hatch) : [], "#b48ce0", true],
  ]);
  if ((s.death_marks || []).length) {
    const c = el("c-equity"), ctx = c.getContext("2d");
    ctx.fillStyle = "#e55";
    s.death_marks.forEach((d, i) => {
      const x = ((i + 1) / (s.death_marks.length + 1)) * c.clientWidth;
      ctx.fillRect(x, 0, 2, 8);
    });
  }
  // --- activity timeline ---------------------------------------------------
  renderTimeline(s);
  // --- neural balance + inferred thresholds --------------------------------
  const bs = s.neural?.balance_series || [];
  const bal = bs.map((r) => r[1]);
  const raw = bs.map((r) => r[2]);
  const anc = (s.neural?.anchor_series || []).map((r) => r[1]);
  const thr = s.neural?.thresholds;
  const series = [[bal, "#7ec8e3"], [raw, "#e0a040"], [anc, "#b48ce0", true]];
  if (thr && thr.approach !== undefined)
    series.push([bs.map(() => thr.approach), "#7ad48a", true]);
  if (thr && thr.avoid !== undefined)
    series.push([bs.map(() => -thr.avoid), "#e88", true]);
  draw(el("c-balance"), series);
  el("p-thresholds").textContent = thr
    ? "inferred thresholds — approach ≥ " + fmt(thr.approach, 4)
      + ", avoid ≤ −" + fmt(thr.avoid, 4)
    : "thresholds: not inferable from receipts";
  // --- intensity histogram ---------------------------------------------------
  const ih = s.neural?.intensity_hist;
  if (ih) hist(el("c-intensity"), ih.counts, ih.edges);
  // --- decision mix + sugar channels ---------------------------------------
  const mix = s.neural?.decision_mix || {};
  const days = Object.keys(mix);
  const mixColors = ["#7ad48a", "#e88", "#e0a040", "#7ec8e3", "#b48ce0"];
  draw(el("c-mix"), buckets.map((b, i) => (
    [days.map((d) => mix[d][b] || 0), mixColors[i], i === 2])));
  const sugar = s.neural?.sugar_daily || {};
  const sdays = Object.keys(sugar);
  el("p-sugar").textContent = sdays.length
    ? "sugar_shock " + sdays.map((d) => d + ": r=" + fmt(sugar[d].reward, 4)
      + " p=" + fmt(sugar[d].punishment, 4) + " h=" + fmt(sugar[d].hunger, 4)
      + " a=" + fmt(sugar[d].arousal, 4)).join("  ")
    : "no sugar_shock events";
  // --- per-day activity bars -------------------------------------------------
  const act = s.daily_activity || {};
  const adays = Object.keys(act);
  drawBars(el("c-activity"), adays.map((d) => d.slice(5)), [
    { vals: adays.map((d) => act[d].orders || 0), color: "#7ec8e3" },
    { vals: adays.map((d) => act[d].structural || 0), color: "#7ad48a" },
  ]);
  // --- aggregate pnl ----------------------------------------------------------
  draw(el("c-realized"), [[(s.realized_series || []).map((r) => r[1]), "#7ad48a"]]);
  el("p-split").textContent = "realized " + fmt(s.realized_total)
    + "  unrealized " + fmt(s.unrealized_total)
    + "  equity " + fmt(s.equity)
    + "  market value " + fmt(s.market_value);
  renderSniff(s);
  // --- sortable tables ----------------------------------------------------------
  Object.keys(TABLES).forEach((id) => {
    const [rows, cols] = TABLES[id](s);
    renderTable(id, rows, cols);
  });
}

async function pollEvents() {
  try {
    const r = await fetch("/api/events?limit=50");
    const j = await r.json();
    const box = el("events");
    box.innerHTML = "";
    j.events.slice().reverse().forEach((e) => {
      const div = document.createElement("div");
      div.className = "ev";
      const type = document.createElement("span");
      type.className = "type"; type.textContent = e.type || "?";
      div.appendChild(type);
      div.appendChild(document.createTextNode("  " + JSON.stringify(e)));
      box.appendChild(div);
    });
  } catch (_) { /* transient */ }
}

document.querySelectorAll("#chips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    filter = chip.dataset.filter;
    document.querySelectorAll("#chips .chip").forEach((c) =>
      c.classList.toggle("active", c === chip));
    if (snap) renderTimeline(snap);
  });
});

document.querySelectorAll("table.sortable th[data-key]").forEach((th) => {
  th.addEventListener("click", () => {
    const id = th.closest("table").id;
    const cur = sorts[id];
    sorts[id] = (cur && cur.key === th.dataset.key)
      ? { key: th.dataset.key, dir: -cur.dir }
      : { key: th.dataset.key, dir: 1 };
    if (snap) {
      const [rows, cols] = TABLES[id](snap);
      renderTable(id, rows, cols);
    }
  });
});

// Keep the glance age ticking between SSE pushes.
setInterval(() => {
  if (snap && snap.last_event_age_s !== null && snap.last_event_age_s !== undefined)
    el("g-age").textContent = fmtAge(
      Math.max(0, snap.last_event_age_s + Date.now() / 1000 - snapAt));
}, 1000);

const es = new EventSource("/api/stream");
es.addEventListener("state", (m) => { render(JSON.parse(m.data)); pollEvents(); });
es.onerror = () => { el("g-status").textContent = "reconnecting"; };
setInterval(pollEvents, 5000);
</script>
</body>
</html>
"""
# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(run_dir: Path) -> FastAPI:
    """Dashboard app serving one run directory's receipts (read-only)."""
    run_dir = Path(run_dir)
    # Docs/OpenAPI disabled: their pages pull a CDN, and nothing may leave
    # the machine (D21).
    app = FastAPI(
        title="Fruit Fly of Wall Street — live dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/state")
    async def state() -> dict:
        return build_state(run_dir)

    @app.get("/api/events")
    async def events(
        limit: int = Query(default=EVENTS_TAIL_DEFAULT, ge=1, le=EVENTS_TAIL_MAX),
    ) -> dict:
        log = _read_events(run_dir)
        return {"events": log[-limit:], "n_total": len(log)}

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(
            sse_states(run_dir),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def sse_states(run_dir: Path) -> AsyncIterator[str]:
    """SSE frames: push a snapshot whenever a receipt file changes.

    Fixed deterministic poll interval; the first frame is emitted
    immediately on connect so a client never waits for the first push.
    """
    sent: tuple | None = None
    while True:
        fp = _fingerprint(run_dir)
        if fp != sent:
            sent = fp
            payload = json.dumps(build_state(run_dir), sort_keys=True)
            yield f"event: state\ndata: {payload}\n\n"
        await asyncio.sleep(POLL_SECONDS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _register() -> None:
    def builder(subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "dashboard",
            help="Serve the live local dashboard for a run directory (D21).",
        )
        parser.add_argument(
            "--run-dir",
            type=Path,
            required=True,
            help="Run directory with equity.csv + events.jsonl (data/runs/...).",
        )
        parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
        parser.add_argument("--port", type=int, default=8765, help="Bind port.")
        parser.set_defaults(func=_cmd_dashboard)

    register_command("dashboard", builder)


def _cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(create_app(args.run_dir), host=args.host, port=args.port, log_level="info")
    return 0


_register()
