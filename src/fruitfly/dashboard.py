"""T14: live local dashboard (DESIGN §12, D18/D21).

FastAPI + uvicorn + Server-Sent Events push, one static page (vanilla JS +
``<canvas>``, no build step, no npm, no JS libraries, nothing loaded
off-machine). Read-only file interface to a run's receipts — ``equity.csv``
plus ``events.jsonl`` under ``data/runs/`` — never attached to the sim
process: the server polls the receipt files on a fixed interval and serves
exactly what is on disk (D21, thin by design).

Panels, all reconstructed from receipts (unknown event types are ignored,
missing fields render as ``—``):

* **Equity** — equity curve + underwater drawdown, hatch-equity line, death
  marks.
* **Neural state** — centered vs raw KC→MBON balance with the approach/avoid
  thresholds *inferred from the decisions those balances triggered* (no
  config file is read), innate-balance anchor series when ``anchor`` events
  exist, per-day decision mix (approach/avoid/neutral/cap/dust), daily
  ``sugar_shock`` channels, encounter-intensity histogram.
* **Trading** — every ``order`` event as a trade row with per-ticker
  realized-P&L aggregation, and open positions rebuilt as FIFO lots (mark =
  last known fill price, an approximation; unrealized P&L = mark − avg cost).
* **Aggregate P&L** — cumulative realized curve, per-day realized bars,
* **Trade credits** — upcoming ``trade_credit`` events, rendered when present.
* **Event log tail**.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from fruitfly.__main__ import register_command

#: SSE poll interval (s): deterministic, file mtime+size change triggers push.
POLL_SECONDS = 1.0

#: Default/max event-log tail length for ``/api/events``.
EVENTS_TAIL_DEFAULT = 50
EVENTS_TAIL_MAX = 1000

#: Max points in downsampled time series (equity, drawdown, balance, P&L).
SERIES_MAX_POINTS = 240

#: Receipt file names inside a run directory.
_EQUITY_CSV = "equity.csv"
_EVENTS_JSONL = "events.jsonl"

#: Embedded static page (vanilla canvas; nothing loaded off-machine).
_STATIC_DIR = Path(__file__).parent / "static"

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


#: Decision ``reason`` → decision-mix bucket; anything else counts neutral.
_DECISION_BUCKETS = {
    "approach": "approach",
    "avoid": "avoid",
    "valence_flip": "avoid",
    "cap": "cap",
    "dust": "dust",
}
_DECISION_MIX_ORDER = ("approach", "avoid", "neutral", "cap", "dust")


def build_state(run_dir: Path) -> dict:
    """Snapshot of one run, reconstructed entirely from its receipt files."""
    equity_rows = _read_equity(run_dir)
    events = _read_events(run_dir)

    state: dict = {
        "run_dir": str(run_dir),
        "status": "live" if (run_dir / _EQUITY_CSV).exists() else "waiting",
        "hatch_equity": None,
        "equity": None,
        "cash": None,
        "n_positions": 0,
        "n_bars": len(equity_rows),
        "latest_ts": equity_rows[-1][0] if equity_rows else None,
        "market_value": None,
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
        "realized_series": [],
        "realized_daily": {},
    }

    decisions: dict[tuple, str] = {}  # (ts, ticker) -> reason
    encounters: list[tuple[str, str, float]] = []  # (ts, ticker, centered bal)
    mix: dict[str, dict[str, int]] = {}
    sugar: dict[str, dict] = {}
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
            state["mbon"] = {
                "ts": ts,
                "ticker": ev.get("ticker"),
                "balance": bal,
                "raw_balance": _num(ev.get("raw_balance")),
                "valence_readout": _num(ev.get("valence_readout")),
                "mbon_rate_hz": _num(ev.get("mbon_rate_hz")),
            }
            if ts is not None and bal is not None:
                encounters.append((ts, ev.get("ticker"), bal))
                state["neural"]["balance_series"].append(
                    [ts, bal, _num(ev.get("raw_balance"))]
                )
            intensity = _num(ev.get("intensity"))
            if intensity is not None:
                counts = state["neural"]["intensity_hist"]["counts"]
                counts[min(max(int(intensity * 10), 0), 9)] += 1
        elif kind == "decision":
            reason = ev.get("reason")
            if isinstance(ts, str):
                decisions[(ts, ev.get("ticker"))] = reason
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
    return state


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
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

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
