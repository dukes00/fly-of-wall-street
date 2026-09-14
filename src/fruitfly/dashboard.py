"""T14: live local dashboard (DESIGN §12, D18/D21).

FastAPI + uvicorn + Server-Sent Events push, one static page (vanilla JS +
``<canvas>``, no build step, no npm, no JS libraries, nothing loaded
off-machine). Read-only file interface to a run's receipts — ``equity.csv``
plus ``events.jsonl`` under ``data/runs/`` — never attached to the sim
process: the server polls the receipt files on a fixed interval and serves
exactly what is on disk (D21, thin by design).

Neuron-activity view (zero-touch): the loop already persists per-encounter
neural readouts in ``events.jsonl`` — ``mbon_rate_hz`` (MBON population
firing rate), ``balance``/``raw_balance`` (the KC→MBON approach/avoid
drive) and ``valence_readout`` — so the dashboard renders neuron activity
from the events the loop emits; no per-bar spike log and no loop change.

Open positions are reconstructed by replaying ``order`` events (the loop
fills at the bar close and only ever sells whole positions, so cumulative
shares and cost basis reproduce ``avg_cost`` exactly). Per-ticker live
marks are NOT in the receipts — only aggregate equity/cash — so the table
shows entry price and weight, not unrealized P&L per name.
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

#: Recent encounters kept for the neuron-activity snapshot.
NEURON_WINDOW = 120

#: Max points in the equity chart series (evenly downsampled).
EQUITY_CHART_POINTS = 240

#: Receipt file names inside a run directory.
_EQUITY_CSV = "equity.csv"
_EVENTS_JSONL = "events.jsonl"

_STATIC_DIR = Path(__file__).with_name("static")


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



def _downsample(rows: list[list[str]], max_points: int) -> list[list]:
    """Evenly thin equity rows to <= max_points, always keeping the last."""
    if len(rows) <= max_points:
        return [[r[0], float(r[1]), float(r[2]), int(r[3])] for r in rows]
    step = -(-len(rows) // max_points)
    picked = rows[::step]
    if rows[-1] is not picked[-1]:
        picked.append(rows[-1])
    return [[r[0], float(r[1]), float(r[2]), int(r[3])] for r in picked]


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
        "positions": [],
        "mbon": None,
        "n_encounters": 0,
        "neuron_activity": {"source": "events.jsonl encounter readouts", "recent": []},
        "neuromod": None,
        "deaths": 0,
        "n_events": len(events),
        "equity_series": _downsample(equity_rows, EQUITY_CHART_POINTS),
    }

    # MBON balance summary + neuron-activity snapshot from encounter events.
    recent: list[dict] = []
    for ev in events:
        kind = ev.get("type")
        if kind == "hatch":
            state["hatch_equity"] = float(ev["hatch_equity"])
        elif kind == "encounter":
            state["n_encounters"] += 1
            state["mbon"] = {
                "ts": ev["ts"],
                "ticker": ev["ticker"],
                "balance": ev["balance"],
                "raw_balance": ev["raw_balance"],
                "valence_readout": ev["valence_readout"],
                "mbon_rate_hz": ev["mbon_rate_hz"],
            }
            recent.append(
                {
                    "ts": ev["ts"],
                    "ticker": ev["ticker"],
                    "balance": ev["balance"],
                    "mbon_rate_hz": ev["mbon_rate_hz"],
                }
            )
        elif kind == "sugar_shock":
            state["neuromod"] = {
                "ts": ev["ts"],
                "reward": ev["reward"],
                "punishment": ev["punishment"],
                "hunger": ev["hunger"],
                "arousal": ev["arousal"],
            }
        elif kind == "death":
            state["deaths"] += 1

    state["neuron_activity"]["recent"] = recent[-NEURON_WINDOW:]

    # Positions from order replay; headline equity/cash from the last row.
    qty: dict[str, int] = {}
    basis: dict[str, float] = {}
    for ev in events:
        if ev.get("type") != "order":
            continue
        ticker, shares, price = ev["ticker"], int(ev["shares"]), float(ev["price"])
        if ev["side"] == "buy":
            qty[ticker] = qty.get(ticker, 0) + shares
            basis[ticker] = basis.get(ticker, 0.0) + shares * price
        elif ev["side"] == "sell":
            qty.pop(ticker, None)
            basis.pop(ticker, None)
    state["positions"] = [
        {"ticker": t, "shares": q, "avg_cost": round(basis[t] / q, 6)}
        for t, q in sorted(qty.items())
        if q > 0
    ]

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
