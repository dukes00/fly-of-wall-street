"""Tests for the T14 live dashboard (offline, synthetic run dir, no network).

Covers the four panel groups reconstructed from receipts by ``build_state``
(equity/drawdown, neural state, trading + FIFO positions, aggregate P&L),
schema tolerance for unknown and upcoming event types (anchor, trade_credit),
the HTTP surface, the SSE push loop, and CLI registration.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from fruitfly.dashboard import POLL_SECONDS, build_state, create_app, sse_states

# --- synthetic run directory (matches the loop's receipt contract) ----------

_EQUITY = (
    "timestamp,equity,cash,n_positions\n"
    "2026-08-18T13:30:00+00:00,100000.00,100000.00,0\n"
    "2026-08-18T13:31:00+00:00,100050.00,75050.00,1\n"
    "2026-08-18T13:32:00+00:00,100100.00,50100.00,2\n"
)
_EXTRA_ROW = "2026-08-18T13:33:00+00:00,100150.00,25150.00,2\n"


def _events() -> list[dict]:
    return [
        {"type": "hatch", "hatch_equity": 100000.0},
        {"type": "wake", "ts": "2026-08-18T13:30:00+00:00"},
        {"type": "anchor", "ts": "2026-08-18T13:30:30+00:00", "balance": 0.0125},
        {
            "type": "encounter", "ts": "2026-08-18T13:31:00+00:00", "ticker": "AAPL",
            "intensity": 0.55, "balance": 0.02, "raw_balance": 0.0325,
            "valence_readout": 1.5, "mbon_rate_hz": 12.5,
        },
        {
            "type": "decision", "ts": "2026-08-18T13:31:00+00:00", "ticker": "AAPL",
            "action": "buy", "reason": "approach",
        },
        {
            "type": "order", "ts": "2026-08-18T13:31:00+00:00", "ticker": "AAPL",
            "side": "buy", "reason": "approach", "shares": 100, "price": 250.0,
            "realized_pnl": None, "cash_after": 75050.0, "n_positions_after": 1,
        },
        {
            "type": "encounter", "ts": "2026-08-18T13:31:30+00:00", "ticker": "MSFT",
            "intensity": 0.15, "balance": -0.03, "raw_balance": -0.0425,
            "valence_readout": -1.2, "mbon_rate_hz": 9.0,
        },
        {
            "type": "decision", "ts": "2026-08-18T13:31:30+00:00", "ticker": "MSFT",
            "action": "avoid", "reason": "valence_flip",
        },
        {
            "type": "order", "ts": "2026-08-18T13:32:00+00:00", "ticker": "MSFT",
            "side": "buy", "reason": "dust", "shares": 50, "price": 500.0,
            "realized_pnl": None, "cash_after": 50100.0, "n_positions_after": 2,
        },
        {
            "type": "order", "ts": "2026-08-18T13:32:30+00:00", "ticker": "AAPL",
            "side": "sell", "reason": "cap", "shares": 40, "price": 260.0,
            "realized_pnl": 100.0, "cash_after": 60500.0, "n_positions_after": 2,
        },
        {
            "type": "trade_credit", "ts": "2026-08-18T13:32:35+00:00", "ticker": "AAPL",
            "realized_pnl": 100.0, "reward": 0.02, "punishment": 0.01,
        },
        {
            "type": "sugar_shock", "ts": "2026-08-18T20:00:00+00:00",
            "reward": 0.0185, "punishment": 0.0, "hunger": 0.0, "arousal": 0.066,
        },
        {"type": "sleep", "ts": "2026-08-18T20:00:05+00:00"},
        {"type": "cancel", "ts": "2026-08-18T20:00:10+00:00", "ticker": "MSFT"},
        {
            "type": "death", "ts": "2026-08-19T13:00:00+00:00",
            "equity": 99000.0, "hatch_equity": 100000.0,
        },
        {"type": "hatch", "hatch_equity": 100000.0},
        {"type": "hologram", "ts": "2026-08-19T13:00:01+00:00"},  # unknown type
    ]


def _write_receipts(run_dir: Path, events: list[dict]) -> None:
    (run_dir / "equity.csv").write_text(_EQUITY)
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in events)
    )


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "backtest_7_2026-08-18_2026-08-18"
    d.mkdir()
    _write_receipts(d, _events())
    return d


@pytest.fixture()
def client(run_dir: Path):
    with TestClient(create_app(run_dir)) as c:
        yield c


# --- routes -----------------------------------------------------------------


def test_index_serves_static_page(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    for panel in ("panel-equity", "panel-neural", "panel-decisions",
                  "panel-trading", "panel-pnl", "panel-positions",
                  "panel-league", "panel-events"):
        assert panel in body
    assert "<canvas" in body
    assert 'new EventSource("/api/stream")' in body


def test_state_route_200_all_panels(client: TestClient) -> None:
    r = client.get("/api/state")
    assert r.status_code == 200
    s = r.json()
    # equity panel
    assert s["equity"] == 100100.0
    assert s["cash"] == 50100.0
    assert s["hatch_equity"] == 100000.0
    assert s["n_bars"] == 3
    assert len(s["equity_series"]) == 3
    assert s["equity_series"][-1] == ["2026-08-18T13:32:00+00:00", 100100.0, 50100.0, 2]
    assert len(s["drawdown_series"]) == 3
    assert s["drawdown_series"][-1][1] == 0.0  # last row is the running peak
    assert s["death_marks"] == [
        {"ts": "2026-08-19T13:00:00+00:00", "equity": 99000.0}
    ]
    assert s["deaths"] == 1
    # neural panel
    n = s["neural"]
    assert n["balance_series"] == [
        ["2026-08-18T13:31:00+00:00", 0.02, 0.0325],
        ["2026-08-18T13:31:30+00:00", -0.03, -0.0425],
    ]
    assert n["anchor_series"] == [["2026-08-18T13:30:30+00:00", 0.0125]]
    assert n["thresholds"] == {"inferred": True, "approach": 0.02, "avoid": 0.03}
    assert n["decision_mix"] == {"2026-08-18": {
        "approach": 1, "avoid": 1, "neutral": 0, "cap": 0, "dust": 0}}
    assert n["sugar_daily"] == {"2026-08-18": {
        "reward": 0.0185, "punishment": 0.0, "hunger": 0.0, "arousal": 0.066}}
    assert n["intensity_hist"]["counts"] == [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    assert s["n_encounters"] == 2
    assert s["mbon"]["balance"] == -0.03
    assert s["mbon"]["raw_balance"] == -0.0425
    assert s["mbon"]["valence_readout"] == -1.2
    assert s["mbon"]["mbon_rate_hz"] == 9.0
    assert s["mbon"]["ticker"] == "MSFT"
    assert s["neuromod"]["reward"] == 0.0185
    # trading panel: every order is a trade row
    assert [(t["ticker"], t["side"], t["shares"], t["price"], t["reason"])
            for t in s["trades"]] == [
        ("AAPL", "buy", 100, 250.0, "approach"),
        ("MSFT", "buy", 50, 500.0, "dust"),
        ("AAPL", "sell", 40, 260.0, "cap"),
    ]
    # FIFO replay: 100 bought, 40 sold -> 60 left at avg cost 250.0
    assert s["open_positions"] == [
        {"ticker": "AAPL", "shares": 60, "avg_cost": 250.0, "mark": 260.0,
         "unrealized": 600.0},
        {"ticker": "MSFT", "shares": 50, "avg_cost": 500.0, "mark": 500.0,
         "unrealized": 0.0},
    ]
    # aggregate P&L panel
    assert s["realized_total"] == 100.0
    assert s["unrealized_total"] == 600.0
    assert s["pnl_by_ticker"] == {"AAPL": 100.0}
    assert s["realized_daily"] == {"2026-08-18": 100.0}
    assert s["realized_series"] == [["2026-08-18T13:32:30+00:00", 100.0]]
    # upcoming trade_credit events render when present
    assert s["trade_credits"] == [{
        "ts": "2026-08-18T13:32:35+00:00", "ticker": "AAPL",
        "realized_pnl": 100.0, "reward": 0.02, "punishment": 0.01,
    }]
    # event log
    assert s["n_events"] == 17


def test_unknown_event_type_is_tolerated(run_dir: Path) -> None:
    s = build_state(run_dir)
    assert s["n_events"] == 17  # counted, not crashed on


def test_missing_fields_render_as_none(run_dir: Path) -> None:
    (run_dir / "events.jsonl").write_text(
        json.dumps({"type": "encounter", "ts": "2026-08-18T13:31:00+00:00"})
        + "\n"
        + json.dumps({"type": "order", "ts": "2026-08-18T13:32:00+00:00",
                      "ticker": "X", "side": "buy", "reason": "approach"})
        + "\n"
    )
    s = build_state(run_dir)
    assert s["mbon"]["balance"] is None
    assert s["mbon"]["mbon_rate_hz"] is None
    assert s["trades"][0]["shares"] is None
    assert s["trades"][0]["price"] is None
    assert s["trades"][0]["realized_pnl"] is None


def test_positions_survive_add_and_close(run_dir: Path) -> None:
    """Buy + add + partial sell replays to the correct remaining book."""
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in [
            {"type": "order", "ts": "t1", "ticker": "AAPL", "side": "buy",
             "reason": "approach", "shares": 100, "price": 10.0,
             "realized_pnl": None},
            {"type": "order", "ts": "t2", "ticker": "AAPL", "side": "buy",
             "reason": "approach", "shares": 100, "price": 20.0,
             "realized_pnl": None},
            {"type": "order", "ts": "t3", "ticker": "AAPL", "side": "sell",
             "reason": "cap", "shares": 150, "price": 25.0,
             "realized_pnl": 1000.0},
            # oversell: consumes past the book without crashing
            {"type": "order", "ts": "t4", "ticker": "AAPL", "side": "sell",
             "reason": "cap", "shares": 100, "price": 30.0,
             "realized_pnl": None},
        ])
    )
    s = build_state(run_dir)
    # FIFO: t3 consumes the 10.0 lot + 50 of the 20.0 lot; t4's oversell
    # drains the rest of the book without crashing.
    assert s["open_positions"] == []
    assert s["realized_total"] == 1000.0
    assert s["n_positions"] == 2  # equity.csv trailing row still says 2


def test_all_routes_200(run_dir: Path) -> None:
    """Headless smoke: a real uvicorn server serves every route; the SSE
    stream's first frame is a full snapshot."""
    import threading
    import time

    import httpx
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(create_app(run_dir), host="127.0.0.1", port=0,
                       log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        base = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        for path in ("/", "/api/state", "/api/events"):
            assert httpx.get(base + path, timeout=5).status_code == 200
        with httpx.stream("GET", base + "/api/stream", timeout=5) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            chunk = next(r.iter_raw())
            assert chunk.startswith(b"event: state\ndata: ")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_events_route_tail(client: TestClient) -> None:
    r = client.get("/api/events")
    assert r.status_code == 200
    j = r.json()
    assert j["n_total"] == 17
    assert len(j["events"]) == 17  # fewer than the default tail of 50
    r2 = client.get("/api/events", params={"limit": 3})
    assert len(r2.json()["events"]) == 3
    assert r2.json()["events"][-1]["type"] == "hologram"  # unknown stays queryable
    assert client.get("/api/events", params={"limit": 0}).status_code == 422
    assert client.get("/api/events", params={"limit": 1001}).status_code == 422




# --- SSE --------------------------------------------------------------------


def _state_of(frame: str) -> dict:
    assert frame.startswith("event: state\ndata: ")
    return json.loads(frame.split("data: ", 1)[1].strip())


def test_sse_generator_emits_state_event(run_dir: Path) -> None:
    """The first SSE frame (emitted immediately on connect) is a snapshot."""
    gen = sse_states(run_dir)
    frame = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        gen.__anext__()
    )
    s = _state_of(frame)
    assert s["n_events"] == 17
    assert s["open_positions"][0]["ticker"] == "AAPL"


def test_sse_pushes_again_after_receipt_change(run_dir: Path) -> None:
    """A changed receipt file triggers a fresh push on the next poll tick."""

    async def drive() -> list[dict]:
        gen = sse_states(run_dir)
        frames = [_state_of(await gen.__anext__())]
        (run_dir / "equity.csv").write_text(_EQUITY + _EXTRA_ROW)
        frames.append(_state_of(await asyncio.wait_for(gen.__anext__(), 5)))
        await gen.aclose()
        return frames

    frames = asyncio.run(drive())
    assert frames[0]["n_bars"] == 3
    assert frames[1]["n_bars"] == 4


def test_stream_route_wired(client: TestClient) -> None:
    routes = {r.path for r in client.app.routes if hasattr(r, "path")}
    assert {"/", "/api/state", "/api/events", "/api/stream"} <= routes


# --- degradation + CLI ------------------------------------------------------


def test_missing_run_dir_degrades_to_waiting(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "nope")) as client:
        s = client.get("/api/state").json()
        assert s["status"] == "waiting"
        assert s["n_bars"] == 0
        assert s["equity_series"] == []
        assert s["trades"] == []
        assert s["open_positions"] == []
        assert client.get("/").status_code == 200
        assert client.get("/api/events").json()["events"] == []


def test_poll_interval_is_constant() -> None:
    """Deterministic fixed poll interval (D21: thin, file-polling server)."""
    assert POLL_SECONDS > 0.0



def test_dashboard_cli_registered() -> None:
    """`fruitfly dashboard --run-dir …` parses to the dashboard handler."""
    from fruitfly.__main__ import build_parser

    args = build_parser().parse_args(
        ["dashboard", "--run-dir", "data/runs/x", "--port", "8765"]
    )
    assert args.run_dir == Path("data/runs/x")
    assert args.port == 8765
    assert args.func.__name__ == "_cmd_dashboard"
