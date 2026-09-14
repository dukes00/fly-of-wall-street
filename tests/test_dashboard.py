"""Tests for the T14 live dashboard (offline, synthetic run dir, no network).

The dashboard is a read-only file interface to run receipts, so the tests
point ``create_app`` at a tiny synthetic ``data/runs/...`` directory (same
receipt contract as ``loop.run_backtest``) and drive it with starlette's
httpx-backed TestClient. Verified: every route answers 200, the five
panels' data fields carry the expected reconstructed values (equity/cash
from equity.csv, positions replayed from order events, MBON balance from
the latest encounter, neuron-activity window, event tail), the SSE
generator pushes state frames (including after a receipt file changes),
and a missing run directory degrades to a ``waiting`` snapshot.

The SSE generator is tested directly rather than over HTTP: starlette 1.6's
TestClient transport buffers the entire response body before returning, so
an infinite SSE stream can never be read through it. The generator is what
``/api/stream`` wraps verbatim; the real socket path is exercised by the
CLI smoke run (``python -m fruitfly dashboard`` + curl), not by tests.
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
        {
            "type": "encounter", "ts": "2026-08-18T13:31:00+00:00", "ticker": "AAPL",
            "intensity": 0.5, "balance": 0.02, "raw_balance": 0.03,
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
            "type": "order", "ts": "2026-08-18T13:32:00+00:00", "ticker": "MSFT",
            "side": "buy", "reason": "approach", "shares": 50, "price": 500.0,
            "realized_pnl": None, "cash_after": 50100.0, "n_positions_after": 2,
        },
        {
            "type": "sugar_shock", "ts": "2026-08-18T20:00:00+00:00",
            "reward": 0.0185, "punishment": 0.0, "hunger": 0.0, "arousal": 0.066,
        },
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
    for panel in ("panel-equity", "panel-positions", "panel-mbon",
                  "panel-neuron", "panel-events"):
        assert panel in body
    assert "<canvas" in body
    assert 'new EventSource("/api/stream")' in body


def test_state_route_200_and_five_panels(client: TestClient) -> None:
    r = client.get("/api/state")
    assert r.status_code == 200
    s = r.json()
    # equity panel
    assert s["equity"] == 100100.0
    assert s["cash"] == 50100.0
    assert s["hatch_equity"] == 100000.0
    assert s["n_bars"] == 3
    assert len(s["equity_series"]) == 3
    # positions panel (reconstructed from order events)
    assert s["n_positions"] == 2
    assert s["positions"] == [
        {"ticker": "AAPL", "shares": 100, "avg_cost": 250.0},
        {"ticker": "MSFT", "shares": 50, "avg_cost": 500.0},
    ]
    # MBON balance panel (latest encounter)
    assert s["mbon"]["balance"] == 0.02
    assert s["mbon"]["raw_balance"] == 0.03
    assert s["mbon"]["valence_readout"] == 1.5
    assert s["mbon"]["mbon_rate_hz"] == 12.5
    assert s["mbon"]["ticker"] == "AAPL"
    assert s["n_encounters"] == 1
    # neuron-activity snapshot
    assert s["neuron_activity"]["recent"] == [
        {"ts": "2026-08-18T13:31:00+00:00", "ticker": "AAPL",
         "balance": 0.02, "mbon_rate_hz": 12.5}
    ]
    # event log panel feed
    assert s["n_events"] == 7
    assert s["deaths"] == 0
    assert s["neuromod"]["reward"] == 0.0185


def test_positions_survive_add_and_close(run_dir: Path) -> None:
    """Buy + add + full sell replays to the correct remaining book."""
    events = _events() + [
        {
            "type": "order", "ts": "2026-08-18T13:33:00+00:00", "ticker": "AAPL",
            "side": "buy", "reason": "add", "shares": 100, "price": 300.0,
            "realized_pnl": None, "cash_after": 20000.0, "n_positions_after": 2,
        },
        {
            "type": "order", "ts": "2026-08-18T13:34:00+00:00", "ticker": "MSFT",
            "side": "sell", "reason": "avoid", "shares": 50, "price": 510.0,
            "realized_pnl": 500.0, "cash_after": 45500.0, "n_positions_after": 1,
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in events)
    )
    s = build_state(run_dir)
    # AAPL: (100*250 + 100*300) / 200 = 275; MSFT fully closed
    assert s["positions"] == [{"ticker": "AAPL", "shares": 200, "avg_cost": 275.0}]
    assert s["n_positions"] == 2  # equity.csv trailing row still says 2


def test_events_route_tail(client: TestClient) -> None:
    r = client.get("/api/events")
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == 7
    assert len(body["events"]) == 7  # default limit above log length
    r = client.get("/api/events", params={"limit": 2})
    body = r.json()
    assert [e["type"] for e in body["events"]] == ["order", "sugar_shock"]
    assert client.get("/api/events", params={"limit": 0}).status_code == 422
    assert client.get("/api/events", params={"limit": 1001}).status_code == 422


# --- SSE --------------------------------------------------------------------


def _state_of(frame: str) -> dict:
    assert frame.startswith("event: state\ndata: ")
    return json.loads(frame.split("data: ", 1)[1].strip())


def test_sse_generator_emits_state_event(run_dir: Path) -> None:
    """The first SSE frame (emitted immediately on connect) is a snapshot."""

    async def first_frame() -> str:
        gen = sse_states(run_dir)
        frame = await asyncio.wait_for(gen.__anext__(), timeout=5.0)
        await gen.aclose()
        return frame

    s = _state_of(asyncio.run(first_frame()))
    assert s["equity"] == 100100.0
    assert s["positions"][0]["ticker"] == "AAPL"


def test_sse_pushes_again_after_receipt_change(run_dir: Path) -> None:
    """A changed receipt file triggers a fresh push on the next poll tick."""

    async def two_frames() -> list[dict]:
        gen = sse_states(run_dir)
        frames = [_state_of(await asyncio.wait_for(gen.__anext__(), timeout=5.0))]
        (run_dir / "equity.csv").write_text(_EQUITY + _EXTRA_ROW)
        frames.append(
            _state_of(
                await asyncio.wait_for(
                    gen.__anext__(), timeout=2 * POLL_SECONDS + 5.0
                )
            )
        )
        await gen.aclose()
        return frames

    frames = asyncio.run(two_frames())
    assert frames[0]["n_bars"] == 3
    assert frames[1]["n_bars"] == 4


def test_stream_route_wired(client: TestClient) -> None:
    routes = {r.path for r in client.app.routes if hasattr(r, "path")}
    assert {"/", "/api/state", "/api/events", "/api/stream"} <= routes


# --- degradation + CLI ------------------------------------------------------


def test_missing_run_dir_degrades_to_waiting(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "nope")) as client:
        assert client.get("/api/state").status_code == 200
        s = client.get("/api/state").json()
        assert s["status"] == "waiting"
        assert s["equity"] is None and s["positions"] == []
        assert client.get("/api/events").status_code == 200
        assert client.get("/api/events").json()["events"] == []


def test_poll_interval_is_constant() -> None:
    """Deterministic fixed poll interval (D21: thin, file-polling server)."""
    assert POLL_SECONDS > 0.0


def test_dashboard_cli_registered() -> None:
    from fruitfly.__main__ import main

    with pytest.raises(SystemExit) as excinfo:
        main(["dashboard", "--help"])
    assert excinfo.value.code == 0
