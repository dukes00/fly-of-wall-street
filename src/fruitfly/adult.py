"""T15: the adult run harness (DESIGN §7 daily loop, §9 adult stage).

The adult fly is the larval backtest loop let loose: it forages across
trading days indefinitely — wake at the NYSE open (D7), one encounter per
1-minute bar, settle sugar/shock -> sleep at the close (D7) — and it must
survive its operator: the process can be SIGKILLed mid-run and resume from a
persistent state file with the receipts left consistent.

**Architecture (dependency injection at the two seams that differ between
replay and live-paper).** Everything else — the per-bar foraging cycle, the
decision rules, the paper accounting, the death/hatch machinery — is ONE code
path shared by both modes:

- **Data seam** (:class:`ReplayFeed` / :class:`LivePaperFeed`): produces
  ``(timestamp, frames)`` pairs. Replay iterates the recorded parquet cache
  (identical construction to ``loop.run_backtest``); live polls real 1-minute
  bars per session minute, gated on the NYSE calendar helpers.
- **Broker seam** (:class:`SimulatedExecution` / :class:`LiveExecution`):
  turns a decision into a fill. Replay runs the loop's pessimistic
  next-bar fill model (:class:`~fruitfly.loop.PendingOrderBook`: orders
  decided on bar t fill on the symbol's next available bar, BUY at that
  bar's HIGH, SELL at its LOW; ``fill_mode="close"`` restores the legacy
  look-ahead-biased same-bar close fill); live submits an
  :class:`~fruitfly.broker.OrderEvent` to the paper-only
  :class:`~fruitfly.broker.AlpacaPaperBroker` and polls the actual fill
  (unchanged by the fill model).

Cash/position accounting stays in the driver (single book of record, same
arithmetic as the loop), so replay receipts are byte-identical to
``run_backtest`` receipts for the same seed/window/config — same files
(``equity.csv`` + ``events.jsonl``), same columns/fields, same event types.
The dashboard (D18/D21) and post-mortem read adult runs unchanged.

**Loop seams reused (documented, not forked).** The per-bar cycle calls
``loop.py``'s pure helpers rather than re-deriving them:
``_features_at`` (feature builder), ``_plume_set`` (plume filter, D3),
``_decide`` (decision map), ``_innate_balance`` (hatch calibration sniff),
``_size_fraction`` (sizing), plus the gain/threshold constants (``SMELL_GAIN``,
``VISION_WINDOW_BARS``, ``NOISE_SIGMA_MV``, ...) and the ``_Position`` book
record. Death parity (v0.6): the death bar clears the per-exit eligibility
snapshots before liquidation (no trade_credit on death) and emits exactly one
``hatch`` after the brain reset, then re-calibrates the anchor — same order as
loop.py.

**Persistence (resume semantics).** :class:`FlyState` captures every mutable
byte of the fly — cash/hatch equity/positions/last marks, hunger and hatch
state, daily accumulators, foraging pointer, innate-balance anchor, plasticity
(weights + eligibility + habituation), LIF sim state, the loop RNG's
bit-generator state, receipt cursors, and the last processed bar — and is
written atomically (tmp file + ``os.replace``, fsync'd) every ``persist_every``
bars. On restart the receipts are truncated back to the persisted cursor
(dropping any torn/partial lines written after the last state save) and the
run continues from the next bar, appending to the same equity.csv/events.jsonl.

Determinism guarantee: because the state file captures the RNG and sim state
too, a resumed run is **byte-identical** to an uninterrupted run of the same
seed/config — same guarantee as the same-seed double-run gate. This covers
resume exactly; live-paper mode is inherently non-deterministic (real fills)
and makes no such claim. The state file records the chassis fingerprint
(:func:`fruitfly.train.chassis_fingerprint`) and the larval artifact's
fingerprint (:func:`fruitfly.train.load_larval_weights` guard): a state file
or artifact trained on a different brain is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fruitfly.__main__ import register_command
from fruitfly.broker import AlpacaPaperBroker, OrderEvent
from fruitfly.data import BASKET, is_regular_session, load_bars
from fruitfly.loop import (
    APPROACH_THR,
    AVOID_THR,
    DEATH_THRESHOLD,
    DEFAULT_FILL_MODE,
    DT_MS,
    FILL_MODE_CLOSE,
    FILL_MODES,
    HUNGER_DRAWDOWN,
    INITIAL_CASH,
    NOISE_SIGMA_MV,
    SHOCK_ADVERSE_PCT,
    SMELL_GAIN,
    VISION_WINDOW_BARS,
    BacktestConfig,
    PendingOrder,
    PendingOrderBook,
    _decide,
    _features_at,
    _innate_balance,
    _plume_set,
    _Position,
    _size_fraction,
    _window,
    resolve_chassis_fields,
)
from fruitfly.neuromod import NeuromodState, Plasticity
from fruitfly.senses import encode_smell, encode_taste, encode_vision
from fruitfly.senses.smell import upn_channels
from fruitfly.sim import LIFSim
from fruitfly.train import chassis_fingerprint, load_larval_weights

__all__ = [
    "AdultConfig",
    "AdultResult",
    "AdultRun",
    "FlyState",
    "LiveExecution",
    "LivePaperFeed",
    "ReplayFeed",
    "SimulatedExecution",
    "next_session_open",
    "run_replay",
]

#: Root directory for adult runs (runtime state; data/ is gitignored).
ADULT_RUNS_DIR = Path("data/adult")

#: State file name inside a run directory.
STATE_FILE = "state.npz"

#: Live-feed warmup: two vision windows of 1-minute history per symbol.
_LIVE_WARMUP_BARS = 2 * VISION_WINDOW_BARS


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdultConfig:
    """One adult run. Trading parameters mirror :class:`loop.BacktestConfig`
    exactly (the defaults ARE the loop's documented constants), plus the
    harness knobs: mode, persistence cadence, and the larval artifact."""

    seed: int
    start: str
    end: str
    #: ``"replay"`` (recorded bars, simulated fills) or ``"live-paper"``
    #: (AlpacaPaperBroker + clock/calendar wake-sleep per D7).
    mode: str = "replay"
    #: Run directory (receipts + state file). Default under ADULT_RUNS_DIR.
    run_dir: str | Path | None = None
    #: Persist the state file every N bars (also at every day settle).
    persist_every: int = 1
    #: Optional larval artifact (T9): initial KC→MBON weights, verified
    #: against the chassis fingerprint before injection.
    larval_weights: str | Path | None = None
    top_k: int = 5
    grudge_top_k: int = 16
    #: Structural looming drive weight (DESIGN v0.6 §5): ``balance_used =
    #: centered + lambda_struct × structural_score`` (mirror of the loop).
    lambda_struct: float = 0.05
    position_cap: int = 10
    ms_per_bar: float = 500.0
    death_threshold: float = DEATH_THRESHOLD
    fees: float = 0.0
    fill_mode: str = DEFAULT_FILL_MODE
    initial_cash: float = INITIAL_CASH
    shock_adverse_pct: float = SHOCK_ADVERSE_PCT
    hunger_drawdown: float = HUNGER_DRAWDOWN
    approach_thr: float = APPROACH_THR
    avoid_thr: float = AVOID_THR
    noise_sigma_mv: float = NOISE_SIGMA_MV
    dt_ms: float = DT_MS
    #: Brain chassis + opt-in STD (mirror of ``loop.BacktestConfig``; the
    #: whole-fly chassis defaults to the calibrated T12b STD parameters).
    chassis: str = "stripped"
    std_beta: float | None = None
    std_tau_rec_ms: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("replay", "live-paper"):
            raise ValueError(f"mode must be 'replay' or 'live-paper', got {self.mode!r}")
        if self.persist_every < 1:
            raise ValueError("persist_every must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.grudge_top_k < 1:
            raise ValueError("grudge_top_k must be >= 1")
        if self.position_cap < 1:
            raise ValueError("position_cap must be >= 1")
        if self.ms_per_bar <= 0:
            raise ValueError("ms_per_bar must be positive")
        if not -1.0 < self.death_threshold < 0.0:
            raise ValueError("death_threshold must be in (-1, 0)")
        if self.fees < 0:
            raise ValueError("fees must be non-negative")
        if self.fill_mode not in FILL_MODES:
            raise ValueError(
                f"fill_mode must be one of {FILL_MODES}, got {self.fill_mode!r}"
            )
        chassis, std_beta, std_tau_rec_ms = resolve_chassis_fields(
            self.chassis, self.std_beta, self.std_tau_rec_ms
        )
        object.__setattr__(self, "chassis", chassis)
        object.__setattr__(self, "std_beta", std_beta)
        object.__setattr__(self, "std_tau_rec_ms", std_tau_rec_ms)

    def backtest_config(self, out_dir: str | Path | None = None) -> BacktestConfig:
        """The equivalent larval :class:`BacktestConfig` (parity reference)."""
        return BacktestConfig(
            seed=self.seed,
            start=self.start,
            end=self.end,
            top_k=self.top_k,
            grudge_top_k=self.grudge_top_k,
            lambda_struct=self.lambda_struct,
            position_cap=self.position_cap,
            ms_per_bar=self.ms_per_bar,
            death_threshold=self.death_threshold,
            fees=self.fees,
            fill_mode=self.fill_mode,
            initial_cash=self.initial_cash,
            shock_adverse_pct=self.shock_adverse_pct,
            hunger_drawdown=self.hunger_drawdown,
            approach_thr=self.approach_thr,
            avoid_thr=self.avoid_thr,
            noise_sigma_mv=self.noise_sigma_mv,
            dt_ms=self.dt_ms,
            chassis=self.chassis,
            std_beta=self.std_beta,
            std_tau_rec_ms=self.std_tau_rec_ms,
            out_dir=out_dir,
        )

    def run_directory(self) -> Path:
        if self.run_dir is not None:
            return Path(self.run_dir)
        stem = f"adult_{self.seed}_{self.start}_{self.end}"
        return ADULT_RUNS_DIR / stem.replace(":", "")

    def _echo(self) -> dict:
        """Config fields recorded in the state file (resume must match)."""
        return {
            "seed": self.seed,
            "mode": self.mode,
            "start": self.start,
            "end": self.end,
            "top_k": self.top_k,
            "grudge_top_k": self.grudge_top_k,
            "lambda_struct": self.lambda_struct,
            "position_cap": self.position_cap,
            "ms_per_bar": self.ms_per_bar,
            "death_threshold": self.death_threshold,
            "fees": self.fees,
            "fill_mode": self.fill_mode,
            "initial_cash": self.initial_cash,
            "shock_adverse_pct": self.shock_adverse_pct,
            "hunger_drawdown": self.hunger_drawdown,
            "approach_thr": self.approach_thr,
            "avoid_thr": self.avoid_thr,
            "noise_sigma_mv": self.noise_sigma_mv,
            "dt_ms": self.dt_ms,
            "chassis": self.chassis,
            "std_beta": self.std_beta,
            "std_tau_rec_ms": self.std_tau_rec_ms,
        }


@dataclass(frozen=True)
class AdultResult:
    """Summary of a finished (or interrupted-then-resumed) adult run."""

    run_dir: Path
    resumed: bool
    n_bars: int
    n_events: int
    n_orders: int
    n_deaths: int
    final_equity: float
    wall_s: float
    #: Final KC→MBON weights of the last-hatched fly.
    final_weights: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Persistent fly state
# ---------------------------------------------------------------------------

_ARRAY_MEMBERS = (
    "weights",
    "eligibility",
    "habituation",
    "daily_spikes",
    "daily_mbon_drive",
    "sim_v",
    "sim_refr",
    "sim_prev",
    "sim_spikes",
)


@dataclass
class FlyState:
    """Every mutable byte of the adult fly (see module docstring).

    Serialized as one ``state.npz``: arrays verbatim, everything else in a
    single JSON member. Writes are atomic (tmp + ``os.replace`` + fsync), so a
    SIGKILL mid-write leaves the previous state intact.
    """

    # Identity guards: the state belongs to one brain and one constitution.
    chassis_fingerprint: str
    config_echo: dict
    larval_fingerprint: str | None = None

    # Book (loop paper accounting).
    cash: float = 0.0
    hatch_equity: float = 0.0
    equity: float = 0.0
    #: Per-exit dopamine (DESIGN v0.6 D6): ticker -> eligibility snapshot
    #: captured at the opening buy/add fill, credited at close.
    entry_elig: dict[str, np.ndarray] = field(default_factory=dict)
    positions: dict[str, _Position] = field(default_factory=dict)
    last_close: dict[str, float] = field(default_factory=dict)
    #: Queued pessimistic next-bar orders (loop parity: ``PendingOrderBook``).
    pending: list = field(default_factory=list)

    # Foraging state.
    cur_day: str | None = None
    pointer: int = 0
    pointer_drawn: bool = False
    anchor: float = 0.0

    # Daily accumulators (reset by settle_and_sleep).
    daily_spikes: np.ndarray | None = None
    daily_mbon_drive: np.ndarray | None = None
    daily_signal_encounters: int = 0
    day_realized: float = 0.0
    day_abs_ret_sum: float = 0.0
    day_abs_ret_n: int = 0
    hunger_armed: bool = True

    # Plasticity + engine state.
    weights: np.ndarray | None = None
    eligibility: np.ndarray | None = None
    habituation: np.ndarray | None = None
    sim_v: np.ndarray | None = None
    sim_refr: np.ndarray | None = None
    sim_prev: np.ndarray | None = None
    sim_spikes: np.ndarray | None = None
    rng_state: dict | None = None

    # Receipt cursors + counters.
    n_bars: int = 0
    n_events: int = 0
    n_orders: int = 0
    n_deaths: int = 0
    #: ISO timestamp of the last persisted bar (resume skips bars <= this).
    last_ts: str | None = None

    # ------------------------------------------------------------------ disk
    def save(self, path: Path) -> None:
        """Write the state atomically: tmp file, fsync, ``os.replace``."""
        path = Path(path)
        payload = {
            "chassis_fingerprint": self.chassis_fingerprint,
            "config_echo": self.config_echo,
            "larval_fingerprint": self.larval_fingerprint,
            "cash": self.cash,
            "hatch_equity": self.hatch_equity,
            "equity": self.equity,
            "positions": {
                t: {"shares": p.shares, "avg_cost": p.avg_cost}
                for t, p in self.positions.items()
            },
            "last_close": self.last_close,
            "pending": [
                {
                    "decision_ts": o.decision_ts.isoformat(),
                    "ticker": o.ticker,
                    "side": o.side,
                    "reason": o.reason,
                    "shares": o.shares,
                }
                for o in self.pending
            ],
            "entry_elig": {
                t: np.asarray(a, dtype=np.float64).tolist()
                for t, a in self.entry_elig.items()
            },
            "cur_day": self.cur_day,
            "pointer": self.pointer,
            "pointer_drawn": self.pointer_drawn,
            "anchor": self.anchor,
            "daily_signal_encounters": self.daily_signal_encounters,
            "day_realized": self.day_realized,
            "day_abs_ret_sum": self.day_abs_ret_sum,
            "day_abs_ret_n": self.day_abs_ret_n,
            "hunger_armed": self.hunger_armed,
            "rng_state": self.rng_state,
            "n_bars": self.n_bars,
            "n_events": self.n_events,
            "n_orders": self.n_orders,
            "n_deaths": self.n_deaths,
            "last_ts": self.last_ts,
        }
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                json=np.array(json.dumps(payload, sort_keys=True)),
                **{name: np.ascontiguousarray(getattr(self, name)) for name in _ARRAY_MEMBERS},
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:  # fsync the directory so the rename itself is durable
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:  # pragma: no cover - best effort on exotic filesystems
            pass

    @classmethod
    def load(cls, path: Path) -> FlyState:
        """Read a state file written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            payload = json.loads(str(z["json"].item()))
            arrays = {name: np.asarray(z[name]) for name in _ARRAY_MEMBERS}
        state = cls(
            chassis_fingerprint=payload["chassis_fingerprint"],
            config_echo=payload["config_echo"],
            larval_fingerprint=payload["larval_fingerprint"],
        )
        state.cash = payload["cash"]
        state.hatch_equity = payload["hatch_equity"]
        state.equity = payload["equity"]
        state.positions = {
            t: _Position(shares=int(p["shares"]), avg_cost=float(p["avg_cost"]))
            for t, p in payload["positions"].items()
        }
        state.last_close = {t: float(v) for t, v in payload["last_close"].items()}
        state.pending = [
            PendingOrder(
                decision_ts=pd.Timestamp(o["decision_ts"]),
                ticker=o["ticker"],
                side=o["side"],
                reason=o["reason"],
                shares=None if o["shares"] is None else int(o["shares"]),
            )
            for o in payload["pending"]
        ]
        state.entry_elig = {
            t: np.asarray(a, dtype=np.float64) for t, a in payload["entry_elig"].items()
        }
        state.cur_day = payload["cur_day"]
        state.pointer = payload["pointer"]
        state.pointer_drawn = payload["pointer_drawn"]
        state.anchor = payload["anchor"]
        state.daily_signal_encounters = payload["daily_signal_encounters"]
        state.day_realized = payload["day_realized"]
        state.day_abs_ret_sum = payload["day_abs_ret_sum"]
        state.day_abs_ret_n = payload["day_abs_ret_n"]
        state.hunger_armed = payload["hunger_armed"]
        state.rng_state = payload["rng_state"]
        state.n_bars = payload["n_bars"]
        state.n_events = payload["n_events"]
        state.n_orders = payload["n_orders"]
        state.n_deaths = payload["n_deaths"]
        state.last_ts = payload["last_ts"]
        for name in _ARRAY_MEMBERS:
            setattr(state, name, arrays[name])
        return state


# ---------------------------------------------------------------------------
# The two adapter seams (replay vs live-paper)
# ---------------------------------------------------------------------------


class ReplayFeed:
    """Data seam, replay mode: recorded parquet cache, backtest semantics.

    ``frames`` is built exactly like ``loop.run_backtest`` (full cache history
    for feature warmup, symbols with at least one bar on/after the start) and
    ``bars()`` yields the sorted union of bar timestamps in the window.
    """

    def __init__(self, config: AdultConfig) -> None:
        self._config = config
        self._frames: dict[str, pd.DataFrame] | None = None

    def frames(self) -> dict[str, pd.DataFrame]:
        if self._frames is None:
            start_ts, _ = _window(self._config.start, self._config.end)
            frames = {
                sym: df
                for sym, df in load_bars(list(BASKET), None, None).items()
                if len(df) and df.index[-1] >= start_ts
            }
            if not frames:
                raise ValueError(f"no cached bars on or after {self._config.start}")
            self._frames = frames
        return self._frames

    def bars(self):
        start_ts, end_ts = _window(self._config.start, self._config.end)
        frames = self.frames()
        timeline = sorted(
            {ts for df in frames.values() for ts in df.index if start_ts <= ts <= end_ts}
        )
        if not timeline:
            raise ValueError(
                f"no bars in [{self._config.start}, {self._config.end}] for the basket"
            )
        for ts in timeline:
            yield ts, frames


class SimulatedExecution:
    """Broker seam, replay mode: fills through the loop's fill engine.

    With the default ``fill_mode="pessimistic_next_bar"`` the seam owns a
    :class:`~fruitfly.loop.PendingOrderBook` over the replay frames
    (:attr:`book`): the driver queues decided orders on it and drains the
    pessimistic next-bar fills (BUY at the execution bar's HIGH, SELL at its
    LOW, gaps/halts skipped) on later bars — the same model as
    ``loop.run_backtest``, so replay receipts stay byte-identical.
    ``fill_mode="close"`` restores the legacy same-bar close fill
    (documented look-ahead-biased escape hatch).
    """

    def __init__(
        self,
        frames: dict[str, pd.DataFrame] | None = None,
        fill_mode: str = DEFAULT_FILL_MODE,
    ) -> None:
        self.fill_mode = fill_mode
        self._book = PendingOrderBook(frames or {}, fill_mode)

    @property
    def book(self) -> PendingOrderBook:
        """The pending-order queue (pending mode; unused in close mode)."""
        return self._book

    def execute(
        self,
        ts: pd.Timestamp,
        ticker: str,
        side: str,
        shares: int,
        reference_price: float,
    ) -> float | None:
        """Immediate fill at the reference price — used only in close mode
        (the pending model routes orders through :attr:`book` instead)."""
        return reference_price


class LivePaperFeed:
    """Data seam, live-paper mode: the D7 clock.

    Sleeps until the next NYSE open (via :func:`next_session_open`), then
    yields one bar per session minute as its minute closes, fetching real
    1-minute bars (yfinance, session-filtered) with enough trailing history
    for the vision window and feature warmup. Between close and the next open
    it yields nothing — the fly sleeps in the aftermarket (D7).
    """

    def __init__(
        self,
        basket: list[str] | None = None,
        warmup_bars: int = _LIVE_WARMUP_BARS,
        poll_s: float = 5.0,
    ) -> None:
        self.basket = list(basket or BASKET)
        self.warmup_bars = warmup_bars
        self.poll_s = poll_s
        self._frames: dict[str, pd.DataFrame] | None = None

    def _fetch(self, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        import yfinance as yf  # deferred: keeps the replay path import-light

        frames: dict[str, pd.DataFrame] = {}
        for sym in self.basket:
            try:
                raw = yf.Ticker(sym).history(period="5d", interval="1m")
            except Exception:  # transient feed errors: skip the symbol this bar
                continue
            if raw.empty:
                continue
            idx = raw.index.tz_convert("UTC")
            raw.index = idx
            mask = np.fromiter(
                (is_regular_session(ts) and ts <= now for ts in idx),
                dtype=bool,
                count=len(idx),
            )
            df = raw.loc[mask, ["open", "high", "low", "close", "volume"]]
            if df.empty:
                continue
            frames[sym] = df.tail(self.warmup_bars)
        return frames

    def frames(self) -> dict[str, pd.DataFrame]:
        """Warmup snapshot at run start (hatch calibration sniff input)."""
        if self._frames is None:
            self._frames = self._fetch(pd.Timestamp.now(tz="UTC"))
        return self._frames

    def bars(self):
        import time as _time

        while True:
            now = pd.Timestamp.now(tz="UTC")
            open_ts = next_session_open(now)
            if now < open_ts:
                _time.sleep(max(0.0, (open_ts - now).total_seconds()) + 2.0)
            minute = next_session_open(pd.Timestamp.now(tz="UTC"))
            session_close = pd.Timestamp(minute.date(), tz="UTC") + pd.Timedelta(hours=20)
            while minute < session_close:
                target = minute + pd.Timedelta(minutes=1)
                now = pd.Timestamp.now(tz="UTC")
                if now < target:
                    _time.sleep((target - now).total_seconds())
                frames = self._fetch(minute)
                if frames:
                    yield minute, frames
                minute += pd.Timedelta(minutes=1)
            # 20:00 UTC: settle/sleep happens at the next day boundary in the
            # driver's cycle; the feed simply goes quiet until the next open.


class LiveExecution:
    """Broker seam, live-paper mode: market orders on the paper account.

    Submits an :class:`~fruitfly.broker.OrderEvent` to the paper-only
    :class:`~fruitfly.broker.AlpacaPaperBroker` and polls until the order
    reaches a terminal state, returning the actual average fill price (or
    ``None`` when the order was rejected/cancelled/timed out — the driver
    then keeps its book unchanged).
    """

    def __init__(self, broker: AlpacaPaperBroker, poll_s: float = 1.0, timeout_s: float = 60.0):
        self.broker = broker
        self.poll_s = poll_s
        self.timeout_s = timeout_s

    def execute(
        self,
        ts: pd.Timestamp,
        ticker: str,
        side: str,
        shares: int,
        reference_price: float,
    ) -> float | None:
        order_id = self.broker.submit(
            OrderEvent(symbol=ticker, side=side, qty=shares, tif="day", type="market")
        )
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            order = self.broker._client.get_order_by_id(order_id)
            status = order.status.value
            if status == "filled":
                return float(order.filled_avg_price)
            if status in ("canceled", "cancelled", "expired", "rejected"):
                return None
            time.sleep(self.poll_s)
        self.broker.cancel(order_id)  # stale: pull the order, skip the fill
        return None


def next_session_open(after: pd.Timestamp) -> pd.Timestamp:
    """The session open governing ``after`` (D7 wake): the open of the
    regular session containing ``after``, else the next NYSE open.

    Uses the shared calendar helper :func:`fruitfly.data.is_regular_session`
    (weekday + holiday + 13:30-20:00 UTC window), so weekend and holiday
    probes are skipped and the answer lands on a real open.
    """
    ts = after.tz_localize("UTC") if after.tzinfo is None else after.tz_convert("UTC")
    if is_regular_session(ts):
        return pd.Timestamp(ts.date(), tz="UTC") + pd.Timedelta(hours=13, minutes=30)
    day = ts.date()
    for _ in range(400):  # ~14 months of calendar before giving up
        candidate = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=13, minutes=30)
        if candidate > ts and is_regular_session(candidate):
            return candidate
        day += timedelta(days=1)
    raise RuntimeError("no NYSE session open found within 400 days")


# ---------------------------------------------------------------------------
# Receipts helpers (shape parity with loop.run_backtest)
# ---------------------------------------------------------------------------


def _truncate_lines(path: Path, keep: int) -> None:
    """Keep the first ``keep`` lines of ``path`` (drops torn/extra tail lines).

    Resume hygiene: the state file is persisted after the bar's receipts are
    flushed, so the files on disk are always at least as long as the cursor;
    anything longer is a torn write from the kill and is dropped here.
    """
    lines = path.read_text().splitlines(keepends=True)
    if len(lines) > keep:
        with open(path, "w") as f:
            f.writelines(lines[:keep])


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


class AdultRun:
    """One persistent adult fly driven bar-by-bar through the loop machinery.

    Fresh start: hatch calibration sniff -> initial ``hatch`` event -> bars.
    Resume (state file present): state restored, receipts truncated to the
    persisted cursor, run continues from the next bar with identical bytes.
    """

    def __init__(
        self,
        config: AdultConfig,
        *,
        feed=None,
        execution=None,
    ) -> None:
        self._cfg = config
        self._feed = feed
        self._execution = execution

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> AdultResult:
        t0 = time.perf_counter()
        cfg = self._cfg
        chassis = _resolve_chassis(cfg)
        self._chassis = chassis
        fingerprint = chassis_fingerprint(chassis)

        pop = chassis.nodes["population"].to_numpy()
        self._upn_rows, self._channels = upn_channels(chassis)
        self._mbon_rows = np.flatnonzero(pop == "MBON")
        self._n = chassis.n_neurons

        run_dir = cfg.run_directory()
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / STATE_FILE
        resumed = state_path.exists()

        if resumed:
            st = FlyState.load(state_path)
            if st.chassis_fingerprint != fingerprint:
                raise ValueError(
                    "state file was written on a different chassis "
                    f"(state {st.chassis_fingerprint[:12]}… != current {fingerprint[:12]}…)"
                )
            if st.config_echo != cfg._echo():
                raise ValueError(
                    "state file config mismatch: refusing to resume a fly with a "
                    f"changed constitution.\nstate: {st.config_echo}\nnow:    {cfg._echo()}"
                )
            for name in ("equity.csv", "events.jsonl"):
                if not (run_dir / name).exists():
                    raise ValueError(
                        f"state file exists but {name} is missing from {run_dir}; "
                        "the receipts and state must be a pair"
                    )
        else:
            st = FlyState(chassis_fingerprint=fingerprint, config_echo=cfg._echo())
        self._st = st

        # --- receipts (same files, same columns/fields as a backtest run) ---
        equity_path = run_dir / "equity.csv"
        events_path = run_dir / "events.jsonl"
        if resumed:
            _truncate_lines(equity_path, 1 + st.n_bars)
            _truncate_lines(events_path, st.n_events)
            equity_file = equity_path.open("a", newline="")
            events_file = events_path.open("a")
        else:
            equity_file = equity_path.open("w", newline="")
            events_file = events_path.open("w")
        self._equity_file = equity_file
        self._events_file = events_file
        self._equity_writer = csv.writer(equity_file, lineterminator="\n")
        if not resumed:
            self._equity_writer.writerow(["timestamp", "equity", "cash", "n_positions"])

        # --- brain (fresh build, then state restore on resume) --------------
        self._mbon_rows = np.flatnonzero(pop == "MBON")
        # Structural looming view (DESIGN v0.6 §5): the LC→MBON connectome
        # synapses, log1p-scaled to mean 1 over their nonzero support (the
        # ``Plasticity.baseline`` convention); None when the chassis has no
        # LC→MBON path. Built exactly like loop.run_backtest.
        lc_rows = np.flatnonzero(pop == "LC-looming")
        w_struct: np.ndarray | None = None
        if lc_rows.size:
            w_lc = np.asarray(
                chassis.adj[lc_rows][:, self._mbon_rows].toarray(), dtype=np.float64
            )
            lc_support = w_lc > 0.0
            if lc_support.any():
                w_lc = np.log1p(w_lc)
                w_lc /= w_lc[lc_support].mean()
                w_struct = w_lc
        self._lc_rows = lc_rows
        self._w_struct = w_struct
        sim = LIFSim(
            chassis,
            dt_ms=cfg.dt_ms,
            seed=cfg.seed,
            std_beta=cfg.std_beta,
        )
        plasticity = Plasticity(chassis, top_k=cfg.grudge_top_k)
        if resumed:
            plasticity.weights[...] = st.weights
            plasticity.eligibility[...] = st.eligibility
            plasticity.habituation[...] = st.habituation
            # Engine state restore (documented seam: LIFSim exposes no public
            # snapshot API; its state is exactly these four arrays).
            sim._v[...] = st.sim_v
            sim._refr[...] = st.sim_refr
            sim._prev[...] = st.sim_prev
            sim._spikes[...] = st.sim_spikes
        else:
            st.cash = cfg.initial_cash
            st.hatch_equity = cfg.initial_cash
            st.equity = cfg.initial_cash
            st.daily_spikes = np.zeros(self._n, dtype=np.int64)
            st.daily_mbon_drive = np.zeros(self._mbon_rows.size, dtype=np.float64)
            if cfg.larval_weights is not None:
                lw = load_larval_weights(cfg.larval_weights, chassis)  # fingerprint guard
                plasticity.weights[...] = lw.weights
                st.larval_fingerprint = lw.fingerprint
        st.weights = plasticity.weights.copy()
        st.eligibility = plasticity.eligibility.copy()
        st.habituation = plasticity.habituation.copy()
        self._plasticity = plasticity
        self._sim = sim

        # The loop owns the only RNG: seeded noise floor + daily rotation.
        rng = np.random.Generator(np.random.PCG64(cfg.seed))
        if resumed:
            rng.bit_generator.state = st.rng_state
        self._rng = rng

        feed = self._feed or (ReplayFeed(cfg) if cfg.mode == "replay" else LivePaperFeed())
        frames = feed.frames()

        execution = self._execution or SimulatedExecution(
            frames=frames, fill_mode=cfg.fill_mode
        )
        self._execution = execution
        # Pending mode (replay + pessimistic model): the driver drains this
        # book bar by bar; close mode / the live seam keep immediate fills.
        self._pending_book = (
            execution.book
            if isinstance(execution, SimulatedExecution)
            and execution.fill_mode != FILL_MODE_CLOSE
            else None
        )
        if resumed and self._pending_book is not None:
            for o in st.pending:  # orders queued before the kill keep waiting
                self._pending_book.submit(o)

        start_ts, end_ts = _window(cfg.start, cfg.end)

        if not resumed:
            # Hatch calibration: one neutral sniff anchors the innate balance.
            st.anchor = self._innate(list(frames))
            st.weights = plasticity.weights.copy()
            self._emit({"type": "hatch", "hatch_equity": round(st.hatch_equity, 6)})

        for ts, bar_frames in feed.bars():
            if st.last_ts is not None and ts <= pd.Timestamp(st.last_ts):
                continue  # already persisted before the kill
            if ts > end_ts:
                break
            day_key = ts.date().isoformat()
            if day_key != st.cur_day:
                if st.cur_day is not None:
                    self._settle_and_sleep(st.cur_day, frames)
                st.cur_day = day_key
                st.pointer_drawn = False
                self._emit({"type": "wake", "ts": ts.isoformat()})
            self._process_bar(ts, bar_frames)
            equity_file.flush()
            events_file.flush()
            st.last_ts = ts.isoformat()
            if st.n_bars % cfg.persist_every == 0:
                self._persist(state_path)


        # Run end: any order whose symbol never printed again is cancelled.
        if self._pending_book is not None and self._pending_book.pending_orders():
            for o in self._pending_book.cancel_all():
                self._cancel(o, pd.Timestamp(st.last_ts), "run_end")
        if st.cur_day is not None:
            self._settle_and_sleep(st.cur_day, frames)

        self._persist(state_path)
        events_file.close()
        return AdultResult(
            run_dir=run_dir,
            resumed=resumed,
            n_bars=st.n_bars,
            n_events=st.n_events,
            n_orders=st.n_orders,
            n_deaths=st.n_deaths,
            final_equity=st.equity,
            wall_s=time.perf_counter() - t0,
            final_weights=plasticity.weights.copy(),
        )

    def _persist(self, state_path: Path) -> None:
        st = self._st
        st.weights = self._plasticity.weights.copy()
        st.pending = (
            self._pending_book.pending_orders() if self._pending_book is not None else []
        )
        st.eligibility = self._plasticity.eligibility.copy()
        st.habituation = self._plasticity.habituation.copy()
        st.sim_v = self._sim._v.copy()
        st.sim_refr = self._sim._refr.copy()
        st.sim_prev = self._sim._prev.copy()
        st.sim_spikes = self._sim._spikes.copy()
        st.rng_state = self._rng.bit_generator.state
        st.save(state_path)

    # -- loop machinery (mirrors loop.run_backtest exactly; seams reused) ---

    def _innate(self, tickers: list[str]) -> float:
        """Hatch calibration sniff (loop seam ``_innate_balance``)."""
        if not tickers:
            return 0.0  # live warmup edge: no data yet -> neutral anchor
        return _innate_balance(
            self._chassis,
            self._plasticity,
            self._sim,
            self._upn_rows,
            self._channels,
            tickers,
            self._cfg.ms_per_bar,
        )

    def _emit(self, event: dict) -> None:
        self._st.n_events += 1
        self._events_file.write(json.dumps(event, sort_keys=True) + "\n")

    def _equity_at(self) -> float:
        st = self._st
        return st.cash + math.fsum(
            p.shares * st.last_close[s] for s, p in st.positions.items()
        )

    def _order(
        self,
        ts: pd.Timestamp,
        o: PendingOrder,
        price: float,
        shares: int,
        realized_pnl: float | None,
        cash_after: float,
    ) -> None:
        st = self._st
        st.n_orders += 1
        self._emit(
            {
                "type": "order",
                "ts": ts.isoformat(),
                "decision_ts": o.decision_ts.isoformat(),
                "ticker": o.ticker,
                "side": o.side,
                "reason": o.reason,
                "shares": shares,
                "price": round(price, 6),
                "realized_pnl": None if realized_pnl is None else round(realized_pnl, 6),
                "cash_after": round(cash_after, 6),
                "n_positions_after": len(st.positions),
            }
        )

    def _cancel(self, o: PendingOrder, ts: pd.Timestamp, cancel_reason: str) -> None:
        self._emit(
            {
                "type": "cancel",
                "ts": ts.isoformat(),
                "decision_ts": o.decision_ts.isoformat(),
                "ticker": o.ticker,
                "side": o.side,
                "reason": o.reason,
                "cancel_reason": cancel_reason,
            }
        )

    def _apply_buy(self, ts: pd.Timestamp, o: PendingOrder, price: float) -> None:
        """Buy/add fill on the paper book (loop parity with ``apply_buy``)."""
        st = self._st
        shares = o.shares
        st.cash -= shares * price * (1.0 + self._cfg.fees)
        pos = st.positions.get(o.ticker)
        if pos is None:
            st.positions[o.ticker] = _Position(shares=shares, avg_cost=price)
        else:
            total = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + shares * price) / total
            pos.shares = total
        # Snapshot the eligibility trace for this position's opening
        # encounter (DESIGN v0.6 D6); overwritten on add — the latest
        # snapshot wins (loop parity with ``apply_buy``).
        st.entry_elig[o.ticker] = self._plasticity.eligibility.copy()
        self._order(ts, o, price, shares, None, st.cash)

    def _apply_sell(self, ts: pd.Timestamp, o: PendingOrder, price: float) -> float | None:
        """Close the WHOLE current position (loop parity with ``apply_sell``);
        returns realized P&L, or ``None`` when there is nothing left to sell."""
        st = self._st
        pos = st.positions.pop(o.ticker, None)
        if pos is None:
            return None
        proceeds = pos.shares * price * (1.0 - self._cfg.fees)
        st.cash += proceeds
        realized = proceeds - pos.shares * pos.avg_cost
        self._order(ts, o, price, pos.shares, realized, st.cash)
        snap = st.entry_elig.pop(o.ticker, None)
        if snap is not None:
            # Per-exit dopamine (DESIGN v0.6 D6, T9 addendum 2026-09-15):
            # normalized by the trade's OWN notional (clipped to [-1, 1]);
            # loop parity with ``apply_sell``.
            notional = pos.shares * pos.avg_cost
            reward = min(1.0, max(0.0, realized) / notional) if notional > 0 else 0.0
            punishment = min(1.0, max(0.0, -realized) / notional) if notional > 0 else 0.0
            self._plasticity.observe_trade(
                NeuromodState(
                    reward=reward, punishment=punishment, hunger=0.0, arousal=0.0
                ),
                snap,
            )
            self._emit(
                {
                    "type": "trade_credit",
                    "ts": ts.isoformat(),
                    "ticker": o.ticker,
                    "realized_pnl": round(realized, 6),
                    "reward": round(reward, 9),
                    "punishment": round(punishment, 9),
                }
            )
        return realized

    def _close_position(self, ts: pd.Timestamp, ticker: str, reason: str) -> float:
        """Immediate close at the last close via the broker seam (live seam,
        close mode, and death liquidation); returns realized P&L."""
        st = self._st
        price = st.last_close[ticker]
        pos = st.positions.pop(ticker)
        fill = self._execution.execute(ts, ticker, "sell", pos.shares, price)
        if fill is None:
            st.positions[ticker] = pos  # live reject: book unchanged
            return 0.0
        proceeds = pos.shares * fill * (1.0 - self._cfg.fees)
        st.cash += proceeds
        realized = proceeds - pos.shares * pos.avg_cost
        self._order(
            ts,
            PendingOrder(decision_ts=ts, ticker=ticker, side="sell", reason=reason),
            fill,
            pos.shares,
            realized,
            st.cash,
        )
        snap = st.entry_elig.pop(ticker, None)
        if snap is not None:
            # Per-exit dopamine (DESIGN v0.6 D6, T9 addendum 2026-09-15):
            # normalized by the trade's OWN notional (clipped to [-1, 1]);
            # loop parity with ``apply_sell``.
            notional = pos.shares * pos.avg_cost
            reward = min(1.0, max(0.0, realized) / notional) if notional > 0 else 0.0
            punishment = min(1.0, max(0.0, -realized) / notional) if notional > 0 else 0.0
            self._plasticity.observe_trade(
                NeuromodState(
                    reward=reward, punishment=punishment, hunger=0.0, arousal=0.0
                ),
                snap,
            )
            self._emit(
                {
                    "type": "trade_credit",
                    "ts": ts.isoformat(),
                    "ticker": ticker,
                    "realized_pnl": round(realized, 6),
                    "reward": round(reward, 9),
                    "punishment": round(punishment, 9),
                }
            )
        return realized

    def _settle_and_sleep(self, day, frames: dict[str, pd.DataFrame]) -> None:
        """Close-of-day: sugar/shock settle -> observe -> sleep (DESIGN §7.3)."""
        cfg, st = self._cfg, self._st
        reward = max(0.0, st.day_realized) / st.hatch_equity
        punishment = max(0.0, -st.day_realized) / st.hatch_equity
        drawdown = max(0.0, 1.0 - st.equity / st.hatch_equity)
        hunger = min(1.0, drawdown / abs(cfg.death_threshold))
        mean_abs_ret = st.day_abs_ret_sum / st.day_abs_ret_n if st.day_abs_ret_n else 0.0
        arousal = min(1.0, mean_abs_ret / 0.01)
        state = NeuromodState(
            reward=reward, punishment=punishment, hunger=hunger, arousal=arousal
        )
        post = np.zeros(self._n, dtype=np.float64)
        if st.daily_signal_encounters:
            post[self._mbon_rows] = st.daily_mbon_drive / st.daily_signal_encounters
        self._plasticity.observe(state, st.daily_spikes.astype(np.float64), post)
        self._emit(
            {
                "type": "sugar_shock",
                "ts": f"{day}T20:00:00+00:00",
                "reward": round(reward, 9),
                "punishment": round(punishment, 9),
                "hunger": round(hunger, 9),
                "arousal": round(arousal, 9),
            }
        )
        self._plasticity.sleep()
        self._emit({"type": "sleep", "ts": f"{day}T20:00:00+00:00"})
        st.day_realized = 0.0
        st.day_abs_ret_sum = 0.0
        # Daily anchor refresh (DESIGN v0.6 T9): re-measure the innate
        # balance with the current (learned) weights — one extra neutral
        # sniff per sleep, deterministic (RNG-free).
        st.anchor = self._innate(list(frames))
        self._emit(
            {
                "type": "anchor",
                "ts": f"{day}T20:00:00+00:00",
                "balance": round(st.anchor, 9),
            }
        )
        st.day_abs_ret_n = 0
        st.daily_spikes.fill(0)
        st.daily_mbon_drive.fill(0)
        st.daily_signal_encounters = 0
        st.hunger_armed = True

    def _process_bar(self, ts: pd.Timestamp, frames: dict[str, pd.DataFrame]) -> None:
        """The per-bar foraging cycle (DESIGN §7 step 2); mirrors
        ``loop.run_backtest`` bar-for-bar with the loop's pure seams."""
        cfg, st = self._cfg, self._st

        # 1. Mark prices (equity marking stays at close prices — marking is
        # not execution), then fill pending orders whose symbol's next
        # available bar is this one (pessimistic next-bar model; loop parity).
        for ticker, df in frames.items():
            i = df.index.get_indexer([ts], method="pad")[0]
            if i >= 0 and df.index[i] == ts:
                st.last_close[ticker] = float(df["close"].iloc[i])

        # 1b. Pending-order fills (loop parity with run_backtest step 1b).
        if self._pending_book is not None:
            for o, fill_price in self._pending_book.drain(ts):
                if o.side == "buy":
                    self._apply_buy(ts, o, fill_price)
                    continue
                realized = self._apply_sell(ts, o, fill_price)
                if realized is None:
                    self._cancel(o, ts, "no_position")
                else:
                    st.day_realized += realized

        # 2. Mechanical exits, independent of encounters (DESIGN §6.5).
        for ticker in sorted(st.positions):
            pos = st.positions[ticker]
            price = st.last_close[ticker]
            pnl_pct = (price / pos.avg_cost - 1.0) * 100.0
            if pnl_pct <= -cfg.shock_adverse_pct:
                if self._pending_book is not None:
                    if not self._pending_book.has_pending(ticker):
                        # Queue the stop-loss: it lands one bar later (the
                        # execution delay is the point — no look-ahead
                        # compensation).
                        self._pending_book.submit(
                            PendingOrder(
                                decision_ts=ts, ticker=ticker, side="sell",
                                reason="shock",
                            )
                        )
                else:
                    st.day_realized += self._close_position(ts, ticker, "shock")
        st.equity = self._equity_at()
        drawdown = max(0.0, 1.0 - st.equity / st.hatch_equity)
        if drawdown >= cfg.hunger_drawdown and st.hunger_armed:
            winners = sorted(
                (
                    (s, (st.last_close[s] / p.avg_cost - 1.0) * 100.0)
                    for s, p in st.positions.items()
                ),
                key=lambda t: (-t[1], t[0]),
            )
            if winners and winners[0][1] > 0.0:
                winner = winners[0][0]
                if self._pending_book is not None:
                    if not self._pending_book.has_pending(winner):
                        self._pending_book.submit(
                            PendingOrder(
                                decision_ts=ts, ticker=winner, side="sell",
                                reason="hunger",
                            )
                        )
                        st.hunger_armed = False
                else:
                    st.day_realized += self._close_position(ts, winner, "hunger")
                    st.hunger_armed = False
        elif drawdown < 0.5 * cfg.hunger_drawdown:
            st.hunger_armed = True

        # 3. Plume filter (DESIGN §6.1): the day's smelliest movers.
        plumes = _plume_set(frames, ts, cfg.top_k)
        for ticker, _ in plumes:
            df = frames[ticker]
            i = df.index.get_indexer([ts], method="pad")[0]
            features, ret1, _ = _features_at(df, i)
            st.day_abs_ret_sum += abs(ret1)
            st.day_abs_ret_n += 1
        if plumes:
            if not st.pointer_drawn:
                # Seeded rotation offset (the loop RNG's daily draw).
                st.pointer = int(self._rng.integers(0, len(plumes)))
                st.pointer_drawn = True
            ticker, intensity = plumes[st.pointer % len(plumes)]
            st.pointer += 1

            # 4. Encounter: render -> odor -> taste -> noise floor -> step.
            df = frames[ticker]
            i = df.index.get_indexer([ts], method="pad")[0]
            window = df.iloc[max(0, i - VISION_WINDOW_BARS + 1) : i + 1]
            inp = encode_vision(window, self._chassis).astype(np.float64)
            features, _, _ = _features_at(df, i)
            inp[self._upn_rows] += SMELL_GAIN * encode_smell(ticker, features, self._chassis)
            if ticker in st.positions:
                pnl_pct = (st.last_close[ticker] / st.positions[ticker].avg_cost - 1.0) * 100.0
                inp += encode_taste(pnl_pct, self._chassis)
            inp += self._rng.standard_normal(self._n) * cfg.noise_sigma_mv
            out = self._sim.step(inp, cfg.ms_per_bar)
            spikes = out["spikes"]
            st.daily_spikes += spikes

            # 5. Readout: learned KC->MBON drive split by valence; balance in
            # [-1, 1] is the decision variable.
            spike_f = spikes.astype(np.float64)
            drive = self._plasticity.mbon_activation(spike_f)
            valence = self._plasticity.mbon_valence
            approach_drive = float(valence[valence > 0.0] @ drive[valence > 0.0])
            avoid_drive = float(-(valence[valence < 0.0] @ drive[valence < 0.0]))
            balance = (approach_drive - avoid_drive) / (
                approach_drive + avoid_drive + 1e-9
            )
            raw_score = float(valence @ drive)
            if approach_drive + avoid_drive > 0.0:
                # Eligibility post-activity: the STRUCTURAL MBON response.
                st.daily_mbon_drive += (
                    self._plasticity.baseline.T @ self._plasticity.kc_activity(spike_f)
                )
                st.daily_signal_encounters += 1
            duration_s = cfg.ms_per_bar / 1000.0
            mbon_rate = (
                float(spikes[self._mbon_rows].sum()) / self._mbon_rows.size / duration_s
            )
            # Center on the innate balance; a silent encounter reads neutral.
            centered = (
                balance - st.anchor if approach_drive + avoid_drive > 0.0 else 0.0
            )
            # Structural looming drive (DESIGN v0.6 §5): the LC→MBON
            # connectome response to this encounter's LC spikes, entering
            # the decision balance at a small fixed λ. Zero when the
            # chassis has no LC→MBON path (or a silent LC population).
            if self._w_struct is not None:
                structural_score = float(
                    self._plasticity.mbon_valence
                    @ (self._w_struct.T @ spikes[self._lc_rows])
                )
            else:
                structural_score = 0.0
            balance_used = centered + cfg.lambda_struct * structural_score

            self._emit(
                {
                    "type": "encounter",
                    "ts": ts.isoformat(),
                    "ticker": ticker,
                    "intensity": round(intensity, 9),
                    "balance": round(centered, 9),
                    "raw_balance": round(balance, 9),
                    "structural_score": round(structural_score, 9),
                    "balance_used": round(balance_used, 9),
                    "valence_readout": round(raw_score, 6),
                    "mbon_rate_hz": round(mbon_rate, 6),
                }
            )

            # 6. Decision + order (fills on the symbol's next available bar).
            held = ticker in st.positions
            price = st.last_close[ticker]
            action, reason = _decide(
                balance_used,
                held,
                len(st.positions) + (
                    self._pending_book.n_buys() if self._pending_book is not None else 0
                ) >= cfg.position_cap,
                cfg.approach_thr,
                cfg.avoid_thr,
            )
            shares = 0
            if action in ("buy", "add"):
                st.equity = self._equity_at()
                notional = _size_fraction(mbon_rate) * st.equity / cfg.position_cap
                shares = int(notional / price) if price > 0.0 else 0
                if shares < 1:
                    action, reason, shares = "pass", "dust", 0
            self._emit(
                {
                    "type": "decision",
                    "ts": ts.isoformat(),
                    "ticker": ticker,
                    "action": action,
                    "reason": reason,
                }
            )
            if action in ("buy", "add"):
                o = PendingOrder(
                    decision_ts=ts, ticker=ticker, side="buy", reason=reason,
                    shares=shares,
                )
                if self._pending_book is not None:
                    self._pending_book.submit(o)
                else:
                    fill = self._execution.execute(ts, ticker, "buy", shares, price)
                    if fill is not None:
                        self._apply_buy(ts, o, fill)
            elif action == "sell":
                if self._pending_book is not None:
                    # Sells close the WHOLE position at fill time.
                    self._pending_book.submit(
                        PendingOrder(
                            decision_ts=ts, ticker=ticker, side="sell",
                            reason=reason,
                        )
                    )
                else:
                    st.day_realized += self._close_position(ts, ticker, reason)

        # 7. Death check (D14): equity <= hatch * (1 + threshold) kills the fly.
        st.equity = self._equity_at()
        if st.equity <= st.hatch_equity * (1.0 + cfg.death_threshold):
            self._emit(
                {
                    "type": "death",
                    "ts": ts.isoformat(),
                    "equity": round(st.equity, 6),
                    "hatch_equity": round(st.hatch_equity, 6),
                }
            )
            st.n_deaths += 1
            # Clear the per-exit snapshots BEFORE liquidation: a dying brain
            # gets no per-trade credit (loop parity with run_backtest).
            st.entry_elig.clear()
            for ticker in sorted(st.positions):
                st.day_realized += self._close_position(ts, ticker, "death_liquidation")
            if self._pending_book is not None:
                for o in self._pending_book.cancel_all():
                    self._cancel(o, ts, "death")  # the brain that placed them is gone
            self._plasticity = Plasticity(self._chassis, top_k=cfg.grudge_top_k)
            self._sim.reset(cfg.seed)
            st.daily_spikes.fill(0)
            st.daily_mbon_drive.fill(0)
            st.daily_signal_encounters = 0
            st.hatch_equity = st.cash
            st.hunger_armed = True
            self._emit({"type": "hatch", "hatch_equity": round(st.hatch_equity, 6)})
            st.equity = st.cash

            # A fresh brain re-calibrates its innate balance.
            st.anchor = self._innate(list(frames))
            st.equity = st.cash

        # 8. Bar receipt.
        self._equity_writer.writerow(
            [ts.isoformat(), f"{st.equity:.2f}", f"{st.cash:.2f}", len(st.positions)]
        )
        st.n_bars += 1


def _load_chassis():
    """Stripped-chassis source (test seam: tests monkeypatch this to a tiny
    synthetic fixture)."""
    from fruitfly.connectome import load_stripped_chassis

    return load_stripped_chassis()


def _load_whole_chassis():
    """Whole-fly chassis source (test seam: tests monkeypatch this to a tiny
    synthetic whole-like fixture).

    Production: the cached whole fly with the stripped chassis's
    population/region labels transferred onto matched bodyIds — the labeled
    whole-fly chassis the T12b replay seam injected (the raw whole-fly cache
    labels every node ``population="whole"``, which the loop's
    population-keyed readouts cannot see).
    """
    from fruitfly.connectome import (
        label_whole_chassis,
        load_stripped_chassis,
        load_whole_fly,
    )

    return label_whole_chassis(load_whole_fly(), load_stripped_chassis())


def _resolve_chassis(config: AdultConfig):
    """Chassis per ``config.chassis``: stripped (default) or whole fly."""
    if config.chassis == "whole":
        return _load_whole_chassis()
    return _load_chassis()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_replay(config: AdultConfig) -> AdultResult:
    """Convenience wrapper: one adult run in replay mode."""
    if config.mode != "replay":
        raise ValueError("run_replay requires mode='replay'")
    return AdultRun(config).run()


def _register() -> None:
    def builder(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "adult",
            help="Persistent adult fly: multi-day replay (or live-paper) with "
            "kill-and-resume state (T15, DESIGN §7/§9).",
        )
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--start", required=True, help="e.g. 2026-08-17")
        p.add_argument("--end", required=True, help="e.g. 2026-09-14")
        p.add_argument("--mode", choices=("replay", "live-paper"), default="replay")
        p.add_argument("--run-dir", default=None)
        p.add_argument(
            "--chassis", choices=("stripped", "whole"), default="stripped",
            help="Brain chassis; 'whole' implies the calibrated T12b STD.",
        )
        p.add_argument("--std-beta", type=float, default=None)
        p.add_argument("--std-tau-rec-ms", type=float, default=None)
        p.add_argument("--persist-every", type=int, default=1)
        p.add_argument("--larval-weights", default=None, help="T9 .npz artifact path")
        p.set_defaults(func=_cmd_adult)

    register_command("adult", builder)


def _cmd_adult(args: argparse.Namespace) -> int:
    config = AdultConfig(
        seed=args.seed,
        start=args.start,
        end=args.end,
        mode=args.mode,
        run_dir=args.run_dir,
        persist_every=args.persist_every,
        larval_weights=args.larval_weights,
        chassis=args.chassis,
        std_beta=args.std_beta,
        std_tau_rec_ms=args.std_tau_rec_ms,
    )
    result = AdultRun(config).run()
    resumed = " (resumed)" if result.resumed else ""
    print(
        f"adult{resumed}: run_dir={result.run_dir} bars={result.n_bars} "
        f"events={result.n_events} orders={result.n_orders} deaths={result.n_deaths} "
        f"final_equity={result.final_equity:.2f} wall={result.wall_s:.1f}s"
    )
    return 0


_register()
