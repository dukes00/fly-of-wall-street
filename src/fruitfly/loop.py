"""T7: the foraging backtest loop (DESIGN §5-§7, D3/D13/D14/D16).

One simulated fly forages a 1-minute-bar market session. Per DESIGN §6 the
fly does not watch a watchlist: every bar, the basket is ranked by plume
intensity (|1-bar return| + relative volume) and the ``top_k`` smelliest
movers become the active plume set; flat stocks are odorless. The fly
samples **one plume per bar**, round-robin through the active set — the
rotation pointer persists across bars so successive bars start at different
plumes, and the starting offset is drawn from the loop's RNG at the first
encounter of each day. One sample per bar keeps the foraging cadence inside
the 1-minute bar budget (the LIF step costs ~0.15-0.45 s wall on the 27,115
-neuron chassis; five samples per bar would blow the 2-day runtime target).

Per encounter (DESIGN §5/§7): render the recent OHLC window (vision), mix the
ticker identity x market-state odor (smell), taste the open position's
unrealized P&L, add the seeded sensory-noise floor, step the LIF chassis for
``ms_per_bar`` of biological time, and read the mushroom body.

**MBON readout path (one documented path).** The LIF engine propagates
through *structural* synapses only; learned KC→MBON weights live in
``Plasticity``. Decisions therefore read the learned drive, not raw MBON
spikes: ``drive = Plasticity.mbon_activation(KC spikes from the sim step)``,
split by MBON valence into approach drive ``A`` and avoidance drive ``R``
(the two halves of ``Plasticity.valence_readout``), and the decision variable
is the scale-free **balance** ``(A - R) / (A + R)`` in [-1, 1] — DESIGN §5's
"approach/avoid balance", robust to encounter intensity, **centered on the
fly's innate balance**: the structural KC→MBON wiring is avoid-biased
(~ -0.35 on the real chassis at every drive level), so at hatch the fly takes
one neutral calibration sniff (mean basket identity, no market state, no
noise) and every encounter's balance is centered on that anchor (the raw
``valence_readout`` and raw balance are also logged for T9). ``balance >
approach_thr``
→ BUY/add, ``balance < -avoid_thr`` → SELL (valence-flip exit on a held
position), otherwise pass. Position size maps the MBON population firing
rate (post-spikes over MBON nodes / duration) through
``tanh(rate / MBON_RATE_SCALE_HZ)``, floored at ``MIN_SIZE_FRAC`` and capped
at 1.0; the per-order notional is ``frac * equity / position_cap`` (D13).
Caveat documented for T9: at the current engine gain constants MBONs are
sub-threshold (KC→MBON quantal PSPs ≈ 0.01 mV/synapse), so MBON firing rates
are ~0 and sizing rides the floor.

Exits (DESIGN §6): a held ticker re-encountered with an avoid balance is
closed (valence flip); any open position whose unrealized P&L drops below
``-shock_adverse_pct`` is closed immediately (sharp adverse move → shock —
the biological stop-loss bypasses the brain); when portfolio drawdown from
hatch equity crosses ``hunger_drawdown`` the fly closes its largest winner
(hunger → take profit), re-arming when drawdown recovers below half the
threshold. Fills are at the current bar's close price, whole shares, with
proportional ``fees`` on each side (default 0).
Cash accounting is paper accounting: buys deduct cash without a balance
check (a bar's close fill can push cash negative and gross exposure past
equity — margin, in effect); equity = cash + marked positions is the only
solvency signal, and D14's death threshold acts on it.

Close of day (DESIGN §7): the day's realized P&L settles into sugar/shock
(``NeuromodState.reward/punishment`` as a fraction of hatch equity), hunger
maps drawdown to [0, 1] hitting 1.0 exactly at the death threshold (D14),
arousal maps the day's mean |1-bar return| to [0, 1]; one
``Plasticity.observe`` consumes the day's accumulated spike counts, then
``Plasticity.sleep()`` consolidates and the fly logs the sleep event.

Death (D14): when equity <= ``hatch_equity * (1 + death_threshold)`` the fly
dies: remaining positions are liquidated at the bar close, a fresh
``Plasticity`` hatches (brain reset), and the new fly's hatch equity is the
post-liquidation equity. Death and hatch are logged as events (T9 exercises
this).

Determinism (hard gate): the engine is RNG-free; the LOOP owns the only RNG,
``np.random.Generator(np.random.PCG64(seed))``, consumed in a fixed order —
one rotation-offset draw per day plus one standard-normal draw per encounter.
The noise floor is small (``noise_sigma_mv`` mV on encoder currents in the
tens-of-mV range; see reports/t7-loop.md for the measured SNR) but it is what
makes different seeds genuinely diverge. Same seed → byte-identical
equity.csv and events.jsonl (json emitted with sorted keys, fixed field
order; equity floats formatted to 2 decimals).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fruitfly.__main__ import register_command
from fruitfly.connectome import Chassis, load_stripped_chassis
from fruitfly.data import BASKET, load_bars
from fruitfly.neuromod import NeuromodState, Plasticity
from fruitfly.senses import (
    build_features,
    encode_smell,
    encode_taste,
    encode_vision,
    identity_profile,
)
from fruitfly.senses.smell import upn_channels
from fruitfly.sim import LIFSim

__all__ = [
    "BacktestConfig",
    "RunResult",
    "run_backtest",
]

# ---------------------------------------------------------------------------
# Documented defaults (T9 calibration candidates unless pinned to a decision)
# ---------------------------------------------------------------------------

#: Root directory for run artifacts.
RUNS_DIR = Path("data/runs")

#: OHLC bars rendered per vision encounter (matches senses.vision GRID_W).
VISION_WINDOW_BARS = 48

#: uPN current multiplier: encoder smell output is a [0, 1) identity profile
#: modulated by market state, far below the 15 mV LIF threshold. The loop
#: scales it up to drive the olfactory pathway. Measured on real 2026-08-18
#: bars: KC yield 0 spikes/encounter at 300, ~19/encounter at 1000 (2% silent
#: encounters), ~58 at 2000 — and per-encounter wall time is flat in the
#: gain, so 1000 is the knee: live circuit, lowest spike volume.
SMELL_GAIN = 1000.0

#: Sensory-noise floor (mV-above-rest, per node, per encounter). The loop's
#: only stochastic element; SNR documented in reports/t7-loop.md.
NOISE_SIGMA_MV = 0.5

#: Decision thresholds on the hatch-anchored approach/avoid balance in
#: [-1, 1] (see reports/t7-loop.md: the raw balance is structurally avoid-
#: biased ~ -0.35; centering on the innate balance puts fresh-fly encounters
#: around 0 with a measured spread of a few hundredths). T9 calibration
#: candidates.
APPROACH_THR = 0.005
AVOID_THR = 0.005

#: Sizing map: fraction = clip(tanh(mbon_rate / MBON_RATE_SCALE_HZ),
#: MIN_SIZE_FRAC, 1.0). MBONs are sub-threshold under the current engine
#: gains, so this currently rides the floor (documented T9 gap).
MBON_RATE_SCALE_HZ = 20.0
MIN_SIZE_FRAC = 0.25

#: Sharp-adverse exit: unrealized P&L (%) beyond this closes the position
#: immediately (biological stop-loss, DESIGN §6).
SHOCK_ADVERSE_PCT = 2.0

#: Hunger exit: portfolio drawdown from hatch equity beyond this closes the
#: largest winner; re-arms when drawdown recovers below half the threshold.
HUNGER_DRAWDOWN = 0.10

#: Death: equity <= hatch_equity * (1 + death_threshold) kills the fly (D14).
DEATH_THRESHOLD = -0.50

#: Simulation substep (ms): 1 ms = 5% of tau_m, half the T3 bench's 0.5 ms
#: — halves wall time per encounter (~0.36 s) so a 2-day run fits the
#: 5-minute budget; dynamics qualitatively unchanged. Starting cash below.
DT_MS = 1.0
INITIAL_CASH = 100_000.0


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """One backtest run. See module docstring and reports/t7-loop.md."""

    seed: int
    start: str
    end: str
    #: Plume-filter width (D3): how many smelliest movers are in the active set.
    top_k: int = 5
    #: Concurrent-position cap (D13).
    position_cap: int = 10
    #: Biological time stepped per 1-minute bar (D16).
    ms_per_bar: float = 500.0
    #: Death threshold on equity vs hatch equity (D14).
    death_threshold: float = DEATH_THRESHOLD
    #: Proportional per-side fee (0 = paper fills).
    fees: float = 0.0
    initial_cash: float = INITIAL_CASH
    shock_adverse_pct: float = SHOCK_ADVERSE_PCT
    hunger_drawdown: float = HUNGER_DRAWDOWN
    approach_thr: float = APPROACH_THR
    avoid_thr: float = AVOID_THR
    noise_sigma_mv: float = NOISE_SIGMA_MV
    dt_ms: float = DT_MS
    #: Override the artifact directory (tests use this to avoid collisions).
    out_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.position_cap < 1:
            raise ValueError("position_cap must be >= 1")
        if self.ms_per_bar <= 0:
            raise ValueError("ms_per_bar must be positive")
        if not -1.0 < self.death_threshold < 0.0:
            raise ValueError("death_threshold must be in (-1, 0)")
        if self.fees < 0:
            raise ValueError("fees must be non-negative")


@dataclass(frozen=True)
class RunResult:
    """Summary of a finished backtest run."""

    run_dir: Path
    n_bars: int
    n_events: int
    n_orders: int
    n_deaths: int
    final_equity: float
    wall_s: float


@dataclass
class _Position:
    shares: int
    avg_cost: float


# ---------------------------------------------------------------------------
# Feature helpers (pure, deterministic)
# ---------------------------------------------------------------------------


def _features_at(df: pd.DataFrame, i: int) -> tuple[dict, float, float]:
    """Market-state features at bar ``i`` via the shared smell feature builder.

    Returns ``(features, ret1, volume_ratio)``: exactly the dict built by
    ``senses.build_features`` (neutral values when history is short), the
    1-bar return as a fraction, and volume relative to the trailing mean
    (``1 + volume_delta``; 1.0 without history). The plume filter and the
    odor mix consume the same numbers — one feature definition, no loop-side
    re-derivation.
    """
    features = build_features(df.iloc[: i + 1])
    ret1 = features["returns"]
    volume_ratio = 1.0 + features["volume_delta"]
    return features, ret1, volume_ratio


def _plume_set(
    frames: dict[str, pd.DataFrame], ts: pd.Timestamp, top_k: int
) -> list[tuple[str, float]]:
    """Rank the basket by plume intensity at ``ts`` (DESIGN §6.1, D3).

    Intensity = |1-bar return| + volume ratio; flat stocks (zero 1-bar
    return) are odorless and skipped entirely. Returns the top ``top_k`` as
    ``(ticker, intensity)``, ties broken by ticker for determinism.
    """
    ranked: list[tuple[float, str]] = []
    for ticker, df in frames.items():
        i = df.index.get_indexer([ts], method="pad")[0]
        if i < 1 or df.index[i] != ts:
            continue
        features, ret1, volume_ratio = _features_at(df, i)
        if ret1 == 0.0:
            continue  # flat -> odorless
        ranked.append((abs(ret1) + volume_ratio, ticker))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    return [(t, s) for s, t in ranked[:top_k]]


def _decide(
    balance: float, held: bool, cap_reached: bool, approach_thr: float, avoid_thr: float
) -> tuple[str, str]:
    """Map the MBON approach/avoid balance to an action (DESIGN §5).

    Returns ``(action, reason)`` with action in {buy, add, sell, pass}.
    """
    if balance > approach_thr:
        if held:
            return "add", "approach"
        if cap_reached:
            return "pass", "cap"
        return "buy", "approach"
    if balance < -avoid_thr:
        if held:
            return "sell", "valence_flip"
        return "pass", "avoid"
    return "pass", "neutral"


def _innate_balance(
    chassis: Chassis,
    plasticity: Plasticity,
    sim: LIFSim,
    upn_rows: np.ndarray,
    channels: np.ndarray,
    tickers: list[str],
    ms_per_bar: float,
) -> float:
    """Hatch calibration sniff: the fly's innate approach/avoid balance.

    The raw KC->MBON balance is structurally avoid-biased (measured ~-0.35
    on the stripped chassis at every drive level): the valence split is a
    documented judgment call (transmitter sign), not a calibrated behavior.
    At hatch the fly takes one neutral sniff — the mean basket identity
    profile with no market-state modulation, no noise — and the measured
    balance becomes the anchor every subsequent encounter is centered on.
    Deterministic (engine RNG-free); one extra sim step per hatch. Silent
    sniff (no KC activity) anchors at 0.0.
    """
    profile = np.mean([identity_profile(t) for t in tickers], axis=0)
    inp = np.zeros(chassis.n_neurons, dtype=np.float64)
    inp[upn_rows] = SMELL_GAIN * profile[channels]
    spikes = sim.step(inp, ms_per_bar)["spikes"].astype(np.float64)
    drive = plasticity.mbon_activation(spikes)
    valence = plasticity.mbon_valence
    approach_drive = float(valence[valence > 0.0] @ drive[valence > 0.0])
    avoid_drive = float(-(valence[valence < 0.0] @ drive[valence < 0.0]))
    if approach_drive + avoid_drive <= 0.0:
        return 0.0
    return (approach_drive - avoid_drive) / (approach_drive + avoid_drive)


def _size_fraction(mbon_rate_hz: float) -> float:
    """MBON firing rate -> position-size fraction (DESIGN §5)."""
    frac = math.tanh(mbon_rate_hz / MBON_RATE_SCALE_HZ)
    return min(1.0, max(MIN_SIZE_FRAC, frac))


def _load_chassis() -> Chassis:
    """Chassis source (test seam: tests monkeypatch this to a tiny fixture)."""
    return load_stripped_chassis()


def _window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse the run window; a date-only ``end`` extends to end of that UTC day."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if len(end) <= 10:  # date only: the whole session of that day
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if end_ts < start_ts:
        raise ValueError(f"end {end!r} precedes start {start!r}")
    return start_ts, end_ts


def _run_dir_name(config: BacktestConfig) -> Path:
    stem = f"backtest_{config.seed}_{config.start}_{config.end}"
    return Path(config.out_dir) if config.out_dir is not None else RUNS_DIR / stem.replace(":", "")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_backtest(config: BacktestConfig) -> RunResult:
    """Run the foraging backtest (DESIGN §7) and write equity.csv + events.jsonl."""
    t0 = time.perf_counter()
    chassis = _load_chassis()
    plasticity = Plasticity(chassis)
    sim = LIFSim(chassis, dt_ms=config.dt_ms, seed=config.seed)
    # The loop owns the only RNG: seeded noise floor + daily rotation offsets.
    rng = np.random.Generator(np.random.PCG64(config.seed))

    pop = chassis.nodes["population"].to_numpy()
    upn_rows, channels = upn_channels(chassis)
    mbon_rows = np.flatnonzero(pop == "MBON")
    n = chassis.n_neurons

    start_ts, end_ts = _window(config.start, config.end)
    # Full cache history for feature warmup; trading is confined to the window.
    frames = {
        sym: df
        for sym, df in load_bars(list(BASKET), None, None).items()
        if len(df) and df.index[-1] >= start_ts
    }
    timeline = sorted(
        {ts for df in frames.values() for ts in df.index if start_ts <= ts <= end_ts}
    )
    if not timeline:
        raise ValueError(f"no bars in [{config.start}, {config.end}] for the basket")

    # Hatch calibration: the fly sniffs a neutral odor once and its innate
    # (structurally avoid-biased) balance becomes the decision anchor.
    anchor = _innate_balance(
        chassis, plasticity, sim, upn_rows, channels, list(frames), config.ms_per_bar
    )

    run_dir = _run_dir_name(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    equity_path = run_dir / "equity.csv"
    events_path = run_dir / "events.jsonl"
    equity_file = equity_path.open("w", newline="")
    events_file = events_path.open("w")
    equity_writer = csv.writer(equity_file, lineterminator="\n")
    equity_writer.writerow(["timestamp", "equity", "cash", "n_positions"])
    n_events = 0
    n_orders = 0

    def emit(event: dict) -> None:
        nonlocal n_events
        events_file.write(json.dumps(event, sort_keys=True) + "\n")
        n_events += 1

    def order(
        ts: pd.Timestamp, ticker: str, side: str, reason: str, shares: int, price: float,
        realized_pnl: float | None, cash_after: float,
    ) -> None:
        nonlocal n_orders
        n_orders += 1
        emit(
            {
                "type": "order",
                "ts": ts.isoformat(),
                "ticker": ticker,
                "side": side,
                "reason": reason,
                "shares": shares,
                "price": round(price, 6),
                "realized_pnl": None if realized_pnl is None else round(realized_pnl, 6),
                "cash_after": round(cash_after, 6),
                "n_positions_after": len(positions),
            }
        )

    # --- mutable run state -------------------------------------------------
    cash = config.initial_cash
    hatch_equity = config.initial_cash
    positions: dict[str, _Position] = {}
    last_close: dict[str, float] = {}
    cur_day: date | None = None
    pointer = 0
    pointer_drawn = False
    daily_spikes = np.zeros(n, dtype=np.int64)
    day_realized = 0.0
    day_abs_ret_sum = 0.0
    day_abs_ret_n = 0
    hunger_armed = True
    n_bars = 0
    n_deaths = 0
    equity = cash

    emit({"type": "hatch", "hatch_equity": round(hatch_equity, 6)})

    def equity_at() -> float:
        return cash + math.fsum(p.shares * last_close[s] for s, p in positions.items())

    def close_position(ts: pd.Timestamp, ticker: str, reason: str) -> float:
        """Sell a position at the last close; returns realized P&L in dollars."""
        price = last_close[ticker]
        pos = positions.pop(ticker)
        nonlocal cash
        proceeds = pos.shares * price * (1.0 - config.fees)
        cash += proceeds
        realized = proceeds - pos.shares * pos.avg_cost
        order(ts, ticker, "sell", reason, pos.shares, price, realized, cash)
        return realized

    def settle_and_sleep(day: date) -> None:
        """Close-of-day: sugar/shock settle -> observe -> sleep (DESIGN §7.3)."""
        nonlocal day_realized, day_abs_ret_sum, day_abs_ret_n, hunger_armed
        reward = max(0.0, day_realized) / hatch_equity
        punishment = max(0.0, -day_realized) / hatch_equity
        drawdown = max(0.0, 1.0 - equity / hatch_equity)
        hunger = min(1.0, drawdown / abs(config.death_threshold))
        mean_abs_ret = day_abs_ret_sum / day_abs_ret_n if day_abs_ret_n else 0.0
        arousal = min(1.0, mean_abs_ret / 0.01)
        state = NeuromodState(
            reward=reward, punishment=punishment, hunger=hunger, arousal=arousal
        )
        plasticity.observe(
            state, daily_spikes.astype(np.float64), daily_spikes.astype(np.float64)
        )
        emit(
            {
                "type": "sugar_shock",
                "ts": f"{day}T20:00:00+00:00",
                "reward": round(reward, 9),
                "punishment": round(punishment, 9),
                "hunger": round(hunger, 9),
                "arousal": round(arousal, 9),
            }
        )
        plasticity.sleep()
        emit({"type": "sleep", "ts": f"{day}T20:00:00+00:00"})
        day_realized = 0.0
        day_abs_ret_sum = 0.0
        day_abs_ret_n = 0
        daily_spikes.fill(0)
        hunger_armed = True

    for ts in timeline:
        day = ts.date()
        if day != cur_day:
            if cur_day is not None:
                settle_and_sleep(cur_day)
            cur_day = day
            pointer_drawn = False
            emit({"type": "wake", "ts": ts.isoformat()})

        # 1. Mark prices (fills and marks are at this bar's close).
        for ticker, df in frames.items():
            i = df.index.get_indexer([ts], method="pad")[0]
            if i >= 0 and df.index[i] == ts:
                last_close[ticker] = float(df["close"].iloc[i])

        # 2. Mechanical exits, independent of encounters (DESIGN §6.5).
        for ticker in sorted(positions):
            pos = positions[ticker]
            price = last_close[ticker]
            pnl_pct = (price / pos.avg_cost - 1.0) * 100.0
            if pnl_pct <= -config.shock_adverse_pct:
                day_realized += close_position(ts, ticker, "shock")
        equity = equity_at()
        drawdown = max(0.0, 1.0 - equity / hatch_equity)
        if drawdown >= config.hunger_drawdown and hunger_armed:
            winners = sorted(
                (
                    (s, (last_close[s] / p.avg_cost - 1.0) * 100.0)
                    for s, p in positions.items()
                ),
                key=lambda t: (-t[1], t[0]),
            )
            if winners and winners[0][1] > 0.0:
                day_realized += close_position(ts, winners[0][0], "hunger")
                hunger_armed = False
        elif drawdown < 0.5 * config.hunger_drawdown:
            hunger_armed = True

        # 3. Plume filter (DESIGN §6.1): the day's smelliest movers.
        plumes = _plume_set(frames, ts, config.top_k)
        for ticker, _ in plumes:
            df = frames[ticker]
            i = df.index.get_indexer([ts], method="pad")[0]
            features, ret1, _ = _features_at(df, i)
            day_abs_ret_sum += abs(ret1)
            day_abs_ret_n += 1
        if plumes:
            if not pointer_drawn:
                # Seeded rotation offset (the loop RNG's daily draw).
                pointer = int(rng.integers(0, len(plumes)))
                pointer_drawn = True
            ticker, intensity = plumes[pointer % len(plumes)]
            pointer += 1

            # 4. Encounter: render -> odor -> taste -> noise floor -> step.
            df = frames[ticker]
            i = df.index.get_indexer([ts], method="pad")[0]
            window = df.iloc[max(0, i - VISION_WINDOW_BARS + 1) : i + 1]
            inp = encode_vision(window, chassis).astype(np.float64)
            features, _, _ = _features_at(df, i)
            inp[upn_rows] += SMELL_GAIN * encode_smell(ticker, features, chassis)
            if ticker in positions:
                pnl_pct = (last_close[ticker] / positions[ticker].avg_cost - 1.0) * 100.0
                inp += encode_taste(pnl_pct, chassis)
            inp += rng.standard_normal(n) * config.noise_sigma_mv
            out = sim.step(inp, config.ms_per_bar)
            spikes = out["spikes"]
            daily_spikes += spikes

            # 5. Readout: learned KC->MBON drive split by valence (see module
            # docstring); balance in [-1, 1] is the decision variable.
            spike_f = spikes.astype(np.float64)
            drive = plasticity.mbon_activation(spike_f)
            valence = plasticity.mbon_valence
            approach_drive = float(valence[valence > 0.0] @ drive[valence > 0.0])
            avoid_drive = float(-(valence[valence < 0.0] @ drive[valence < 0.0]))
            balance = (approach_drive - avoid_drive) / (
                approach_drive + avoid_drive + 1e-9
            )
            raw_score = float(valence @ drive)
            duration_s = config.ms_per_bar / 1000.0
            mbon_rate = float(spikes[mbon_rows].sum()) / mbon_rows.size / duration_s
            # Center on the innate balance; a silent encounter (no KC
            # activity) carries no signal and reads neutral.
            centered = (
                balance - anchor if approach_drive + avoid_drive > 0.0 else 0.0
            )

            emit(
                {
                    "type": "encounter",
                    "ts": ts.isoformat(),
                    "ticker": ticker,
                    "intensity": round(intensity, 9),
                    "balance": round(centered, 9),
                    "raw_balance": round(balance, 9),
                    "valence_readout": round(raw_score, 6),
                    "mbon_rate_hz": round(mbon_rate, 6),
                }
            )

            # 6. Decision + order (fills at this bar's close).
            held = ticker in positions
            price = last_close[ticker]
            action, reason = _decide(
                centered, held, len(positions) >= config.position_cap,
                config.approach_thr, config.avoid_thr,
            )
            shares = 0
            if action in ("buy", "add"):
                equity = equity_at()
                notional = _size_fraction(mbon_rate) * equity / config.position_cap
                shares = int(notional / price) if price > 0.0 else 0
                if shares < 1:
                    action, reason, shares = "pass", "dust", 0
            emit(
                {
                    "type": "decision",
                    "ts": ts.isoformat(),
                    "ticker": ticker,
                    "action": action,
                    "reason": reason,
                }
            )
            if action == "buy":
                cost = shares * price * (1.0 + config.fees)
                cash -= cost
                positions[ticker] = _Position(shares=shares, avg_cost=price)
                order(ts, ticker, "buy", reason, shares, price, None, cash)
            elif action == "add":
                cost = shares * price * (1.0 + config.fees)
                cash -= cost
                pos = positions[ticker]
                total = pos.shares + shares
                pos.avg_cost = (pos.avg_cost * pos.shares + shares * price) / total
                pos.shares = total
                order(ts, ticker, "buy", reason, shares, price, None, cash)
            elif action == "sell":
                day_realized += close_position(ts, ticker, reason)

        # 7. Death check (D14): equity <= hatch * (1 + threshold) kills the fly.
        equity = equity_at()
        if equity <= hatch_equity * (1.0 + config.death_threshold):
            emit(
                {
                    "type": "death",
                    "ts": ts.isoformat(),
                    "equity": round(equity, 6),
                    "hatch_equity": round(hatch_equity, 6),
                }
            )
            n_deaths += 1
            for ticker in sorted(positions):
                day_realized += close_position(ts, ticker, "death_liquidation")
            plasticity = Plasticity(chassis)
            sim.reset(config.seed)
            daily_spikes.fill(0)
            hatch_equity = cash
            hunger_armed = True
            emit({"type": "hatch", "hatch_equity": round(hatch_equity, 6)})
            equity = cash

            # A fresh brain re-calibrates its innate balance.
            anchor = _innate_balance(
                chassis, plasticity, sim, upn_rows, channels, list(frames),
                config.ms_per_bar,
            )
            emit({"type": "hatch", "hatch_equity": round(hatch_equity, 6)})
            equity = cash
        # 8. Bar receipt.
        equity_writer.writerow(
            [ts.isoformat(), f"{equity:.2f}", f"{cash:.2f}", len(positions)]
        )
        n_bars += 1

    if cur_day is not None:
        settle_and_sleep(cur_day)

    equity_file.close()
    events_file.close()
    return RunResult(
        run_dir=run_dir,
        n_bars=n_bars,
        n_events=n_events,
        n_orders=n_orders,
        n_deaths=n_deaths,
        final_equity=equity,
        wall_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _register() -> None:
    def builder(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "backtest", help="Run the foraging backtest loop (DESIGN §7)."
        )
        p.add_argument("--seed", type=int, required=True, help="Loop RNG seed.")
        p.add_argument("--start", required=True, help="Inclusive ISO date/timestamp (UTC).")
        p.add_argument("--end", required=True, help="Inclusive ISO date/timestamp (UTC).")
        p.add_argument("--top-k", type=int, default=5, help="Plume-filter width (D3).")
        p.add_argument(
            "--position-cap", type=int, default=10, help="Concurrent-position cap (D13)."
        )
        p.add_argument(
            "--ms-per-bar", type=float, default=500.0, help="Biological time per bar (D16)."
        )
        p.add_argument(
            "--death-threshold", type=float, default=-0.50,
            help="Death drawdown on hatch equity (D14).",
        )
        p.add_argument("--fees", type=float, default=0.0, help="Proportional per-side fee.")
        p.add_argument("--out-dir", default=None, help="Override the artifact directory.")
        p.set_defaults(func=_cmd_backtest)

    register_command("backtest", builder)


def _cmd_backtest(args: argparse.Namespace) -> int:
    config = BacktestConfig(
        seed=args.seed,
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        position_cap=args.position_cap,
        ms_per_bar=args.ms_per_bar,
        death_threshold=args.death_threshold,
        fees=args.fees,
        out_dir=args.out_dir,
    )
    result = run_backtest(config)
    print(
        f"backtest seed={config.seed} {config.start}..{config.end}\n"
        f"  artifacts : {result.run_dir} (equity.csv, events.jsonl)\n"
        f"  bars      : {result.n_bars}\n"
        f"  events    : {result.n_events} ({result.n_orders} orders)\n"
        f"  deaths    : {result.n_deaths}\n"
        f"  equity    : {result.final_equity:.2f}\n"
        f"  wall time : {result.wall_s:.1f}s"
    )
    return 0


_register()
