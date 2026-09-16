"""T7: the foraging backtest loop (DESIGN §5-§7, D3/D13/D14/D16).

One simulated fly forages a 1-minute-bar market session. Per DESIGN §6 the
fly does not watch a watchlist: every bar, the basket is ranked by plume
intensity (|20-bar momentum| × 20 + relative volume — direction-agnostic,
DESIGN v0.6) and the ``top_k`` smelliest movers become the active plume set;
dead stocks (zero 1-bar return AND zero 20-bar momentum) are odorless. The fly
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
``valence_readout`` and raw balance are also logged for T9). The anchor is
re-measured at every sleep (T9, v0.6 daily refresh): as learning shifts the
innate bias, the centering drifts back with it instead of going stale.
``balance >
approach_thr``
→ BUY/add, ``balance < -avoid_thr`` → SELL (valence-flip exit on a held
position), otherwise pass. Two v0.6 additions ride on the centered balance:
the LC→MBON *structural looming* response (connectome synapses, log1p-scaled
like ``Plasticity.baseline``) enters at a fixed weight —
``balance_used = centered + lambda_struct × structural_score`` — so a
crash-shaped bar moves the decision even with a silent learned readout; and
the decision consumes ``balance_used``, while the encounter receipt logs both
the raw pieces and the combined value. Position size maps the MBON population firing
rate (post-spikes over MBON nodes / duration) through
``tanh(rate / MBON_RATE_SCALE_HZ)``, floored at ``MIN_SIZE_FRAC`` and capped
at 1.0; the per-order notional is ``frac * equity / position_cap`` (D13).
Caveat documented for T9: at the current engine gain constants MBONs are
sub-threshold (KC→MBON quantal PSPs ≈ 0.01 mV/synapse), so MBON firing rates
are ~0 and sizing rides the floor.

Exits (DESIGN §6): a held ticker re-encountered with an avoid balance is
closed (valence flip); any open position whose unrealized P&L drops below
``-shock_adverse_pct`` triggers a close (sharp adverse move → shock —
the biological stop-loss bypasses the brain); when portfolio drawdown from
hatch equity crosses ``hunger_drawdown`` the fly closes its largest winner
(hunger → take profit), re-arming when drawdown recovers below half the
threshold.

TRAINING2 §5 adds a mechanical exit grid on the same pessimistic next-bar
queue: a per-position trailing stop (``trailing_stop_pct`` below the max
close since entry, tracked on ``_Position.high_water``), an ATR stop
(``atr_stop_mult`` × the 20-bar close-to-close return σ below entry), and
bools gating the valence-flip (``valence_flip_exit``) and hunger
(``hunger_exit``) exits. §5.1's T-1 exit-forecast gate (``exit_forecast``)
scores each held position with the encounter pipeline WITHOUT taste —
``exit_score = -balance_used`` — and sells when the score clears
``exit_score_thr``; the default cadence reuses the encountered ticker's own
encounter readout (zero extra sim steps — taste is dead code: PAM/PPL1 are
sign-0 and never spike), while ``exit_score_every_bar`` scores held
tickers the encounter path missed (up to ``position_cap`` extra LIF steps
per bar; Phase-B budget experiment).

**Execution model (pessimistic next-bar fills).** An order decided during
bar ``t`` is PENDING and executes on the symbol's NEXT available bar (the
symbol's own next bar; gaps/halts are skipped, and an order unfilled at a
session close carries into the next session). No look-ahead: the decision
uses only bar-``t``-and-earlier data, the fill consumes only later bars.
Fills are pessimistically intra-bar — BUY at the execution bar's HIGH, SELL
at its LOW (parameter-free and provably conservative: any real fill lies
within [low, high], so this can only understate performance). Sells close
the WHOLE current position at fill time; the position cap (D13) counts open
positions plus queued buys so fills can never exceed it. Equity marking
stays at close prices — marking is not execution. Whole shares, with
proportional ``fees`` on each side (default 0 — commission-free
brokerage). Pending orders are cancelled with a ``cancel`` event when the
fly dies (the brain that placed them is discarded) and at run end if their
symbol never prints again. Order receipts carry both bars (``decision_ts``
and execution ``ts``) plus the executed price for fill-quality audits.
``config.fill_mode="close"`` restores the previous same-bar close fills —
documented as LOOK-AHEAD-BIASED and kept only as a comparison escape hatch.
Cash accounting is paper accounting: buys deduct cash without a balance
check (a next-bar fill can push cash negative and gross exposure past
equity — margin, in effect); equity = cash + marked positions is the only
solvency signal, and D14's death threshold acts on it.

Close of day (DESIGN §7): the day's realized P&L settles into sugar/shock
(``NeuromodState.reward/punishment`` as a fraction of hatch equity), hunger
maps drawdown to [0, 1] hitting 1.0 exactly at the death threshold (D14),
arousal maps the day's mean |1-bar return| to [0, 1]; one
``Plasticity.observe`` consumes the day's accumulated spike counts, then
``Plasticity.sleep()`` consolidates, the anchor is re-measured, and the fly
logs the sleep + anchor events. Complementing the diffuse daily ritual, each
position close emits a precise ``trade_credit`` (D6, v0.6): the eligibility
snapshot taken at the opening buy/add is credited through
``Plasticity.observe_trade`` with a dopamine gate built from the trade's
realized P&L. Death liquidation clears the snapshots — a dying brain gets no
per-exit credit.

Death (D14): when equity <= ``hatch_equity * (1 + death_threshold)`` the fly
dies: remaining positions are liquidated immediately at the death bar's
last close (settle semantics unchanged by the fill model), the dying fly's
pending orders are cancelled, a fresh ``Plasticity`` hatches (brain reset),
and the new fly's hatch equity is the post-liquidation equity. Death and
hatch are logged as events (T9 exercises this).

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
from fruitfly.connectome import Chassis, load_stripped_chassis, load_whole_fly
from fruitfly.data import BASKET, load_bars, parse_basket_file, resolve_basket
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
    "DEFAULT_FILL_MODE",
    "FILL_MODES",
    "FILL_MODE_CLOSE",
    "BacktestConfig",
    "PendingOrder",
    "PendingOrderBook",
    "RunResult",
    "resolve_chassis_fields",
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

#: Calibrated short-term-depression parameters for the whole-fly chassis
#: (T12b sweep, reports/t12b-apl-std.md: beta=0.1, tau_rec=500 ms revives KC
#: yield to stripped-brain level). These are the DEFAULTS for
#: ``chassis="whole"`` so a whole-fly run cannot forget them.
WHOLE_CHASSIS_STD_BETA = 0.1
WHOLE_CHASSIS_STD_TAU_REC_MS = 500.0

#: Fill models. ``"pessimistic_next_bar"`` (default): an order decided on
#: bar t is pending and fills on the symbol's next available bar — BUY at
#: that bar's HIGH, SELL at its LOW (parameter-free, provably conservative:
#: any real fill lies within [low, high], so performance can only be
#: understated). ``"close"``: legacy same-bar close fills — documented as
#: LOOK-AHEAD-BIASED (the decision sees bar-t data and fills at bar t's
#: close); kept only as a comparison escape hatch.
DEFAULT_FILL_MODE = "pessimistic_next_bar"
FILL_MODES = ("pessimistic_next_bar", "close")
FILL_MODE_CLOSE = "close"

#: Entry-credit objective knobs (TRAINING2 §2). The defaults are the no-op
#: config: ``entry_credit="off"`` disables the whole machinery byte-for-byte.
ENTRY_CREDIT_MODES = ("off", "forecast", "advantage")
TRADE_CREDIT_MODES = ("realized", "forecast", "mix")
DAILY_OBSERVE_MODES = ("on", "off")

#: Option B volatility-tercile edges (TRAINING2 §2B — fixed constants, NOT
#: data-derived): 20-bar close-to-close return std-dev bucket edges. T9
#: calibration candidates.
VOL_TERCILE_EDGES = (0.001, 0.002)


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
    #: Grudge-protection width (T9, v0.6): strongest learned KC→MBON
    #: associations the sleep consolidation pass preserves.
    grudge_top_k: int = 16
    #: Structural looming drive weight (DESIGN v0.6 §5): the LC→MBON
    #: structural response enters the decision balance at this fixed λ:
    #: ``balance_used = centered + lambda_struct × structural_score``.
    lambda_struct: float = 0.05
    #: Concurrent-position cap (D13).
    position_cap: int = 10
    #: Biological time stepped per 1-minute bar (D16).
    ms_per_bar: float = 500.0
    #: Death threshold on equity vs hatch equity (D14).
    death_threshold: float = DEATH_THRESHOLD
    #: Proportional per-side fee (0 = paper fills).
    fees: float = 0.0
    #: Fill model: ``"pessimistic_next_bar"`` (default) or the legacy
    #: look-ahead-biased ``"close"`` (see ``DEFAULT_FILL_MODE`` above).
    fill_mode: str = DEFAULT_FILL_MODE
    initial_cash: float = INITIAL_CASH
    shock_adverse_pct: float = SHOCK_ADVERSE_PCT
    hunger_drawdown: float = HUNGER_DRAWDOWN
    approach_thr: float = APPROACH_THR
    avoid_thr: float = AVOID_THR
    noise_sigma_mv: float = NOISE_SIGMA_MV
    dt_ms: float = DT_MS
    #: Override the artifact directory (tests use this to avoid collisions).
    out_dir: str | Path | None = None
    #: Restored plasticity (T9 larval training): initial KC→MBON weights,
    #: shape-(n_kc, n_mbon) float64 (e.g. ``load_larval_weights`` output).
    #: ``None`` (default) keeps the structural baseline — behavior unchanged.
    initial_weights: np.ndarray | None = None
    #: Brain chassis: ``"whole"`` (default since 2026-09-16, D22 shootout
    #: verdict — the full proofread fly, labeled for the loop's
    #: population-keyed readouts, see ``_load_whole_chassis``) or
    #: ``"stripped"`` (the fast dev/test lane).
    chassis: str = "whole"
    #: Opt-in short-term synaptic depression (T12b). ``None`` = off (the
    #: engine stays bit-identical to the pre-STD one). On ``chassis="whole"``
    #: an unset parameter defaults to the calibrated T12b values
    #: (``WHOLE_CHASSIS_STD_BETA`` / ``WHOLE_CHASSIS_STD_TAU_REC_MS``).
    std_beta: float | None = None
    std_tau_rec_ms: float | None = None
    #: Explicit basket override (TRAINING2 §3): the parsed ``--basket-file``
    #: symbol list; ``None`` (default) runs the module ``BASKET``. Guarded
    #: against the frozen never-seen eval basket by ``resolve_basket``.
    basket: list[str] | None = None
    #: Entry-credit objective (TRAINING2 §2): ``"off"`` (default — no entry
    #: credits at all), ``"forecast"`` (Option A: supervised entry-forecast
    #: reward) or ``"advantage"`` (Option B: bucket-EMA advantage gating).
    entry_credit: str = "off"
    #: Forecast horizon in 1-min bars (TRAINING2 §2): pins the eval window
    #: (§6 G-A.1) and the artifact meta. The runtime credit sites are the
    #: position close and the day settle (spec-pinned, horizon-uncapped).
    horizon_bars: int = 30
    #: Option A tanh scale on forward returns: ``a = tanh(r_fwd / r_scale)``.
    r_scale: float = 0.005
    #: Option B tanh scale on the advantage: ``a = tanh((r_fwd - b) / a_scale)``.
    a_scale: float = 0.003
    #: Option B baseline EMA rate: ``b <- (1 - alpha) * b + alpha * r_fwd``.
    baseline_alpha: float = 0.05
    #: Missed-winner weight (TRAINING2 §2A): approached-but-passed buys AND
    #: avoided run-ups (r_fwd > 0) are punished at this weight.
    miss_weight: float = 0.25
    #: Correct-avoid weight (TRAINING2 §2A): avoided encounters with
    #: ``r_fwd < 0`` earn positive credit at this weight.
    avoid_correct_weight: float = 0.5
    #: Close-fill entry-credit content (TRAINING2 §2A): the incumbent
    #: realized/notional gate (``"realized"``), the forward-return forecast
    #: (``"forecast"``), or the blend (``"mix"``).
    trade_credit_mode: str = "realized"
    #: ``"mix"`` blend weight: ``(1 - w) * realized + w * forecast``.
    mix_weight: float = 0.5
    #: Daily pooled sugar/shock ritual (D6): ``"on"`` (default) or
    #: ``"off"`` — the ablation skips ONLY the ``plasticity.observe`` call.
    daily_observe: str = "on"
    #: Identity attenuation (TRAINING2 §3): scales the smell identity term
    #: AND the ``_innate_balance`` anchor-sniff profile (which bypasses
    #: ``encode_smell``); 1.0 = incumbent behavior, 0.0 = state-only odor.
    id_scale: float = 1.0
    #: Option B warm start (TRAINING2 §2B): restored bucket baselines
    #: (an artifact-meta round-trip), keys ``(sign, vol_bucket, dir_bucket)``
    #: tuples — string ``"sign|vol|dir"`` keys and 3-sequences are coerced.
    initial_baselines: dict[tuple[int, int, int], float] | None = None
    #: Asymmetric gate gains (TRAINING2 §4): reward_gain > punishment_gain
    #: makes a winner teach more than an equal-size loss. Defaults 1.0/1.0
    #: = the incumbent symmetric gate (bit-identical no-op). Plumbed to
    #: ``Plasticity.__init__`` (which already accepted them; the loop
    #: previously dropped them).
    reward_gain: float = 1.0
    punishment_gain: float = 1.0
    #: Mechanical exit grid (TRAINING2 §5.2). ``None`` = off (no-op).
    #: Trailing stop: sell when price falls this many percent below the
    #: per-position high-water mark (max close since entry, tracked on
    #: ``_Position.high_water``).
    trailing_stop_pct: float | None = None
    #: ATR stop: sell when ``price < entry - k * sigma20 * price`` where
    #: ``sigma20`` is ``build_features``' 20-bar close-to-close return
    #: std-dev (the existing ATR proxy; no true-range helper).
    atr_stop_mult: float | None = None
    #: Keep/drop the ``_decide`` sell-on-avoid exit (valence flip).
    valence_flip_exit: bool = True
    #: Keep/drop the close-largest-winner hunger exit.
    hunger_exit: bool = True
    #: T-1 exit-forecast gate (TRAINING2 §5.1): score each held position
    #: with the encounter pipeline WITHOUT taste (``exit_score =
    #: -balance_used``) and sell when the score clears ``exit_score_thr``.
    exit_forecast: bool = False
    #: Exit-forecast threshold on ``-balance_used``; required when
    #: ``exit_forecast`` is on, unused (``None``) while it is off.
    exit_score_thr: float | None = None
    #: Exit-forecast scoring cadence: ``False`` (default) scores only bars
    #: where the held ticker IS the encountered one (zero extra sim steps);
    #: ``True`` scores every held ticker every bar — up to
    #: ``position_cap`` extra LIF steps per bar (Phase-B budget experiment).
    exit_score_every_bar: bool = False

    def __post_init__(self) -> None:
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
        if self.entry_credit not in ENTRY_CREDIT_MODES:
            raise ValueError(
                f"entry_credit must be one of {ENTRY_CREDIT_MODES}, got {self.entry_credit!r}"
            )
        if self.trade_credit_mode not in TRADE_CREDIT_MODES:
            raise ValueError(
                f"trade_credit_mode must be one of {TRADE_CREDIT_MODES}, "
                f"got {self.trade_credit_mode!r}"
            )
        if self.trailing_stop_pct is not None and self.trailing_stop_pct <= 0.0:
            raise ValueError("trailing_stop_pct must be positive when set")
        if self.atr_stop_mult is not None and self.atr_stop_mult <= 0.0:
            raise ValueError("atr_stop_mult must be positive when set")
        if self.exit_forecast and self.exit_score_thr is None:
            raise ValueError("exit_forecast requires exit_score_thr")
        if self.exit_score_every_bar and not self.exit_forecast:
            raise ValueError("exit_score_every_bar requires exit_forecast")
        if self.daily_observe not in DAILY_OBSERVE_MODES:
            raise ValueError(
                f"daily_observe must be one of {DAILY_OBSERVE_MODES}, got {self.daily_observe!r}"
            )
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.r_scale <= 0.0:
            raise ValueError("r_scale must be positive")
        if self.a_scale <= 0.0:
            raise ValueError("a_scale must be positive")
        if not 0.0 < self.baseline_alpha <= 1.0:
            raise ValueError("baseline_alpha must be in (0, 1]")
        if self.miss_weight < 0.0 or self.avoid_correct_weight < 0.0:
            raise ValueError("miss_weight and avoid_correct_weight must be non-negative")
        if not 0.0 <= self.mix_weight <= 1.0:
            raise ValueError("mix_weight must be in [0, 1]")
        if not 0.0 <= self.id_scale <= 1.0:
            raise ValueError("id_scale must be in [0, 1]")
        if self.reward_gain < 0.0 or self.punishment_gain < 0.0:
            raise ValueError("reward_gain and punishment_gain must be non-negative")
        if self.initial_baselines is not None:
            restored: dict[tuple[int, int, int], float] = {}
            for key, value in self.initial_baselines.items():
                parts = key.split("|") if isinstance(key, str) else key
                restored[tuple(int(p) for p in parts)] = float(value)
            object.__setattr__(self, "initial_baselines", restored)
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
    #: Final KC→MBON weights of the last-hatched fly (T9 larval training
    #: reads this to persist ``data/fly-larval-weights.npz``). A fresh,
    #: never-trained run returns the structural baseline.
    final_weights: np.ndarray | None = None
    #: Option B bucket baselines at run end (TRAINING2 §2B): the trained
    #: EMA dict for the artifact-meta round-trip; ``None`` unless
    #: ``entry_credit == "advantage"``.
    baselines: dict[tuple[int, int, int], float] | None = None


@dataclass
class _Position:
    """One open position plus its mechanical-exit state (TRAINING2 §5.2)."""

    shares: int
    avg_cost: float
    #: High-water mark: max close since entry — the trailing-stop
    #: reference. Initialized at the entry fill price, raised each bar.
    high_water: float = 0.0


@dataclass
class _EntryRecord:
    """One signal-bearing encounter in the entry-credit ledger (TRAINING2 §2).

    The eligibility compact factors the encounter's ``eligibility_driver``
    increment as (habituation-gated KC activity, MBON post activity,
    structural support): the full (n_kc, n_mbon) trace equals
    ``kc_activity[:, None] * post[None, :]`` masked to ``support`` — ~33 KB
    per encounter instead of a 3.1 MB full-trace copy (spec §2 memory note).
    ``Plasticity.observe_trade_compact`` rebuilds the outer product at
    credit time. The KC activity carries the habituation gate AS OF the
    encounter (habituation moves between capture and credit).
    """

    #: Ledger index — the ``supersedes`` handle on entry_credit events.
    rec_id: int
    decision_ts: pd.Timestamp
    ticker: str
    #: float64 (n_kc,): ``kc_activity(spike_f) * habituation`` at capture.
    kc_activity: np.ndarray
    #: float64 (n_mbon,): the structural MBON response (eligibility post).
    post: np.ndarray
    #: bool (n_kc, n_mbon): the shared read-only plasticity support mask.
    support: np.ndarray
    #: The encounter bar's close — a price the decision already had.
    base_price: float
    #: "buy" | "pass_signal" (finalized after the encounter's decision).
    action: str
    #: "pending" | "credited@settle" | "final" (single-credit state machine).
    credit_state: str = "pending"
    #: The settle credit (reward, punishment) a later close supersedes.
    provisional: tuple[float, float] | None = None
    #: Option B state key (sign, vol_bucket, dir_bucket); None under Option A.
    bucket: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class PendingOrder:
    """A market order decided on one bar, awaiting its next-bar execution."""

    #: The bar whose data produced the decision (receipts audit field).
    decision_ts: pd.Timestamp
    ticker: str
    #: "buy" | "sell".
    side: str
    reason: str
    #: Fixed share count for buys. Sells leave this ``None``: they close the
    #: WHOLE current position at fill time (the book can change while the
    #: order waits, so decision-time sizing would be wrong).
    shares: int | None = None


class PendingOrderBook:
    """The pessimistic next-bar fill queue — the loop's fill engine.

    An order decided on bar ``t`` of a symbol fills on that symbol's NEXT
    available bar (skipping gaps/halts); BUY fills at that bar's HIGH, SELL
    at its LOW. An order unfilled at a session close simply stays queued
    and carries into the next session. Pure and deterministic (no wall
    clock, no RNG): the same bar data always produces the same fills, so
    the same-seed byte-identity gate holds.
    """

    def __init__(
        self, frames: dict[str, pd.DataFrame], fill_mode: str = DEFAULT_FILL_MODE
    ) -> None:
        if fill_mode not in FILL_MODES:
            raise ValueError(f"fill_mode must be one of {FILL_MODES}, got {fill_mode!r}")
        self._frames = frames
        self._fill_mode = fill_mode
        self._pending: list[PendingOrder] = []

    @property
    def fill_mode(self) -> str:
        return self._fill_mode

    def submit(self, order: PendingOrder) -> None:
        """Queue an order decided on the current bar."""
        self._pending.append(order)

    def has_pending(self, ticker: str) -> bool:
        """True while any order for ``ticker`` is still queued (the
        mechanical-exit guards use this to avoid duplicate triggers)."""
        return any(o.ticker == ticker for o in self._pending)

    def n_buys(self) -> int:
        """Queued buy count: the position cap (D13) counts open positions
        plus queued buys, so fills can never exceed the cap."""
        return sum(1 for o in self._pending if o.side == "buy")

    def pending_orders(self) -> list[PendingOrder]:
        return list(self._pending)

    def drain(self, ts: pd.Timestamp) -> list[tuple[PendingOrder, float]]:
        """Orders whose symbol's next available bar is ``ts``, FIFO, paired
        with the pessimistic fill price (high for buys, low for sells)."""
        fills: list[tuple[PendingOrder, float]] = []
        for o in self._pending:
            df = self._frames.get(o.ticker)
            if df is None or ts <= o.decision_ts or ts not in df.index:
                continue
            price = float(df.at[ts, "high" if o.side == "buy" else "low"])
            fills.append((o, price))
        if fills:
            filled = {id(o) for o, _ in fills}
            self._pending = [o for o in self._pending if id(o) not in filled]
        return fills

    def cancel_all(self) -> list[PendingOrder]:
        """Drop and return every queued order (death / run-end cancels)."""
        orders = self._pending
        self._pending = []
        return orders


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

    Intensity = |20-bar momentum| × 20 + volume ratio (DESIGN v0.6 amendment):
    the 20-bar move expressed in 1-bar-equivalent magnitude plus the volume
    ratio. Direction-agnostic — persistent movers of either sign surface, not
    just last-bar jumps. A dead ticker (``ret1 == 0`` AND ``mom20 == 0``) is
    odorless and skipped entirely; a flat last bar on an otherwise-moving
    ticker still smells. Returns the top ``top_k`` as ``(ticker, intensity)``,
    ties broken by ticker for determinism.
    """
    ranked: list[tuple[float, str]] = []
    for ticker, df in frames.items():
        i = df.index.get_indexer([ts], method="pad")[0]
        if i < 1 or df.index[i] != ts:
            continue
        features, ret1, volume_ratio = _features_at(df, i)
        mom20 = features["mom20"]
        if ret1 == 0.0 and mom20 == 0.0:
            continue  # dead -> odorless
        ranked.append((abs(mom20) * 20.0 + volume_ratio, ticker))
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
    id_scale: float = 1.0,
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
    # Identity attenuation (TRAINING2 §3): the anchor sniff bypasses
    # ``encode_smell``, so id_scale must scale THIS input directly. At 0.0
    # the sniff is silent and the anchor degenerates gracefully to 0.0
    # (centered == raw balance, the documented silent-anchor behavior).
    inp[upn_rows] = SMELL_GAIN * id_scale * profile[channels]
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


def _entry_bucket(features: dict[str, float]) -> tuple[int, int, int]:
    """Option B state key (TRAINING2 §2B): ``(sign(mom20), vol tercile, rsi >= 50)``.

    Volatility terciles use the FIXED constants :data:`VOL_TERCILE_EDGES`
    (not data-derived); the momentum sign is 0 for a momentum-less mover;
    the RSI direction split sits at 50. Deterministic, coarse, float64-free.
    """
    mom20 = features.get("mom20", 0.0)
    sign = 0 if mom20 == 0.0 else (1 if mom20 > 0.0 else -1)
    vol = features.get("volatility", 0.0)
    vol_bucket = (
        0 if vol < VOL_TERCILE_EDGES[0]
        else (1 if vol < VOL_TERCILE_EDGES[1] else 2)
    )
    return (sign, vol_bucket, 1 if features.get("rsi", 50.0) >= 50.0 else 0)


def _entry_gate_input(
    config: BacktestConfig,
    r_fwd: float,
    baselines: dict[tuple[int, int, int], float],
    bucket: tuple[int, int, int] | None,
) -> float:
    """The entry-credit gate input ``a`` (TRAINING2 §2).

    Option A: the absolute forecast ``tanh(r_fwd / r_scale)``. Option B: the
    advantage ``tanh((r_fwd - b) / a_scale)`` against the bucket baseline —
    the PRE-update EMA value; the caller updates the baseline only after the
    credit is computed (no target leakage into its own prediction).
    """
    if config.entry_credit == "advantage":
        return math.tanh((r_fwd - baselines.get(bucket, 0.0)) / config.a_scale)
    return math.tanh(r_fwd / config.r_scale)


def _update_baseline(
    baselines: dict[tuple[int, int, int], float],
    bucket: tuple[int, int, int],
    r_fwd: float,
    alpha: float,
) -> None:
    """Option B EMA update (TRAINING2 §2B): ``b <- (1 - a) * b + a * r_fwd``.

    Called strictly AFTER the credit that consumed the pre-update value.
    """
    baselines[bucket] = (1.0 - alpha) * baselines.get(bucket, 0.0) + alpha * r_fwd


def _exit_forecast_readout(
    chassis: Chassis,
    plasticity: Plasticity,
    sim: LIFSim,
    upn_rows: np.ndarray,
    w_struct: np.ndarray | None,
    lc_rows: np.ndarray,
    anchor: float,
    config: BacktestConfig,
    ticker: str,
    df: pd.DataFrame,
    i: int,
    rng: np.random.Generator,
) -> float:
    """T-1 exit-forecast score (TRAINING2 §5.1): the encounter pipeline
    WITHOUT taste for held ticker ``ticker`` at bar ``i`` — vision + smell
    + the seeded noise floor, one LIF step, then the centered balance with
    the structural looming drive, negated (``exit_score = -balance_used``).

    Consumes ``rng`` (one noise draw) and advances the sim/STD state — the
    documented Phase-B budget cost of the every-bar cadence. The default
    cadence never calls this: it reuses the encountered ticker's own
    encounter readout (taste is dead code — PAM/PPL1 are sign-0 and never
    spike — so that readout IS the without-taste recomputation, bit for
    bit).
    """
    n = chassis.n_neurons
    window = df.iloc[max(0, i - VISION_WINDOW_BARS + 1) : i + 1]
    inp = encode_vision(window, chassis).astype(np.float64)
    features, _, _ = _features_at(df, i)
    inp[upn_rows] += SMELL_GAIN * encode_smell(
        ticker, features, chassis, id_scale=config.id_scale
    )
    inp += rng.standard_normal(n) * config.noise_sigma_mv
    out = sim.step(inp, config.ms_per_bar)
    spike_f = out["spikes"].astype(np.float64)
    drive = plasticity.mbon_activation(spike_f)
    valence = plasticity.mbon_valence
    approach_drive = float(valence[valence > 0.0] @ drive[valence > 0.0])
    avoid_drive = float(-(valence[valence < 0.0] @ drive[valence < 0.0]))
    balance = (approach_drive - avoid_drive) / (
        approach_drive + avoid_drive + 1e-9
    )
    centered = (
        balance - anchor if approach_drive + avoid_drive > 0.0 else 0.0
    )
    if w_struct is not None:
        structural = float(
            plasticity.mbon_valence @ (w_struct.T @ out["spikes"][lc_rows])
        )
    else:
        structural = 0.0
    return -(centered + config.lambda_struct * structural)


def _load_chassis() -> Chassis:
    """Stripped-chassis source (test seam: tests monkeypatch this to a tiny
    synthetic fixture)."""
    return load_stripped_chassis()


def _load_whole_chassis() -> Chassis:
    """Whole-fly chassis source (test seam: tests monkeypatch this to a tiny
    synthetic whole-like fixture).

    Production: the cached whole fly with the stripped chassis's
    population/region labels transferred onto matched bodyIds — the labeled
    whole-fly chassis the T12b replay seam injected. The raw whole-fly cache
    labels every node ``population="whole"``; the loop's sensory encoders and
    KC/MBON readouts key off the population column.
    """
    from fruitfly.connectome import label_whole_chassis

    return label_whole_chassis(load_whole_fly(), load_stripped_chassis())


def _resolve_chassis(config: BacktestConfig) -> Chassis:
    """Chassis per ``config.chassis``: whole fly (default, D22) or stripped."""
    if config.chassis == "whole":
        return _load_whole_chassis()
    return _load_chassis()


def resolve_chassis_fields(
    chassis: str, std_beta: float | None, std_tau_rec_ms: float | None
) -> tuple[str, float | None, float | None]:
    """Validate + normalize the (chassis, STD) config triple.

    Shared by :class:`BacktestConfig` and the adult harness config. The
    whole-fly chassis gets the calibrated T12b STD defaults
    (:data:`WHOLE_CHASSIS_STD_BETA` / :data:`WHOLE_CHASSIS_STD_TAU_REC_MS`)
    for any STD parameter left unset, so a whole-fly run cannot forget them;
    elsewhere the STD parameters must be given together or not at all (the
    default: off — the engine stays bit-identical to the pre-STD one).
    """
    if chassis not in ("stripped", "whole"):
        raise ValueError(f"chassis must be 'stripped' or 'whole', got {chassis!r}")
    if chassis == "whole":
        if std_beta is None:
            std_beta = WHOLE_CHASSIS_STD_BETA
        if std_tau_rec_ms is None:
            std_tau_rec_ms = WHOLE_CHASSIS_STD_TAU_REC_MS
    if (std_beta is None) != (std_tau_rec_ms is None):
        raise ValueError("std_beta and std_tau_rec_ms must be given together")
    if std_beta is not None and not 0.0 < std_beta < 1.0:
        raise ValueError("std_beta must be in (0, 1)")
    if std_tau_rec_ms is not None and std_tau_rec_ms <= 0:
        raise ValueError("std_tau_rec_ms must be positive")
    return chassis, std_beta, std_tau_rec_ms


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
    chassis = _resolve_chassis(config)
    plasticity = Plasticity(
        chassis,
        top_k=config.grudge_top_k,
        reward_gain=config.reward_gain,
        punishment_gain=config.punishment_gain,
    )
    sim = LIFSim(
        chassis,
        dt_ms=config.dt_ms,
        seed=config.seed,
        std_beta=config.std_beta,
        std_tau_rec_ms=config.std_tau_rec_ms,
    )
    if config.initial_weights is not None:
        restored = np.asarray(config.initial_weights, dtype=np.float64)
        if restored.shape != plasticity.weights.shape:
            raise ValueError(
                f"initial_weights shape {restored.shape} does not match the "
                f"chassis KC→MBON view {plasticity.weights.shape}"
            )
        plasticity.weights[...] = restored
    # The loop owns the only RNG: seeded noise floor + daily rotation offsets.
    rng = np.random.Generator(np.random.PCG64(config.seed))

    pop = chassis.nodes["population"].to_numpy()
    upn_rows, channels = upn_channels(chassis)
    mbon_rows = np.flatnonzero(pop == "MBON")
    # Structural looming view (DESIGN v0.6 §5): the LC→MBON connectome
    # synapses as a dense float64 matrix, log1p-scaled to mean 1 over their
    # nonzero support (the ``Plasticity.baseline`` convention). A chassis
    # without LC-looming cells — or without LC→MBON synapses — leaves the
    # structural score at 0 for every encounter.
    lc_rows = np.flatnonzero(pop == "LC-looming")
    w_struct: np.ndarray | None = None
    if lc_rows.size:
        w_lc = np.asarray(chassis.adj[lc_rows][:, mbon_rows].toarray(), dtype=np.float64)
        lc_support = w_lc > 0.0
        if lc_support.any():
            w_lc = np.log1p(w_lc)
            w_lc /= w_lc[lc_support].mean()
            w_struct = w_lc
    n = chassis.n_neurons

    start_ts, end_ts = _window(config.start, config.end)
    # Effective basket (TRAINING2 §3): the explicit ``--basket-file``
    # override or the module BASKET, guarded against the frozen never-seen
    # eval basket (missing guard file = no-op). ``_innate_balance`` follows
    # the parsed list through ``frames`` keys.
    basket = resolve_basket(
        config.basket if config.basket is not None else list(BASKET)
    )
    frames = {
        sym: df
        for sym, df in load_bars(basket, None, None).items()
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
        chassis, plasticity, sim, upn_rows, channels, list(frames),
        config.ms_per_bar, id_scale=config.id_scale,
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
        ts: pd.Timestamp, o: PendingOrder, price: float, shares: int,
        realized_pnl: float | None, cash_after: float,
    ) -> None:
        nonlocal n_orders
        n_orders += 1
        emit(
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
                "n_positions_after": len(positions),
            }
        )

    def cancel(o: PendingOrder, ts: pd.Timestamp, cancel_reason: str) -> None:
        emit(
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

    # --- mutable run state -------------------------------------------------
    cash = config.initial_cash
    hatch_equity = config.initial_cash
    positions: dict[str, _Position] = {}
    last_close: dict[str, float] = {}
    # Per-exit dopamine (DESIGN v0.6 D6): ticker -> eligibility snapshot
    # captured at the position's opening buy/add fill, credited at close.
    entry_elig: dict[str, np.ndarray] = {}
    pending_book = PendingOrderBook(frames, config.fill_mode)
    cur_day: date | None = None
    pointer = 0
    pointer_drawn = False
    daily_spikes = np.zeros(n, dtype=np.int64)
    daily_mbon_drive = np.zeros(mbon_rows.size, dtype=np.float64)
    # Intraday eligibility trace (DESIGN v0.6 D6 fix): the live
    # ``plasticity.eligibility`` trace is consumed (zeroed) by ``sleep()``
    # and only advanced by the daily ``observe()``, so it is always
    # all-zero intraday. The loop therefore composes its own trace from the
    # pure ``Plasticity.eligibility_driver`` increments — this is the trace
    # a buy snapshots for the per-exit credit.
    intraday_elig = np.zeros(
        (plasticity.kc_index.size, plasticity.mbon_index.size), dtype=np.float64
    )
    # Entry-credit ledger (TRAINING2 §2): one record per signal-bearing
    # encounter with a compact eligibility snapshot (never a full-trace
    # copy). ``ledger_by_key`` links buy fills back to their decision
    # record; ``open_entry`` mirrors ``entry_elig`` — the RECORD behind
    # each open position's opening buy/add (latest add wins).
    encounter_ledger: list[_EntryRecord] = []
    ledger_by_key: dict[tuple[str, pd.Timestamp], _EntryRecord] = {}
    open_entry: dict[str, _EntryRecord] = {}
    filled_ids: set[int] = set()
    # Option B bucket baselines (TRAINING2 §2B): a deterministic online
    # EMA, optionally warm-started from the artifact-meta round-trip.
    baselines: dict[tuple[int, int, int], float] = dict(
        config.initial_baselines or {}
    )
    daily_signal_encounters = 0
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

    def apply_buy(ts: pd.Timestamp, o: PendingOrder, price: float) -> None:
        """Buy/add fill on the paper book: whole shares, proportional fees,
        cash deducted without a balance check (documented margin)."""
        nonlocal cash
        shares = o.shares
        cash -= shares * price * (1.0 + config.fees)
        pos = positions.get(o.ticker)
        if pos is None:
            positions[o.ticker] = _Position(
                shares=shares, avg_cost=price, high_water=price
            )
        else:
            total = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + shares * price) / total
            pos.shares = total
        # Snapshot the INTRADAY eligibility trace for this position's
        # opening encounter (DESIGN v0.6 D6). Overwritten on add — the
        # latest snapshot wins, because the realized P&L at close is
        # credited to the most recent entry decision for the position.
        # Not ``plasticity.eligibility``: the live trace is zero intraday
        # (sleep consumes it; only the daily observe advances it), so the
        # loop's intraday trace is the real eligibility source.
        entry_elig[o.ticker] = intraday_elig.copy()
        rec = ledger_by_key.get((o.ticker, o.decision_ts))
        if rec is not None:
            # The opening encounter's ledger record backs this position's
            # close credit (latest add wins — mirrors entry_elig above);
            # every filled record earns its own settle credit if the
            # position is still open at settle.
            open_entry[o.ticker] = rec
            filled_ids.add(rec.rec_id)
        order(ts, o, price, shares, None, cash)

    def apply_sell(ts: pd.Timestamp, o: PendingOrder, price: float) -> float | None:
        """Close the WHOLE current position at ``price``; returns realized
        P&L, or ``None`` when there is nothing left to sell (the caller
        cancels the order)."""
        nonlocal cash
        pos = positions.pop(o.ticker, None)
        if pos is None:
            return None
        proceeds = pos.shares * price * (1.0 - config.fees)
        cash += proceeds
        realized = proceeds - pos.shares * pos.avg_cost
        order(ts, o, price, pos.shares, realized, cash)
        # Per-exit dopamine (DESIGN v0.6 D6, T9 addendum 2026-09-15): the
        # trade's realized P&L normalized by the trade's OWN notional
        # (clipped to [-1, 1]) — a trade must teach in proportion to its
        # own outcome, not the portfolio's size (hatch-equity scaling
        # left the gate ~40x too weak to overcome pessimistic-fill spread
        # costs). Death liquidation clears the snapshots first, so it
        # never reaches this path.
        notional = pos.shares * pos.avg_cost
        reward = min(1.0, max(0.0, realized) / notional) if notional > 0 else 0.0
        punishment = min(1.0, max(0.0, -realized) / notional) if notional > 0 else 0.0
        snap = entry_elig.pop(o.ticker, None)
        if snap is not None:
            # Incumbent per-exit dopamine (DESIGN v0.6 D6): the full
            # eligibility snapshot through ``observe_trade`` — untouched by
            # the entry-credit machinery (the entry credit below rides its
            # own compact snapshot; the spec keeps the incumbent event).
            plasticity.observe_trade(
                NeuromodState(
                    reward=reward, punishment=punishment, hunger=0.0, arousal=0.0
                ),
                snap,
            )
            emit(
                {
                    "type": "trade_credit",
                    "ts": ts.isoformat(),
                    "ticker": o.ticker,
                    "realized_pnl": round(realized, 6),
                    "reward": round(reward, 9),
                    "punishment": round(punishment, 9),
                }
            )
        # --- entry-credit close site (TRAINING2 §2A) ---------------------
        rec = open_entry.pop(o.ticker, None)
        if rec is not None and config.entry_credit != "off":
            # Forward return to the exit fill: base = the ENCOUNTER bar's
            # close (a price the decision already had); the fill bar is
            # strictly after the decision bar (pessimistic next-bar queue
            # or the close-mode escape hatch's later encounter) — the
            # anti-lookahead invariant, structurally enforced.
            r_fwd = price / rec.base_price - 1.0
            mode = config.trade_credit_mode
            event_a: float | None
            if mode == "realized":
                final_reward, final_punishment, event_a = reward, punishment, None
            else:
                a = _entry_gate_input(config, r_fwd, baselines, rec.bucket)
                event_a = a
                if mode == "forecast":
                    final_reward, final_punishment = max(0.0, a), max(0.0, -a)
                else:  # mix: (1 - w) * realized + w * forecast
                    final_reward = (
                        (1.0 - config.mix_weight) * reward
                        + config.mix_weight * max(0.0, a)
                    )
                    final_punishment = (
                        (1.0 - config.mix_weight) * punishment
                        + config.mix_weight * max(0.0, -a)
                    )
            supersedes: int | None = None
            credit_reward, credit_punishment = final_reward, final_punishment
            ev_reward, ev_punishment = final_reward, final_punishment
            if rec.credit_state == "credited@settle":
                # Single-credit last-credit-wins (TRAINING2 §2A): the close
                # credit REPLACES the settle forecast via a delta — the net
                # per-snapshot gate credit equals the final outcome with no
                # weight rollback. The component deltas may be NEGATIVE
                # (signed), so the gate receives the net dopamine
                # difference split into the non-negative reward/punishment
                # channels NeuromodState validates; the three-factor update
                # is linear in the gate, so d_final = d_provisional +
                # d_delta exactly. The event log records the signed
                # component deltas, so the two events sum to the final
                # outcome.
                reward_delta = final_reward - rec.provisional[0]
                punishment_delta = final_punishment - rec.provisional[1]
                net_gate = reward_delta - punishment_delta
                credit_reward = net_gate if net_gate > 0.0 else 0.0
                credit_punishment = -net_gate if net_gate < 0.0 else 0.0
                ev_reward, ev_punishment = reward_delta, punishment_delta
                supersedes = rec.rec_id
            plasticity.observe_trade_compact(
                (rec.kc_activity, rec.post, rec.support),
                credit_reward, credit_punishment,
            )
            rec.credit_state = "final"
            rec.provisional = (final_reward, final_punishment)
            if rec.bucket is not None:
                _update_baseline(baselines, rec.bucket, r_fwd, config.baseline_alpha)
            event = {
                "type": "entry_credit",
                "ts": ts.isoformat(),
                "ticker": o.ticker,
                "decision_ts": rec.decision_ts.isoformat(),
                "credit_ts": ts.isoformat(),
                "base_price": round(rec.base_price, 6),
                "price": round(price, 6),
                "r_fwd": round(r_fwd, 9),
                "a": None if event_a is None else round(event_a, 9),
                "reward": round(ev_reward, 9),
                "punishment": round(ev_punishment, 9),
                "mode": mode,
                "action": "buy",
            }
            if supersedes is not None:
                event["supersedes"] = supersedes
            emit(event)
        return realized

    def close_position(ts: pd.Timestamp, ticker: str, reason: str) -> float:
        """Immediate close at the last close — death liquidation (semantics
        unchanged by the fill model) and the fill_mode="close" escape hatch."""
        price = last_close[ticker]
        realized = apply_sell(
            ts,
            PendingOrder(
                decision_ts=ts, ticker=ticker, side="sell", reason=reason
            ),
            price,
        )
        assert realized is not None  # the caller only closes held positions
        return realized

    def settle_and_sleep(day: date) -> None:
        """Close-of-day: sugar/shock settle -> observe -> sleep (DESIGN §7.3)."""
        nonlocal day_realized, day_abs_ret_sum, day_abs_ret_n, hunger_armed
        nonlocal daily_mbon_drive, daily_signal_encounters, anchor
        # Entry-credit settle site (TRAINING2 §2A): BEFORE the daily pooled
        # ritual. Every still-pending record is credited exactly once from
        # ``last_close`` — a price already public to the loop (no
        # look-ahead): an encounter on a day's final bar gets r_fwd = 0 and
        # zero credit (the anti-lookahead invariant).
        if config.entry_credit != "off":
            credit_ts = pd.Timestamp(f"{day}T20:00:00+00:00")
            for rec in encounter_ledger:
                if rec.credit_state != "pending":
                    continue
                price = last_close[rec.ticker]
                r_fwd = price / rec.base_price - 1.0
                if rec.action == "buy" and rec.rec_id in filled_ids:
                    # Still-open filled buy: settle forecast credit; a
                    # later close supersedes it via the delta rule
                    # (single-credit last-credit-wins).
                    a = _entry_gate_input(config, r_fwd, baselines, rec.bucket)
                    reward_e, punishment_e = max(0.0, a), max(0.0, -a)
                    rec.credit_state = "credited@settle"
                    rec.provisional = (reward_e, punishment_e)
                    if rec.bucket is not None:
                        _update_baseline(
                            baselines, rec.bucket, r_fwd, config.baseline_alpha
                        )
                else:
                    # Avoid-side credit (TRAINING2 §2A): correctness of the
                    # avoid — the same weights for approached-but-passed
                    # buys and avoided run-ups. An unfilled buy is
                    # effectively a pass: the fly never got the position.
                    if r_fwd < 0.0:
                        reward_e = config.avoid_correct_weight * max(
                            0.0, math.tanh(abs(r_fwd) / config.r_scale)
                        )
                        punishment_e = 0.0
                    elif r_fwd > 0.0:
                        reward_e = 0.0
                        punishment_e = config.miss_weight * max(
                            0.0, math.tanh(r_fwd / config.r_scale)
                        )
                    else:
                        reward_e = punishment_e = 0.0
                    # A pass record has no position to close later — it can
                    # never be superseded: final directly.
                    rec.credit_state = "final"
                    a = math.tanh(r_fwd / config.r_scale)
                    if rec.bucket is not None:
                        _update_baseline(
                            baselines, rec.bucket, r_fwd, config.baseline_alpha
                        )
                plasticity.observe_trade_compact(
                    (rec.kc_activity, rec.post, rec.support),
                    reward_e, punishment_e,
                )
                emit(
                    {
                        "type": "entry_credit",
                        "ts": credit_ts.isoformat(),
                        "ticker": rec.ticker,
                        "decision_ts": rec.decision_ts.isoformat(),
                        "credit_ts": credit_ts.isoformat(),
                        "base_price": round(rec.base_price, 6),
                        "price": round(price, 6),
                        "r_fwd": round(r_fwd, 9),
                        "a": round(a, 9),
                        "reward": round(reward_e, 9),
                        "punishment": round(punishment_e, 9),
                        "mode": "settle",
                        "action": rec.action,
                    }
                )
        reward = max(0.0, day_realized) / hatch_equity
        punishment = max(0.0, -day_realized) / hatch_equity
        drawdown = max(0.0, 1.0 - equity / hatch_equity)
        hunger = min(1.0, drawdown / abs(config.death_threshold))
        mean_abs_ret = day_abs_ret_sum / day_abs_ret_n if day_abs_ret_n else 0.0
        arousal = min(1.0, mean_abs_ret / 0.01)
        state = NeuromodState(
            reward=reward, punishment=punishment, hunger=hunger, arousal=arousal
        )
        # T9 un-silencing: raw MBON spikes are always zero at the current
        # engine gains (T7 gap), which would keep the plasticity eligibility
        # trace at zero forever — learning could never happen. Post activity
        # is therefore the day's mean STRUCTURAL MBON response
        # (``baseline.T @`` KC activity — the MBON's innate receptive field),
        # averaged over signal-bearing encounters. Weight-independent, so
        # the three-factor update cannot feed back into itself.
        post = np.zeros(n, dtype=np.float64)
        if daily_signal_encounters:
            post[mbon_rows] = daily_mbon_drive / daily_signal_encounters
        if config.daily_observe == "on":
            plasticity.observe(state, daily_spikes.astype(np.float64), post)
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
        # Sleep consumes the intraday trace along with the live one: the
        # day's co-activity is consolidated, tomorrow starts fresh.
        intraday_elig.fill(0.0)
        # Daily anchor refresh (DESIGN v0.6 T9): re-measure the innate
        # balance with the current (learned) weights — one extra neutral
        # sniff per sleep, deterministic (RNG-free) — so learned drift in
        # the structural bias does not go stale in the centering.
        anchor = _innate_balance(
            chassis, plasticity, sim, upn_rows, channels, list(frames),
            config.ms_per_bar, id_scale=config.id_scale,
        )
        emit(
            {
                "type": "anchor",
                "ts": f"{day}T20:00:00+00:00",
                "balance": round(anchor, 9),
            }
        )
        day_realized = 0.0
        day_abs_ret_sum = 0.0
        day_abs_ret_n = 0
        daily_spikes.fill(0)
        daily_mbon_drive.fill(0)
        daily_signal_encounters = 0
        hunger_armed = True

    for ts in timeline:
        day = ts.date()
        if day != cur_day:
            if cur_day is not None:
                settle_and_sleep(cur_day)
            cur_day = day
            pointer_drawn = False
            emit({"type": "wake", "ts": ts.isoformat()})

        # 1. Mark prices (equity marking stays at close prices — marking is
        # not execution), then fill pending orders whose symbol's next
        # available bar is this one (pessimistic next-bar model).
        for ticker, df in frames.items():
            i = df.index.get_indexer([ts], method="pad")[0]
            if i >= 0 and df.index[i] == ts:
                last_close[ticker] = float(df["close"].iloc[i])

        # 1b. Pending-order fills: an order decided on an earlier bar
        # executes now iff its symbol prints here (gaps/halts skipped;
        # a session close just carries the order into the next session).
        for o, fill_price in pending_book.drain(ts):
            if o.side == "buy":
                apply_buy(ts, o, fill_price)
                continue
            realized = apply_sell(ts, o, fill_price)
            if realized is None:
                cancel(o, ts, "no_position")  # closed before the fill landed
            else:
                day_realized += realized

        # 2. Mechanical exits, independent of encounters (DESIGN §6.5,
        # TRAINING2 §5.2): the incumbent shock stop plus the exit grid —
        # trailing stop, ATR stop — all through the same pessimistic
        # next-bar queue (no new fill semantics). First trigger wins.
        for ticker in sorted(positions):
            pos = positions[ticker]
            price = last_close[ticker]
            # High-water mark: max close since entry (trailing reference).
            pos.high_water = max(pos.high_water, price)
            pnl_pct = (price / pos.avg_cost - 1.0) * 100.0
            reason: str | None = None
            if pnl_pct <= -config.shock_adverse_pct:
                reason = "shock"
            elif (
                config.trailing_stop_pct is not None
                and price
                <= pos.high_water * (1.0 - config.trailing_stop_pct / 100.0)
            ):
                reason = "trailing_stop"
            elif config.atr_stop_mult is not None:
                # σ20: build_features' 20-bar close-to-close return
                # std-dev — the existing ATR proxy. Without 20 bars of
                # history the estimate is the neutral 0.0: no ATR stop
                # (no volatility data, no stop).
                stop_df = frames[ticker]
                stop_i = stop_df.index.get_indexer([ts], method="pad")[0]
                sigma20 = _features_at(stop_df, stop_i)[0]["volatility"]
                if (
                    sigma20 > 0.0
                    and price < pos.avg_cost - config.atr_stop_mult * sigma20 * price
                ):
                    reason = "atr_stop"
            if reason is None:
                continue
            if pending_book.fill_mode == FILL_MODE_CLOSE:
                day_realized += close_position(ts, ticker, reason)
            elif not pending_book.has_pending(ticker):
                # Queue the stop-loss: it lands one bar later (the
                # execution delay is the point — no look-ahead compensation).
                pending_book.submit(
                    PendingOrder(
                        decision_ts=ts, ticker=ticker, side="sell",
                        reason=reason,
                    )
                )
        equity = equity_at()
        drawdown = max(0.0, 1.0 - equity / hatch_equity)
        if config.hunger_exit and drawdown >= config.hunger_drawdown and hunger_armed:
            winners = sorted(
                (
                    (s, (last_close[s] / p.avg_cost - 1.0) * 100.0)
                    for s, p in positions.items()
                ),
                key=lambda t: (-t[1], t[0]),
            )
            if winners and winners[0][1] > 0.0:
                winner = winners[0][0]
                if pending_book.fill_mode == FILL_MODE_CLOSE:
                    day_realized += close_position(ts, winner, "hunger")
                    hunger_armed = False
                elif not pending_book.has_pending(winner):
                    pending_book.submit(
                        PendingOrder(
                            decision_ts=ts, ticker=winner, side="sell",
                            reason="hunger",
                        )
                    )
                    hunger_armed = False
        elif drawdown < 0.5 * config.hunger_drawdown:
            hunger_armed = True

        # 3. Plume filter (DESIGN §6.1): the day's smelliest movers.
        encountered: str | None = None  # T-1 cadence bookkeeping (§5.1)
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
            inp[upn_rows] += SMELL_GAIN * encode_smell(
                ticker, features, chassis, id_scale=config.id_scale
            )
            if ticker in positions:
                pnl_pct = (last_close[ticker] / positions[ticker].avg_cost - 1.0) * 100.0
                inp += encode_taste(pnl_pct, chassis)
            inp += rng.standard_normal(n) * config.noise_sigma_mv
            out = sim.step(inp, config.ms_per_bar)
            spikes = out["spikes"]
            encountered = ticker
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
            if approach_drive + avoid_drive > 0.0:
                # Eligibility post-activity: the STRUCTURAL MBON response
                # (baseline.T x KC activity — the MBON's innate receptive
                # field), not the learned drive. The learned drive is
                # proportional to the current weights, so using it as the
                # eligibility input is a positive feedback loop (delta ∝ W,
                # verified to explode over 20 training days); the structural
                # response is weight-independent and O(1-10) on the real
                # chassis (mean 2.85 per encounter), keeping daily deltas
                # bounded by the dopamine gate.
                structural_post = plasticity.baseline.T @ plasticity.kc_activity(
                    spike_f
                )
                daily_mbon_drive += structural_post
                daily_signal_encounters += 1
                # Intraday eligibility: decay the trace, add this
                # encounter's driver — the exact increment a zero-gate
                # ``observe`` would apply (habituation is untouched
                # intraday, so the drivers track the live trace's path).
                post = np.zeros(n, dtype=np.float64)
                post[mbon_rows] = structural_post
                intraday_elig = plasticity.eligibility_decay * intraday_elig + (
                    plasticity.eligibility_driver(spike_f, post)
                )
                # Ledger capture (TRAINING2 §2): EVERY signal-bearing
                # encounter gets a record with the compact eligibility
                # snapshot — the KC activity carries the CURRENT habituation
                # gate so the rebuilt outer product equals this encounter's
                # driver increment exactly.
                rec = _EntryRecord(
                    rec_id=len(encounter_ledger),
                    decision_ts=ts,
                    ticker=ticker,
                    kc_activity=plasticity.kc_activity(spike_f)
                    * plasticity.habituation,
                    post=structural_post,
                    support=plasticity.support,
                    base_price=last_close[ticker],
                    action="pass_signal",  # finalized after the decision
                    bucket=(
                        _entry_bucket(features)
                        if config.entry_credit == "advantage" else None
                    ),
                )
                encounter_ledger.append(rec)
                ledger_by_key[(ticker, ts)] = rec
            duration_s = config.ms_per_bar / 1000.0
            mbon_rate = float(spikes[mbon_rows].sum()) / mbon_rows.size / duration_s
            # Center on the innate balance; a silent encounter (no KC
            # activity) carries no signal and reads neutral.
            centered = (
                balance - anchor if approach_drive + avoid_drive > 0.0 else 0.0
            )
            # Structural looming drive (DESIGN v0.6 §5): the LC→MBON
            # connectome response to this encounter's LC spikes, entering the
            # decision balance at a small fixed λ. Zero when the chassis has
            # no LC→MBON path (or a silent LC population).
            if w_struct is not None:
                structural_score = float(
                    plasticity.mbon_valence @ (w_struct.T @ spikes[lc_rows])
                )
            else:
                structural_score = 0.0
            balance_used = centered + config.lambda_struct * structural_score

            emit(
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
            held = ticker in positions
            price = last_close[ticker]
            action, reason = _decide(
                balance_used, held,
                len(positions) + pending_book.n_buys() >= config.position_cap,
                config.approach_thr, config.avoid_thr,
            )
            if action == "sell" and not config.valence_flip_exit:
                # TRAINING2 §5.2 ablation: drop the sell-on-avoid exit
                # without touching anything else (the encounter reads on).
                action, reason = "pass", "valence_flip_off"
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
            rec = ledger_by_key.get((ticker, ts))
            if rec is not None:
                # Finalize the record's action after the decision (a dust
                # downgraded buy counts as a pass).
                rec.action = "buy" if action in ("buy", "add") else "pass_signal"
            if action in ("buy", "add"):
                o = PendingOrder(
                    decision_ts=ts, ticker=ticker, side="buy", reason=reason,
                    shares=shares,
                )
                if pending_book.fill_mode == FILL_MODE_CLOSE:
                    apply_buy(ts, o, price)  # legacy same-bar close fill
                else:
                    pending_book.submit(o)
            elif action == "sell":
                if pending_book.fill_mode == FILL_MODE_CLOSE:
                    day_realized += close_position(ts, ticker, reason)
                else:
                    # Sells close the WHOLE position at fill time.
                    pending_book.submit(
                        PendingOrder(
                            decision_ts=ts, ticker=ticker, side="sell",
                            reason=reason,
                        )
                    )
            # T-1 exit-forecast gate (TRAINING2 §5.1), default cadence:
            # score the encountered position only, reusing this bar's
            # encounter readout (zero extra sim steps; taste never reaches
            # the readout, so ``balance_used`` IS the without-taste
            # recomputation). ``exit_score = -balance_used``.
            if (
                config.exit_forecast
                and held
                and action != "sell"
                and config.exit_score_thr is not None
                and -balance_used > config.exit_score_thr
                and not pending_book.has_pending(ticker)
            ):
                if pending_book.fill_mode == FILL_MODE_CLOSE:
                    day_realized += close_position(ts, ticker, "exit_forecast")
                else:
                    pending_book.submit(
                        PendingOrder(
                            decision_ts=ts, ticker=ticker, side="sell",
                            reason="exit_forecast",
                        )
                    )

        # 6b. T-1 every-bar cadence (TRAINING2 §5.1, Phase-B budget
        # experiment): score held tickers the encounter path did not
        # score — up to position_cap extra LIF steps per bar, each with
        # its own RNG noise draw and STD-state perturbation. The
        # encountered ticker was already scored in the encounter block.
        if config.exit_forecast and config.exit_score_every_bar:
            for e_ticker in sorted(positions):
                if e_ticker == encountered or pending_book.has_pending(e_ticker):
                    continue
                e_df = frames[e_ticker]
                e_i = e_df.index.get_indexer([ts], method="pad")[0]
                score = _exit_forecast_readout(
                    chassis, plasticity, sim, upn_rows, w_struct, lc_rows,
                    anchor, config, e_ticker, e_df, e_i, rng,
                )
                if score > config.exit_score_thr:
                    if pending_book.fill_mode == FILL_MODE_CLOSE:
                        day_realized += close_position(
                            ts, e_ticker, "exit_forecast"
                        )
                    else:
                        pending_book.submit(
                            PendingOrder(
                                decision_ts=ts, ticker=e_ticker, side="sell",
                                reason="exit_forecast",
                            )
                        )

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
            # No per-exit credit for death liquidation: clear the opening
            # snapshots before the liquidation closes reach apply_sell.
            entry_elig.clear()
            open_entry.clear()
            filled_ids.clear()
            # A dying brain gets no further credits: pending ledger records
            # are sealed without credit (the plasticity below is a fresh
            # brain) and the Option B baselines reset with it.
            for rec in encounter_ledger:
                if rec.credit_state == "pending":
                    rec.credit_state = "final"
            baselines.clear()
            baselines.update(config.initial_baselines or {})
            for ticker in sorted(positions):
                day_realized += close_position(ts, ticker, "death_liquidation")
            for o in pending_book.cancel_all():
                cancel(o, ts, "death")  # the brain that placed them is gone
            plasticity = Plasticity(
                chassis,
                top_k=config.grudge_top_k,
                reward_gain=config.reward_gain,
                punishment_gain=config.punishment_gain,
            )
            sim.reset(config.seed)
            daily_spikes.fill(0)
            daily_mbon_drive.fill(0)
            intraday_elig.fill(0.0)  # fresh brain = fresh trace
            daily_signal_encounters = 0
            hatch_equity = cash
            hunger_armed = True
            emit({"type": "hatch", "hatch_equity": round(hatch_equity, 6)})
            equity = cash

            # A fresh brain re-calibrates its innate balance.
            anchor = _innate_balance(
                chassis, plasticity, sim, upn_rows, channels, list(frames),
                config.ms_per_bar, id_scale=config.id_scale,
            )
        # 8. Bar receipt.
        equity_writer.writerow(
            [ts.isoformat(), f"{equity:.2f}", f"{cash:.2f}", len(positions)]
        )
        n_bars += 1

    # Run end: any order whose symbol never printed again is cancelled.
    for o in pending_book.cancel_all():
        cancel(o, timeline[-1], "run_end")

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
        final_weights=plasticity.weights.copy(),
        baselines=(dict(baselines) if config.entry_credit == "advantage" else None),
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
        p.add_argument(
            "--chassis", choices=("stripped", "whole"), default="stripped",
            help="Brain chassis; 'whole' implies the calibrated T12b STD "
            "(beta=0.1, tau_rec=500 ms) unless overridden.",
        )
        p.add_argument(
            "--std-beta", type=float, default=None,
            help="Short-term-depression depletion fraction (whole-fly default 0.1).",
        )
        p.add_argument(
            "--std-tau-rec-ms", type=float, default=None,
            help="STD recovery time constant in ms (whole-fly default 500).",
        )
        p.add_argument(
            "--basket-file", default=None,
            help="Basket file (TRAINING2 §3): one symbol per line, '#'"
            " comments; replaces the module BASKET for this run (guarded"
            " against the frozen never-seen eval basket).",
        )
        p.add_argument(
            "--entry-credit", choices=ENTRY_CREDIT_MODES, default="off",
            help="Entry-credit objective (TRAINING2 §2): off / forecast (A)"
            " / advantage (B).",
        )
        p.add_argument(
            "--trade-credit-mode", choices=TRADE_CREDIT_MODES, default="realized",
            help="Close-fill entry-credit content (TRAINING2 §2A).",
        )
        p.add_argument(
            "--horizon-bars", type=int, default=30,
            help="Forecast horizon in 1-min bars (TRAINING2 §2).",
        )
        p.add_argument("--r-scale", type=float, default=0.005,
                       help="Option A tanh scale on forward returns.")
        p.add_argument("--a-scale", type=float, default=0.003,
                       help="Option B tanh scale on the advantage.")
        p.add_argument("--baseline-alpha", type=float, default=0.05,
                       help="Option B baseline EMA rate.")
        p.add_argument("--miss-weight", type=float, default=0.25,
                       help="Missed-winner weight (TRAINING2 §2A).")
        p.add_argument("--avoid-correct-weight", type=float, default=0.5,
                       help="Correct-avoid reward weight (TRAINING2 §2A).")
        p.add_argument("--mix-weight", type=float, default=0.5,
                       help="trade_credit_mode=mix blend weight.")
        p.add_argument(
            "--daily-observe", choices=DAILY_OBSERVE_MODES, default="on",
            help="Daily pooled sugar/shock ritual (D6) on/off ablation.",
        )
        p.add_argument("--reward-gain", type=float, default=1.0,
                       help="Dopamine gate reward gain (TRAINING2 §4).")
        p.add_argument("--punishment-gain", type=float, default=1.0,
                       help="Dopamine gate punishment gain (TRAINING2 §4).")
        p.add_argument("--id-scale", type=float, default=1.0,
                       help="Smell identity attenuation (TRAINING2 §3).")
        p.add_argument(
            "--trailing-stop-pct", type=float, default=None,
            help="Trailing stop: percent below the max close since entry"
            " (TRAINING2 §5.2).",
        )
        p.add_argument(
            "--atr-stop-mult", type=float, default=None,
            help="ATR stop: k in price < entry - k*sigma20*price"
            " (TRAINING2 §5.2).",
        )
        p.add_argument(
            "--valence-flip-exit",
            action=argparse.BooleanOptionalAction, default=True,
            help="Sell a held position on an avoid balance (TRAINING2 §5.2).",
        )
        p.add_argument(
            "--hunger-exit",
            action=argparse.BooleanOptionalAction, default=True,
            help="Close the largest winner at the hunger drawdown"
            " (TRAINING2 §5.2).",
        )
        p.add_argument(
            "--exit-forecast", action="store_true",
            help="T-1 exit-forecast gate (TRAINING2 §5.1); requires"
            " --exit-score-thr.",
        )
        p.add_argument(
            "--exit-score-thr", type=float, default=None,
            help="Exit-forecast threshold on -balance_used.",
        )
        p.add_argument(
            "--exit-score-every-bar", action="store_true",
            help="Exit-forecast every-bar cadence (Phase-B budget experiment).",
        )
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
        chassis=args.chassis,
        std_beta=args.std_beta,
        std_tau_rec_ms=args.std_tau_rec_ms,
        basket=(
            parse_basket_file(args.basket_file) if args.basket_file else None
        ),
        entry_credit=args.entry_credit,
        trade_credit_mode=args.trade_credit_mode,
        horizon_bars=args.horizon_bars,
        r_scale=args.r_scale,
        a_scale=args.a_scale,
        baseline_alpha=args.baseline_alpha,
        miss_weight=args.miss_weight,
        avoid_correct_weight=args.avoid_correct_weight,
        mix_weight=args.mix_weight,
        daily_observe=args.daily_observe,
        id_scale=args.id_scale,
        reward_gain=args.reward_gain,
        punishment_gain=args.punishment_gain,
        trailing_stop_pct=args.trailing_stop_pct,
        atr_stop_mult=args.atr_stop_mult,
        valence_flip_exit=args.valence_flip_exit,
        hunger_exit=args.hunger_exit,
        exit_forecast=args.exit_forecast,
        exit_score_thr=args.exit_score_thr,
        exit_score_every_bar=args.exit_score_every_bar,
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
