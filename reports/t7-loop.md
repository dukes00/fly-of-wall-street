# T7 — Foraging Backtest Loop

**Module:** `src/fruitfly/loop.py` · **CLI:** `uv run python -m fruitfly backtest --seed S --start DATE --end DATE [--top-k K] [--position-cap N] [--ms-per-bar MS] [--death-threshold X] [--fees F] [--out-dir DIR]` · **Artifacts:** `data/runs/backtest_{seed}_{start}_{end}/` — `equity.csv` (`timestamp, equity, cash, n_positions`, one row per bar, equity/cash at 2 decimals) and `events.jsonl` (one JSON object per line, `json.dumps(..., sort_keys=True)` → deterministic field order; types: `hatch, wake, encounter, decision, order, sugar_shock, sleep, death`).

## Loop shape (DESIGN §7, D3/D13/D14/D16)

Per 1-minute bar: mark prices → mechanical exits → plume filter → **one encounter** → decision/order → death check → equity row. Day boundary: settle realized P&L → `NeuromodState` → `Plasticity.observe` (daily spike sums) → `Plasticity.sleep()`.

- **Plume filter (D3):** intensity = `|1-bar return| + volume_ratio` (`volume_ratio` = volume / trailing 20-bar mean). A flat stock (`ret == 0`) is odorless and never enters the set. Top `top_k` (default 5), ties broken by ticker. Features come from the shared builder `senses.build_features` (returns / RSI-14 / volatility / volume_delta) — one definition for smell, plume ranking and T8.
- **Round-robin cadence:** one plume per bar. The rotation pointer persists across bars; its daily starting offset is drawn from the loop RNG at the first plume-bearing bar. Rationale: one LIF encounter costs ~0.36 s wall on the 27,115-neuron chassis; five per bar would put a 2-day run at ~25 min, blowing the runtime budget. One sample/minute also keeps the fly inside D16's 1-minute bar cadence.
- **Exits (§6.5):** sharp adverse move — unrealized P&L ≤ −`shock_adverse_pct` (default 2%) closes immediately, no brain involved (biological stop-loss); hunger — drawdown from hatch equity ≥ `hunger_drawdown` (default 10%) closes the largest winner, re-arming when drawdown recovers below half the threshold; valence flip — a held ticker re-encountered with balance < −`avoid_thr` is closed.

## MBON readout path (one documented path)

The LIF engine propagates only structural synapses; learned KC→MBON weights live in `Plasticity`. Decisions therefore read the **learned drive**, not raw MBON spikes:

1. `drive = Plasticity.mbon_activation(KC spikes from the sim step)` — the same quantity `Plasticity.valence_readout` aggregates; the raw `valence_readout` value is also logged per encounter (`valence_readout` field).
2. Split by MBON valence: `A` = drive over approach MBONs (+1), `R` = drive over avoidance MBONs (−1); decision variable = **balance** `(A − R)/(A + R)` in [−1, 1] — DESIGN §5's "approach/avoid balance", scale-free so encounter intensity doesn't rescale it.
3. **Hatch anchoring.** The raw balance is structurally avoid-biased: measured −0.35 to −0.37 at every drive level on the real chassis (the valence split is the transmitter-sign judgment call from `fruitfly.neuromod`, not a calibrated behavior). An uncentered readout would make a fresh fly avoid everything, forever. At every hatch the fly takes one **calibration sniff** — mean basket identity profile, neutral market state, no noise, one sim step — and its innate balance becomes the anchor; decisions use `balance − anchor`. A silent sniff (no KC activity) anchors at 0.
4. `balance > approach_thr` → BUY/add; `balance < −avoid_thr` → SELL (valence-flip exit on a held position); otherwise pass. Silent encounters (zero KC activity) carry no signal and always pass.

**Defaults: `approach_thr = avoid_thr = 0.005`.** Measured centered-balance distribution over 36 encounters (8 symbols, 2026-08-18 real bars, fresh plasticity): min −0.093, median +0.008, p75 +0.028, max +0.050; 53% above +0.005, 28% below −0.005 — so a fresh fly acts on roughly 80% of encounters and passes the rest. These are **T9 calibration candidates**.

### Sizing (DESIGN §5) — documented T9 gap

Size fraction = `clip(tanh(mbon_rate / MBON_RATE_SCALE_HZ), MIN_SIZE_FRAC, 1.0)` from the MBON population firing rate (post-spikes over MBON nodes / duration); per-order notional = `frac × equity / position_cap`. **Caveat:** at the current engine constants MBONs never spike — KC→MBON quantal PSPs are ≈0.01 mV/synapse (0.01 × median 5 synapses), ~3 orders of magnitude below the 15 mV threshold — so `mbon_rate` is ~0 and sizing rides the `MIN_SIZE_FRAC = 0.25` floor (2.5% of equity per order at cap 10). T9 options: raise `SYN_CONDUCTANCE_MV`, lower `V_TH`, or rate-model the sizing input.

## Calibrated constants (measured, T9 candidates)

| Constant | Default | Measurement on real 2026-08-18 bars |
|---|---:|---|
| `SMELL_GAIN` | 1000 | KC yield/encounter: 0 at 300, ~19 at 1000 (2% silent), ~58 at 2000, ~470 at 3000; wall time roughly flat in gain → 1000 is the knee: live circuit, lowest spike volume. Encoder smell output is a [0,1) identity profile, ~3 orders below the 15 mV LIF threshold — the gain is the loop's bridge, and **it saturates the uPN population at refractory-limited rates**, largely destroying ticker identity in the current regime (T9 gap: land uPNs in the linear range, e.g. via `SYN_CONDUCTANCE_MV`). |
| `noise_sigma_mv` | 0.5 | Sensory-noise floor per node per encounter. SNR: encounter drives sit ~20–30 mV above KC threshold with ±0.5 mV jitter → SNR ≈ 40–60 (≈3% of spike threshold). Small enough to preserve signal, large enough that threshold-adjacent spike counts differ per seed — **this is what makes seeds genuinely diverge** (same-seed runs are byte-identical). |
| `approach_thr` / `avoid_thr` | ±0.005 | see readout section above |
| `dt_ms` | 1.0 | 5% of tau_m; half the T3 bench's 0.5 ms, halves wall time (~0.36 s/encounter → 2-day run ≈ 4.8 min, inside the 5-minute budget) with qualitatively unchanged dynamics. |
| `ms_per_bar` | 500 | biological time per 1-min bar (D16). |
| `shock_adverse_pct` | 2.0 | 1-min-bar stop-loss scale |
| `hunger_drawdown` | 0.10 | closes the largest winner at 10% drawdown |
| `death_threshold` | −0.50 | D14, pinned |
| `position_cap` | 10 | D13, pinned |

Rotation: `pointer` persists across bars, `plumes[pointer % len(plumes)]`; the daily offset = `rng.integers(0, len(plumes))` at the day's first plume-bearing bar. RNG consumption order (the loop owns the only RNG, `PCG64(seed)`): per day one offset draw, per encounter one `standard_normal(n)` noise draw — nothing else consumes it.

## Real-run receipts (2026-08-18..19, full 22-symbol basket, 780 bars)

Fills at the bar's close price, whole shares, fees default 0. CLI:
`python -m fruitfly backtest --seed 7 --start 2026-08-18 --end 2026-08-19`.

- **Seed 7 double run, byte-identical:** equity.csv md5 `306e456fe55d1f208be4547822b5c28f`, events.jsonl md5 `2628e6bc86192f62f7790bfa9e557681` in both runs (and identical to a third run executed under heavy concurrent CPU load — determinism holds across load). Wall time 4m34.8s / 4m35.1s (274 s each, dt=1 ms) — inside the <5-min budget.
- **Seed 8 diverges:** equity md5 `e62ff49f7783235679b0f5c82f21848b`, events md5 `87bb959039a4fb890c201f086bf54be0`; 1857 events / 290 orders vs seed 7's 1892 / 325; final equity 102,329.60 vs 103,063.29.
- **Seed 7 run shape:** 780 bars, 780 encounters (one per bar), 325 orders, 0 deaths, final equity 103,063.29 (+3.06%), day-2 settled reward 0.0185 (1.85% of hatch equity), hunger 0, arousal 0.066.
- **Cap invariant from the log:** replaying orders across events.jsonl — max 10 open positions, 0 cap violations; equity.csv `n_positions` column max 10. Cash ran negative during the session (paper margin, see module docstring) while equity stayed positive; per-order notional stayed at the 2.5%-of-equity floor.
- Event-type coverage in the real log: hatch 1, wake 2, encounter 780, decision 780, order 325, sugar_shock 2, sleep 2.

## What T9 should tune

1. **Thresholds ±0.005** on the anchored balance — sweep for decision quality (hit rate vs. next-bar return).
2. **`SMELL_GAIN` + `SYN_CONDUCTANCE_MV` jointly** — land the olfactory pathway in its linear range so ticker identity and market state actually modulate KC activity (currently saturated); this also un-silences MBONs.
3. **Sizing map** — once MBON rates are live, calibrate `MBON_RATE_SCALE_HZ` / `MIN_SIZE_FRAC` against the firing-rate distribution.
4. **`shock_adverse_pct` / `hunger_drawdown`** — position-level vs. portfolio-level risk budget.
5. **Death/hatch exercise** — the death path (liquidation, fresh `Plasticity`, re-anchored innate balance, hatch equity reset) is implemented and log-visible (`death`/`hatch` events); T9 should push equity below −50% and verify lifecycle continuity.
6. **Anchor freshness** — the innate-balance anchor is taken at hatch only; re-anchoring per day (weights drift with learning, sleep decays them) is a candidate.
7. **Foraging cadence** — one encounter/bar is a runtime constraint; if wall time drops (engine gains), consider `top_k` encounters per bar with the same persistent rotation.