# The Fruit Fly of Wall Street — DESIGN.md

**Version:** 0.6 (2026-09-15)
**Status:** Ideation complete — all decisions settled. Pre-implementation.
**Tagline:** *166,691 neurons. Zero emotions.*

An emulated fruit fly brain trades intraday markets. It sees candlesticks, smells indicators, feels profit as sugar and loss as shock, gets hungry in drawdown, and sleeps when the market closes. The in-fiction fund entity is **Compound Eye Capital**.

---

## 1. Concept

The fly is not a portfolio optimizer. It is a forager. It drifts through the market, encounters odor plumes (tickers), and for each makes one decision — approach or avoid — based on learned associations. In ML terms: a contextual bandit that experiences reward as sugar.

Best-case outcome (the joke's jackpot): the fly beats the average active fund manager (SPIVA: most underperform their index over long horizons).

## 2. Brain chassis

- Connectome: complete male Drosophila CNS (Berg et al., Cell 2026 — Google/Janelia). 166,691 neurons, ~125M synapses, brain + optic lobes + ventral nerve cord, fruitless/doublesex annotated.
- Simulation recipe: leaky integrate-and-fire, weights from synapse counts, excitatory/inhibitory from predicted transmitter (Shiu et al., Nature 2024 style).
- Two run modes:
  - **Stripped chassis** — retina + optic lobe + olfactory pathway + mushroom body + descending readout. Trains fast. The dev/test lane.
  - **Whole fly** — all 166,691 neurons, all the time. The live engine (D22) and the art piece.
- Run mode (revised, D22): **whole-fly is the live brain**. The whole-fly LIF engine gained opt-in short-term synaptic depression on inhibitory terminals (calibrated β=0.1, τ_rec=500 ms) — required because tonic APL/DPM feedback inhibition otherwise pins KCs below threshold in a fatique-free quantal model. With STD, the whole fly fires KCs at stripped-brain yield and trades (~1.26 s/bar ≈ 8 min per 390-bar day on M1 — live-speed feasible). The larval stage now trains **directly on the whole-fly chassis**; the stripped chassis remains the fast dev/test lane. The transplant path (D10 Phase B, D19) is **dropped**: measured transplant agreement was 72.82% < 90%, and direct training makes transfer — and the agreement gate — moot. Evaluation duty passes to held-out replay on extended IEX 1-min history plus live paper receipts. Stripped-vs-whole head-to-head on the in-sample replay window: whole-fly +$874 vs stripped +$488 (2 days, statistically inconclusive).

## 3. Senses

- **Vision:** candlestick chart rendered at ommatidia resolution → photoreceptor array → optic lobe. T4/T5 direction-selective cells extract price drift; looming-sensitive cells catch vertical spikes (crash/rally detection). DIRECTION/LOOMING gains recalibrated (v0.6): looming-sensitive LC cells reach nonzero spike yield on crash-like bars (previously 0 spikes in both brains — measured, T12b). T4/T5→KC is 0 synapses (connectome fact) — vision reaches the readout only through the looming→MBON structural path and the smell state vector.
- **Smell:** indicators mixed as a virtual odor across ~50 glomeruli → mushroom body. Two components (settled, D5; amended 2026-09-15, v0.6):
  - **Identity** — fixed glomerular signature per ticker ("AAPL smells like AAPL").
  - **State** — market features: returns, RSI, volatility, volume delta, plus multi-bar momentum: `mom20` (20-bar simple return, close[-1]/close[-21]−1) and `slope20` (OLS slope per bar of close over the last 20 bars, relative to price level), both trailing-only. The 1-bar-only state starved the readout of horizon; multi-bar features make the time-ticker signal meaningful. Features are available evidence, not a strategy — the fly may learn momentum, mean-reversion, or neither.
  - Decomposition enables both specific associations ("AAPL = sugar") and general ones ("anything this volatile = shock"). This is what transfers learning across tickers.
- **Taste (settled, D15 — in v1):** unrealized P&L of the open position, sweet/bitter. The only sense representing what the fly currently owns; enables exit from slow bleed. Also required for the crypto handoff mechanic (§10).

## 4. Internal state (neuromodulation)

| State | Biological channel | Market mapping |
|---|---|---|
| Reward | PAM dopamine neurons (sugar) | Realized profit |
| Punishment | PPL1 dopamine neurons (shock) | Realized loss |
| Hunger | Starvation state (raises reward drive) | Portfolio drawdown → risk appetite rises |
| Arousal | Octopamine | Market volatility |
| Sleep | Consolidation + memory decay | Market closed; decay doubles as regime forgetting |

Learning: dopamine-gated plasticity at Kenyon-cell → MBON synapses. One-trial learning is biological, not a bug: the fly forms grudges and favorites.

## 5. Decisions (trade action)

- Readout: mushroom body output neurons (MBONs), ~35 cells, approach/avoid balance.
- Mapping: approach > threshold → BUY/add; avoid > threshold → SELL/skip; neutral → pass.
- Structural vision drive (v0.6): the looming→MBON path carries real connectome synapses (~152/154); its response enters the decision balance at a small fixed λ (`lambda_struct`, default 0.05): `balance_used = (learned_balance − anchor) + λ_struct × structural_lc_mbon_score`.
- Position size ← MBON firing rate.
- Latency: biological reaction ~50–100 ms; sim can clock faster than real time. Intraday scalping is physically legal.

## 6. Position selection — the foraging loop

1. **Universe: plume-filtered (settled, D3; amended 2026-09-15, v0.6).** No curated watchlist. Each ticker emits a plume with intensity = |20-bar return| × 20 + volume ratio (20-bar movement expressed in 1-bar-equivalent magnitude; volume ratio unchanged). Direction-agnostic — persistent movers of either sign surface. A dead ticker (ret1 = 0 and mom20 = 0) is odorless; the universe self-filters to the day's movers (top-K smelliest of the index).
2. **Encounter.** One plume at a time, round-robin through the active set. Per encounter: smell (identity × state) + vision (recent chart stream).
3. **Decision.** MBON approach/avoid as in §5.
4. **Portfolio** = set of plumes currently approached. Cap **N=10** concurrent positions (settled, D13; per D4 survivability direction). Revocable; calibrate during the larval stage.
5. **Exit.** Valence flip → close. Sharp adverse move → shock → immediate avoidance (biological stop-loss). Hunger → close a winner to realize sugar (drawdown makes the fly take profits).

Free behaviors, no code required:

- **Habituation** — stimuli that predict nothing fade from attention. Automatic noise filtering.
- **One-trial personality** — a ticker that shocks the fly once gets avoided hard. Watchlists emerge from plasticity, not config.

## 7. Daily loop

1. Market opens → fly wakes.
2. Per 1-minute bar (D16, calibration candidate): render chart → mix odor → step sim → read MBONs → emit paper order.
3. Close: settle P&L → administer sugar/shock → fly sleeps (consolidation + decay).
   - Per-exit dopamine (D6; amended 2026-09-15, v0.6; scale amended 2026-09-15, T9): when a position closes with realized P&L r on notional `N = shares × avg_cost`, the eligibility snapshot taken at the opening buy/add is credited with a dopamine gate built from r/N clipped to [-1, 1] (same three-factor rule; reward > 0 potentiates approach-MBON associations of the opening encounter, punishment > 0 the opposite). Trade-level normalization — a trade teaches in proportion to its own outcome; hatch-equity scaling measured ~40x too weak against pessimistic-fill spread costs (t9b gate, 2026-09-15). The close-of-day sugar/shock settle remains the designed diffuse ritual (day realized P&L over hatch equity); per-exit is the precise trade-level credit.
   - Sleep calibration (T9, v0.6): the innate-balance anchor is refreshed at every sleep (daily re-centering, one extra neutral sniff; deterministic) instead of hatch-only. Grudge protection top_k drops 64 → 16 (config: `grudge_top_k`).

## 8. Data (free only, hard constraint)

- Market: **US equities** (settled, D12) — market close = natural sleep per D7. Crypto deferred to the Wall & Street experiment (§10).
- Granularity: 1-minute bars to start (D16, calibration candidate).
- Backtests: yfinance / stooq intraday.
- Live paper: Alpaca free tier (IEX feed, US equities).
- Crypto via ccxt websockets remains free and real-time, but 24/7 — no close, no sleep. Not in v1.

## 9. Lifecycle

- **Larval stage** = backtest training on historical data.
- **Adult stage** = live paper account.
- **Death:** drawdown starves the fly past a threshold → it dies → new fly hatches with fresh plasticity. Emergent, not scripted: mortality falls out of the hunger state. Threshold: **−50% equity to start** (settled, D14); calibrate later.
- **Swarm:** v1 is a single fly (settled, D17). Swarm reuses the hatch machinery; candidate v2 experiment.

## 10. Crypto alternative (parked, D9)

If crypto: two flies on shifts — **Wall** (day) and **Street** (night). Same connectome, separate plasticity state files, circadian gating via the mapped clock-neuron network.

- Handoff policy: **taste the inheritance (provisional pick).** The waking fly senses open positions via the taste channel (unrealized P&L). It never knows why the trade exists, only how it feels.
- Emergent content: US and Asia sessions have genuinely different volatility regimes → the two flies will actually specialize and disagree.
- Complication cost: 2× state, one handoff rule.

## 11. Scoreboard

Benchmarks, tracked from day one:

1. S&P 500 buy-and-hold
2. Average active fund manager (SPIVA baseline — the joke's jackpot)
3. Monkey-with-darts (random)
4. Logistic regression on identical features (the scientific control)

## 12. Output

Not a content product (settled, D11). The artifact is the run itself plus its receipts: a **simple live local dashboard** (settled, D18 — nothing fancy: current equity, open positions, MBON balance, neuron-activity view, event log) and an end-of-run post-mortem (fly lifespan, cause of death, P&L vs scoreboard). No PDF statements (D18). Distribution: friends, maybe one Reddit post (dashboard screenshots suffice). No streaming, no audience mechanics.

## 13. Stack sketch

- Connectome data: Neuroglancer/CAVE download (male CNS) or FlyWire Codex.
- Sim: LIF whole-brain (Shiu recipe) or lean mushroom-body rate model with dopamine-gated plasticity. Target machine: MacBook Pro M1, Apple Silicon (D20). No Raspberry Pi target.
- Data: yfinance / stooq (backtests). Broker: Alpaca paper (live).
- Dashboard (D21): FastAPI + uvicorn backend, Server-Sent Events push, one static page with vanilla JS + `<canvas>`. No build step, no npm, no JS libraries. Read-only file interface to run receipts — never attaches to the sim process. Thin by design (D18).

## 14. Decisions log

| D | Date | Decision |
|---|---|---|
| D1 | 2026-09-11 | Project name: **The Fruit Fly of Wall Street** |
| D2 | 2026-09-11 | In-fiction fund entity: **Compound Eye Capital** |
| D3 | 2026-09-11 | Universe: plume-filtered index (emergent attention), no curated watchlist. **Amended 2026-09-15 (v0.6):** plume intensity = \|20-bar return\| × 20 + volume ratio; direction-agnostic; dead ticker (ret1 = 0 ∧ mom20 = 0) odorless |
| D4 | 2026-09-11 | Position cap: higher N for survivability (exact value → P2) |
| D5 | 2026-09-11 | Dual sensory channels: vision (candlesticks) + smell (identity × state indicators). **Amended 2026-09-15 (v0.6):** state vector gains multi-bar momentum — mom20 (20-bar simple return) and slope20 (OLS slope/20 bars, price-relative), trailing-only |
| D6 | 2026-09-11 | Stimuli: profit = sugar (PAM), loss = shock (PPL1), hunger = drawdown. **Amended 2026-09-15 (v0.6):** per-exit dopamine added — closing a position credits the opening encounter's eligibility snapshot from realized P&L; daily sugar/shock settle remains the diffuse ritual |
| D7 | 2026-09-11 | Intraday operation; fly sleeps in aftermarket |
| D8 | 2026-09-11 | Free data only — no paid feeds |
| D9 | 2026-09-11 | Crypto shift pair (Wall & Street) parked as alternative; provisional handoff = taste channel |
| D10 | 2026-09-11 | Chassis: phased — stripped-chassis training harness, then transplant KC→MBON weights into whole-fly for the live run |
| D11 | 2026-09-11 | Not a content product: share results with friends + maybe one Reddit post; no streaming |
| D12 | 2026-09-11 | Market: US equities (Alpaca free tier, natural sleep at close); crypto deferred |
| D13 | 2026-09-11 | Position cap N=10 (per D4 survivability direction); calibrate in larval stage |
| D14 | 2026-09-11 | Death threshold: −50% equity to start; calibrate later |
| D15 | 2026-09-11 | Taste channel included in v1 (unrealized P&L) |
| D16 | 2026-09-11 | Granularity: 1-minute bars to start, subject to calibration |
| D17 | 2026-09-11 | Single fly in v1; swarm deferred |
| D18 | 2026-09-11 | Reporting: simple live local dashboard; no PDF statements |
| D19 | 2026-09-11 | Transplant: direct KC→MBON weight copy + behavioral validation (≥90% decision agreement on held-out days). Duke deferred; my call, revocable |
| D20 | 2026-09-11 | Target hardware: MacBook Pro M1 (Apple Silicon); no Raspberry Pi target |
| D21 | 2026-09-11 | Dashboard stack: FastAPI + uvicorn + SSE push; single static page, vanilla JS + canvas; no build step, no JS libraries; read-only file interface to run receipts |
| D22 | 2026-09-14 | **Supersedes D19 (and D10 Phase B).** Transplant dropped; whole-fly (166,700 neurons, male CNS v1.0) is the live brain, trained directly via larval-stage plasticity. Enabler: opt-in short-term synaptic depression on inhibitory terminals in the LIF engine (β=0.1, τ_rec=500 ms), without which APL/DPM feedback inhibition silences all KCs in a fatigue-free quantal model. Stripped chassis (27,115 neurons) becomes the fast dev/test lane. Signed off by Duke in session. Evidence: reports/t12-transplant.md, reports/t12b-apl-std.md |

## 15. Parking lot (non-blocking)

Empty as of v0.3 — P2–P9 all settled (D12–D19). New open questions land here.

## 16. Changelog

- **v0.1 (2026-09-11):** Initial doc. Settled: naming (D1, D2), plume-filtered universe (D3), higher position cap (D4), dual-channel senses (D5), stimulus mapping (D6), intraday+sleep (D7), free-data constraint (D8), crypto pair parked (D9). Sections 1–15 established.
- **v0.2 (2026-09-11):** D10 settled P1 — phased chassis (stripped training → whole-fly weight transplant). D11: de-contented — no streaming; output = receipts, daily statement, post-mortem (§12 rewritten, P8 recast as reporting tech). P9 added (transplant fidelity). Remaining parking-lot items presented with defaults.
- **v0.3 (2026-09-11):** All parking-lot items settled: US equities (D12), N=10 (D13), death at −50% equity (D14), taste in v1 (D15), 1-min bars (D16), single fly (D17), live local dashboard over PDF (D18), transplant = copy + validate ≥90% (D19, revocable, my call on Duke's defer). Ideation complete. Next phase: implementation planning.
- **v0.4 (2026-09-11):** D20: target hardware — MacBook Pro M1; Raspberry Pi clause removed (§13). D21: dashboard stack — FastAPI + SSE + vanilla canvas, file-interface, no build step (§13). Implementation plan written: plan/TASKS.md v0.2 (15 tasks, larval harness first).
- **v0.5 (2026-09-14):** D22 — whole-fly is the live brain, trained directly; transplant (D19) dropped; STD added to the LIF engine (§2 rewritten). Implementation status: all 15 tasks of plan/TASKS.md v0.1 complete (M1–M4), plus the APL/STD revision. Extended IEX 1-min history acquisition for held-out evaluation in progress.
- **v0.6 (2026-09-15):** Duke stopped both trainings after the no-look-ahead audit + offline regressions showed the decision variable carried ~zero forward-return signal (r ≈ 0.003–0.02) and pass rates ratcheted 94 → 98%. Root cause: momentum/horizon information dies between vision and the readout (sub-threshold gains, 0 T4/T5/LC→KC synapses, readout excludes structural input) and credit assignment pools a day under one dopamine scalar. Ratified package: multi-bar state features (§3), per-exit dopamine (§7.3), widened plume ranker (§6.1), daily anchor refresh + grudge top-k 16 (§7.3), looming gain fix + λ structural drive (§3/§5). Emergence principle: the fly may learn momentum, mean-reversion, or neither — the umwelt no longer forbids it.

---

*Next step: confirm the stack assumption (Python 3.11+, plan/TASKS.md §1) and dispatch T1 — repo scaffold. Plan: plan/TASKS.md — larval-stage harness first (T2 connectome extraction → T4 sensory encoders → T5 plasticity → T7 backtest loop).*
