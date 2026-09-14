# The Fruit Fly of Wall Street — TASKS.md

**Version:** 0.1 (2026-09-11)
**Source of truth:** `DESIGN.md` v0.3 (all decisions settled, D1–D19). The design wins on any conflict.
**Execution model:** one task = one omp courier round (brief in `.omp-<task>-brief.md`, run via `scripts/acp_courier.py`), plus at most one fix round. Hermes is courier and verifier — it does not implement alongside omp.
**Status legend:** `TODO` | `DISPATCHED` | `REVIEW` | `DONE` | `BLOCKED`

---

## 1. Stack assumption (confirm before T1)

| Item | Default | Note |
|---|---|---|
| Language | Python 3.11+ (userland venv, PEP 668) | numpy + scipy sparse for LIF; matches yfinance/stooq/alpaca-py ecosystem and the Raspberry Pi MB-only target (§13) |
| Package layout | `pyproject.toml`, `src/fruitfly/`, `pytest` + `ruff` | — |
| Data cache | parquet/CSV under `data/` (gitignored) | — |
| Dashboard | stdlib `http.server` or FastAPI + vanilla JS (§13, D18) | thin by design |
| Repo | `git init` in this directory; DESIGN.md stays tracked | done as part of T1 |

Open items that need Duke, none blocking T1–T12:

1. **Stack veto** — speak before T1 dispatch if you want Rust instead.
2. **Alpaca paper API keys** — needed at T13. User action.
3. **CAVE/FlyWire Codex token** — possibly needed at T2 for connectome download. T2 brief must name a no-auth fallback (FlyWire Codex public snapshot); if both fail, the round surfaces a question.

## 2. Milestones

| M | Name | Exit criterion |
|---|---|---|
| M1 | Larval harness foundation | T1–T4 done: connectome subgraph loads, LIF core steps deterministically, all three senses encode |
| M2 | Learning + market loop | T5–T7 done: seeded backtest runs end-to-end over historical 1-min data, byte-identical double-run |
| M3 | Larval evidence | T8–T10 done: trained weight artifact, calibrated D13/D14/D16 values, scoreboard + post-mortem produced |
| M4 | Adult stage | T11–T15 done: whole-fly sim validated ≥90% (D19), Alpaca paper live loop, dashboard, run receipts |

## 3. Waves (dispatch order)

Sequential courier rounds unless noted. Max two lanes in parallel; never two omp sessions in one worktree.

- **W1:** T1 (alone — everything depends on it)
- **W2:** T2 ∥ T6 (independent: connectome vs market data)
- **W3:** T3 ∥ T4 (T4 codes against provisional population dims; T2's report finalizes them — re-sync T4 at review if dims moved)
- **W4:** T5 → T7 (plasticity before the loop that uses it)
- **W5:** T8 → T9 → T10 (each consumes the previous output)
- **W6:** T11 (whole-fly sim; may start any time after T3 if a lane is free)
- **W7:** T12 → T13 ∥ T14 → T15

## 4. Task ledger

| ID | Title | DESIGN anchors | Depends on | Status |
|---|---|---|---|---|
| T1 | Repo scaffold + conventions | §13 | — | TODO |
| T2 | Connectome acquisition + stripped-chassis extraction | §2, D10 | T1 | TODO |
| T3 | LIF sim core | §2 | T2 | TODO |
| T4 | Sensory encoders (vision, smell, taste) | §3, D5, D15 | T2, T3 | TODO |
| T5 | Neuromodulation + dopamine-gated plasticity | §4, D6 | T3 | TODO |
| T6 | Market data layer | §8, D8, D12, D16 | T1 | TODO |
| T7 | Foraging backtest loop | §5, §6, §7, D3, D13, D16 | T3, T4, T5, T6 | TODO |
| T8 | Scoreboard + baselines | §11 | T7 | TODO |
| T9 | Larval training + calibration | §9, D13, D14, D16 | T7, T8 | TODO |
| T10 | Post-mortem generator | §12 | T9 | TODO |
| T11 | Whole-fly sim | §2, D10 | T3 | TODO |
| T12 | Transplant + behavioral validation | §2, D19 | T9, T11 | TODO |
| T13 | Alpaca paper adapter | §8, D12 | T1 | TODO |
| T14 | Live local dashboard | §12, D18 | T7 | TODO |
| T15 | Adult run harness (wake/sleep, death/rebirth, receipts) | §7, §9, D7, D14, D17 | T12, T13, T14 | TODO |

## 5. Task details

Each block lists: deliverables, acceptance (machine-checkable), and the anchors the brief MUST quote verbatim. Every task also inherits the standing brief rules in §6.

### T1 — Repo scaffold + conventions
- **Deliverables:** `git init`; `pyproject.toml`; `src/fruitfly/` package skeleton; `pytest` + `ruff` wired; `.gitignore` (`data/`, `.omp-*.log`, `__pycache__`); `AGENTS.md` (conventions: seeded determinism everywhere, headless CLI for every runnable artifact, no git writes by agents, brief protocol); stub `python -m fruitfly --help`.
- **Acceptance:** `pytest` green; `ruff check` clean; `python -m fruitfly --help` exits 0.

### T2 — Connectome acquisition + stripped-chassis extraction
- **Anchors:** §2 (brain chassis, stripped chassis definition), D10.
- **Deliverables:** raw connectome cache under `data/connectome/` (FlyWire Codex public snapshot primary; CAVE fallback); extraction script for the stripped chassis — retina/optic lobe (photoreceptors, T4/T5, looming-sensitive cells), olfactory pathway (~50 glomeruli, Kenyon cells, MBONs, PAM/PPL1 dopamine neurons), descending readout; weights from synapse counts; excitatory/inhibitory sign from predicted transmitter (Shiu et al. style, §2); `fruitfly.connectome` loader module; `reports/t2-connectome.md` stats report (per-region neuron counts, synapse counts, transmitter coverage, ~35 MBON count check per §5).
- **Acceptance:** all named populations present and non-empty; loader output hash-stable across two runs; report states the exact neuron count of the stripped chassis (the number every later brief re-quotes).

### T3 — LIF sim core
- **Anchors:** §2 (simulation recipe), §5 (latency note: sim may clock faster than real time).
- **Deliverables:** `fruitfly.sim` — sparse LIF engine over the stripped chassis; configurable dt; seeded stepping; headless step API; perf benchmark CLI.
- **Acceptance:** unit tests for single-neuron spiking and excitatory vs inhibitory sign; two same-seed runs produce byte-identical spike logs; benchmark: one full trading day (390 one-minute steps) through the stripped chassis in ≤10 min wall on this laptop (state measured time in report).

### T4 — Sensory encoders (vision, smell, taste)
- **Anchors:** §3 (all three senses), D5 (identity × state smell), D15 (taste in v1).
- **Deliverables:** `fruitfly.senses.vision` — candlestick chart render at ommatidia resolution → photoreceptor input (T4/T5 direction = price drift; looming channel = vertical spikes); `fruitfly.senses.smell` — fixed identity signature per ticker × state features (returns, RSI, volatility, volume delta) → ~50-glomerulus activation; `fruitfly.senses.taste` — unrealized P&L → sweet/bitter input; golden fixtures on synthetic charts.
- **Acceptance:** encoder output dimensions match the T2 input populations exactly; identity signature is constant per ticker and distinct across tickers; fixtures reproduce byte-identical; looming channel fires on a synthetic crash spike and stays silent on flat data.

### T5 — Neuromodulation + dopamine-gated plasticity
- **Anchors:** §4 (state table + learning rule), D6 (stimulus mapping).
- **Deliverables:** `fruitfly.neuromod` — PAM ← realized profit, PPL1 ← realized loss, hunger ← drawdown (raises reward drive), octopamine ← volatility; dopamine-gated plasticity at KC→MBON synapses; sleep operator (consolidation + memory decay).
- **Acceptance:** one-trial learning test (one shock → avoidance on re-exposure); habituation test (unpredictive stimulus fades); hunger test (drawdown measurably raises reward drive); sleep test (decay applied, doubles as regime forgetting).

### T6 — Market data layer
- **Anchors:** §8 (free only, hard constraint), D8, D12 (US equities), D16 (1-min bars).
- **Deliverables:** `fruitfly.data` — yfinance/stooq 1-minute bar ingestion; local cache; US market calendar + regular-session filter; validation report (gaps, timezone correctness, per-symbol bar counts); fetch CLI.
- **Acceptance:** cached bars for a stated test basket (≥20 S&P 500 symbols, ≥20 trading days); validation report shows zero out-of-session bars; two fetches of the same window are byte-identical after caching; no paid feed anywhere in the dependency tree.

### T7 — Foraging backtest loop
- **Anchors:** §5 (MBON readout → order mapping), §6 (plume filter, encounter, N=10 cap, exits), §7 (daily loop), D3, D13, D16.
- **Deliverables:** `fruitfly.loop` — plume filter (top-K smelliest movers of the index per bar); round-robin encounters (one plume at a time: smell + vision); MBON approach/avoid → BUY/SELL/pass; position size from MBON firing rate; portfolio cap N=10; exits (valence flip; sharp adverse move → shock → immediate avoidance; hunger → close a winner); per-bar cycle: render → mix odor → step sim → read MBONs → emit paper order; close-of-day settle → sugar/shock → sleep; headless CLI `python -m fruitfly.backtest --seed S --start ... --end ...` writing equity curve + full event log.
- **Acceptance:** two same-seed runs byte-identical (equity + event log); position cap never exceeded in the log; event log covers wake, encounters, decisions, orders, sugar/shock, sleep; different seeds diverge (sanity).

### T8 — Scoreboard + baselines
- **Anchors:** §11 (four benchmarks).
- **Deliverables:** `fruitfly.scoreboard` — S&P 500 buy-and-hold; monkey-with-darts (seeded random, same universe and windows); logistic regression control on identical state features; SPIVA baseline as a static cited reference row; report writer comparing any fly run against all four over the same window.
- **Acceptance:** report renders all four columns for a T7 output; random and logistic baselines reproduce byte-identical at fixed seed; logistic features are provably the same feature vector the smell channel uses.

### T9 — Larval training + calibration
- **Anchors:** §9 (larval stage, death, single fly), D13 (N=10, calibrate), D14 (−50% equity death, calibrate), D16 (bar granularity, calibration candidate).
- **Deliverables:** training driver over historical windows; calibration sweep over {position cap, death threshold, bar granularity} around the settled defaults; trained KC→MBON weight artifact (`data/fly-larval-weights.npz` or equivalent); `reports/t9-calibration.md` with the picked values and reasons; death → hatch-with-fresh-plasticity path exercised.
- **Acceptance:** artifact loads into T7 and changes decisions (sanity probe); calibration table covers ≥3 values per swept parameter; death at −50% equity demonstrably kills the fly and hatches a new one with reset plasticity; chosen values recorded as a decision addendum for Duke (D13/D14/D16 updates need his sign-off, not the agent's).

### T10 — Post-mortem generator
- **Anchors:** §12 (post-mortem content).
- **Deliverables:** `fruitfly.postmortem` — from a run's event log: fly lifespan, cause of death, P&L vs scoreboard, notable one-trial grudges/favorites (§6 free behaviors).
- **Acceptance:** runs on the T9 larval output; produces the report file end-to-end; cites event-log line ranges for the stated cause of death.

### T11 — Whole-fly sim
- **Anchors:** §2 (whole fly = art piece / promo mode; all 166,691 neurons, all the time), D10 (Phase B).
- **Deliverables:** full male-CNS chassis loading (same T2 pipeline, no subgraph filter); sparse LIF over ~125M synapses; same encoder/readout interface as the stripped chassis; perf + memory report.
- **Acceptance:** full connectome loads and steps; neuron count = 166,691 in the load report; same-seed determinism; report states achievable sim rate and RAM on this laptop; if LIF misses the budget, the round documents it and falls back to the lean rate model per §13 — recorded deviation, not silence.

### T12 — Transplant + behavioral validation
- **Anchors:** §2 (Phase B transplant), D19 (direct KC→MBON copy + ≥90% decision agreement on held-out days; fine-tune only if below).
- **Deliverables:** weight-transplant tool (stripped → whole-fly KC→MBON); validation driver replaying held-out market days through both brains; `reports/t12-transplant.md` with per-day agreement table.
- **Acceptance:** agreement metric = fraction of bars with identical BUY/SELL/pass decisions; report shows ≥90% mean agreement, or exactly one documented fine-tune round and its result; D19 is revocable — the report states which branch was taken.

### T13 — Alpaca paper adapter
- **Anchors:** §8 (Alpaca free tier, IEX feed, US equities), D12.
- **Deliverables:** `fruitfly.broker` — Alpaca paper account adapter: IEX feed intake, paper order submit/cancel, position and account state sync; mapping from T7 order events to broker calls; config via env vars (keys never committed).
- **Acceptance:** end-to-end paper smoke: submit → fill → position visible → close, all on the paper account; zero code paths capable of reaching a live-money endpoint (grep-proof in the report); blocked until Duke provides paper API keys.

### T14 — Live local dashboard
- **Anchors:** §12 (dashboard content), D18 (simple, local, no PDF).
- **Deliverables:** local web view reading run state: current equity, open positions, MBON balance, neuron-activity view, event log tail; auto-refresh.
- **Acceptance:** serves on localhost; headless smoke test fetches every route (HTTP 200) and asserts the five panels render with live data from a replayed T7 run; nothing leaves the machine (no external calls).

### T15 — Adult run harness
- **Anchors:** §7 (daily loop), §9 (adult stage, death, single fly), D7 (sleeps aftermarket), D14, D17.
- **Deliverables:** `fruitfly.adult` — market-calendar wake/sleep driver; persistent fly state files (plasticity survives restarts); death → hatch cycle; receipts (per-day event log + equity) identical in shape between replay mode and live-paper mode.
- **Acceptance:** multi-day replay run over recorded data produces receipts; kill-and-resume mid-day continues from state file; death during replay triggers hatch with fresh plasticity and logs it; live mode differs from replay only at the data/broker adapter seam.

## 6. Standing orchestration rules (every dispatch)

These restate the omp skill's rules as project convention; the skill remains authoritative.

1. **One task = one brief + one courier run.** Brief at `.omp-t<N>-brief.md` per the omp skill template; anchor paragraph at top; quote the task's DESIGN anchors verbatim.
2. **Worktree before brief.** `git worktree add` first; brief written inside the worktree; never a plain directory.
3. **Model pin.** Current pin per the omp skill at dispatch time; restated in the brief header and on all four role flags. Verify via session jsonl after dispatch.
4. **Read-only paths.** DESIGN.md and this file are read-only for the agent. Calibration value *changes* (T9) are proposals for Duke, not edits.
5. **No agent git writes.** Work lands uncommitted; Hermes verifies (runs the acceptance commands itself, same-seed double-runs, greps claimed features) and commits.
6. **Two-pass review** on every task PR before merge; fix rounds carry detector proofs.
7. **Autonomy window.** Any dispatch Duke cannot watch gets: work until <absolute UTC>, no questions, pick the spec-closest option, log the choice.
8. **Determinism is a hard gate** for every runnable artifact: same seed → byte-identical output, verified by the courier, twice.
9. **Free-data constraint is absolute** (D8): a dependency that phones a paid API is an automatic reject.

## 7. Calibration register

Settled starting values; T9 owns tuning evidence; Duke signs off changes.

| Param | Current | Source | Calibration candidate? |
|---|---|---|---|
| Position cap | 10 | D13 | yes (larval) |
| Death threshold | −50% equity | D14 | yes |
| Bar granularity | 1 min | D16 | yes |
| Transplant rule | copy + ≥90% agreement | D19 | revocable, my call per Duke's defer |
| Plume top-K | unset | §6 | T7 proposes, T9 tunes |
| MBON approach/avoid thresholds | unset | §5 | T7 proposes, T9 tunes |

## 8. Next action

1. Duke confirms/vetoes the stack assumption (§1).
2. Dispatch T1 (brief per §6).

*Changelog: v0.1 (2026-09-11) — initial plan from DESIGN.md v0.3. 15 tasks, 4 milestones, 7 waves.*
