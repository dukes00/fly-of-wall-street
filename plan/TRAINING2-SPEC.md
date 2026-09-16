# TRAINING2 SPEC — entry skill + realized P&L

**Status:** RATIFIED by Duke 2026-09-16 (A1–A9 as DESIGN v0.7; §10 checklist closed; `MISS_WEIGHT=0.25`; objective = active trader choosing on information — A vs B decided empirically in Phase 0). Corrective pass after design review: the six review blockers are pinned in place (avoid-side credit, single-credit last-credit-wins rule, T-1 exit-scoring cost/default, overlap-robust G-A.1 statistics, artifact meta contract, turnover guard G-A.3).
**Grounding:** all file:line refs are against `src/fruitfly/` at HEAD. Runtime claims trace to measured numbers: stripped ≈ 2–3 min/trading-day, whole ≈ 8.3 min/day, whole-fly eval ≈ 40 min/day (M1 Pro, 16 GB). Data: Alpaca IEX 1m free (D8), 2024+2025 fetched (~40 symbol-years), IEX depth verified to 2024+; ≥2016 unverified.

**PHASE-0 HEAD-TO-HEAD VERDICT (2026-09-16): Option B (advantage gating) wins** — `reports/h2h-avsb-phase0.md`. Stripped lane, 10 days, seed 7: B leads on the ratified objective (positive sign-stable entry signal +0.023/+1.02 with subsample agreement, trades/day 1.77× reference inside the G-A.3 band, win/loss 1.80, paired +0.87%/day vs reference t=+2.15); A's entry signal is ~zero with subsample sign disagreement. G-A.1 fails for both at this power (effective n ≈ 400 — expected); Phase A proceeds with B + the §6 Phase-A grid, G-A.1 re-checked per cell, capture preservation tracked as the §9.2 sentinel.
---

## EXECUTIVE SUMMARY

1. The fly already learned **when to be in the market** (regime utilization: win/loss 0.23→2.04 in training; OOS up-capture 1.20 / down-capture 0.34, stable over 40 OOS days) but not **what to buy** (forward returns after buys ≈ 0 or negative; 30m post-buy edges ≤ 0 in 7/8 OOS buckets; `balance_used` corr ≈ 0 with forward returns).
2. Nothing in the current objective rewards picking winners: the dopamine gate sees pooled daily P&L (`settle_and_sleep`, loop.py:826-834) and per-exit realized P&L (loop.py:787-795) — both are portfolio/trade outcomes, not entry forecasts.
3. Two competing objectives are specified (§2): **A. supervised entry-forecast reward** (per-encounter forward return into the gate) and **B. advantage/contrastive gating** (gate fires on return *relative to a state-conditional baseline*). They are distinct mechanisms; both hook at the same two credit sites but change different bookkeeping.
4. The asymmetry is currently evaporated at the exit: closed-trade P&L is negative in all 8 OOS cells (the fly closes losers and rides marks). §5 upgrades exits (taste channel is dead code today — taste currents land on PAM/PPL1, which are sign-0 — zero outgoing weights, so their spikes never propagate (sim.py:24-28) and the loop reads no PAM/PPL1 activity (neuromod.py:15-19)).
5. Eval power is the third lever: 5 days/cell cannot resolve a 0.1–0.2%/day edge at observed σ. §7 mandates ≥20 days/period, paired A/B tests, one never-seen basket.
6. **Corrective pass (this revision) pins the six review blockers:** avoid-side credit semantics (§2A — correct avoids earn positive credit scaled by |r_fwd|, missed run-ups punished at `MISS_WEIGHT`), the single-credit **last-credit-wins** rule (a realized close credit replaces the settle forecast credit via a delta credit, §2A), the T-1 exit-forecast cheap default (score exits only on bars where the held ticker IS the encountered ticker; every-bar scoring is a Phase-A budget experiment, §5.1), overlap-robust G-A.1 statistics (HAC/Newey–West lag-30 + non-overlapping subsample; effective n ≈ ticker-days ≈ 400, §6), the artifact meta contract (`train_larval` meta records the new knobs and Option B baselines; `eval_brains` asserts meta ↔ config, §2 + Phase 0), and the turnover guard G-A.3 (trades/day within [0.5×, 2.0×] of the reference band, §6).

---

## 1. PROBLEM STATEMENT

Decomposition of the fly's behavior, from the 40-day OOS analysis:

| Component | Evidence | Verdict |
|---|---|---|
| Exit/regime disposition | win/loss ratio 0.23 → 2.04 over training; OOS up-capture 1.20, down-capture 0.34, stable across 40 OOS days | **learned** |
| Entry selection | forward return after buys ≈ 0 or negative; 30m forward edges ≤ 0 in 7/8 OOS buckets | **coin flip** |
| Decision variable informativeness | `balance_used` corr ≈ 0 with forward returns (the variable that drives buys, loop.py:1040) | **no entry signal** |
| Realization | closed-trade realized P&L negative in **all 8** OOS cells; unrealized marks carry the asymmetry | **asymmetry not converted to P&L** |

Why the objective cannot fix entry by itself: the only training signals are
(a) **daily pooled** P&L → `Plasticity.observe` (neuromod.py:216) at `settle_and_sleep` (loop.py:845) — one scalar for the whole day, diluted across every encounter;
(b) **per-exit realized** P&L → `Plasticity.observe_trade` (neuromod.py:263) at `apply_sell` (loop.py:790) — credits the opening encounter with `realized/notional`, but only for trades that actually closed, and rewards *outcome*, not *selection* (a buy in a rising tape is rewarded for the tape, not the pick).

Consequence: gradient pressure goes to "be long when the tape rises" (which the fly learned) and never to "this ticker, not that one." TRAINING2's job: make the gate see **ticker-relative forward returns**, and make exits realize the captured asymmetry.

---

## 2. OBJECTIVE REDESIGN — two competing options

Both options share the same plumbing (an encounter credit ledger in `run_backtest`) and differ in the gate computation. Both respect the credit-timing rule:

> **Anti-lookahead rule (hard constraint, applies to both options).** The forward return may enter the gate only *after* the information that defines it exists: credit is emitted at the earliest of (i) the position's close fill (already async, next-bar pessimistic model, loop.py:901-912) and (ii) day settle `settle_and_sleep` (loop.py:822), computed from `last_close` values of bars **strictly after** the encounter bar (an encounter on a day's final bar gets r_fwd = 0 from its own close — zero credit, no leak). Never as a same-encounter input to the decision (the decision path loop.py:1040-1064 never sees it). The forward return uses the **encounter bar's close** as base — a price the decision already had — so no same-bar execution fiction is introduced. The existing `close` fill-mode escape hatch (loop.py:216-222) is out of scope for all training runs.

### Shared plumbing (both options)

New module-level state in `run_backtest` (loop.py:593), mirroring `entry_elig` (loop.py:712):

```python
# entry = (decision_ts, ticker, eligibility_snapshot (copy of intraday_elig),
#          base_price=last_close[ticker], action)   # action ∈ {buy, add, pass_signal}
encounter_ledger: list[_EntryRecord]
```

- Snapshot `intraday_elig.copy()` for **every signal-bearing encounter** (the `approach_drive + avoid_drive > 0` branch, loop.py:999-1022), not only buys. `intraday_elig` already exists and is the correct trace (loop.py:719-727).
- A new pure helper in `loop.py` (or `neuromod.py`) computes the credit and calls `Plasticity.observe_trade` (neuromod.py:263) — the mechanism is already exactly right: it applies the three-factor update against a snapshot without touching the live trace or habituation (neuromod.py:282-294).
- New event `{"type": "entry_credit", ...}` emitted alongside the existing `trade_credit` (loop.py:796-805).
- Determinism: unchanged in kind — float64, no new RNG (Option B's baseline is a deterministic EMA; no hashing of wall-clock, `stable_u64`-free).
- **Artifact meta contract (additive-safe, pinned).** `train_larval` meta (train.py:144-150) records every new knob: `id_scale`, `trade_credit_mode`, `entry_credit`, `horizon_bars`, `r_scale`, `miss_weight`, `avoid_correct_weight`, `daily_observe`, plus — Option B only — the bucket baselines (`initial_baselines`, JSON-safe dict). The meta writer is additive (existing loaders ignore unknown meta keys today). `scripts/eval_brains.py` asserts the artifact's meta matches the runtime config (basket, knobs) before evaluating any arm and refuses a mismatch, so a stale artifact can never be scored under different credit semantics.

### Option A — supervised entry-forecast reward

**Mechanism:** per signal-bearing encounter, the gate receives `f(r_fwd)` — the realized forward return of the *encountered ticker* from the encounter bar's close to the credit time — independent of whether the fly traded.

```python
H_BARS = 30            # forecast horizon in 1-min bars (matches the 30m OOS edge metric)
# r_fwd = (P_exit_or_settle / base_price) - 1     base_price = encounter bar's close
# a = tanh(r_fwd / R_SCALE)            # per-encounter gate input, Option A
# bought  (any buy encounter)          : reward += max(0,  a) ; punishment += max(0, -a)
# avoided, r_fwd < 0  (correct avoid)  : reward    += AVOID_CORRECT_WEIGHT * max(0, -a)
# avoided, r_fwd > 0  (missed run-up)  : punishment += MISS_WEIGHT * a
R_SCALE = 0.005              # 0.5% return → tanh ≈ 0.46; typical 30m IEX move scale
MISS_WEIGHT = 0.25           # missed-winner weight — approached-but-passed buys AND avoided run-ups
AVOID_CORRECT_WEIGHT = 0.5   # correct-avoid reward weight (positive credit scaled by |r_fwd|)
```

- For a **buy** that was filled and closed within the horizon: credit at `apply_sell` replaces (or augments, knob `trade_credit_mode ∈ {"realized", "forecast", "mix"}`) the current `realized/notional` gate (loop.py:787-795). `forecast` uses `r_fwd` to the exit fill price — the same signal up to the entry-side pessimistic-fill spread (`avg_cost` is the next-bar HIGH, not the encounter close) and fees; mode `mix` blends them.
- For a **buy** still open at day settle, or a **pass/avoid** encounter: credit at `settle_and_sleep` (loop.py:822, before `plasticity.observe` at loop.py:845) using `last_close[ticker]` at settle — a price already public to the loop, no look-ahead.
- **Single-credit rule (last-credit-wins, pinned).** Each `_EntryRecord` carries `credit_state ∈ {"pending", "credited@settle", "final"}` and contributes **exactly one effective gate credit**. A settle-time forecast credit marks the entry `credited@settle`; if the position later closes, the close credit **replaces** the settle forecast credit — realized data is strictly better than a forecast, so the last (realized) credit wins. Replacement is implemented as a delta credit: `observe_trade` receives the difference `realized − provisional` (one extra Hadamard against the stored compact snapshot), so the net per-snapshot gate credit equals the realized outcome with no weight rollback; the event log records both credits, the replacing one carrying `supersedes: <event_id>`. The Option B bucket EMA updates **exactly once per entry**, only at its final credit (the close credit, or the settle credit for never-bought encounters, which never close); forecast credits on entries still open defer their baseline update to the close, and a terminal sweep at backtest end finalizes anything still open. Double-crediting a snapshot (settle forecast + close realized both applied in full) is a gate bug; the offline audit test asserts no snapshot ever receives two non-delta credits.
- **Avoid-side credit (pinned).** Avoid/pass encounters are the majority of signal-bearing encounters — leaving them uncredited would starve the gate of most of its gradient — so they are credited by *correctness of the avoid*: an avoid followed by a **downward** forward return (`r_fwd < 0`) was correct and earns positive credit scaled by the realized move, `AVOID_CORRECT_WEIGHT · tanh(|r_fwd|/R_SCALE)`; an avoid followed by a run-up (`r_fwd > 0`) was a missed winner — a learning signal, not a crime — punished at `MISS_WEIGHT · tanh(r_fwd/R_SCALE)`, the same weight approached-but-passed buys receive. The asymmetry (correct avoids 0.5 vs missed winners 0.25) encodes the behavior prior that the pass-rate ratchet (DESIGN v0.6 changelog: 94→98% pass) makes under-entering the dominant failure mode; both weights are sweep knobs. This makes the "more proactive fly" lever trainable end-to-end: the gate learns *negative* weights for approached-but-avoided states that subsequently ran, pushing `approach_thr` behavior without hand-tuning (loop.py:180-181, `_decide` loop.py:457).

**What changes:** loop.py gains `_EntryRecord` + ledger + `_credit_entry()`; `BacktestConfig` gains `entry_credit: bool`, `horizon_bars: int = 30`, `r_scale: float`, `miss_weight: float`, `avoid_correct_weight: float`, `trade_credit_mode: str`. `Plasticity` unchanged (the gate math in `_gate`, neuromod.py:209-213, is untouched; only the `NeuromodState` reward/punishment inputs change).

**Runtime cost:** zero extra sim steps — credit is O(1) float ops per encounter against a snapshot; `observe_trade` cost is one (4064×97) Hadamard per credit, same as today's per-exit path. Ledger memory: one eligibility snapshot is 4064×97 float64 ≈ 3.1 MB — buy snapshots live in `entry_elig` (≤ position_cap=10 → ~31 MB), but pass-encounter snapshots accumulate until settle (up to ~390 × 3.1 MB ≈ 1.2 GB peak as full trace copies); store compact per-encounter increments instead (KC-activity + post vectors, ~33 KB each) and rebuild the support-masked outer product at credit. Training wall-clock: unchanged (~2–3 min/day stripped).

**What could break:** (i) look-ahead — mitigated by the rule above; write an offline audit test that replays a day and asserts every `entry_credit` references prices strictly after its `decision_ts` bar (timestamp ≥ `decision_ts` + 1 bar; the encounter bar's close is the documented base). (ii) Sign conflict with the daily pooled ritual (loop.py:826-834): a winning day can carry negative per-encounter credits; add knob `daily_observe ∈ {"on", "off"}` and ablate — the daily diffuse gate is plausibly the dilution culprit and should be toggleable for the A/B. (iii) Horizon mismatch: `H_BARS=30` must match the eval metric (30m forward edge) or the fly optimizes the wrong window. (iv) Pass-encounter credits increase updates/day roughly 2× (passes currently get zero credit) — watch `sleep_decay`/`top_k` (neuromod.py:119-120, loop.py:241) for grudge saturation.

### Option B — advantage / contrastive gating (predictive plasticity)

**Mechanism:** the gate fires on the **advantage** — realized forward return minus a state-conditional expectation — so the fly learns "better than the state predicts," not "positive." The regime beta (the thing it already learned) is subtracted out; only selection skill survives the gate.

```python
# State key (deterministic, coarse): bucket = (sign(mom20), vol_bucket, dir_bucket)
#   vol_bucket = 0/1/2 by volatility tercile threshold (fixed constants, not data-derived)
#   dir_bucket = 0/1 by rsi < / >= 50
# baseline: EMA per bucket, float64, persisted in RunResult and saved to the artifact meta
#   b_new = (1 - ALPHA) * b_old + ALPHA * r_fwd        ALPHA = 0.05
# a = r_fwd - b[bucket]
# reward     = max(0,  tanh(a / A_SCALE))
# punishment = max(0, -tanh(a / A_SCALE))
A_SCALE = 0.003
```

- **Credit sites:** identical to Option A (ledger → `observe_trade` at close / settle). The difference is purely in the gate input: `a` (advantage) instead of `r_fwd`.
- **Where the baseline lives:** a small dict in the loop (`dict[tuple[int,int,int], float]`), initialized to 0.0 (i.e. Option A semantics on the first observation per bucket, converging to contrastive), updated at every credit — a deterministic online model, updated *after* the credit is computed (the prediction used for credit is always the pre-update one; no target leakage into its own update).
- Baseline warmth: persisted via `train_larval` meta (train.py:212-248) so multi-seed/multi-window runs can warm-start; artifact format gains one JSON-safe field in the provenance dict (train.py:144-150 meta contract — new field is additive, loaders ignore unknown meta today).

**What changes:** same loop plumbing as A, plus the bucket EMA (~30 lines, no new module). `Plasticity` unchanged.

**Runtime cost:** negligible (one dict lookup + one EMA per credit). Same wall-clock as A.

**What could break:** (i) bucket starvation — with 12 buckets and ~1 encounter/bar, early buckets are cold; the zero-init ramp handles it but the first training day is effectively Option A. (ii) Baseline absorbs the very beta we want to keep: the fly's OOS value is regime capture (up 1.20 / down 0.34); if the advantage gate strips regime drift, the *exit* side must carry the regime position (§5) or the combined system loses its edge. This is the central A-vs-B tradeoff — hence run both. (iii) Determinism holds (float EMA, fixed op order, no RNG), but the baseline makes runs order-dependent *within* a window — the artifact replay (`initial_weights`, loop.py:605-612) must persist the baseline too or a restored fly credits with a cold baseline; add `initial_baselines` to `BacktestConfig` mirroring `initial_weights`.

### Comparison

| | A: entry-forecast | B: advantage |
|---|---|---|
| Gate input | absolute `r_fwd` | `r_fwd − E[bucket]` |
| Teaches | "buy winners" | "buy *relative* winners" |
| Keeps regime beta in the gate | yes | no (must move to exits) |
| New state | ledger only | ledger + bucket EMA (persisted) |
| Failure mode | rewards bull-tape beta, duplicates what's learned | strips the beta that IS the edge |
| Risk | sign conflict with daily ritual | cold-start + baseline persistence |

**Recommendation:** implement both behind `BacktestConfig` knobs; Phase A/B of §6 trains A first (simpler, matches the "supervised entry-forecast" lever ranked #1 in the sweep analysis), then B as the challenger. The paired eval in §7 decides.

---

## 3. WIDER TICKER POOL

**Current:** `BASKET` is 22 hardcoded symbols (data.py:29-33); `_plume_set` ranks exactly this dict (loop.py:430-454); `run_backtest` loads the whole basket (loop.py:637-641).

**Target: N = 22 → 40 train / 20 held-out eval (or 60 train / 20 held-out).**

### Mechanics

1. **Basket files, not code.** New `--basket-file data/market/train40.txt` (one symbol per line, `#` comments) parsed into a module-level list that replaces `BASKET` references at the three use sites: `run_backtest` (loop.py:639), `_innate_balance`'s ticker list (loop.py:651 — already receives `list(frames)` keys, so it follows automatically), and `scripts/eval_brains.py`'s `_basket_override` (scripts/eval_brains.py:415-437, already exists — the backtest CLI is the only gap). Deterministic: file content is committed alongside the run receipt; `train_larval` meta records the basket list (train.py:144-150).
2. **Data budget.** Cache contract: one parquet per symbol, `data/market/{SYMBOL}_1m.parquet` (data.py:3-8, `cache_path` data.py:45). Already fetched: 2024+2025, ~40 symbol-years of IEX 1m. Adding 18–38 symbols × 2 years ≈ +36–76 symbol-years. IEX 1m free feed is ~2–5% of consolidated volume — expect sparse bars for mid-caps (the cache keeps gaps as gaps, data.py:7-8; `_plume_set`'s dead-ticker rule, loop.py:450-451, already handles no-print bars). **Pre-commit gate:** verify IEX depth reaches at least the earliest training window before committing symbols (fetch 1 month for a candidate mid-cap and count bars/day; require ≥ 300 of 390).
3. **Sim cost scaling — the cheap direction.** The loop takes **one encounter per bar** (pointer rotation through the `top_k` plume set, loop.py:957-971), so encounters/day ≈ bars with a nonempty plume set (≈ all ~390 bars/day with the current 22 mega-caps) — **not** `top_k` × bars. The per-encounter cost (vision encode + smell + LIF step, loop.py:974-984) is independent of pool size N. Widening N only adds: (a) `load_bars` read time (parquet, one-time per run), (b) `_plume_set` ranking O(N) per bar — float ops, ~µs. At fixed `top_k=5`, **aggregate encounters/day and therefore the 2–3 min/day (stripped) / 8.3 min/day (whole) budgets are unchanged**. Raising `top_k` does not multiply sim cost either: it lengthens the rotation cycle (each ticker encountered every ~top_k bars — per-ticker encounter frequency ∝ 1/top_k) and adds O(N) ranking per bar; encounter *coverage* (which tickers surface, how often) is set by the pool × top_k interaction, not the pool alone.
4. **Curriculum.** Phase A trains on 40 (mix of the current 22 mega-caps + 18 mid-caps across sectors); eval baskets: (i) the 22-name current basket (in-distribution), (ii) a **20-name never-seen basket** — symbols that appear in *no* training run, ever (list committed before Phase A starts, training runs assert against it via the basket-file). This is the honest test of whether entry skill is identity-based or state-based.
5. **Credit-dilution guard.** Wider pool at fixed `top_k` doesn't raise encounter count, but the ledger credits more distinct tickers per day. Watch the `grudge_top_k=16` budget (loop.py:241, `sleep` neuromod.py:297-319): with more associations competing for 16 protected slots, consolidation turnover rises. Sweep `grudge_top_k ∈ {16, 32}` in Phase A.

### Identity attenuation (determinism-safe)

The sweep analysis found identity features contribute nothing that generalizes (basket effect flips sign between periods) — consistent with the smell decomposition's design intent (DESIGN.md §3, D5: identity enables specific associations, state enables general ones; the data says the specific ones are noise).

Options, in order of preference:

| Mechanism | Where | Determinism |
|---|---|---|
| **Identity scale** `ID_SCALE ∈ [0,1]`: `encode_smell` returns `base * modulator * ID_SCALE` (senses/smell.py:187-213; scale applied to the identity `base[channels]` term only) | `senses/smell.py:213` | pure constant, no RNG |
| **Hash-profile dropout**: per encounter, zero a seeded fraction of identity glomeruli using the loop's owned RNG (`rng`, loop.py:614 — the loop owns the only RNG by contract, loop.py:613) | loop.py:979 | seeded, replayable |
| **Drop identity entirely** (state-only odor) | `encode_smell` | trivially deterministic |

`ID_SCALE=0` is the cleanest ablation and the first thing to sweep (it's one multiply at one call site). Note `_innate_balance` (loop.py:497-499) sniffs the *mean identity profile* **directly — it does not call `encode_smell`**: an ID_SCALE multiply inside `encode_smell` alone leaves the hatch sniff and the daily anchor refresh (loop.py:861-868) identity-driven. To get the documented silent-anchor behavior (sniff returns 0.0, loop.py:505-506), ID_SCALE must also scale the `_innate_balance` profile input — the centering then degenerates gracefully (centered = raw balance). Flag this interaction in the T9 calibration register (§4).

---

## 4. REWARD CALIBRATION — winner-rewarding

Current gate symmetry: `reward_gain = punishment_gain = 1.0` (neuromod.py:117-118), gate `d = reward_drive − punishment_gain·punishment` (neuromod.py:209-213), and `reward_drive` multiplies by hunger (neuromod.py:205-207). Per-exit credit is symmetric in |realized|/notional (loop.py:788-789).

Changes (all as `BacktestConfig` knobs plumbed to `Plasticity.__init__` kwargs — the constructor already takes them, neuromod.py:107-121; `run_backtest` currently constructs `Plasticity(chassis, top_k=...)` at loop.py:597 and drops the rest):

1. **Asymmetric gate.** `reward_gain > punishment_gain` (start 1.5 / 1.0): a winner teaches more than an equal-size loss. Rationale: the fly's failure mode is *not entering* winners (pass-rate ratchet, DESIGN v0.6 changelog: 94→98% pass), not over-entering losers; the loss side is already handled by the shock exit (loop.py:914-930). Keep the asymmetry out of the daily ritual unless ablation says otherwise (Option A knob `daily_observe` above).
2. **Per-exit credit retention.** Keep `realized/notional` normalization (loop.py:787-789) — it fixed the 40× gate-scale bug (T9 addendum, loop.py:779-786 comment). With Option A `trade_credit_mode="mix"`, blend: `r = (1-w)·realized/notional + w·tanh(r_fwd/R_SCALE)`, `w` knob.
3. **Advantage baseline.** Option B's bucket EMA (§2B) is the principled version; a cheap approximation for the Phase-A config: subtract the day's basket mean forward return from `r_fwd` before `tanh` — one accumulator, same credit sites.
4. **Winner bonus (optional knob, not default):** `winner_bonus`: add a constant `+b` to the gate when `r_fwd > 0` and the fly *bought* — makes one-trial approach learning fire harder on genuine winners (the three-factor rule already learns in one trial when the gate is strong, neuromod.py:21-27 docstring). Evaluate only if the asymmetric gate alone under-drives.

**T9 calibration register changes** (`reports/t9-calibration.md` §7 is the register format): add rows for `reward_gain`, `punishment_gain`, `r_scale`, `miss_weight`, `avoid_correct_weight`, `trade_credit_mode`, `daily_observe`, `grudge_top_k`, `id_scale`, `horizon_bars` — each with: swept values, window, seed, marginal effect, picked value + reason, and the same deviation-flagging discipline (proposals only until Duke ratifies; the report's §2 pattern).

---

## 5. EXIT UPGRADE — convert asymmetry into realized P&L

Diagnosis: realized closed-trade P&L < 0 in all 8 OOS cells while unrealized marks carry the up-capture — the fly exits winners too early (hunger closes the largest winner, loop.py:933-953; valence_flip on a temporary dip) and holds losers until the −2% shock (loop.py:914-930). The `balance_used` asymmetry (up 1.20 / down 0.34) is being *given back* at the close.

### 5.1 The taste channel is dead — revitalize or replace

`encode_taste` (senses/taste.py:47-69) drives currents into PAM/PPL1 rows. But PAM/PPL1 are **sign-0 nodes: they integrate input and never propagate spikes** (neuromod.py:15-19; taste.py:11-14). The loop adds taste to the input at each held-position encounter (loop.py:980-982) and the current is then read by nothing — `spikes[mbon_rows]` (loop.py:1024) and the KC/MBON readout (loop.py:990-998) never see it. Taste cannot influence any decision today. Two options:

- **T-1 (replace, recommended): explicit exit-forecast gate.** Add a per-held-position check at the mechanical-exit block (loop.py:914-930): at each held ticker's bar, compute the state features (loop.py:961-962 pattern) and a learned exit score = `-balance_used` recomputed from the same encounter pipeline *without* taste; if `exit_score > exit_thr` (new config knob), submit a sell (`reason="exit_forecast"`). This trains against the same supervised objective as §2 — the entry gate's negative class is literally the exit signal. **Cost (corrected):** scoring every held position on every bar costs up to `position_cap` (=10) extra readouts/bar and — the real cost — a readout for a held ticker that is *not* the encountered ticker needs its own LIF step with that ticker's odor input (the loop.py:974-984 path), consuming spikes and advancing STD state for the whole fly; it is not a free side-readout. **Cheap default (pinned): score exits only on bars where the held ticker IS the encountered ticker** — rotation surfaces each held ticker every ~`top_k` bars, and those bars' spikes/readout are reused at zero marginal sim cost. The every-bar multi-step variant goes behind an explicit Phase-A budget experiment (`exit_score_every_bar: bool = False`); enable it only if the encountered-only default measurably under-performs and the extra-step budget is re-measured.
- **T-2 (revitalize): make taste reach the readout.** Sum taste currents into the *KC input* rows (pre-MBON) instead of PAM/PPL1, i.e. `inp[upn_rows] += ...`-adjacent hook at loop.py:979-982, with a gain knob. This is a DESIGN change (D15's mapping, taste.py:4-14) and a bigger behavioral risk (taste current saturates at 50/node (senses/taste.py:33-34) while a 2% unrealized move is only ≈1.0/node (50·tanh(0.02)) — ~0.1% of a strong uPN drive (SMELL_GAIN × profile up to 1000); the real risk is gain calibration against the 15 mV KC threshold, not flooding). Defer unless T-1 fails.

### 5.2 Mechanical exit grid

Add `BacktestConfig` knobs, all evaluated in the mechanical exit block (loop.py:914-930, same pessimistic next-bar queue — no new fill semantics):

| Knob | Semantics | Default today |
|---|---|---|
| `trailing_stop_pct` | sell when price falls `x%` below max close since entry (per-position high-water mark tracked in `_Position`, loop.py:324-327) | none (new) |
| `atr_stop_mult` | sell when price < entry − k·σ20·price — `build_features` `volatility` is the 20-bar close-to-close return std-dev (senses/smell.py:48, 101-123), a close-to-close ATR proxy; a true-range ATR needs a new helper (windows at senses/smell.py:80-81) | none (new) |
| `valence_flip_exit: bool` | keep/drop the `_decide` sell-on-avoid path (loop.py:470-472) | on |
| `hunger_exit: bool` | keep/drop the close-largest-winner behavior (loop.py:933-953) | on |
| `exit_score_every_bar` | T-1 exit-forecast scoring cadence: off = score only on bars where the held ticker is the encountered one (cheap default, zero extra sim steps); on = every bar, up to `position_cap` extra sim steps/bar + whole-fly STD perturbation (Phase-A budget experiment only) | off (new) |

Experiments (Phase B): (E1) trailing 1% / 2% / 3%; (E2) ATR 1.5× / 2×; (E3) `valence_flip_exit=off` + trailing stop (does the valence flip close temporary dips the trail would ride?); (E4) hold-to-close (all discretionary exits off, shock only — isolates how much realized P&L the exits destroy). Acceptance for the winner: **median closed-trade P&L > 0 across OOS cells** while up/down-capture stays ≥ 1.0 / ≤ 0.5.

---

## 6. TRAINING PROTOCOL — phased, with acceptance gates

All commands assume `uv` + src-layout. Basket files are new artifacts (§3); `--basket-file` on the backtest/train CLIs is a Phase-0 implementation item. `train_larval` (train.py:212) persists the artifact; `--chassis stripped` is the fast lane (loop.py:270-274).

### Phase 0 — plumbing (no training)
Implement: ledger + credit paths (§2), config knobs (§2/§4/§5), `--basket-file`, basket fetch + IEX depth verification (§3), `id_scale` (§3), eval-protocol additions (§7). Offline regression: existing `tests/test_train.py` byte-identical determinism receipts must still pass with knobs at current defaults (train.py double-train byte-identity, reports/t9-calibration.md §3). **Gate: no default-behavior drift (bit-identical equity.csv with all new knobs at no-op values).** **Gate (artifact meta):** `train_larval` meta records `id_scale`, `trade_credit_mode`, the `entry_credit` knob set (`horizon_bars`, `r_scale`, `miss_weight`, `avoid_correct_weight`, `daily_observe`), and Option B `initial_baselines`; `scripts/eval_brains.py` asserts artifact meta ↔ runtime config before evaluating and refuses a mismatch.

### Phase A — fast iteration loop (stripped, 40 symbols, short window)
- Window: 10 trading days (2025 window inside fetched 2024–2025 data; pick a two-regime window: one up-week + one down-week). Seed 7.
- Command shape (per config cell):
  `uv run python -m fruitfly backtest --seed 7 --start 2025-06-02 --end 2025-06-13 --chassis stripped --basket-file data/market/train40.txt --entry-credit forecast --r-scale 0.005 --id-scale 0.0`
  followed by `scripts/calibrate.py train` to persist the artifact (train.py:212-248).
- Cost: 2–3 min/day stripped × 10 days ≈ **20–30 min/config**; grid of ~12 configs (2 objectives × 3 gate calibrations × 2 id_scale) ≈ 1 day of serial runs, parallelizable (the T9 sweep ran 5 workers).
- **Acceptance gate G-A (must pass before Phase B), computed on the training window from events.jsonl:**
  1. **Entry signal gate (G-A.1, overlap-robust):** Spearman correlation between `balance_used` at decision time and the forward 30m return of the encountered ticker, tested with **Newey–West (HAC) standard errors, lag = H_BARS = 30** — adjacent encounters share overlapping 30-bar forward windows and the same tape, so raw per-encounter n (≈ 3,900 per 10-day window) overstates the evidence; the effective n is bounded by independent ticker-days ≈ 40 tickers × 10 days = **400**, not 3,900. Robustness cross-check: non-overlapping subsample — one encounter per ticker per 30-bar window (raw n ≈ 5,200, still ticker-day-clustered) — must agree in sign. **Gate: |HAC t| > 2** on the training window; the subsample t is reported alongside. This is where 0.1% edges are resolvable; the eval-power problem in §7 applies to portfolio returns, not per-encounter correlations.
  2. **Behavior gate:** executed-trade win/loss ratio > 1.0 on the training window (today: 2.04 post-training — don't regress it).
  3. **Turnover guard (G-A.3):** executed trades/day within [0.5×, 2.0×] of the reference config's band (incumbent v0.6 config, same window/seed/basket) **and** per-encounter buy rate (buys/encounter) within the same factor band — so per-exit or per-encounter credit cannot be gamed by churn (harvesting close-credit frequency with many tiny trades). Outside the band → gate fails regardless of the other statistics.
  4. **Learning gate:** fresh-vs-trained decision-change rate > 0 (the T9 probe pattern, reports/t9-calibration.md §5) and no death-regime pathology (`n_deaths` ≤ baseline).

### Phase B — exit integration + whole-fly confirmation
- Take the Phase-A winner; sweep exit grid E1–E4 (§5.2) on **stripped**, 10 days, same window.
- Confirm the combined config (entry + exit) on **whole fly**, 10 days: 8.3 min/day × 10 ≈ **85 min/config**.
- **Gate G-B:** median closed-trade realized P&L > 0 on the training window; up/down-capture (vs ^GSPC, computed from equity.csv per-day returns) stays ≥ 1.0 / ≤ 0.5; deaths unchanged.

### Phase C — multi-seed, longer window
- ≥3 seeds × 20 trading days, stripped for sweep, whole-fly for the final 2–3 configs only. Stripped: 3 × 20 × 2.5 min ≈ 2.5 h; whole-fly: 3 × 20 × 8.3 min ≈ 8.3 h. Persist one artifact per seed (naming: `data/runs/train2/{config}/{seed}.npz` via `train_larval` `out_path`).
- **Gate G-C:** the Phase-B winner's *median-across-seeds* training-window HAC t-gate (G-A.1), turnover guard (G-A.3), and win/loss hold; cross-seed variance reported. Only then proceed to §7 eval.

---

## 7. EVAL PROTOCOL — power analysis and sweep design

### Power

Detecting a daily-return edge μ vs 0 with daily σ (two-sided α=0.05, 80% power) needs

```
n ≥ ( (z_{α/2} + z_{power}) · σ / μ )²  =  (2.80 · σ / μ)²
```

| σ (daily) | μ = 0.1%/day | μ = 0.2%/day | μ = 0.5%/day |
|---|---|---|---|
| 0.5% | 196 days | 49 days | 8 days |
| 1.0% | 784 days | 196 days | 31 days |

**Honest conclusion:** at σ ≈ 0.5–1%, *unpaired absolute-return* tests cannot resolve 0.1%/day at ≤ 40 days. The protocol therefore leans on two cheaper statistics:

1. **Per-encounter signal tests** (G-A.1): raw n = encounters (≈ 3,900 per 10 days), but with HAC lag-30 / ticker-day clustering the effective n is ≈ 400 (§6 G-A.1) → a 0.1% mean forward-return edge with per-encounter σ ≈ 0.3–0.5% still resolves at effective n ≈ 70–200 independent clusters. This is the primary *training-time* gate; the overlap-robust statistics are mandatory, not optional.
2. **Paired A/B comparisons** (same window, same seed, config vs control): the day-level difference series σ_d is typically far below σ because both arms share the tape; paired t over 20 days is the primary *portfolio-level* gate. Report the paired σ_d measured in Phase A before trusting Phase C conclusions.

### Sweep design

- Harness: `scripts/eval_brains.py --start ... --end ... --basket ... --days ...` (CLI at scripts/eval_brains.py:445-509; per-arm receipts + per-day artifacts under `RESULTS_DIR` = `data/runs/brain-shootout`, scripts/eval_brains.py:43). Extend `RECENT_POOL` (scripts/eval_brains.py:50, currently 10) — the held-out day sampler must draw from ≥ 40 days.
- **Cells:** ≥ 20 days/period × ≥ 2 periods (one from 2024 data, one from 2025 — regime diversity) × ≥ 2 baskets (the 22-name current basket + a 20-name never-seen basket, §3.4). Minimum 4 cells per config; whole-fly fresh-fly-per-day replay ≈ 40 min/day → one 20-day cell ≈ **13.3 h**; budget accordingly (run cells overnight, arms in parallel where RAM allows — 16 GB bounds parallelism to ~2–3 whole-fly processes).
- **Primary statistic:** pooled paired t (config vs incumbent artifact) across the 40+ days; secondary: up/down-capture per cell; tertiary: median closed-trade P&L per cell (the §5 acceptance metric).
- **Report:** extend `render_report` (scripts/eval_brains.py:272) with per-cell paired-t columns and capture ratios; write to `reports/train2-eval.md`.

---

## 8. DESIGN v0.7 AMENDMENTS (each flagged for Duke sign-off)

| # | Section | Proposed direction | Decisions touched |
|---|---|---|---|
| A1 | §6.1 Universe | Universe = plume-filtered pool from a **declared basket file** (40–60 train + held-out eval baskets) rather than the hardcoded 22-name list; plume-intensity formula unchanged | **D3** |
| A2 | §3 Smell | Add an **identity attenuation** parameter: identity glomerular profile scaled by `id_scale ∈ [0,1]` (or dropped); state features unchanged; anchor-sniff interaction documented (`id_scale` must also scale the `_innate_balance` sniff input, which bypasses `encode_smell`; silent at `id_scale=0`) | **D5** |
| A3 | §7.3 Learning | Per-exit dopamine generalized: the gate at credit time may carry a **supervised entry-forecast component** (forward return of the encountered ticker, credited only after close/settle; avoid-side encounters credited — correct avoids rewarded at `AVOID_CORRECT_WEIGHT`, missed run-ups punished at `MISS_WEIGHT`; each snapshot credited exactly once, last-credit-wins) and optionally an **advantage baseline** (state-conditional EMA, persisted in the artifact); daily pooled sugar/shock becomes ablatable | **D6** |
| A4 | §4 gate | `reward_gain`/`punishment_gain` become calibrated (asymmetric) rather than fixed 1.0/1.0 | D6 (T9 register) |
| A5 | §6.5 Exit | Mechanical exits extended: trailing/ATR stops and an explicit exit-forecast path alongside valence_flip; taste channel's PAM/PPL1 mapping acknowledged as readout-inert (documented dead channel) pending D15 revisit | **D15**, §6.5 |
| A6 | §7.3 Credit timing | New hard rule: forward-return rewards may only be computed from bars strictly after the decision bar and credited at close-fill or day settle; audited by test | new sub-decision of D6 |
| A7 | Eval | Held-out evaluation: ≥20 days/period, ≥2 periods, ≥2 baskets (one never-seen); paired t primary | process, no D-number |
| A8 | §6 G-A | **Turnover guard (G-A.3):** trades/day and buys/encounter within [0.5×, 2.0×] of the reference config's band; **G-A.1 made overlap-robust** (HAC lag-30, effective n ≈ ticker-days) | process, no D-number |
| A9 | §2/§6 | **Artifact meta contract:** `train_larval` meta records the new credit knobs and Option B baselines; `eval_brains` asserts artifact meta ↔ runtime config | process, no D-number |

---

## 9. RISKS & OPEN QUESTIONS

1. **Lookahead in reward shaping.** The whole v0.6 stop was a no-look-ahead audit. Any forward-return reward is one bug away from leaking. Mitigations: the §2 timing rule as a hard invariant + a dedicated audit test (no `entry_credit` with `credit_ts < decision_ts + 1 bar`; a next-bar close-fill credit at exactly `decision_ts` + 1 bar is the designed earliest site); forward returns recomputed offline from the parquet cache and diffed against event payloads).
2. **Regime confound.** All training windows live inside 2024–2025 fetched data; a bull-heavy curriculum teaches long-beta, and the advantage objective (B) may strip exactly the regime skill that produced up 1.20/0.34. Mitigation: two-period eval, one never-seen basket, report capture ratios per cell — and treat "capture collapsed but realized P&L improved" as an ambiguous result requiring Duke review, not a win.
3. **Credit-assignment dilution at higher encounter rates.** If `top_k` rises with the pool (§3.3), the per-day credit count stays ≈ one per bar but per-ticker credit frequency falls (∝ 1/`top_k`); `sleep_decay=0.5` + `grudge_top_k=16` consolidation may churn associations faster than they consolidate. Mitigation: keep `top_k=5` in Phase A; sweep `grudge_top_k` explicitly.
4. **Runtime budgets.** Whole-fly eval is 40 min/day: the §7 minimum protocol is ~13 h per config-cell; a 4-cell × 4-config comparison ≈ 8–9 days of serial wall-clock. Stripped eval is ~10× cheaper but unrepresentative (D22 made whole the live brain). Mitigation: stripped for elimination, whole-fly only for the top-2 configs; overnight parallel cells.
5. **Baseline persistence (Option B).** A restored artifact with a cold baseline mis-credits on replay; `initial_baselines` must round-trip through the npz meta. Until then B is train-only.
6. **RATIFIED OBJECTIVE (Duke, 2026-09-16):** the goal is NOT the best trading bot — it is a fly that is an **active trader choosing its picks based on information, not randomly**. Consequences: (a) Option A vs B is decided EMPIRICALLY in Phase 0 — both run head-to-head on the stripped lane (~20-30 min/config) and the G-A gates (balance↔forward-return signal, activity inside the turnover band, win/loss > 1) pick the winner; (b) "active" is a first-class acceptance property (G-A.3 band is a floor as well as a ceiling); (c) alpha-maximization is explicitly NOT the success criterion — an information-driven pick rate with positive expectancy is.
7. **Open questions for Duke:** (i) is `MISS_WEIGHT > 0` (penalizing missed winners — approached-but-passed buys **and** avoided run-ups) acceptable as a behavior prior, or should misses be unsupervised? (`AVOID_CORRECT_WEIGHT = 0.5` for correct avoids is pinned-by-spec and swept, not a Duke decision); (ii) confirm the never-seen 20-name basket before Phase A starts (it must stay unseen by *all* training runs, including sweeps); (iii) appetite for whole-fly eval wall-clock (~2 overnight runs per eval round).

---

## 10. DECISIONS-FOR-DUKE CHECKLIST

- [x] Ratify A1–A9 (§8) as DESIGN v0.7 — **RATIFIED by Duke 2026-09-16**; changelog entry added (DESIGN.md §16 v0.7).
- [x] Pick primary objective for Phase A: **decided empirically in Phase 0** (§9.6, ratified) — A and B run head-to-head on the stripped lane; G-A gates pick the winner.
- [x] Set `MISS_WEIGHT` policy — **0.25 ratified** (governs approached-but-passed buys AND avoided run-ups, §2A). Pinned-by-spec: `AVOID_CORRECT_WEIGHT = 0.5`, single-credit last-credit-wins, T-1 encountered-ticker-only scoring default, overlap-robust G-A.1, artifact meta contract, G-A.3 turnover band.
- [x] `daily_observe` ablation allowed — **ratified** (A3: daily pooled ritual becomes ablatable; default "on", ablated in Phase A grid).
- [x] Approve the 40-name training pool + commit the never-seen 20-name eval basket — **provisionally frozen** in `baskets/train40.txt` + `baskets/eval20-neverseen.txt` at Phase-0 close (all names listed pre-2024; eval20 disjoint from every trained basket; Duke holds veto until the Phase-0 commit).
- [x] Approve IEX depth verification spend — **approved** (Phase 0 pre-commit gate: ≥300/390 median bars/day, report `reports/iex-depth.md`).
- [x] Approve exit grid E1–E4 scope (§5.2) and the T-1 exit-forecast path (§5.1) — **ratified via A5**.
- [x] Accept the eval wall-clock budget (~13 h/config-cell whole-fly; top-2 configs only) — **accepted** (Duke keeps the dashboard open during runs; runs proceed overnight-parallel where RAM allows).
- [x] Acceptance gates G-A/G-B/G-C (§6) as the phase-advance criteria — **ratified**; Phase 0 additionally runs the ratified A-vs-B head-to-head through the G-A gates.
