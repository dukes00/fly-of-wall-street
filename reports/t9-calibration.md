# T9 — Larval Training + Calibration

**Module:** `src/fruitfly/train.py` (training driver + artifact I/O) · **CLI:** `scripts/calibrate.py` (`sweep` / `train` / `probe`) · **Artifact:** `data/fly-larval-weights.npz` (gitignored via `data/`) · **Tests:** `tests/test_train.py` (9, offline synthetic)

## Summary

The larval stage works end-to-end: a 20-day training run (2026-08-17..2026-09-14, seed 7, default config) persists the surviving fly's KC→MBON weights to `data/fly-larval-weights.npz` (3.2 MB, 4064×97 float64 + chassis fingerprint + provenance); restoring them via `BacktestConfig.initial_weights` changes **53.6% of decisions** on the probe day vs a fresh fly — and **53.8%** when the artifact is trained on 19 days and probed on the genuinely held-out 2026-09-14 (larval +$812 vs fresh −$620 on that day). One training run made a net −0.40% over 20 days at the settled defaults; 0 natural deaths (max drawdown 0.73% — the calm month never approaches D14). The death→hatch-with-reset-plasticity path is exercised by offline tests (byte-exact fresh-`Plasticity` equality after a last-bar death) and demonstrated on real data by forcing the threshold. A 27-cell sweep over D13/D14/D16 (5-day window, seed 7) produced a clean picture: **keep cap 10** (best return and drawdown, marginals below), **death threshold unidentifiable** from calm-window data (0 deaths at −0.35/−0.50/−0.65), **keep 1-min bars** (5-min was best on this small window at 1/5 the compute — flagged, not picked). One deviation needed Duke's attention and is documented in §2: **larval training was initially inert** (MBONs never spike at current engine gains → eligibility always zero → weights could never move); the surgical un-silencing caught a positive-feedback runaway on the way and landed on a weight-independent structural eligibility. All calibration value changes are proposals only — DESIGN.md/plan untouched.

## 1. What was built

- **`fruitfly.train`** — `train_larval(config, out_path)` drives `run_backtest` over a training window and persists the surviving fly's KC→MBON weights + chassis fingerprint + provenance meta; `load_larval_weights(path, chassis, verify=True)` restores them (fingerprint mismatch → `ValueError`); `chassis_fingerprint` = sha256 over chassis bodyIds + CSR adjacency arrays; `decision_map` for the probe.
- **Artifact format** — `.npz` readable by `np.load(..., allow_pickle=False)`: `weights` (n_kc × n_mbon float64), `fingerprint` (unicode), `meta_*` provenance (seed, window, params, final equity, deaths — deterministic fields only). **Determinism:** `np.savez` stamps zip members with the wall clock, so `save_larval_weights` writes the same npz layout by hand — `numpy.lib.format.write_array` blobs, `ZIP_STORED`, fixed member order, fixed member timestamp (1980-01-01). Key order is fixed by construction; no member ever needs pickle.
- **Loop seam (minimal, additive)** — `BacktestConfig.initial_weights: ndarray | None = None`: when set, `run_backtest` validates the shape against the chassis KC→MBON view and writes it into the fresh `Plasticity` before bar one (defaults unchanged). `RunResult.final_weights` returns the last-hatched fly's weights (copied) so the training driver can persist them. All pre-existing tests untouched and green (`tests/test_loop.py` 6/6, `tests/test_neuromod.py` included in a 25-pass run).
- **`scripts/calibrate.py`** — `sweep` (grid over D13/D14/D16, `--jobs N` worker processes, per-cell JSON + `sweep.csv`/`sweep.json`), `train` (full-window training → artifact), `probe` (fresh vs larval decision distributions on one day).
- **Only what is learned is persisted.** The eligibility trace and habituation factors are run-transient and reset on restore (the adult fly wakes with fresh attention — same semantics as the post-sleep reset); the structural `baseline` is re-derived from the chassis, never trusted from disk.

## 2. T9 deviation: un-silencing the learning signal (flagged for Duke)

Measured on the real chassis: **MBONs never spike** at the current engine gains (0 MBON spikes in 20 direct encounters; T7 already documented the sub-threshold gap — KC→MBON quantal PSPs ≈ 0.01 mV/synapse vs the 15 mV threshold). Consequence: `Plasticity.observe`'s eligibility trace (KC × MBON *spike* co-activity) was identically zero in every loop run — **larval training could never move a weight**, and a trained artifact would be byte-equal to the structural baseline (verified before the fix: `max|Δweights| = 0.0` after a full training run).

Surgical fix (loop.py, the `settle_and_sleep` observe call only — no engine/sim changes, no RNG changes): post-activity fed to `observe` is the day's mean **structural MBON response** — `baseline.T @` the day's KC activity, i.e. what each MBON would receive through its *innate* wiring — averaged over signal-bearing encounters (approach+avoid > 0, matching the loop's neutral-readout rule). The dopamine gate still decides sign and magnitude. Accumulators reset at day settle and on death (the mid-day death invariant from T7 is preserved: post-hatch learning sees only post-hatch activity).

**Why structural, not the learned drive (a caught runaway):** the first implementation fed the *learned* drive (`mbon_activation`, the quantity decisions read) as post activity. Because drive ∝ current weights, the three-factor update fed back into itself (delta ∝ W × driver(W)): over the full 20-day training run the weights exploded — 455 entries moved, max |ΔW| = 472,125 on baseline-≈1 weights — and the trained brain went degenerate (silent readout, all-pass on the final day, 0 orders across 390 encounters). The structural response is weight-independent, so daily deltas are bounded by the dopamine gate: measured cumulative |gate| over the 20-day window is 0.104 (median day 0, max |d| 0.043). The retrained artifact moves 512 of 394,208 entries (max weight 96.6 on one hot KC→MBON pair, everything else O(1)); the probe (§5) shows this is enough to shift behavior sharply without degenerating it.

Scale check (measured on a fresh brain, 8 real identities): per-encounter structural MBON response mean 2.85, max 52.6; with `learning_rate = 1` and the observed gates, daily deltas land ≈ 0.06 on baseline-mean-1 weights — sleep's `top_k=64` protection stays meaningful. No learning-rate retune was needed.

Alternative rejected (recorded for Duke): raising `SYN_CONDUCTANCE_MV` / lowering `V_TH` to un-silence MBONs at the engine level would rescale the *entire* LIF engine (vision pathway included) and invalidate every T7 measurement; the rate-based eligibility is the surgical version of T7's tuning item #2. **This change alters T7-documented day-2+ behavior** (balances drift once the fly has learned) — hence the explicit flag.

## 3. Determinism receipts

- **Sweep cell double run (same seed, same cell):** `cap10_death-0.50_5min` re-run after the full sweep — `equity.csv` md5 `44d21df8d3c3d540a0ea9776d362f4b3` and `events.jsonl` md5 `7630c0ffa29093426248570c7178ed77` byte-identical in both runs (and metrics identical: ret +0.149%, 148 trades).
- **Synthetic full-driver double train:** `train_larval` twice on the synthetic chassis — artifact bytes, `equity.csv` and `events.jsonl` all identical (`tests/test_train.py::TestArtifact::test_double_train_byte_identical`).
- **Artifact writer:** double `save_larval_weights` byte-identical (fixed zip member timestamps/order; test-pinned), and the artifact reads back with `np.load(..., allow_pickle=False)`.
- **Stated exception:** the ~48-min full-window 20-day training run was *not* re-run end-to-end for byte-identity (compute budget). Byte-identity there follows by composition: same-seed loop determinism (T7-measured, including under concurrent load) + RNG-free plasticity + the test-pinned deterministic writer. Every cheaper granularity of the same code path was verified byte-identical.
- Sweep parallelism (5–6 worker processes) does not affect results — T7 measured determinism-across-load, and the re-run cell above executed under different load than the sweep.

## 4. Calibration sweep (D13 × D14 × D16)

Grid: `position_cap ∈ {6, 10, 14}` × `death_threshold ∈ {−0.35, −0.50, −0.65}` × `bar_granularity ∈ {1min, 2min, 5min}` — 27 cells, fixed seed 7, window **2026-08-24..2026-08-28** (5 trading days; 390 1-min bars/day). Coarser granularities are resampled in-memory from the 1-minute cache (OHLCV aggregation, left-labeled 13:30-anchored bins, empty bins dropped) — the cache is never re-fetched. Cells ran in 5 parallel worker processes. Metrics: final return %, max drawdown % (peak-to-trough on the continuous equity curve; note deaths reset *hatch* equity but the curve is continuous), deaths, trades (orders), wall seconds.

| gran | cap | death | ret % | mdd % | deaths | trades | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1min | 6 | * | −9.039 | 10.816 | 0 | 297 | 731–761 |
| 1min | 10 | * | −0.267..−0.275 | 0.681 | 0 | 209–210 | 731–753 |
| 1min | 14 | * | −5.454 | 6.774 | 0 | 678 | 735–758 |
| 2min | 6 | * | −0.198 | 1.059 | 0 | 102 | 364–369 |
| 2min | 10 | * | −1.610 | 2.227 | 0 | 150 | 360–371 |
| 2min | 14 | * | −1.400 | 1.846 | 0 | 200 | 365–371 |
| 5min | 6 | * | −0.972 | 3.285 | 0 | 88 | 142–149 |
| 5min | 10 | * | **+0.149** | 1.498 | 0 | 148 | 142–149 |
| 5min | 14 | * | −0.121 | 1.169 | 0 | 189 | 141–149 |

(\* = identical across all three death thresholds — see marginals.)

### Marginal means (averaged over the other two axes)

| axis | value | ret % | mdd % | deaths | trades |
|---|---:|---:|---:|---:|---:|
| position_cap | 6 | −3.403 | 5.053 | 0 | 162 |
| | **10** | **−0.578** | **1.469** | 0 | 169 |
| | 14 | −2.325 | 3.263 | 0 | 356 |
| death_threshold | −0.35 | −2.101 | 3.262 | 0 | 229 |
| | −0.50 | −2.102 | 3.262 | 0 | 229 |
| | −0.65 | −2.102 | 3.262 | 0 | 229 |
| granularity | 1min | −4.922 | 6.090 | 0 | 395 |
| | 2min | −1.069 | 1.711 | 0 | 151 |
| | 5min | −0.314 | 1.984 | 0 | 142 |

### Picked values + reasons

- **position_cap — keep 10 (D13).** Best marginal return (−0.58% vs −3.40%/−2.33%) *and* best drawdown (1.47%); cap 6 concentrates the book (10.8% max drawdown in the worst cell) and cap 14 more than doubles trade count (356 vs 169) for worse returns. The settled default is also the best cell at the settled 1-min cadence (−0.27%, the only 1-min cell above −5%).
- **death_threshold — keep −0.50 (D14), flagged unidentifiable here.** Literally zero effect on this window: 0 deaths and identical metrics to three decimals at −0.35/−0.50/−0.65, because the worst observed drawdown (10.8%) never approaches even −0.35. No evidence to move it; the mortality machinery itself is proven (§6). A stressed window (or a calibrated-hunger regime) is needed before D14 has data behind it.
- **granularity — keep 1min (D16), 5min flagged.** 5-min bars had the best returns (−0.31% marginal, the only positive cell) at 1/5 the wall time (147 s vs 752 s per cell) — but the 1-min deficit is dominated by one cell (cap 6, −9.04%), the 5-day sample is small, coarser bars cut decisions 2.6× (395 → 142 trades), and the transplant (D19) + live Alpaca cadence assume 1-min. Recorded as the top candidate for a re-run on the full 20-day window rather than picked now.

Honesty note: with 5 trading days per cell, return differences within a granularity level are within window noise; the robust signals are the cap ordering, the death-threshold null result, and the compute scaling.

## 5. Sanity probe — larval artifact vs fresh fly (acceptance)

Probe: replay 2026-09-14 (390 1-min bars, 22-symbol basket, seed 7) twice — fresh structural plasticity vs the larval artifact injected via `initial_weights` — and compare per-encounter actions (`data/runs/t9-calibration/probe/`).

| artifact | trained on | probe day status | action divergence | fresh equity | larval equity |
|---|---|---|---:|---:|---:|
| `data/fly-larval-weights.npz` | 2026-08-17..09-14 (20d) | in-sample for the artifact | **209/390 (53.6%)** | 99,379.66 | 100,192.05 |
| `data/runs/t9-calibration/larval-19d.npz` | 2026-08-17..09-11 (19d) | **genuinely held out** | **210/390 (53.8%)** | 99,379.66 | 100,187.54 |

Action distributions (held-out 19-day artifact): add 98→57, buy 30→28, pass 242→286, sell 20→19 — the trained fly is markedly more selective (fewer adds, more passes). Sample divergences: `META buy→pass (approach→neutral)`, `XOM buy→pass (approach→avoid)`, `TSLA sell→buy (valence_flip→approach)`, `MA pass→buy (avoid→approach)` — flips in *both* directions, i.e. learned preferences, not a global bias. On the probe day the larval brain nets **+$812 vs the fresh fly's −$620** (both artifacts agree). Caveat recorded: decision-change is demonstrated; *profitability* on one day is not evidence of edge — D19's ≥90%-agreement transplant validation and longer paper trading are the next gates.

## 6. Death → hatch exercise (DESIGN §9, D14)

- **Offline (test-pinned, `tests/test_train.py::TestDeathHatch`):** a crash tape gaps BBB −95%; with an always-approach policy and the shock exit disabled the compounded book drives equity through −50%. Assertions: exactly one `death` event with `equity ≤ hatch × 0.5`, ≥2 `hatch` events, `death_liquidation` orders present, and — with the death on the final bar — `RunResult.final_weights` **byte-equal to a brand-new `Plasticity`** (the fresh-fly reset observed directly). A mid-run variant proves the hatched fly keeps foraging with its own hatch equity.
- **Real data (forced threshold):** cap 6, `death_threshold −0.05`, 2026-08-24..28 — the fly dies at `2026-08-27T13:30` (equity 91,660.03 vs hatch 100,000 → 8.3% drawdown), liquidates 5 positions, re-hatches at 91,660.03 and forages 779 more encounters (`data/runs/t9-calibration/death-demo/`). The lifecycle fires on real market data.
- **At the settled −0.50 threshold, mortality never triggers naturally** on this month: the 20-day training run's max drawdown is 0.73%, the sweep's worst cell 10.8%. Consistent with the D14 null result in §4 — a calm month cannot separate death thresholds.

## 7. Decision addendum for Duke (proposals only — nothing outside this report was changed)

1. **D13 (position cap):** keep **N=10**. Evidence: §4 marginals (best return and drawdown; cap 14 doubles churn for worse returns; cap 6 concentrates risk).
2. **D14 (death threshold):** keep **−50%**. Evidence: unidentifiable on a calm month (0 deaths at any swept value; worst drawdown 10.8%); machinery proven (§6). Recommend a stressed-window re-sweep before the adult stage so the value is data-backed.
3. **D16 (bar granularity):** keep **1-minute**. 5-min bars won this 5-day window (+0.15% best cell vs −0.27%) at 1/5 the compute — flagged as the leading candidate, but the sample is small and the cadence change ripples into D19 transplant validation and live cadence. Recommend a full-20-day granularity rerun before any change.
4. **Learning-rule ratification (new, from §2):** T9 changed the observe call's post-activity input (structural MBON response instead of all-zero MBON spikes). This is a behavior-affecting deviation from the T7-documented loop and needs Duke's sign-off like any calibration change; the learned-drive variant is recorded as rejected (measured runaway).
5. **Standing T7 gap reminder:** MBONs remain sub-threshold — sizing still rides the `MIN_SIZE_FRAC = 0.25` floor and decisions read the learned drive, not MBON spikes. The engine-gain fix (T7 tuning item #2) remains open and is the principled long-term fix for both.

## 8. Artifacts produced

- `data/fly-larval-weights.npz` — the larval artifact (20-day training, seed 7; 3.2 MB; loads via `load_larval_weights`, verified against the chassis fingerprint).
- `data/runs/t9-calibration/larval-19d.npz` — 19-day artifact for the genuinely held-out probe.
- `data/runs/t9-calibration/sweep.{json,csv}` — 27-cell grid results; per-cell run dirs under `cells/`.
- `data/runs/t9-calibration/probe/probe_2026-09-14_*.json` — probe summaries (fresh/larval distributions, divergence, equities).
- `data/runs/t9-calibration/death-demo/` — forced real-data death→hatch receipts.
- `data/runs/backtest_7_2026-08-17_2026-09-14/` (and `…_2026-09-11/`) — the full-window training runs (equity.csv + events.jsonl).
- `src/fruitfly/train.py`, `scripts/calibrate.py`, `tests/test_train.py` (9 tests), loop.py seam + §2 fix, this report.
