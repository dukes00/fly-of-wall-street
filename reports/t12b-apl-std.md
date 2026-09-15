# T12b — APL/DPM short-term depression: whole-fly KC revival and D19 re-validation

**Window:** 2026-09-11 .. 2026-09-14 (replay) · **Seed:** 7 · **Artifact:** `data/fly-larval-weights.npz` · **STD:** beta=0.1, tau_rec=500.0 ms · **Script:** `scripts/validate_transplant.py` (`--std-beta` / `--std-tau-rec` / `--calibrate-std`)

Follow-up to reports/t12-transplant.md: T12 measured the whole-fly readout silent — tonic APL/DPM feedback inhibition pins all 4064 KCs below threshold (median 50 APL inhibitory synapses per KC vs 97 uPN excitatory) because quantal LIF coupling never fatigues. T12b adds **opt-in short-term synaptic depression (STD)** to the LIF engine's inhibitory terminals and re-runs the D19 validation.

## Mechanism (opt-in, default off)

- Per presynaptic node, a depression factor `f` in [0, 1] scales that node's outgoing weights (`w[i,:] * f[i]`); only inhibitory terminals are tracked (51,744 nodes with outgoing edges in the whole fly). Excitatory and modulatory terminals keep `f = 1` exactly — excitatory synapses are untouched.
- Fixed per-substep op order: (1) deliver the previous substep's spikes scaled by `f`; (2) integrate, refractory clamp, threshold; (3) recover every inhibitory factor toward 1 with the exact exponential form `f = 1 - (1 - f) * exp(-dt / tau_rec)` (the exponential is precomputed once — no per-substep transcendentals; `f == 1` is an exact fixed point, so undepressed terminals never drift); (4) deplete the terminals that spiked this substep: `f *= (1 - beta)`.
- A spike is therefore delivered at its depleted amplitude, and recovery accrues per substep until the next spike. Determinism: fixed op order, float64, no RNG, no wall clock. With STD disabled (the default) the propagation step uses the unscaled spike vector, so existing runs are **bit-identical** to the pre-STD engine — pinned by the full test suite plus a dedicated bit-identity test.

## Calibration — (beta, tau_rec) sweep on replayed real bars

- Calibration bars (XOM, 2026-09-11 session, identical loop drive per bar, seeded noise): `2026-09-11 14:09:00+00:00`, `2026-09-11 14:35:00+00:00`, `2026-09-11 15:53:00+00:00`, `2026-09-11 17:36:00+00:00`, `2026-09-11 19:20:00+00:00`

| Brain | KC spikes/bar | MBON drive | APL spikes | LC/T4/T5 spikes | step wall (s) |
|---|---|---|---|---|---|
| stripped (no STD) | 49.2 | 813.7 | 0 | 0 | 0.262 |
| whole-fly (no STD) | 0.0 | 0.0 | 334 | 0 | 0.925 |
| **whole-fly (chosen STD)** | **49.6** | 9253.2 | 334 | 0 (x1.00 of no-STD) | 1.066 |

Full grid (mean over the calibration bars; vision ratio = LC/T4/T5 spike yield vs the no-STD whole-fly baseline):

| beta | tau_rec (ms) | KC spikes | MBON drive | APL spikes | vision spikes | vision x |
|---|---|---|---|---|---|---|
| 0.1 | 25 | 0.0 | 0.0 | 334 | 0 | 1.000 |
| 0.1 | 50 | 6.0 | 2670.7 | 334 | 0 | 1.000 |
| 0.1 | 100 | 22.8 | 6251.5 | 334 | 0 | 1.000 |
| 0.1 | 200 | 34.6 | 7761.0 | 334 | 0 | 1.000 |
| 0.1 | 500 | 49.6 | 9253.2 | 334 | 0 | 1.000 |
| 0.2 | 25 | 6.0 | 2670.7 | 334 | 0 | 1.000 |
| 0.2 | 50 | 28.8 | 6727.3 | 334 | 0 | 1.000 |
| 0.2 | 100 | 42.2 | 8374.5 | 334 | 0 | 1.000 |
| 0.2 | 200 | 54.0 | 9578.9 | 334 | 0 | 1.000 |
| 0.2 | 500 | 59.0 | 9884.2 | 334 | 0 | 1.000 |
| 0.3 | 25 | 23.6 | 6382.9 | 334 | 0 | 1.000 |
| 0.3 | 50 | 39.2 | 8191.7 | 334 | 0 | 1.000 |
| 0.3 | 100 | 51.8 | 9413.8 | 334 | 0 | 1.000 |
| 0.3 | 200 | 58.2 | 9884.3 | 334 | 0 | 1.000 |
| 0.3 | 500 | 63.0 | 10719.2 | 334 | 0 | 1.000 |
| 0.5 | 25 | 44.4 | 8430.4 | 334 | 0 | 1.000 |
| 0.5 | 50 | 56.4 | 9726.7 | 334 | 0 | 1.000 |
| 0.5 | 100 | 59.0 | 9901.6 | 334 | 0 | 1.000 |
| 0.5 | 200 | 62.2 | 10642.5 | 334 | 0 | 1.000 |
| 0.5 | 500 | 63.0 | 10672.0 | 334 | 0 | 1.000 |
| 0.7 | 25 | 58.0 | 9865.1 | 334 | 0 | 1.000 |
| 0.7 | 50 | 62.2 | 10642.5 | 334 | 0 | 1.000 |
| 0.7 | 100 | 62.4 | 10646.8 | 334 | 0 | 1.000 |
| 0.7 | 200 | 63.0 | 10672.0 | 334 | 0 | 1.000 |
| 0.7 | 500 | 64.4 | 10915.7 | 334 | 0 | 1.000 |
| 0.9 | 25 | 62.6 | 10666.0 | 334 | 0 | 1.000 |
| 0.9 | 50 | 63.0 | 10672.0 | 334 | 0 | 1.000 |
| 0.9 | 100 | 64.4 | 10915.7 | 334 | 0 | 1.000 |
| 0.9 | 200 | 64.8 | 10938.3 | 334 | 0 | 1.000 |
| 0.9 | 500 | 65.6 | 10952.1 | 334 | 0 | 1.000 |

Chosen: **beta = 0.1, tau_rec = 500 ms** — KC yield 49.6/bar vs the stripped brain's 49.2/bar at the same drive, with the LC/T4/T5 pathway at x1.00 of its no-STD activity (floor 0.90).

## Whole-fly KC revival vs gain (DIAG bar)

| SMELL_GAIN x | stripped KC (no STD) | whole-fly KC (STD) | whole-fly KC (no STD) | whole-fly MBON drive (STD) |
|---|---|---|---|---|
| 1 | 52 | 50 | 0 | 9195.0 |
| 2 | 74 | 60 | 0 | 10135.5 |
| 4 | 75 | 60 | 0 | 10135.5 |
| 8 | 76 | 67 | 0 | 11938.2 |

- Measured caveats: the transplanted KC→MBON drive under STD (9253 summed activation/bar) is ~11x the stripped brain's — the loop's decision variable is the valence-normalized balance, so the scale difference does not enter the BUY/SELL/pass comparison directly. LC/T4/T5 spike yield is 0 in BOTH brains under this loop drive (vision is delivered sub-threshold at the loop gain), so the vision check is vacuous here — it measures that STD did not create or destroy visual activity on these bars, not that a vision-driven regime is preserved.

## Decision agreement (per day, D19 metric)

| Day | Bars decided | BUY/SELL/pass agreement | Ticker agreement | Action + ticker |
|---|---|---|---|---|
| 2026-09-11 | 390 | 65.64% | 100.00% | 65.64% |
| 2026-09-14 | 390 | 80.00% | 100.00% | 80.00% |
| **Mean (day-weighted)** | — | **72.82%** | — | — |

- Stripped brain orders: 127 of 780 decided bars; whole-fly (STD) orders: 163 of 780.

## Sim-rate impact

- Whole-fly replay **with STD**: 986 s for 780 bars = 1.26 s/bar (loop dt = 1.0 ms).
- Engine-level step cost on the calibration bars: no-STD 0.925 s/bar vs STD 1.066 s/bar (x1.15) — the per-substep recovery pass over the inhibitory subset is the only overhead.
- T12 reference points: whole-fly LIF at 0.281x real time (1.78 s/bar at dt = 0.5 ms); the loop runs dt = 1.0 ms.

## Determinism receipts (same-seed replay byte-identical)

| Run | events.jsonl sha256 | equity.csv sha256 | equity |
|---|---|---|---|
| stripped_a | `d7688a454309f111…` | `54c55e9a25604db6…` | 100488.36 |
| stripped_b | `d7688a454309f111…` | `54c55e9a25604db6…` | 100488.36 |
| wholefly_a | `fd75b280474f276e…` | `95eb782e52be6123…` | 100874.28 |
| wholefly_b | `fd75b280474f276e…` | `95eb782e52be6123…` | 100874.28 |

- Stripped replay A ≡ B (byte-identical): **yes**
- Whole-fly replay A ≡ B (byte-identical): **yes**

## D19 branch recommendation (for human ratification)

Mean decision agreement **72.82%** < 90% despite revived KC activity (whole-fly KC yield 49.6/bar under STD): the whole-fly brain now acts but disagrees with the stripped engine often enough that a live cutover is not warranted. This report **proposes demo-mode decisions** — the whole-fly brain (with STD) decides in promo/art mode while the live engine stays the stripped chassis (D10 Phase A). DESIGN.md is not edited here; the human ratifies.

