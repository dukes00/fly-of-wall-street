# T12 — Weight Transplant + Behavioral Validation (D19)

**Window:** 2026-09-11 .. 2026-09-14 (replay) · **Seed:** 7 · **Artifact:** `data/fly-larval-weights.npz` · **Module:** `fruitfly.transplant` · **Script:** `scripts/validate_transplant.py`

## The transplant (D19, direct copy)

| Metric | Measured |
|---|---|
| Stripped weight view (KC × MBON) | 4064 × 97 |
| Whole-fly weight view (KC × MBON) | 4064 × 97 |
| KC bodyId match rate | 100.00% (4064 KCs) |
| MBON bodyId match rate | 100.00% (97 MBONs) |
| Matched (KC, MBON) pairs | 394208 |
| Trained values copied verbatim | 61210 (mass 63423.1) |
| Baseline-filled (whole support, no artifact synapse) | 0 |
| Dropped (trained > 0, whole fly lacks the synapse) | 0 |
| Synapse counts identical over matched pairs | yes |
| Artifact fingerprint verified vs stripped chassis | yes |

Both chassis are extracted from the same MaleCNS v1.0 release: the stripped chassis is a bodyId subset of the whole fly, and the KC→MBON structural submatrix (synapse counts) is identical over the matched pairs, so every trained nonzero weight lands on a real whole-fly synapse. The transplant itself is verifiably correct. Whole-fly nodes absent from the stripped chassis would keep their baseline weight from the whole-fly synapse count (log1p-compressed, mean-1-scaled) — the mechanism exists and is unit-tested, and the measured match rates show it was not needed: the stripped chassis already contains every annotated Kenyon cell and MBON in the release.

## Replay setup

- Identical loop config for both brains: same seed, same window, same loop constants (``ms_per_bar=500``, ``dt_ms=1.0``, ``top_k=5``, ``position_cap=10``), ``initial_weights`` = trained artifact (stripped) / transplanted weights (whole fly).
- Whole-fly brain: population labels transferred onto matched bodyIds (``label_whole_chassis``); the loop's chassis seam points at the labeled whole-fly chassis (166,700-neuron LIF, ~1-4 s per encounter wall on the M1 target; T11: 0.281× real time at dt=0.5 ms). The full 2-day / 390-bar-per-day replay at full resolution stays within budget — **no bar subsampling was needed**.
- Determinism: each brain replayed twice at the same seed; the run receipts must be byte-identical.
- In-sample caveat, documented: the market cache spans 2026-08-17..2026-09-14 — exactly the larval training window — so strictly out-of-sample bars do not exist in the cache. The final two trading sessions are the closest available held-out replay.
- RNG note, documented: the loop's per-encounter noise draw consumes a chassis-sized number of variates, so after the first day the two brains' rotation offsets can diverge (a different plume sampled at the same bar). Agreement is therefore keyed by bar timestamp; the ticker-agreement column quantifies plume-choice divergence.

## Decision agreement (per day)

| Day | Bars decided | BUY/SELL/pass agreement | Ticker agreement | Action + ticker |
|---|---|---|---|---|
| 2026-09-11 | 390 | 79.49% | 100.00% | 79.49% |
| 2026-09-14 | 390 | 87.95% | 100.00% | 87.95% |
| **Mean (day-weighted)** | — | **83.72%** | — | — |

## Why the whole-fly readout is silent (measured root cause)

778 of 780 whole-fly encounters in the replay are **silent** (``valence_readout = 0``); the 2 remaining encounters carry negligible signal and still read ``pass``. Net effect: zero KC spikes, so the transplanted KC→MBON drive — however correct — multiplies zero, and the whole-fly brain emits **no orders in 780 bars** (the stripped brain places 127) — every decision is ``pass`` by loop construction (``centered = 0`` on a silent encounter). The cause is in the circuit, not the weights. The stripped chassis (T2) deliberately excludes everything outside its task populations — among them the mushroom body's feedback inhibitors **APL** and **DPM**. The whole fly contains them, and with the loop's direct-uPN sensory injection they clamp the KC population below threshold:

- APL in stripped chassis: **absent** · APL neurons in whole fly: 2 (GABAergic, sign −1).
- Synaptic budget per median KC (whole-fly adjacency, matched bodyIds): 50 APL inhibitory synapses vs 97 uPN excitatory synapses. Both sides fire tonically during an encounter (uPNs and APL spike every ~3 substeps at the loop gain), so APL's ~−0.5 mV per spike holds the KC membrane below the 15 mV threshold regardless of the sensory gain — the uPN drive saturates (uPNs are already near their firing ceiling) while APL keeps pace.
- Gain sweep on one replayed bar (XOM @ 2026-09-11 14:35:00+00:00, identical drive into both brains, seeded noise):

| SMELL_GAIN × | stripped KC spikes | stripped MBON drive | whole-fly KC spikes | whole-fly MBON drive |
|---|---|---|---|---|
| 1 | 50 | 828.3 | 0 | 0.0 |
| 2 | 74 | 1234.0 | 0 | 0.0 |
| 4 | 75 | 1252.0 | 0 | 0.0 |
| 8 | 76 | 1270.0 | 0 | 0.0 |

The transplanted weights cannot matter while their presynaptic population is clamped: plasticity eligibility is KC × MBON co-activity (measured 0 in every whole-fly encounter), so a fine-tuning round through the whole-fly LIF is provably a no-op.

## D19 branch

Mean decision agreement **83.72%** < 90% → the direct-copy branch **fails the behavioral gate**. A fine-tune round is ruled out on measured grounds (zero KC activity ⇒ zero plasticity eligibility ⇒ weights cannot move; see the root-cause section above). **Branch: D19's Phase-B transplant into the whole-fly LIF sim is REVOKED.** The stripped chassis remains the working engine (D10 Phase A), consistent with T11's recorded budget deviation (whole-fly LIF at 0.281× real time is an offline/art mode). The adult-stage (Phase B) carrier for transplanted plasticity, if revisited, is the lean mushroom-body rate model over the real KC→MBON/PAM/PPL1 subgraph flagged in reports/t11-wholefly.md — with the KC-silence finding as its first design constraint. The transplanted weight matrix itself is verified correct (100% bodyId match, verbatim copy, synapse counts identical) and remains available in `fruitfly.transplant.transplant_weights` for that carrier.

## Determinism receipts (same-seed replay byte-identical)

| Run | events.jsonl sha256 | equity.csv sha256 | equity |
|---|---|---|---|
| stripped_a | `d7688a454309f111…` | `54c55e9a25604db6…` | 100488.36 |
| stripped_b | `d7688a454309f111…` | `54c55e9a25604db6…` | 100488.36 |
| wholefly_a | `047fd50ba29a26c7…` | `d4630b932bab3f50…` | 100000.00 |
| wholefly_b | `047fd50ba29a26c7…` | `d4630b932bab3f50…` | 100000.00 |

- Stripped replay A ≡ B (byte-identical): **yes**
- Whole-fly replay A ≡ B (byte-identical): **yes**

