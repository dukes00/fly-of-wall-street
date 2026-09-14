# T11 — Whole-Fly Sim

**Source:** MaleCNS v1.0 (FlyEM/HHMI Janelia et al., Cell 2026), same raw tables as the stripped chassis (T2). Module: `fruitfly.connectome.load_whole_fly` (existing; unchanged by T11). Benchmark: `scripts/bench_wholefly.py`. Cache pair: `data/connectome/whole-fly.parquet` (16 MB) + `whole-fly-edges.parquet` (781 MB), gitignored.

## Neuron count (quote this number)

**166,700 neurons** — the final MaleCNS v1.0 release figure (verified locally in T2: exactly 166,700 rows with non-null `superclass`). DESIGN's **166,691** was the bioRxiv preprint figure; it predates final proofreading (9 segments were resolved to neurons). Quote 166,700; keep 166,691 attached to the DESIGN v0.4 tagline as the preprint number.

## Cache build (streamed, measured)

`uv run python scripts/extract_chassis.py --whole-fly` — the raw 1 GB connectome-weights feather is scanned via `pyarrow.dataset` in 4,194,304-row batches (no full-table materialization), intra-whole-fly edges kept by searchsorted index mapping, duplicates aggregated by summed weight, sorted canonically by (pre, post).

| Metric | Measured |
|---|---|
| Build time (cold, full scan) | 45.2 s (rebuild: 43.9 s) |
| Build peak RSS | **2.83 GiB** (`/usr/bin/time -l`, 16 GB machine) |
| Cached-load time | 0.30 s |
| Cache sizes | 16 MB nodes + 781 MB edges |
| Neurons | 166,700 |
| Canonical edges (unique pre→post pairs) | 25,582,938 |
| Synapses (summed weights) | **124,177,617** — matches DESIGN §2's "~125M synapses" |
| Index width | int32 — 166,700 < 2³¹, no overflow possible (max pair weight 2,591 ≪ 2³¹ as well) |
| Weight dtype | int64 in cache/CSR → **float64** effective weights in LIFSim (`sign[i] * 0.01 * synapses`). float64 fit comfortably (peak RSS 1.77 GiB in-sim); the float32 fallback was **not** needed, so no op-order pinning was required |
Of the 151,856,684 raw edge rows, 126,273,746 connect bodies outside the 166,700 proofread neurons (fragments/unproofread segments) and are dropped; the kept 25,582,938 rows are already unique (pre, post) pairs (the duplicate-aggregation step is a no-op here), averaging 4.85 synapses/pair → 124,177,617 synapses.


## Determinism check

Two **independent** `LIFSim` instances (seeds 1 and 2 — the engine consults no RNG, seeds are inert) stepped the same whole-fly chassis with the same input current for 100 ms (200 substeps at dt = 0.5 ms), logging the full per-substep per-node spike-count vector every substep. The two logs are **byte-identical**:

- run 1 sha256(log) = run 2 sha256(log) = `83394fb0d7e2f61efc403db559f880082e6867a803ad332e4ed946c4e270b403`
- 12,340 spikes each run; reproducible via `uv run python scripts/bench_wholefly.py --determinism`

Same result guarantee extends to the loop layer: the engine is RNG-free, so any seed-to-seed divergence must (and can only) come from the loop's seeded sensory-noise floor.

## Performance + memory (whole-fly LIF, dt = 0.5 ms)

Benchmark: 10 bars × 500 ms simulated time, deterministic tonic drive (18 mV + sinusoidal ±6 mV, `sin(2πk/39)` bar modulation) on 20% of the sensory populations the encoders drive — T4/T5/LC-looming/uPN selected by cell type (whole-fly nodes carry `population="whole"`, so the stripped-chassis population labels don't apply; uPNs resolved via the glomerulus-lineage suffix, M_ multiglomerulars excluded). 3,008 driven cells.

| Metric | Measured |
|---|---|
| Cache load | 0.30 s |
| LIFSim init (signed float64 CSR build) | 0.28 s |
| Wall per bar (500 ms sim) | 1.31 s (bar 1) → 2.21 s (bar 10), mean 1.78 s |
| Achievable sim rate | **0.281× real time** |
| Total spikes | 1,009,002 (~56.7k spikes/s wall; activity ramps 70.8k→120.9k spikes/bar as the driven population recruits postsynaptic targets) |
| Peak RSS in-sim | **1.77 GiB** |
| Process survived | yes — no OOM, 4.5 GiB headroom under the 16 GB RAM |

## Budget decision — recorded deviation

Budget (this machine, 16 GB): usable whole-fly LIF requires **≥ 0.5× real time** and **≤ 8 GiB RSS**.

| Criterion | Result |
|---|---|
| ≥ 0.5× real time | **MISS** — 0.281× |
| ≤ 8 GiB RSS | OK — 1.77 GiB |

**The rate budget is missed: whole-fly LIF runs ~0.28× real time, i.e. a 390-bar trading day of sim time costs ~23 h of wall time.** Memory is fine; throughput is the binding constraint — 124M synapses × 1000 substeps/bar of CSR propagation cannot keep pace with 1-minute bars on the M1 target.

Per DESIGN §13 ("lean mushroom-body rate model with dopamine-gated plasticity" as the alternative to whole-brain LIF), the fallback is therefore **recorded as a deviation, not silently applied**:

- **Larval/training stage (Phase A, D10): unaffected** — the stripped chassis (27,115 neurons, 3.9M synapses) runs ~4.9× real time and remains the working engine.
- **Whole-fly art/promo mode (§2): usable offline only** — 0.28× real time is fine for rendered demos (5 s of fly time in 17.8 s wall) but not for live intraday operation.
- **Phase B transplant (D19):** whole-fly adult-stage operation should run the lean mushroom-body rate model over the real KC→MBON/PAM/PPL1 subgraph (4,161 MB neurons + DANs, fully contained in the whole-fly cache) with transplanted plasticity weights, reserving full LIF for offline validation replays. This matches the phased intent of D10/D19 and does not change any decision above — flagged here for the loop/integration owner.

## Reproduction

```bash
uv run python scripts/extract_chassis.py --whole-fly          # build cache (~45 s, 2.8 GiB peak)
uv run python scripts/bench_wholefly.py --determinism         # determinism proof (~3 s)
uv run python scripts/bench_wholefly.py                       # perf/memory bench (~19 s)
```
