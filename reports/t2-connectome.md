# T2 — Connectome Extraction: Stripped Chassis

**Source:** MaleCNS v1.0 (FlyEM/HHMI Janelia et al., Cell 2026) — complete adult male *Drosophila* CNS, CC-BY 4.0. See `data/connectome/MANIFEST.md` for provenance, checksums and download URLs. Raw tables: body-annotations (211,577 rows), body-neurotransmitters (1,835,518), connectome-weights (151,856,684 edges, 1.05 GB).

**Module:** `src/fruitfly/connectome.py` · **Regeneration CLI:** `uv run python scripts/extract_chassis.py [--whole-fly]` · **Cache:** `data/connectome/stripped-chassis.parquet` + `stripped-chassis-edges.parquet` (gitignored).

## Stripped-chassis neuron count (quote this number)

**27,115 neurons**, 1,362,557 canonical edges, 3,885,700 synapses. Every later brief re-quotes 27,115.

The stripped chassis (DESIGN §2, D10 phase A) is the induced subgraph over the task-relevant populations below. Intermediate optic-lobe neurons (L1/L2/L3, Mi, Tm, …) are deliberately excluded — direction selectivity is computed by the T3/T4 sensory encoders in stripped mode; the whole-fly mode (T11) keeps all 166,700 proofread neurons.

## Population dims (T4 codes against these)

| Population | Selection predicate (exact) | Cells |
|---|---|---:|
| Photoreceptors (R1–R8) | `type` regex `^R[1-8]` (R1–R6, R7p/R7y/R7d/R7_unclear, R8p/R8y/R8d/R8_unclear, R7R8_unclear) | 6,091 |
| T4 (direction-selective, medulla) | `type` regex `^T4` (T4a–d + 4 T4_unclear) | 6,865 |
| T5 (direction-selective, lobula) | `type` regex `^T5` (T5a–d + 1 T5_unclear) | 6,720 |
| Looming-sensitive lobula (LC) | `type` regex `^LC(4\|21)(_.*)?$` or `^LC10` — LC4 (wide-field looming, 126), LC21 (153), LC10a/b/c-1/c-2/d/e/unclear (960). LC40/41/43/44 deliberately excluded (not looming-selective) | 1,239 |
| uPN (olfactory, antennal lobe) | `class == 'ALPN'` and not `type.startswith('M_')` (multiglomerular excluded) | 391 |
| Glomeruli represented | unique glomerulus prefix of uPN types (`_(adPN\|lPN\|vPN\|lvPN)\d*$` stripped); 87 named + 4 null-type uPNs → 88 channels. Note: DESIGN §3 says "~50 glomeruli" — this dataset annotates 88 uPN channels (including lineage-split periglomerular duplicates such as DA1_lPN/DA1_vPN both mapping to DA1). T3 should code against the 88-channel list in the cache meta. | 88 |
| Kenyon cells (KC) | `class == 'Kenyon_Cell'` (KCab-, KCa'b'-, KCg- families) | 4,064 |
| MBONs | `class == 'MBON'` — actual: **97 cells, 37 type names** (MBON01–MBON35 plus MBON15-like/17-like/25-like; most bilateral pairs, a few >2 cells). DESIGN §5's "~35 cells" matches the ~35 MBON *types per hemisphere*; the bilateral total is 97. T5's approach/avoid readout should code against 97 nodes. | 97 |
| PAM cluster (DAN, reinforcement) | `class == 'DAN'` and `type` regex `^PAM` (PAM01–PAM15) | 316 |
| PPL1 cluster (DAN, avoidance) | `class == 'DAN'` and `type` regex `^PPL1` (PPL101–PPL108) | 16 |
| Descending neurons (readout) | `superclass in {'descending_neuron', 'descending_neuron_tbc'}` (DNa/DNb/DNp/DNge/DNpe/DNg…, 480 type names) | 1,316 |
| **Total** | | **27,115** |

Selection basis: 166,700 proofread neurons (non-null `superclass`); predicates evaluated in table order, first match wins. PPL2 DANs (8 cells) are not part of either reinforcement cluster and are excluded.

## Per-region neuron counts

| Region | Neurons | Populations |
|---|---:|---|
| retina | 6,091 | photoreceptors |
| medulla | 6,865 | T4 |
| lobula | 7,959 | T5 + LC-looming |
| antennal-lobe | 391 | uPNs |
| mushroom-body | 4,161 | KC + MBON |
| protocerebrum | 332 | PAM + PPL1 |
| brain | 1,316 | descending neurons |

## Synapse counts

Total: 3,885,700. Key pathway flux (summed weights):

| Pathway | Synapses |
|---|---:|
| uPN → KC | 388,843 |
| KC → MBON | 463,640 |
| T4/T5 → LC-looming | 6,396 |
| photoreceptor → T4/T5 | **0** |

**R→T4/T5 = 0 is expected, not a bug:** photoreceptors reach T4/T5 through lamina (L1/L2/L3) and medulla (Mi/Tm) relays, which the stripped chassis deliberately excludes. T3's visual encoder must inject motion/looming signals directly into T4/T5 and LC nodes (or a future relay layer must be added). Photoreceptor out-synapses (49,213) land almost entirely outside the chassis in this mode.

Per-population in/out flux (synapses): photoreceptor 49,213 out; T4 194,944 out / 191,921 in; T5 185,227 / 182,643; LC 186,673 / 152,835; uPN 435,528 / 43,865; KC 1,905,035 / 1,777,116; PAM 210,070 / 219,477; PPL1 62,453 / 83,347; MBON 52,845 / 531,233; DN 603,712 / 654,048. Edges to neurons outside the chassis are dropped (the KC→MBON plasticity triangle PN→KC→MBON and DAN feedback are fully contained). 3 self-loops retained.

## Transmitter sign coverage

Sign rule: acetylcholine → **+1**; GABA, glutamate, glycine → **−1**; dopamine/serotonin/octopamine/histamine/unresolved → **0** (modulators handled by the neuromod layer, not generic LIF).

NT assignment tiers (first resolved wins): ① `consensus_nt` (the dataset's curated reconciliation), ② body-level `predicted_nt`, ③ cell-type `celltype_predicted_nt`. Tier usage on the chassis: consensus 27,077 (99.86%), predicted 20, celltype 2, none 16.

| Sign | Neurons | % |
|---|---:|---:|
| +1 excitatory | 20,168 | 74.38% |
| −1 inhibitory | 484 | 1.78% |
| 0 unknown/modulatory | 6,463 | 23.84% |

The 23.84% unknown = histaminergic photoreceptors (6,091, sign-inverting synapses handled encoder-side), dopaminergic PAM/PPL1 (332), plus residual unclear/other (mainly DNs: octopamine/serotonin/unclear).

Documented judgment calls:

- **Glutamate treated inhibitory** per the Shiu et al. 2024 whole-brain convention; contested for some cell types (e.g. some glutamatergic MBONs may be excitatory via GluCl1 receptors). 104 DNs and 26 MBONs are glutamatergic — revisit if behavior looks inverted.
- **Kenyon cells resolve to acetylcholine** via the `consensus_nt` tier: the Shiu-style predictor mislabels all 4,058 predicted KC bodies as "dopamine" (a known dense-synapse failure mode), but the dataset's own consensus column corrects this. Using predicted-first would have made every KC dopaminergic (sign 0) and broken LIF signaling through the MB.
- **Histamine → 0** (contract rule: others/modulatory → 0), even though photoreceptor→lamina synapses are functionally sign-inverting. T3's retinal encoder compensates.
- 92 uPNs are GABAergic (legit: multiglomerular inhibitory PNs exist in the AL).

## 166,700 vs 166,691 discrepancy

The bioRxiv preprint reports 166,691 neurons; the final v1.0 release reports 166,700. Verified locally: `body-annotations` has **exactly 166,700 rows with non-null `superclass`** — the release number is correct for this data (the preprint figure predates final proofreading; 9 segments were resolved to neurons). No fallback needed; whole-fly = 166,700.

## Loader contract & determinism

```python
from fruitfly.connectome import load_stripped_chassis, load_whole_fly
chassis = load_stripped_chassis()   # Chassis(nodes, adj, meta)
chassis.nodes  # bodyId, type, instance, somaSide, population, region, neurotransmitter, sign — sorted by bodyId
chassis.adj    # scipy.sparse.csr_matrix, pre→post, weight = synapse count, shape (27115, 27115)
chassis.meta   # population dims, glomeruli list, region/sign counts, predicates, tier counts
fly = load_whole_fly()  # same shape, all 166,700 proofread neurons (reads its own cache pair)
```

- Cached-load time: **~0.02 s** (acceptance: <60 s). Cold extraction scans the 1 GB edge table in 4M-row chunks via `pyarrow.dataset` (searchsorted index mapping): ~6 s end-to-end.
- Determinism: sorted bodyIds, edges aggregated (duplicate pre/post pairs summed), sorted canonically by (pre, post), meta JSON with `sort_keys`. Re-extraction reproduces byte-identical caches:
  - `stripped-chassis.parquet` sha256 `de295fda1234bb286be190e20bc479908503114cf60a4353d8ab629d141f96dc`
  - `stripped-chassis-edges.parquet` sha256 `d6d4eba4c2e0a21c70d8cf28ea42a860e144eea6137c947ed81049be1e43788e`
- Two loader runs produce identical sha256 over (nodes frame, CSR `indptr`/`indices`/`data`) — verified on the real artifact and in `tests/test_connectome.py` (offline, synthetic fixture parquet with the raw table schemas).
- Whole-fly caching: `load_whole_fly` reads raw tables only on first call, then persists its own `whole-fly.parquet` / `whole-fly-edges.parquet` pair (166,700 nodes; ~150M edges, int32 index space) — same deterministic construction. Building it scans the full edge table; budget ~2 min and several GB RAM. Not built as part of T2 (T11 triggers it lazily).
