# Phase A — Option B calibration grid (stripped, 10 days, seed 7, train40)

**Date:** 2026-09-16 · Window 2025-06-02..2025-06-13 · chassis stripped · `baskets/train40.txt`
**Base config (Phase-0 winner, spec §2B):** `--entry-credit advantage --trade-credit-mode mix --daily-observe off --reward-gain 1.5`; reference for all gates = the incumbent-config run (`data/runs/h2h-reference`).

## Grid + gate scoreboard (scripts/gates.py; G-A.1 = HAC lag-30 Spearman t + non-overlapping subsample)

| Cell | Knob delta vs base | G-A.1 t (sub) | W/L | Turnover ×(tr/day, buys/enc) | G-A.4 chg | Paired vs ref (%/day, t) | Final $ |
|---|---|---|---|---|---|---|---|
| b-baseline | — (= h2h-optionB) | +1.02 (+1.25) fail | 1.80 ✓ | 1.77× / 1.14× ✓ | 27.6% | +0.87 (+2.15) | 101,615 |
| b-alpha-10 | `baseline_alpha 0.05→0.10` | +1.30 (+1.80) fail | 2.07 ✓ | 1.59× / 1.06× ✓ | 25.9% | +0.78 (+1.96) | 100,714 |
| b-mode-realized | `trade_credit_mode mix→realized` | +0.98 (+1.28) fail | 1.67 ✓ | 1.20× / 1.40× ✓ | 32.6% | +0.71 (+1.61) | 99,849 |
| b-id0 | `id_scale 1.0→0.0` | +0.14 (+0.06) fail | **0.11 FAIL** | **0.11× / 0.13× FAIL** | 17.3% | +0.71 (+1.72) | 99,577 |
| b-punish050 | `punishment_gain 1.0→0.5` | +1.90 (+1.89) fail | 1.79 ✓ | 1.25× / 0.76× ✓ | 23.6% | **+1.16 (+2.96)** | 104,247 |
| **b-tight-miss** | `miss_weight 0.25→0.5, avoid_correct 0.5→0.75` | **+2.20 (+2.55) PASS** | 1.55 ✓ | 1.61× / 0.97× ✓ | 25.5% | +0.96 (+2.29) | 102,395 |
| pa-combo | tight-miss + punish050 | +1.06 (+1.33) fail | 1.44 ✓ | 1.15× / 0.70× ✓ | 22.3% | +1.10 (+2.66) | 103,701 |

All cells: 0 deaths, G-A.4 pass (fresh-vs-trained decision change 17–33%).

## Findings

1. **Winner by the ratified referee: `b-tight-miss`** — the only cell clearing G-A.1 (full t=+2.20, subsample +2.55, sign-stable) with every other gate green. Teaching the gate *harder about avoided winners* (higher miss-weight + correct-avoid weight) is the lever that creates the entry signal the redesign exists for.
2. **Punishment softening pays the portfolio but not the signal**: `punish050` has the best paired outcome (+1.16%/day vs ref, t=+2.96) and nearly clears G-A.1 (+1.90) — a candidate co-knob for later phases, but it fails the primary gate as a standalone.
3. **The two levers don't stack** (combo t=+1.06 vs 2.20/1.90 alone) — the advantage-gate credit budget saturates; avoid-side calibration is the scarcer resource.
4. **`id_scale=0` is architecturally dead**: smell encoding is multiplicative on the identity profile, so zeroing identity collapses the *state* channels too — the fly goes catatonic (win/loss 0.11, trades 0.11× reference, far below the activity band). Identity attenuation below ~0.5 is unusable without restructuring `encode_smell` (state-only odor is not representable in the current umwelt). Recorded for the T9 register; Phase C id sweep should use {1.0, 0.5} only.
5. `mix` ≥ `realized` on the signal gate (1.02 vs 0.98) and paired (+0.87 vs +0.71) — the forecast component in close-credit adds information; keep `mix`.
6. Faster advantage baseline (`alpha 0.10`) improves win/loss (2.07) and G-A.1 (+1.30) slightly — worth revisiting with longer windows; not a Phase-A decision.

## Phase-advance decision (spec §6 G-A)

Phase A passes → **Phase B (exit integration) proceeds on `b-tight-miss`**:

```
--entry-credit advantage --trade-credit-mode mix --daily-observe off \
--reward-gain 1.5 --miss-weight 0.5 --avoid-correct-weight 0.75
```

G-B target (§6): median closed-trade realized P&L > 0 on the training window; up/down-capture ≥ 1.0 / ≤ 0.5; deaths unchanged. Note: 10-day windows remain under-powered for G-A.1 discrimination (effective n ≈ 400); the tight-miss PASS at this window is strong evidence but Phase C re-arms the gate on ≥20-day × multi-seed windows.

## Reproduction

Each cell: the base command in `reports/h2h-avsb-phase0.md` plus the knob delta above, `--out-dir data/runs/pa-<cell>`; gates via `uv run python scripts/gates.py --run-dir data/runs/pa-<cell> --reference data/runs/h2h-reference --market-dir data/market/history`.

## Addendum — G-B early signal from the one completed exit cell (local stop 2026-09-16)

Training was halted locally (Duke: 16× faster whole-fly on a second machine; campaign continues there). One exit-grid cell had completed: `pb-e1-trail1` (tight-miss winner + `--trailing-stop-pct 1.0`). G-B statistics (median closed-trade P&L, FIFO-paired; capture vs SPY daily closes):

| run | trades (W/L) | median closed P&L | Σ P&L | up-cap | down-cap |
|---|---|---|---|---|---|
| pb-e1-trail1 (winner + 1% trail) | 529 (277/246) | **+$0.84** | +$1,926 | 0.43 | −0.19 |
| pa-b-baseline (winner, no stops) | 616 (327/286) | **+$1.36** | +$1,964 | 0.51 | +0.19 |
| h2h-reference (incumbent) | 481 (179/302) | **−$8.40** | −$6,432 | −0.05 | 3.55 |

**Finding:** the entry-credit fix alone already flips median closed-trade P&L positive (G-B's primary criterion) — the reference fly shows the documented pathology (median negative, capture inverted 3.55 down / −0.05 up). The 1% trail keeps G-B positive but dilutes the median (0.84 vs 1.36): the exit grid's remaining job is proving *no dilution* (trail 2–3%, ATR, valence-off, hold-to-close) rather than rescue. Whole-fly confirm + remaining cells → machine 2.
