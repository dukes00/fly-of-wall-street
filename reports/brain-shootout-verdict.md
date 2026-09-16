# Brain shootout verdict — D22 live-brain decision

**Date:** 2026-09-16. **Training:** both arms, v0.6 brain, seed 7,
2026-03-02..06-30 (83 trading days, fair pessimistic fills, no look-ahead —
audited `reports/audit-lookahead.md`). **Held-out eval:**
`scripts/eval_brains.py`, 10 unseen days 2026-08-28..09-11
(`reports/brain-shootout.md`, auto-generated table).

## Result

| arm | held-out ret% | held-out dd% | trades | deaths | training curve |
|---|---:|---:|---:|---:|---|
| SPX buy-hold | −0.957 | 2.014 | — | — | −0.957% over the same days |
| stripped | −0.199 | 1.215 | 1,741 | 0 | −16.3% final; bent to +0.30%/day in the last third |
| **whole** | **−0.097** | 1.291 | 1,879 | 0 | −21.1% final; +0.005%/day trend, no sustained positive regime |

Both brains beat the market on the held-out window. Whole-fly lost less
(−0.097% vs −0.199%) and captured more of the up days (Sep 3: +0.96% vs
+0.56%); stripped ran slightly shallower drawdown (1.215% vs 1.291%).

## Decision (D22 criterion, pre-registered by Duke)

*"Whole-fly unless stripped performs much better."* Stripped did not perform
much better — whole edged it on return (−0.097% vs −0.199%) with comparable
drawdown. **D22 stands: the whole fly is the live brain.** The adult default
chassis and artifact are flipped to
`data/fly-whole-weights.npz` (commit history: "D22 shootout verdict").

## What each brain learned (training-window forensics)

Both runs converge on the same two lessons and no third:

1. **Exit disposition, not entry selection.** Forward returns after buys ≈ 0
   in every phase for both brains; the win/loss payoff flipped from <1 to >1
   in both (whole 0.74 → 1.28; stripped 0.57 → 1.37 via P3) through the
   valence-flip exit channel, exactly the behavior the per-exit dopamine
   (D6 v0.6) can credit.
2. **Preferences expressed in hold duration.** Buy counts stay near-uniform;
   profits concentrate in calm names held long (both: BRK-B, MSFT, JNJ;
   stripped adds SPY/PG; whole bleeds META/TSLA/AVGO throughout).

Whole-vs-stripped contrast: the whole fly is the same animal with a heavier,
less consolidated dopaminergic footprint — 30% more trades, half the P3
median hold, valence-flip exits used less (12.9% vs 20.5%), anchor ended at
−0.93 vs stripped's −0.24.

## Caveats (documented, not hidden)

- Single seed per arm; the market reversed −4.7% → +10% inside the training
  window, so part of each run's mid-course improvement is regime beta. The
  exit-asymmetry change is the only behaviorally attributable learning
  signal; multiple seeds would settle the rest.
- The 22-symbol basket is selected as of Sep 2026 and replayed back to Mar
  2026 — survivorship bias, applies equally to all arms.
- The whole-fly training window return (−21.1%) includes tuition; the
  artifact's held-out behavior is the decision basis.
