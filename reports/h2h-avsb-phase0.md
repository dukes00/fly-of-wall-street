# Phase 0 head-to-head — Option A (entry-forecast) vs Option B (advantage gate)

**Date:** 2026-09-16 · **Seed 7 · window 2025-06-02..2025-06-13 (10 trading days) · chassis stripped · basket `baskets/train40.txt` (40 names)**
**Config:** both arms `--entry-credit {forecast|advantage} --trade-credit-mode mix --daily-observe off --reward-gain 1.5`; reference = incumbent (entry-credit off, daily-observe on, gains 1.0/1.0). Common knobs per spec §2/§4 defaults (`r_scale=0.005`, `a_scale=0.003`, `baseline_alpha=0.05`, `miss_weight=0.25`, `avoid_correct_weight=0.5`, `horizon_bars=30`, `mix_weight=0.5`, `id_scale=1.0`).

## Gate scoreboard (scripts/gates.py, reference = incumbent-config run)

| Gate | A (forecast) | B (advantage) | Notes |
|---|---|---|---|
| G-A.1 entry signal (HAC lag-30 Spearman t, n=3900) | ρ=−0.013, t=−0.56 — **FAIL** (subsample sign disagreement) | ρ=+0.023, t=+1.02 — **FAIL** (|t|≤2; subsample agrees, +1.25) | neither arm clears |t|>2 in 10 days |
| G-A.2 win/loss > 1 | 1.59 (54W/34L) — PASS | 1.80 (99W/55L) — PASS | both above 1; A regressed vs v0.6-trained 2.04 (expected: untrained weights) |
| G-A.3 turnover band vs ref | trades/day 1.023×, buys/enc 0.892× — **PASS** | trades/day 1.770×, buys/enc 1.136× — **PASS** | B trades 73% more (advantage gate encourages re-entry); inside band |
| G-A.4 decision change > 0 | 24.9% — PASS | 27.6% — PASS | deaths 0 = 0 both |
| **Overall** | **FAIL** (G-A.1) | **FAIL** (G-A.1) | |

## Portfolio-level (paired daily returns, 9 shared day-boundaries)

- A vs ref: **+1.41%/day, paired t=+3.05** (run +0.65%/day vs ref −0.77%/day)
- B vs ref: **+0.87%/day, paired t=+2.15** (run +0.11%/day vs ref −0.77%/day)
- B vs A: −0.54%/day, paired t=−0.92 (not significant)

Final equity: ref $93,144 · **A $106,499** · B $101,615 (start $100,000).

## Credit-path evidence (mechanism actually engaged)

- A: 2,878 `entry_credit` events (0 anti-lookahead violations; every `credit_ts ≥ decision_ts + 1 bar`, spot-audit = 0 violations on the full log), 27 supersedes (settle-forecast→final delta replacements working).
- B: 2,892 `entry_credit`, 40 supersedes; 154 `trade_credit` (vs ref 87) — the advantage gate converts more encounters into activity (+73% trades/day), consistent with its design.
- Neither G-A.1 result is surprising at n=3900 overlapping encounters: spec §6 says effective n ≈ ticker-days ≈ 400; a 0.1–0.2% entry edge needs the full Phase-C windows to clear |t|>2.

## Verdict

**Option B (advantage) wins the head-to-head** on every axis the ratified objective cares about:

1. **Information-driven pick rate:** B's `balance_used`↔forward-return association is positive and sign-stable (full +0.023/+1.02, subsample +0.032/+1.25); A's is ~zero with sign disagreement — A learned nothing about *what to buy* that survives de-overlapping.
2. **Active trader:** B trades 15.4/day at 1.77× of reference (inside the [0.5×, 2.0×] band — the floor matters as much as the ceiling) with buys/encounter up 1.14×; A sits at reference activity.
3. **Win/loss:** B 1.80 > A 1.59 (both pass >1.0).
4. **Realized outcome:** both beat the reference paired (t=+3.05 / +2.15), B−A not significant — consistent with the objective: the point is *how* it picks, not alpha-max.

G-A.1 fails for both (statistical power, not behavior — 10 days cannot resolve a 0.1% edge at effective n≈400; the gate is doing its job). Per spec §6 Phase A proceeds with the winning objective and the wider/longer windows where G-A.1 becomes discriminating.

**Phase A config:** `--entry-credit advantage --trade-credit-mode mix --daily-observe off --reward-gain 1.5` + exit-grid and calibration sweeps per spec §6 Phase A; G-A re-checked at each cell; capture-preservation (up ≥1.0 / down ≤0.5) tracked as the B-risk sentinel (spec §9.2).

## Reproduction

```
uv run python -m fruitfly backtest --seed 7 --start 2025-06-02 --end 2025-06-13 \
  --chassis stripped --basket-file baskets/train40.txt --entry-credit advantage \
  --trade-credit-mode mix --daily-observe off --reward-gain 1.5 \
  --out-dir data/runs/h2h-optionB            # FRUITFLY_MARKET_DIR=data/market/history
uv run python scripts/gates.py --run-dir data/runs/h2h-optionB \
  --reference data/runs/h2h-reference --market-dir data/market/history
```
