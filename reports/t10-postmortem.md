# T10 Post-mortem — backtest_7_2026-08-17_2026-09-14

Final equity 99,600.63 (-0.40% on 100,000.00 hatch capital). The four-benchmark table below is rendered by `fruitfly.scoreboard.compare_run` over the identical window and bar grid.

## Lifespan

One row per hatch -> death (or end-of-log) segment. ``hatch`` events carry no timestamp, so a life's birth is its first timestamped event (the session ``wake``). *Bars alive* counts equity.csv rows in the segment.

| Life | Hatched (log) | Hatch equity | Born (first ts) | Died | Death (log) | Bars alive |
|---|---|---:|---|---|---|---:|
| 1 | events.jsonl:1 | 100,000.00 | 2026-08-17T13:30:00+00:00 | alive at end of log | — | 7800 |

Equity grid: 7800 bars, 2026-08-17 13:30:00+00:00 .. 2026-09-14 19:59:00+00:00; event log: 15847 lines.

## Cause of death

**None — the fly was alive at the end of the log.**

- 0 death events; final equity 99,600.63 at 2026-09-14 19:59:00+00:00 (-0.40% vs hatch equity 100,000.00).
- Minimum equity over the run: 99,398.00 (0.60% below hatch) — versus the default D14 death rule (equity <= hatch equity x 0.50).
- The final session closed with the usual sugar_shock and sleep at 2026-09-14T20:00:00+00:00 (events.jsonl:15063-15847).

## P&L vs scoreboard (T8 benchmarks)

- Window: 2026-08-17 .. 2026-09-14T23:59:59.999999 (inclusive), capital 100,000
- Grid: fly equity curve, 7800 bars (2026-08-17 13:30:00+00:00 .. 2026-09-14 19:59:00+00:00)
- Benchmarks computed on the same universe and bar grid; S&P 500 interpolated from daily ^GSPC closes.

## Scoreboard

| Benchmark | Final return % | Max drawdown % |
|---|---:|---:|
| Fly (this run) | -0.40% | -0.73% |
| S&P 500 buy-and-hold | -1.61% | -2.01% |
| Monkey-with-darts (seed 7, rebalance: daily (first bar of each session)) | +0.01% | -1.40% |
| Logistic control (smell features, train 70% / trade 30%, threshold 0.5) | +5.68% | -1.48% |
| SPIVA: active large-cap funds underperforming S&P 500 (static reference) | 1y 78.78% / 3y 66.84% / 5y 88.96% / 15y 89.93% underperforming | — |

## Run events

decision: 7799, encounter: 7799, hatch: 1, order: 188, sleep: 20, sugar_shock: 20, wake: 20
- deaths: 0, hatches: 1

## Logistic control model

- Features: returns, rsi, volatility, volume_delta — built by `fruitfly.senses.smell.build_features` (the smell channel's shared builder).
- sklearn LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000), features standardized with train-set mean/std
- Coefficients: returns=-0.0430, rsi=-0.0015, volatility=+0.0281, volume_delta=+0.0078, intercept=-0.0263
- Train rows: 119617 (up-rate 0.493); test bars: 2340
- Test accuracy (direction): 0.516 over 51438 rows.

## SPIVA reference

- S&P Dow Jones Indices, SPIVA U.S. Scorecard Year-End 2025 (data as of Dec. 31, 2025)
- Source URL: https://www.spglobal.com/spdji/en/research-insights/spiva/ (accessed 2026-09-15)
- Static reference row, not computed from market data. Percentages are net-of-fees returns of ALL large-cap active funds (incl. merged/liquidated) vs the S&P 500.

## One-trial grudges and favorites

Heuristic (deterministic; exact definitions):

```
encounter shock : pair with balance < 0 (avoid side of the MBON
                  approach/avoid readout; decision typically `avoid`).
one-trial grudge: the ticker's FIRST encounter+decision pair after a
                  pending shock, when the INNATE readout (raw_balance,
                  hatch-anchored, pre-plasticity) is >= +0.005 — the
                  loop's APPROACH_THR, the innate reflex says approach —
                  yet the LEARNED readout (balance, post KC->MBON
                  plasticity) is <= -0.005 and the decision is `avoid`.
                  Plasticity flipped an innately positive odor into
                  avoidance after a single aversive trial. Scored once
                  per pending shock; a renewed shock updates it, any
                  other outcome (buy, cap-blocked pass, neutral
                  readout) resolves it.
approach trial  : pair with balance >= +0.005 and decision `buy`.
favorite        : ticker with >= 3 approach trials
                  (repeated strong approach).
```

### Grudges (one-trial post-shock avoidance)

None detected.

### Favorites (repeated strong approach)

| Ticker | Approach trials | Total buys |
|---|---:|---:|
| MSFT | 10 | 10 |
| PG | 4 | 4 |
| ABBV | 3 | 3 |
| AVGO | 3 | 3 |
| BRK-B | 3 | 3 |
| META | 3 | 3 |
| XOM | 3 | 3 |

## Event tallies

decision: 7,799, encounter: 7,799, hatch: 1, order: 188, sleep: 20, sugar_shock: 20, wake: 20.

- deaths: 0, hatches: 1
