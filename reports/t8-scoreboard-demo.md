# T8 Scoreboard — backtest_42_2026-08-17_2026-08-21

- Window: 2026-08-17 .. 2026-08-21 (inclusive), capital 100,000
- Grid: fly equity curve, 1950 bars (2026-08-17 13:30:00+00:00 .. 2026-08-21 19:59:00+00:00)
- Benchmarks computed on the same universe and bar grid; S&P 500 interpolated from daily ^GSPC closes.

## Scoreboard

| Benchmark | Final return % | Max drawdown % |
|---|---:|---:|
| Fly (this run) | +1.21% | -5.97% |
| S&P 500 buy-and-hold | -0.91% | -1.34% |
| Monkey-with-darts (seed 42, rebalance: daily (first bar of each session)) | -8.22% | -10.51% |
| Logistic control (smell features, train 70% / trade 30%, threshold 0.5) | +8.61% | -2.54% |
| SPIVA: active large-cap funds underperforming S&P 500 (static reference) | 1y 78.78% / 3y 66.84% / 5y 88.96% / 15y 89.93% underperforming | — |

## Run events

death: 1, decision: 20, hatch: 1, order: 20, sleep: 20, sugar_shock: 20, wake: 20
- deaths: 1, hatches: 1

## Logistic control model

- Features: returns, rsi, volatility, volume_delta — built by `fruitfly.senses.smell.build_features` (the smell channel's shared builder).
- sklearn LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000), features standardized with train-set mean/std
- Coefficients: returns=+0.0038, rsi=+0.1875, volatility=-0.0027, volume_delta=+0.0315, intercept=-0.0603
- Train rows: 5380 (up-rate 0.485); test bars: 585
- Test accuracy (direction): 0.515 over 2336 rows.

## SPIVA reference

- S&P Dow Jones Indices, SPIVA U.S. Scorecard Year-End 2025 (data as of Dec. 31, 2025)
- Source URL: https://www.spglobal.com/spdji/en/research-insights/spiva/ (accessed 2026-09-15)
- Static reference row, not computed from market data. Percentages are net-of-fees returns of ALL large-cap active funds (incl. merged/liquidated) vs the S&P 500.
