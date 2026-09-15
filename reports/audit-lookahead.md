# No-look-ahead audit — pre-training relaunch

**Date:** 2026-09-15. **Scope:** final code at `2fbc7d5` (fair fills). Three
read-only audits over the fill model, sensory features + plume filter, and
train/eval separation. Every finding cites file:line.

## Verdict: no look-ahead blockers. Trainings cleared to relaunch.

## 1. Fill model — OK

- `src/fruitfly/loop.py:367` — `PendingOrderBook.drain` requires the
  execution bar strictly AFTER the decision bar and an actual print:
  `if df is None or ts <= o.decision_ts or ts not in df.index: continue`.
  A bar-t decision can earliest fill on the symbol's next printed bar.
- Drain runs in step 1b (`loop.py:792`), before mechanical exits and new
  decisions; no drain path reads future bars.
- `loop.py:369` — parameter-free pessimistic bound: buy at exec-bar HIGH,
  sell at exec-bar LOW. `fill_mode="close"` is the only look-ahead path,
  opt-in, validated against `FILL_MODES`, documented as a comparison hatch.
- IEX gaps: a missing `ts` is skipped, order survives until the symbol's
  next real bar; fill price is always read at execution ts (no stale-price
  path). Cancelled on death / run end / `no_position`.
- Position cap: `loop.py:927` — cap check counts open positions PLUS queued
  buys (`pending_book.n_buys()`); fills land before the decision step, so
  the queue race is closed. Cap cannot be exceeded.

## 2. Features + plume filter — OK

- `build_features` (`senses/smell.py:113-146`): RSI uses the last
  `rsi_period + 1` closes ending at t (correct 14-period trailing);
  volatility over the last 20 returns ending at t; volume-delta base window
  explicitly excludes the last bar. All trailing-inclusive.
- `_features_at` (`loop.py:385-389`) slices `df.iloc[: i + 1]` — never t+1.
  Pad-indexing (`get_indexer([ts], method="pad")`) is guarded by
  `df.index[i] == ts` checks, so no silent earlier-bar substitution either.
- Vision (`senses/vision.py:81-104`): 48-bar trailing window, right-aligned
  render; looming = last column range vs mean of earlier columns in the
  same window. No future bars.
- Plume ranking (`loop.py:411-419`): bar-t features only. Sound with
  next-bar fills; no other encounter input touches future data.
- Resample caveat (benign): `scripts/calibrate.py:67-69` labels aggregated
  bars `label="left", closed="left"` — a bar labeled t contains data
  through t+minutes. Benign because fills execute on the NEXT bar, after
  the information bar closes.

## 3. Train/eval separation — OK, with guards added

- Logistic control (`scoreboard.py:249-253, 263`): mean/std fit on TRAIN
  rows only; eval split transformed with train stats, never refit;
  `random_state=0` pinned; lbfgs deterministic. Monkey-darts uses seeded
  `default_rng`.
- `scripts/calibrate.py:213-218`: train window is exactly the CLI
  `--start/--end`, identical code path for both arms; decisions confined to
  `[start, end]` (`loop.py:597-599`); full cache loaded only for trailing
  warmup; no plume/feature caches on disk. Nothing from the eval window can
  enter training.
- **Fixed:** `scripts/eval_brains.py` now raises if the selected held-out
  days do not strictly postdate the artifact's training window end
  (`lw.meta["end"]`) — previously a stale/truncated cache could silently
  evaluate on training days.
- **Fixed:** `run_backtest` death branch emitted the `hatch` event and reset
  `equity = cash` twice (once before, once after anchor re-calibration);
  duplicate removed.
- Receipt determinism: `json.dumps(..., sort_keys=True)`, fixed float
  formatting, artifact md5, both-arms-same-days cross-check.

## Documented caveats (known, unfixable or accepted)

1. **Survivorship/selection bias:** the 22-symbol BASKET is chosen as of
   Sep 2026 and replayed back to Mar 2026. All window returns (fly arms,
   SPX, baselines) share this bias; it does not compare unfairly between
   arms, but absolute numbers are optimistic.
2. **Stale pending buys:** unfilled buys carry no max-age (`loop.py:366`) —
   a symbol that halts for days fills its queued buy whenever it next
   prints, far from the decision context. Performance-conservative; sells
   self-heal via `no_position` cancel.
3. **Fractional control split:** the logistic control uses a 70/30
   fractional split over the compared window (`scoreboard.py:211-214`), not
   the fly's date-based train/eval windows. It is the T8 comparison control,
   leakage-free, but does not encode the DESIGN dates.
4. **One-bar label bleed in the control:** the last train bar's label is the
   next bar's return (`scoreboard.py:233-243`) — single row, directionally
   trivial, no feature/normalization leakage.
5. **IEX volume:** fills bound to IEX-matched bars (~2–5% of consolidated
   volume); missing minutes are real gaps. ^GSPC absent — SPY/SPX proxy.
