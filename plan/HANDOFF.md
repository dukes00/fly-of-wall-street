# HANDOFF — Fresh Orchestrator Session

**Written:** 2026-09-14, end of session 1. **Branch:** `main` @ `5b20764`, everything pushed to `github.com/dukes00/fly-of-wall-street`. **Suite:** 193 passed + 1 env-gated skip; `ruff check .` clean.

**Standing rule:** DESIGN.md v0.5 is the source of truth; D22 (whole-fly is the live brain, trained directly; transplant dropped) is ratified by Duke in session. Duke has signed off editing DESIGN.md with a changelog note when decisions change.

## What exists (all committed)

- Full T1–T15 build complete (plan/TASKS.md v0.1): connectome → LIF → senses → plasticity → loop → scoreboard → training → post-mortem → transplant → dashboard → adult harness.
- `data/connectome/` — MaleCNS v1.0 raw (166,700 neurons, 124.2M synapses, no-auth GCS) + stripped-chassis (27,115 neurons) + whole-fly labeled caches. Gitignored; regenerate via `scripts/extract_chassis.py`.
- `data/market/` — Yahoo 1m cache 2026-08-17..09-14 + `data/market/history/` — Alpaca IEX 1m 2026-03-02..09-11 (135 trading days, 21 symbols, 966,569 bars). Resumable fetcher `data/market/_fetch_history_alpaca.py`. **IEX caveat:** IEX-matched volume only (~2–5% of consolidated); missing minutes are real gaps; ^GSPC absent (index, not on Alpaca — SPY is the proxy).
- Fair-fill execution model (this session's last commit): orders decided on bar *t* execute on the symbol's next available bar; BUY at execution-bar HIGH, SELL at LOW (parameter-free pessimistic bound). `fill_mode="close"` is the documented look-ahead-biased escape hatch. Fees = 0.0 by Duke's call (commission-free brokerage; live Alpaca fills absorb real spread automatically).
- Opt-in STD (short-term synaptic depression) in `sim.py` — `chassis="whole"` implies calibrated β=0.1, τ_rec=500ms. Without it APL/DPM feedback silences all KCs (reports/t12b-apl-std.md).
- Whole-fly brain works: KCs fire at stripped yield, places trades, 1.26 s/bar (~8 min per 390-bar day), byte-identical same-seed runs.
- Eval harness: `scripts/eval_brains.py` (one command per arm, deterministic JSON receipts + report), `reports/brain-shootout.md` skeleton with real SPX column (held-out 10-day window 2026-08-28..09-11: SPX −0.957%, dd 2.014%), both fly arms PENDING.

## Work queue (in order — do NOT skip the audit)

### 1. No-look-ahead audit (Duke explicitly requested; do before relaunching training)
Dispatch a read-only auditor (scout/reviewer) over the final code. Known surface to verify:
- Fill model: next-bar pessimistic (just landed — audit it, especially the pending-queue drain order in `run_backtest` step 1b and cap counting including queued buys).
- Features: `build_features` (RSI/vol/volume-delta trailing windows) and `_features_at` must use bars ≤ decision bar only; vision window trailing.
- Plume filter: same-bar close for ranking is OK *because* fills are next-bar; confirm no ranking uses future data.
- Train/eval separation: logistic control in `scoreboard.py` — scaler fit on train split ONLY.
- **Known unfixable caveat — document, don't hide:** the 22-symbol BASKET is chosen as of Sep 2026 and replayed back to Mar 2026 → survivorship/selection bias. State it in the shootout report.
- Training/eval windows: train 2026-03-02..06-30, eval 2026-08-28..09-11 (held out). Never evaluate on the training window.
Deliverable: audit findings appended to reports/brain-shootout.md or reports/audit-lookahead.md; fix anything found before training.

### 2. Relaunch both trainings on fair fills (Duke-approved, sub-12h)
Whole-fly (~10–11 h):
```
FRUITFLY_MARKET_DIR=data/market/history nohup uv run python scripts/calibrate.py train \
  --chassis whole --start 2026-03-02 --end 2026-06-30 --seed 7 \
  --out data/fly-whole-weights.npz --run-dir data/runs/train-whole > /tmp/wholefly_train.log 2>&1 &
```
Stripped control arm (identical window/seed; ~1 h): same command with `--chassis stripped --out data/fly-stripped-hist-weights.npz --run-dir data/runs/train-stripped`.

**Lessons baked into these commands (do not relearn):**
- `--run-dir` is MANDATORY for parallel arms: `calibrate.py train` defaults to the shared `data/runs/backtest_{seed}_{start}_{end}` and two arms truncate each other's receipts (this already corrupted one run).
- Killing a training: kill the whole process tree (`uv run` spawns a child python; `kill $parent` orphans it — check `ps aux | grep calibrate`).
- Verify liveness after launch: `tail /tmp/wholefly_train.log` shows the banner, then receipts appear in the run dir; a full 390-bar day takes ~8–9 min for whole-fly.
- IEX gaps mean fewer bars than 390/day; GSPC is absent from history (loop skips symbols with empty frames — verified).

### 3. Brain shootout (after both artifacts exist)
```
FRUITFLY_MARKET_DIR=data/market/history uv run python scripts/eval_brains.py \
  --chassis stripped --artifact data/fly-stripped-hist-weights.npz
FRUITFLY_MARKET_DIR=data/market/history uv run python scripts/eval_brains.py \
  --chassis whole --artifact data/fly-whole-weights.npz
```
Stripped arm ≈ 25 min; whole-fly arm ≈ 1.5–2 h (10 held-out days). Report: `reports/brain-shootout.md`. Then present Duke: which brain for live paper. Duke's stated criterion: whole-fly unless stripped performs much better.

### 4. Flip adult default brain to whole-fly (per D22) + final gates + push
Default `AdultConfig`/`BacktestConfig` chassis to `"whole"` (or wire the adult CLI/artifact to `data/fly-whole-weights.npz`), update README tagline claims if needed, rerun full `pytest` + `ruff` + all six CLI `--help`s, commit, push.

### 5. Pending market-dependent item
`scripts/live_smoke.py` — one paper order submit/fill/close on Alpaca; runnable only when market is open (next: Mon 2026-09-15 13:30 UTC). Keys in gitignored `.env`; never print/commit them.

## Verification cheat sheet
```
uv run pytest -q          # 193 passed + 1 skipped expected
uv run ruff check .       # clean
uv run python -m fruitfly --help   # plus backtest/fetch-data/postmortem/dashboard/adult
```

## Standing conventions
- Agents: no git writes, no project-wide gates/formatters, targeted tests only; orchestrator verifies + commits.
- Determinism is a hard gate: same seed → byte-identical receipts (PCG64 caveat: byte-identity holds within this numpy version).
- Calibration value changes are proposals for Duke (T9 addendum in reports/t9-calibration.md), except D22 which he ratified.
- Whole-fly runs must keep STD defaults (β=0.1, τ_rec=500ms) — config enforces this.
- 16 GB RAM: whole-fly sim ~1.8 GB; don't run two whole-fly evals simultaneously.
