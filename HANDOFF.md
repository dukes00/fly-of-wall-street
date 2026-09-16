# HANDOFF — TRAINING2 campaign (DESIGN v0.7)

**Date:** 2026-09-16 · **Branch:** `main` (dukes00/fly-of-wall-street) · **Owner context:** Duke runs the campaign on a second machine (~16× faster whole-fly than the original M1). This file is the turnkey continuation doc: what exists, what was decided, what to run next.

---

## 0. Machine-2 bootstrap

```bash
git clone https://github.com/dukes00/fly-of-wall-street && cd fly-of-wall-street
uv sync                     # Python >= 3.11
uv run pytest -q            # sanity: expect "314 passed, 1 skipped"
uv run ruff check .         # expect: clean
```

`data/` is gitignored (D8). Two ways to populate it:

- **Copy (fastest, one-time):** `rsync -a data/ machine2:fly-of-wall-street/data/` from any initialized host (≈3 GB: connectome 2.6 GB, market 206 MB, weights ~20 MB, runs 110 MB — `runs/` optional, receipts only). Also copy `.env` (Alpaca keys; gitignored).
- **Refetch:** `uv run python -m fruitfly fetch-data` (recent 2026 window, Yahoo) and the Alpaca IEX deep-history fetcher for 2024–2025 — see **AGENTS.md §7 CLI quick reference** (env vars `FETCH_START`/`FETCH_END`/`FETCH_BASKET`, resumable ledger, run legs sequentially).

CLI reference for everything else: **AGENTS.md §7**. Layman's project tour: **README.md**.

> Determinism (AGENTS.md rule 1) is per-machine: same seed → byte-identical receipts on the same host/BLAS. Do NOT byte-compare receipts across hosts; the G-A/G-B statistics are robust to ulp drift.

---

## 1. Where the campaign stands

| Phase | Status | Verdict / artifact |
|---|---|---|
| Phase 0 plumbing | ✅ committed (`0de9954`) | entry-credit ledger (A forecast / B advantage), knobs, basket files, meta contract, G-A gate script |
| Phase 0 head-to-head | ✅ committed (`c6aff95`) | **Option B (advantage gating) wins** — `reports/h2h-avsb-phase0.md` |
| Phase A calibration | ✅ committed (`4136e06`) | **Winner: `tight-miss`** — `reports/phaseA-calibration.md` |
| Phase B exits | ⏳ partially run locally | 1 of 7 cells done; **early G-B finding in `reports/phaseA-calibration.md` addendum** (`e5a0890`) |
| Phase C multi-seed + §7 eval | not started | protocol in `plan/TRAINING2-SPEC.md` §6/§7 |

**Frozen decisions (Duke-ratified, do not relitigate):** objective = *active trader choosing picks on information, not alpha* (spec §9.6); A vs B decided empirically → B; `MISS_WEIGHT=0.25` base (0.5 in the winning cell); never-seen eval basket frozen in `baskets/eval20-neverseen.txt` (never train on it — guards are live in backtest + eval); `id_scale=0` is architecturally dead (smell state is multiplicative on identity → catatonic fly; only sweep {1.0, 0.5}).

**Phase-A winner config (everything below builds on it):**

```bash
WINNER="--seed 7 --start 2025-06-02 --end 2025-06-13 --chassis stripped \
  --basket-file baskets/train40.txt --entry-credit advantage \
  --trade-credit-mode mix --daily-observe off --reward-gain 1.5 \
  --miss-weight 0.5 --avoid-correct-weight 0.75"
```

---

## 2. What to run next (in order)

All commands run from repo root; `export FRUITFLY_MARKET_DIR=data/market/history` for 2024/25 windows. Each run: **distinct `--out-dir`** (backtest) / `--run-dir` (train).

### 2a. Phase B — exit grid remainder (spec §5.2, E1–E4)

E1-trail1 (trailing 1%) is done (receipts may be rsynced or rerun — 25 min each here; ~2 min on machine 2):

```bash
run() { uv run python -m fruitfly backtest $WINNER --out-dir "data/runs/pb-$1" ${=2} \
        > "data/runs/pb-$1.log" 2>&1 && echo "done pb-$1"; }   # zsh: ${=2}; bash: $2
run e1-trail2 --trailing-stop-pct 2.0
run e1-trail3 --trailing-stop-pct 3.0
run e2-atr15  --atr-stop-mult 1.5
run e2-atr20  --atr-stop-mult 2.0
run e3-valoff --valence-flip-exit off --trailing-stop-pct 2.0
run e4-holdclose --valence-flip-exit off --hunger-exit off
```

Reference run (G-A.3 turnover band): `data/runs/h2h-reference` (incumbent config, same window/seed — regenerate if not rsynced: same `$WINNER` minus the four credit knobs, `--entry-credit off` etc.; exact command in `reports/h2h-avsb-phase0.md`).

### 2b. G-B gate on each cell (spec §6)

```bash
uv run python scripts/gates.py --run-dir data/runs/pb-<cell> \
  --reference data/runs/h2h-reference --market-dir data/market/history
```

Then closed-trade statistics (the G-B primary):

```bash
uv run python - <<'EOF'
import json, numpy as np, pandas as pd
def pnls(run):
    orders = {}
    for line in open(f"data/runs/{run}/events.jsonl"):
        ev = json.loads(line)
        if ev.get("type") == "order":
            orders.setdefault(ev["ticker"], []).append(ev)
    out = []
    for tk, obs in orders.items():
        lots = []
        for o in obs:
            if o["side"] == "buy": lots.append({"s": o["shares"], "p": o["price"]})
            elif o["side"] == "sell":
                rem = o["shares"]
                while rem > 0 and lots:
                    L = lots[0]; take = min(rem, L["s"])
                    out.append((o["price"] - L["p"]) * take)
                    L["s"] -= take; rem -= take
                    if L["s"] == 0: lots.pop(0)
    return np.array(out)
def eq(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.groupby(df["timestamp"].dt.date)["equity"].last().pct_change().dropna()
spy = pd.read_parquet("data/market/history/SPY_1m.parquet")
spy["timestamp"] = pd.to_datetime(spy["timestamp"])
spyd = spy.groupby(spy["timestamp"].dt.date)["close"].last().pct_change().dropna()
import sys
for run in sys.argv[1:]:
    p = eq(f"data/runs/{run}/equity.csv"); t = pnls(run)
    sh = p.index.intersection(spyd.index); xs, bs = p[sh], spyd[sh]
    up = xs[bs > 0].mean() / bs[bs > 0].mean() if (bs > 0).any() else None
    dn = xs[bs < 0].mean() / bs[bs < 0].mean() if (bs < 0).any() else None
    print(f"{run:18s} n={len(t):4d} median={np.median(t):+8.2f} sum={t.sum():+10.2f} "
          f"up={up and round(up,2)} down={dn and round(dn,2)}")
EOF
```

Pick the exit winner: **median closed P&L > 0, no capture dilution vs `pa-b-baseline`** (that run: median +$1.36, up-cap 0.51 / down-cap 0.19). Local evidence so far: the entry-credit fix alone already achieves G-B-positive; the 1% trail dilutes (median +$0.84) — so prefer the *simplest* exit config that doesn't dilute.

### 2c. Whole-fly confirmation (spec §6 Phase B)

Combined winner config, `--chassis whole`, 10 days (~85 min on M1, ~5–6 min on machine 2). Check: G-B still positive, deaths 0, capture holds. **This decides whether the v0.7 artifact replaces `data/fly-whole-weights.npz` as the live brain.**

### 2d. Phase C — multi-seed, longer windows (spec §6)

≥3 seeds × 20 trading days, stripped for sweeps; whole-fly for the top-2 configs. Persist artifacts:

```bash
uv run python scripts/calibrate.py train --chassis stripped \
  --start 2025-03-03 --end 2025-03-28 --seed <S> \
  --out data/runs/train2/<config>/<S>.npz --run-dir data/runs/train2/<config>/<S> \
  <winning knob set>
```

Gate G-C: median-across-seeds HAC t (G-A.1), turnover band, win/loss hold; report cross-seed variance.

### 2e. §7 eval protocol (the honest test)

≥20 days/period × 2 periods (2024-03-04..08-30, 2025-03-03..08-29 both fetched) × 2 baskets (in-dist: train40 names; **never-seen**: `baskets/eval20-neverseen.txt`). `uv run python scripts/eval_brains.py --chassis whole --artifact <winner.npz> --start S --end E --days 20 --basket <20 names> --results-dir D --report reports/train2-eval.md`. Primary statistic: pooled paired t vs the incumbent v0.6 artifact on shared days; secondary: per-cell up/down-capture; tertiary: median closed P&L. With the 16× box, run whole-fly evals everywhere (~2.5 min/day → a 20-day cell ≈ 50 min).

---

## 3. Gate criteria (how to read results)

- **G-A.1** entry signal: `|HAC t| > 2` (lag-30) AND non-overlapping subsample agrees in sign — the redesign's whole point; 10-day windows are borderline-powered (effective n ≈ 400), 20-day × multi-seed is the real test.
- **G-A.2** executed win/loss > 1.0. **G-A.3** trades/day AND buys/encounter within [0.5×, 2.0×] of the reference — activity band is a floor too ("active trader" is the objective).
- **G-A.4** decision-change rate > 0, deaths ≤ reference.
- **G-B** median closed-trade P&L > 0 on the training window; up/down-capture ≥ 1.0 / ≤ 0.5; deaths unchanged.
- **Ambiguous-result rule (spec §9.2):** "capture collapsed but realized P&L improved" is NOT a win — flag for Duke.

**Baselines to beat** (local run inventory, same window/seed unless noted): reference incumbent −0.77%/day, median trade −$8.40; h2h Option B +0.11%/day; Phase-A winner tight-miss +0.96%/day (t=+2.29 vs ref), win/loss 1.55; v0.6 whole-fly artifact: OOS +0.145%/day (t=0.88), capture 1.20/0.34.

---

## 4. Caveats & gotchas

- Dashboard: `uv run python -m fruitfly dashboard --run-dir data/runs/<dir> --port 8765` — the status dot is wall-clock-grounded (green = writer emitting now).
- Hub-run processes on the M1 needed `pty: true` to flush stdout; receipts (`equity.csv` growth) are the reliable liveness signal.
- G-A.1's subsample uses first-by-ts per ticker per 30-bar window — deterministic.
- `2025-03-10` has no IEX bars for any symbol (source-side outage; gaps are preserved by design).
- Never train on `baskets/eval20-neverseen.txt` names — the guards raise; don't bypass them.
- If adding data: depth-gate new symbols (≥300/390 median bars/day; see `reports/iex-depth.md` — most mid-caps fail on IEX).
