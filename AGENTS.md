# AGENTS.md — Conventions for the Fruit Fly of Wall Street

1. **Seeded determinism everywhere.** Any stochastic component MUST take an
   explicit seed argument. Same seed → byte-identical output artifacts, always.
   Never rely on hidden global RNG state.
2. **Headless CLI for every runnable artifact.** Anything a user can run MUST
   be exposed as a `python -m fruitfly <subcommand>` command (registered via
   the subparser registry in `src/fruitfly/__main__.py`). No notebook-only or
   REPL-only pipelines.
3. **Agents never write git.** Agents MUST NOT run git commands (commit,
   branch, merge, stash, etc.). The orchestrator owns all VCS operations.
4. **Free-data-only constraint (D8).** All market and connectome data MUST
   come from free, publicly available sources. Never require paid API keys or
   licensed datasets; `data/` is gitignored and never committed.
5. **src layout + uv.** The package lives at `src/fruitfly/`. Use
   [`uv`](https://docs.astral.sh/uv/) as the environment and dependency
   manager (`uv sync`, `uv run ...`). Python >= 3.11.
6. **pytest pythonpath note.** There is no editable install: pytest is
   configured with `pythonpath = ["src"]` in `pyproject.toml`. Run `pytest`
   (or `uv run pytest`) from the repo or worktree root so imports resolve.

7. **CLI quick reference (run from repo root via `uv run`; `uv sync` first).**

| Command | What it does |
|---|---|
| `python -m fruitfly fetch-data [--symbols A,B] [--days 25 \| --start S --end E]` | Yahoo 1m bars for the recent window → `data/market/{SYM}_1m.parquet` (2026 cache) |
| `python -m fruitfly backtest --seed 7 --start S --end E --chassis {stripped,whole} [--basket-file F] [--out-dir D]` | one training-window run (full knob list: `--entry-credit {off,forecast,advantage}`, `--trade-credit-mode {realized,forecast,mix}`, `--daily-observe {on,off}`, `--reward-gain/--punishment-gain`, `--miss-weight/--avoid-correct-weight`, `--id-scale`, exit knobs `--trailing-stop-pct/--atr-stop-mult/--exit-forecast`, ...) |
| `uv run python scripts/calibrate.py train --chassis C --start S --end E --seed 7 --out data/fly-X-weights.npz --run-dir data/runs/train-X` | persist a trained artifact (+ full meta). MANDATORY distinct `--run-dir` per arm |
| `uv run python scripts/eval_brains.py --chassis C --artifact NPZ --start S --end E --days N [--basket A,B] [--results-dir D] [--report R]` | head-to-head held-out eval; asserts artifact meta ↔ runtime config (A9) |
| `uv run python scripts/gates.py --run-dir D --reference REF --market-dir data/market/history [--json OUT]` | G-A acceptance gates (exit 0 all-pass) |
| `python -m fruitfly adult --seed 7 --start S --end E --mode {replay,live-paper} --chassis C [--larval-weights NPZ]` | multi-day kill-and-resume fly (live-paper needs Alpaca paper keys in `.env`) |
| `python -m fruitfly dashboard --run-dir D [--port 8765]` | live local dashboard |
| `python -m fruitfly postmortem --run-dir D` | end-of-run report |

**Deep-history fetcher (Alpaca IEX 1m, free tier, D8):**
`data/market/_fetch_history_alpaca.py` (gitignored, env-driven). Writes
`data/market/history/{SYM}_1m.parquet` for 2024–2026 windows; resumable
ledger `.progress.json` (rerun skips complete symbol-windows; run legs
SEQUENTIAL — the ledger is not race-safe).

```bash
# .env (gitignored) needs: APCA_API_KEY_ID / APCA_API_SECRET_KEY (Alpaca free tier)
FETCH_START=2024-01-01 FETCH_END=2025-12-31 \
FETCH_BASKET="MMM,CAT,DE,BA,LMT,UNP,UPS,COP,SLB,PFE" \
uv run --with alpaca-py,pandas,pyarrow,python-dotenv \
  python data/market/_fetch_history_alpaca.py
```

Notes: `FRUITFLY_MARKET_DIR=data/market/history` selects the 2024/25 cache for
backtests/evals (default `data/market` = the 2026 Yahoo cache). Alpaca IEX
bars cover IEX-matched volume only (~2% of consolidated) — sparse/dead minutes
are gaps and stay gaps (never zero-filled). `BRK-B`→`BRK.B` is mapped;
`^GSPC` is Yahoo-only (SPY is the tradable proxy).

**New host bootstrap:** clone + `uv sync` + `uv run pytest -q` (314 passed + 1
skipped expected). `data/` is gitignored — either `rsync -a data/` from an
initialized host (≈3 GB: connectome 2.6 GB, market 206 MB, weights ~20 MB) or
rebuild via the two fetchers above. Seeded determinism (rule 1) is
per-machine: same seed → byte-identical receipts on the same host/BLAS; do
NOT byte-compare receipts across hosts.
