# The Fruit Fly of Wall Street

**166,691 neurons. Zero emotions.**

A simulated fruit-fly brain — rebuilt from the real fly's wiring diagram —
trades the US stock market one minute at a time. It doesn't read news or run a
standard trading strategy. It sees candlestick charts, smells market
indicators, feels profit as sugar and loss as electric shock, gets hungry in a
drawdown, and sleeps when the market closes. Everything is deterministic: run
it twice with the same seed and you get byte-identical results.

> **This is a research project, not financial advice, and not a product.**
> Out-of-sample profit is **not** demonstrated — see
> [Where we are](#where-we-are). The honest, ratified goal (v0.7) is not a
> money printer but an **active trader that picks based on information rather
> than at random**.

## Meet the fly

The "fly" is a computer simulation of a fruit fly's brain built from the
complete male *Drosophila* connectome (Berg et al., Cell 2026 — Google/Janelia
FlyWire): the full list of **166,691 neurons** and roughly **125 million
synaptic connections** between them, simulated as leaky integrate-and-fire
neurons whose connection strengths come from measured synapse counts. So this
is not a cartoon of a brain — it is the actual wiring of a real fly's nervous
system, running against stock data. It is also *not* a full human-level brain:
it's a fly's, scaled honestly. When it "decides", only a few thousand of those
neurons are actually talking.

Two chassis sizes exist:

- **Whole fly** — all 166,691 neurons, all the time. This is the real brain
  (decision D22). It's slow: about 8 minutes of computing per simulated
  trading day on a MacBook M1.
- **Stripped chassis** — only the wiring the fly needs to see, smell, and
  decide (retina, optic lobe, olfactory pathway, mushroom body): 27,115
  neurons. Trains about 3x faster and is used for fast experiments.

The in-fiction fund is called **Compound Eye Capital**.

## A day of trading

The market session runs on 1-minute bars ("bars" = one candlestick each).
Here's what happens on every single bar:

1. **What can the fly smell? (the plume filter).** There is no watchlist.
   Every ticker in the current basket emits an "odor plume" whose intensity is
   how much it has been moving: |20-bar return| × 20 plus relative volume. A
   stock that isn't moving is odorless and invisible. The five smelliest
   movers become today's active set — the fly only considers stocks that are
   actually going somewhere, in either direction.

2. **One encounter per bar.** The fly visits one plume at a time,
   round-robin. For that one stock it:
   - **Sees** — the recent candlestick chart is rendered at fly-eye
     resolution; direction-selective cells in the optic lobe pick up price
     drift, and "looming"-sensitive cells fire on vertical spikes (crashes and
     rallies).
   - **Smells** — market features (recent return, momentum over 20 bars,
     RSI, volatility, volume change) are mixed into a virtual odor, blended
     with a fixed per-ticker "identity" scent (AAPL smells like AAPL). This
     identity/state split is what lets lessons learned about one volatile
     stock transfer to another.
   - **Tastes** — if the fly already owns this stock, its unrealized
     profit/loss registers as a sweet or bitter taste.

3. **The decision.** The odor excites a tiny network called the **mushroom
   body** (the fly's learning and memory center — Kenyon cells feeding ~35
   output neurons, MBONs, split into "approach" and "avoid" halves). The
   balance between approach and avoid is a number from −1 to +1, re-centered
   on the fly's innate bias so a fresh fly sits at roughly zero. The rule:
   - balance above threshold → **BUY** (or add to a position);
   - balance below threshold → **SELL** if it holds the stock, otherwise
     **pass**;
   - neutral → **pass**.
   Position size comes from how hard the MBONs are firing, up to a cap of 10
   concurrent positions. A crash-shaped bar also nudges the decision through a
   fixed, hard-wired (non-learned) structural pathway from the optic lobe.

4. **Execution is deliberately pessimistic.** An order decided on bar *t*
   fills on the *next* bar — buys at that bar's high, sells at its low. Any
   real-world fill would be no worse, so measured performance can only be
   understated, never flattered.

5. **Exits and death.** Positions close when the learned balance flips
   against them, when a sharp adverse move (~2%) triggers the biological
   stop-loss (shock), or when hunger (portfolio drawdown > 10%) makes the fly
   cash in its biggest winner. If equity drops 50% below the hatch level, the
   fly **dies**: everything is liquidated and a fresh, untrained fly hatches.
   Death isn't scripted — it falls out of the hunger state.

6. **The close of day.** Settle the day's P&L, administer sugar or shock, and
   the fly sleeps: memories consolidate, some decay.

## How it learns

Real flies learn by **dopamine**: these are the reward chemicals that act like
a "sugar" signal when something good happens and a "shock" signal when
something bad does. In the simulation, dopamine gates the strengthening or
weakening of the connections between Kenyon cells and MBONs. Concretely:

- **Realized profit** releases sugar (reward dopamine); **realized loss**
  releases shock (punishment dopamine).
- **Per-exit credit** — when a position closes, the exact encounter that
  opened it is credited with the trade's own outcome, so a single trade
  teaches in proportion to how it went.
- **Per-encounter credit (v0.7)** — every signal-bearing encounter is scored
  on what the stock did *after* the decision: correct avoids are rewarded,
  buying a later winner is rewarded, missing a run-up is mildly punished.
  Crucially, the credit can only use prices from *after* the decision bar —
  a no-look-ahead rule enforced by audit.
- **Advantage gating (Option B, the Phase-0 winner)** — the gate fires on how
  much better (or worse) a stock did than a running baseline for its
  market-state bucket, not on raw profit. This teaches "pick the *relative*
  winner", separating selection skill from simply riding a rising market.
- **Sleep** consolidates memories, protects the strongest associations
  ("grudges" and "favorites" form after a single trial — one bad fill and the
  fly avoids that stock hard), refreshes the innate-balance anchor, and lets
  old associations decay so the fly can forget stale market regimes.

**Why the objective was redesigned.** A v0.6 audit caught the earlier fly
learning only *when* to be in the market (it rode up-markets and sat out
down-markets) but flipping a coin on *what* to buy, and exiting winners too
early while riding losers. The reward signal was a single pooled daily number,
so nothing taught it to prefer one stock over another. The ratified v0.7
objective is therefore explicitly **not** "maximize returns": it is to produce
an **active trader whose picks are driven by information**, with gates that
check whether the decision variable actually correlates with what happens
next.

## How training works

The lifecycle is a metaphor, not decoration:

- **Larval stage** = training on historical data (backtests). The brain's
  learned connections are saved as a `.npz` artifact.

Setup (requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                       # create venv and install dependencies
uv run python -m fruitfly --help   # CLI overview
```

Training is phased, each phase with statistical acceptance gates:

- **Phase 0 (done)** — plumbing plus a head-to-head: two candidate learning
  rules (A: supervised entry-forecast vs B: advantage gating) raced on 10
  stripped-chassis days.
- **Phase A** — the winning objective plus calibration sweeps on the stripped
  chassis, ~20–30 min per configuration.
- **Phase B** — exit-behavior upgrades, then a whole-fly confirmation run.
- **Phase C** — multiple seeds, longer windows, then held-out evaluation.

Every run takes a `--seed`; same seed, same bytes, always. Training baskets
are declared in files (e.g. `baskets/train40.txt`), including a 20-name
"never-seen" basket reserved for honest out-of-sample testing.

Example training-window command (from the Phase-0 head-to-head):

```bash
uv run python -m fruitfly backtest --seed 7 --start 2025-06-02 --end 2025-06-13 \
  --chassis stripped --basket-file baskets/train40.txt --entry-credit advantage \
  --trade-credit-mode mix --daily-observe off --reward-gain 1.5 \
  --out-dir data/runs/h2h-optionB

uv run python scripts/gates.py --run-dir data/runs/h2h-optionB \
  --reference data/runs/h2h-reference --market-dir data/market/history
```

## Running it

All commands are `python -m fruitfly <subcommand>` (headless CLI, free data
only — no paid feeds).

```bash
# fetch 1-minute bars into the local parquet cache, then validate
uv run python -m fruitfly fetch-data --symbols AAPL,MSFT --days 25

# one backtest (the larval loop, as above)
uv run python -m fruitfly backtest --seed 7 --start 2025-06-02 --end 2025-06-13 --chassis whole

# the adult fly: multi-day run that survives being killed and resumed
# (replay mode by default; live-paper submits to an Alpaca paper account)
uv run python -m fruitfly adult --seed 7 --start 2025-06-02 --end 2025-06-13 \
  --mode replay --chassis whole --larval-weights data/fly-whole-weights.npz

# the live local dashboard: equity, positions, decisions, event feed
uv run python -m fruitfly dashboard
# then open http://127.0.0.1:8765 in a browser

# end-of-run post-mortem: lifespan, cause of death, P&L vs the scoreboard
uv run python -m fruitfly postmortem
```

Tests and lint:

```bash
uv run pytest
uv run ruff check .
```

Project layout: `src/fruitfly/` (the package, src layout), `tests/` (pytest;
`pythonpath = ["src"]`, no editable install), `data/` (downloaded datasets,
gitignored). See `AGENTS.md` for project conventions (seeded determinism,
free-data-only constraint, uv + src layout).

## Where we are

Honest status, v0.7 (2026-09-16):

- **Built and working**: the connectome-derived brain (stripped 27,115 /
  whole 166,691 neurons), the full foraging loop with pessimistic fills,
  sugar/shock learning with per-exit and per-encounter credit, sleep
  consolidation, death/rebirth, kill-and-resume adult runs, the dashboard, and
  seeded determinism.
- **Phase 0 verdict (measured, 10 stripped days, seed 7)**: **Option B —
  advantage gating — won**. Its decision signal was positive and stable, and
  it traded 1.77x more than the reference (still inside the activity band)
  with a win/loss ratio of 1.80. Option A's entry signal was indistinguishable
  from zero. Neither arm cleared the strict statistical entry-signal gate in
  only 10 days — that gate needs Phase C's longer windows.
- **Not demonstrated**: any out-of-sample profit edge. The earlier out-of-sample
  sweep found none; the current goal is an **information-driven active trader**,
  not alpha. Closed-trade profits are still negative across the measured
  out-of-sample cells (the fly gives back its up-market capture at the exit),
  which is exactly what the Phase-B exit upgrades target.
- **Next**: a wider ticker pool (40 training + held-out eval baskets), exit
  upgrades (trailing/ATR stops, an explicit exit-forecast path), multi-seed
  Phase C, and held-out evaluation with paired statistics on a never-seen
  basket.

Want the full detail? Read `DESIGN.md` (the design doc: the fly metaphor,
senses, decisions, lifecycle, decision log) and `plan/TRAINING2-SPEC.md` (the
ratified training objective, phased protocol, and evaluation gates).
