# Fruit Fly of Wall Street

**166,691 neurons. Zero emotions.**

A connectome-driven, emotion-free market simulator: run a simulated Drosophila
brain against real financial data. No sentiment analysis, no hype — just wiring
diagrams and prices, with full seeded determinism (same seed → byte-identical
output).

## Quickstart

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# create venv and install dependencies
uv sync

# CLI overview
uv run python -m fruitfly --help

# run tests / lint
uv run pytest
uv run ruff check .
```

## Layout

- `src/fruitfly/` — the package (src layout)
- `tests/` — pytest suite (`pythonpath = ["src"]`, no editable install needed)
- `data/` — downloaded datasets (gitignored, free sources only)

See `AGENTS.md` for project conventions and `DESIGN.md` for the architecture.
