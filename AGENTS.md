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
