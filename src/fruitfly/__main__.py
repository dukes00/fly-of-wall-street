"""Fruit Fly of Wall Street.

A connectome-driven, emotion-free approach to markets: run a simulated
Drosophila brain (166,691 neurons) against real financial data, using
seeds for full determinism instead of sentiment or hype.

This module is the headless CLI entry point. Runnable artifacts register
their subcommands here via the registry below.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from fruitfly import __version__

if __name__ == "__main__":  # runpy (`python -m fruitfly`): allow the
    # circular import fruitfly.data -> fruitfly.__main__ below.
    sys.modules.setdefault("fruitfly.__main__", sys.modules[__name__])


CommandBuilder = Callable[[argparse._SubParsersAction], None]
#: Registry of subcommands. Modules register builders here at import time.
_COMMANDS: dict[str, CommandBuilder] = {}


def register_command(name: str, builder: CommandBuilder) -> None:
    """Register a subcommand builder under ``name``."""
    _COMMANDS[name] = builder


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with all registered subcommands."""
    import fruitfly.data  # noqa: F401  # registers the fetch-data command
    import fruitfly.loop  # noqa: F401  # registers the backtest command

    parser = argparse.ArgumentParser(
        prog="python -m fruitfly",
        description="Fruit Fly of Wall Street: 166,691 neurons. Zero emotions.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for builder in _COMMANDS.values():
        builder(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
