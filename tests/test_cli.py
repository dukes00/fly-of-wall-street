"""Tests for the headless CLI entry point."""

from __future__ import annotations

import pytest

from fruitfly.__main__ import main


def test_help_exits_zero_and_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m fruitfly --help` must exit 0 and print usage."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()
