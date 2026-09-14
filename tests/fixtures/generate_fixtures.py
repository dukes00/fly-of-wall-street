#!/usr/bin/env python
"""Regenerate the T4 senses golden fixtures (run from the repo root):

    uv run python tests/fixtures/generate_fixtures.py

Writes ``vision_golden.npz``: the encode_vision outputs (float64, aligned to
the synthetic chassis node order) for the three synthetic charts, stored in a
fixed key order. np.savez is deterministic in array contents and member
order; the test suite asserts byte-identical *array* reproduction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts  # noqa: E402

from fruitfly.senses import encode_vision  # noqa: E402

OUT = Path(__file__).resolve().parent / "vision_golden.npz"


def main() -> None:
    chassis = charts.make_chassis()
    arrays = {
        name: encode_vision(chart(), chassis)
        for name, chart in (("chart_flat", charts.chart_flat),
                            ("chart_up", charts.chart_up),
                            ("chart_crash", charts.chart_crash))
    }
    np.savez(OUT, **arrays)  # fixed member order = insertion order
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
