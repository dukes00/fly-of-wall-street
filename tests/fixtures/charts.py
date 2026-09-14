"""Shared builders for the T4 senses tests and golden fixtures.

Fully offline: a synthetic chassis with the exact real population dims
(27,115 nodes, reports/t2-connectome.md) and three synthetic 48-bar OHLC
charts (flat, steady uptrend, crash spike). Pure and deterministic — the
golden fixtures in this directory are byte-reproducible from these builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fruitfly.connectome import Chassis

#: Exact stripped-chassis population dims (reports/t2-connectome.md).
POPULATION_DIMS = {
    "photoreceptor": 6_091,
    "T4": 6_865,
    "T5": 6_720,
    "LC-looming": 1_239,
    "uPN": 391,
    "KC": 4_064,
    "MBON": 97,
    "PAM": 316,
    "PPL1": 16,
    "DN": 1_316,
}
assert sum(POPULATION_DIMS.values()) == 27_115

_N_GLOMERULI = 88  # "" + 87 named channels
_N_BARS = 48

_SIGN = {"photoreceptor": 0, "T4": 1, "T5": 1, "LC-looming": 1, "uPN": 1,
         "KC": 1, "MBON": 1, "PAM": 0, "PPL1": 0, "DN": 0}
_REGION = {"photoreceptor": "retina", "T4": "medulla", "T5": "lobula",
           "LC-looming": "lobula", "uPN": "antennal-lobe", "KC": "mushroom-body",
           "MBON": "mushroom-body", "PAM": "protocerebrum", "PPL1": "protocerebrum",
           "DN": "brain"}
_NT = {"photoreceptor": ("histamine", 0), "T4": ("acetylcholine", 1),
       "T5": ("acetylcholine", 1), "LC-looming": ("acetylcholine", 1),
       "uPN": ("acetylcholine", 1), "KC": ("acetylcholine", 1),
       "MBON": ("glutamate", -1), "PAM": ("dopamine", 0), "PPL1": ("dopamine", 0),
       "DN": ("octopamine", 0)}

_LINEAGES = ("adPN", "lPN", "vPN", "lvPN")


def glomeruli() -> list[str]:
    """88 channel names: "" (null-type uPNs) + 87 synthetic named channels."""
    return [""] + [f"GLM{i:02d}" for i in range(1, _N_GLOMERULI)]


def make_chassis() -> Chassis:
    """Synthetic chassis mirroring the real stripped-chassis structure.

    Nodes sorted by bodyId (RangeIndex, like the real loader), empty CSR
    adjacency (the encoders do not read edges), meta with the population dims
    and the 88-glomeruli channel list.
    """
    glist = glomeruli()
    rows = []
    body_id = 10_000
    for pop, count in POPULATION_DIMS.items():
        nt, sign = _NT[pop]
        for k in range(count):
            if pop == "uPN":
                ch = k % _N_GLOMERULI
                name = "" if ch == 0 else glist[ch]
                ntype = f"{name}_{_LINEAGES[k % 4]}" if name else ""
            elif pop == "T4":
                ntype = f"T4{'abcd'[k % 4]}"
            elif pop == "T5":
                ntype = f"T5{'abcd'[k % 4]}"
            elif pop == "LC-looming":
                ntype = ("LC4", "LC21", "LC10a", "LC10b")[k % 4]
            else:
                ntype = f"{pop}{k}"
            rows.append(
                {
                    "bodyId": body_id,
                    "type": ntype,
                    "instance": f"{ntype}_{body_id}",
                    "somaSide": "L" if k % 2 == 0 else "R",
                    "population": pop,
                    "region": _REGION[pop],
                    "neurotransmitter": nt,
                    "sign": sign,
                }
            )
            body_id += 1
    nodes = pd.DataFrame(rows)
    meta = {
        "kind": "synthetic-for-tests",
        "n_neurons": len(nodes),
        "population_dims": dict(POPULATION_DIMS),
        "n_glomeruli": _N_GLOMERULI,
        "glomeruli": glist,
    }
    adj = sp.csr_matrix((len(nodes), len(nodes)), dtype=np.int64)
    return Chassis(nodes=nodes, adj=adj, meta=meta)


def chart_flat(n_bars: int = _N_BARS) -> pd.DataFrame:
    """Perfectly flat window: every price 100.0 — zero drive everywhere."""
    flat = np.full(n_bars, 100.0)
    return pd.DataFrame({"open": flat, "high": flat, "low": flat, "close": flat})


def chart_up(n_bars: int = _N_BARS) -> pd.DataFrame:
    """Steady bullish drift: +0.5 per bar, small consistent ranges."""
    close = 100.0 + 0.5 * np.arange(n_bars, dtype=np.float64)
    open_ = close - 0.5
    return pd.DataFrame(
        {"open": open_, "high": close + 0.2, "low": open_ - 0.2, "close": close}
    )


def chart_crash(n_bars: int = _N_BARS) -> pd.DataFrame:
    """Quiet range-bound tape, then a vertical crash spike on the last bar."""
    open_ = np.full(n_bars, 100.0)
    close = np.full(n_bars, 99.9)
    high = np.full(n_bars, 100.1)
    low = np.full(n_bars, 99.8)
    # Crash bar: opens at the tape, closes far below on a huge range.
    open_[-1], close[-1] = 99.9, 87.0
    high[-1], low[-1] = 100.2, 86.5
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
