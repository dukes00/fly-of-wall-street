#!/usr/bin/env python
"""T2: regenerate the connectome caches from the raw feather tables.

Builds the deterministic stripped-chassis cache pair
(``data/connectome/stripped-chassis*.parquet``) by scanning the 1 GB
connectome-weights table in chunks, and prints the population summary the
T2 report quotes. Pass ``--whole-fly`` to also build the whole-fly cache
pair (166,700 neurons, full edge table — slower).

Deterministic: two runs produce identical cache bytes.

Run from the repo root:  uv run python scripts/extract_chassis.py
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fruitfly.connectome import (  # noqa: E402
    CHASSIS_EDGES,
    CHASSIS_NODES,
    DATA_DIR,
    build_stripped_chassis,
    build_whole_fly,
)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="connectome data dir")
    ap.add_argument("--whole-fly", action="store_true", help="also build whole-fly cache")
    args = ap.parse_args()

    chassis = build_stripped_chassis(args.data_dir, cache=True)
    meta = chassis.meta
    print(f"== Stripped chassis: {meta['n_neurons']:,} neurons, "
          f"{meta['n_edges']:,} edges, {meta['total_synapses']:,} synapses ==")
    print("\nPopulation dims:")
    for pop, count in meta["population_dims"].items():
        print(f"  {pop:<14} {count:>7,}")
    print(f"  glomeruli:     {meta['n_glomeruli']:>7}")
    print("\nRegion counts:")
    for region, count in meta["region_counts"].items():
        print(f"  {region:<14} {count:>7,}")
    print("\nNT sign coverage:")
    for sign, count in meta["sign_counts"].items():
        print(f"  sign {sign}: {count:>7,} ({100 * count / meta['n_neurons']:.2f}%)")

    data_dir = args.data_dir
    for name in (CHASSIS_NODES, CHASSIS_EDGES):
        path = os.path.join(data_dir, name)
        print(f"{name}: {os.path.getsize(path):,} bytes  sha256={_sha256(path)}")

    if args.whole_fly:
        whole = build_whole_fly(args.data_dir, cache=True)
        print(f"\n== Whole fly: {whole.meta['n_neurons']:,} neurons, "
              f"{whole.meta['n_edges']:,} edges ==")
    print("\ndone.")


if __name__ == "__main__":
    main()
