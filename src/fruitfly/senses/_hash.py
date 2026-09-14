"""Deterministic string hashing for sensory encoders.

Every derived quantity in the senses package (direction preference, glomerular
profile, target-node gains, ...) comes from a BLAKE2b hash of an ASCII key.
BLAKE2b is platform-stable and unseeded-per-process, so two runs with the same
inputs produce byte-identical outputs — the determinism hard gate. Python's
builtin ``hash()`` must never be used here (per-process randomization).
"""

from __future__ import annotations

import hashlib
import struct

_MASK64 = float(2**64)


def stable_u64(key: str) -> int:
    """Uniform uint64 from an ASCII key (BLAKE2b-64, big-endian)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return struct.unpack(">Q", digest)[0]


def stable_uniform(key: str, lo: float, hi: float) -> float:
    """Uniform float in ``[lo, hi)`` from an ASCII key."""
    return lo + (hi - lo) * (stable_u64(key) / _MASK64)
