"""Deterministic named random substreams.

Every random decision draws from `substream(seed, "name", ...)`, so adding a new
random consumer never shifts the numbers another consumer sees. Calm and windy
sequences of one tree derive camera jitter and wind phases from the *tree* seed,
which is what makes them paired.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def substream(seed: int, *names) -> np.random.Generator:
    entropy = [int(seed) & 0xFFFFFFFFFFFFFFFF] + [_name_key(str(n)) for n in names]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
