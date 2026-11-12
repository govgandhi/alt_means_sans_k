"""Corpus for the population theory: every token an independent draw in proportion to
strength.

This is exactly the process the step-0 theory line describes (walks on the expected
graph of the
strength-preserving null, started in proportion to strength). Same shape as the walk
corpora:
10 x n sequences of 80 tokens. Input: data/derived/step0/real_walks_s0.npz (for
strengths).
Output: data/derived/step0/pop_walks_s{k}.npz, seeds 0-4.
Run: python scripts/step0/population_corpus.py --data data/derived/step0
"""

import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
real = np.load(a.data / "real_walks_s0.npz")
strength = real["strength"]
shape = real["walks"].shape
for k in range(5):
    tokens = np.random.default_rng(3000 + k).choice(
        len(strength), size=shape, p=strength / strength.sum()
    )
    np.savez_compressed(
        a.data / f"pop_walks_s{k}.npz",
        walks=tokens.astype(np.int32),
        strength=strength,
        ids=real["ids"],
    )
print("written", shape)
