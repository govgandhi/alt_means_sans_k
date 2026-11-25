"""The step-0 (Fig. 1) theory line uses Gensim's own retention and noise tables."""

import sys
from pathlib import Path

import pytest

for dependency in ("numpy", "gensim"):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "step0"))
import theory  # noqa: E402


@pytest.mark.parametrize("column", ["A", "SW"])
def test_theory_matches_gensim_subsampling_and_noise(column: str) -> None:
    sample = theory.SETTINGS[column]["sample"]
    exponent = theory.SETTINGS[column]["exponent"]
    rng = np.random.default_rng(0)
    corpus = [[f"n{i}" for i in rng.zipf(1.3, 200) % 300] for _ in range(200)]
    model = Word2Vec(
        sg=1, negative=5, sample=sample, ns_exponent=exponent, min_count=5, workers=1
    )
    model.build_vocab(corpus)
    counts = model.wv.expandos["count"].astype(float)

    keep = theory.keep_probability(counts, sample)
    stored = model.wv.expandos["sample_int"] / (2**32 - 1)
    assert keep.min() < 0.5  # frequent tokens are actually subsampled
    np.testing.assert_allclose(keep, stored, rtol=0, atol=1e-9)

    q = counts**exponent / (counts**exponent).sum()
    table = np.diff(np.r_[0, model.cum_table.astype(np.int64)]) / model.cum_table[-1]
    np.testing.assert_allclose(q, table, rtol=0, atol=1e-8)
