"""Guard the no-filter intervention and graph-level uncertainty calculation."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

for dependency in (
    "numpy",
    "gensim",
    "matplotlib",
    "networkx",
    "sklearn",
    "threadpoolctl",
):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_subsampling_origin as experiment  # noqa: E402
import simulate_norm_frequency as original  # noqa: E402
from gensim_training import train_checked  # noqa: E402


@pytest.mark.parametrize("threshold", [0.0, 3e-4, 1e-3, 3e-3])
def test_intervention_matches_original_corrected_trainer(threshold: float) -> None:
    walks = [[2, 0, 2, 1, 2], [2, 0, 2, 0, 2], [1, 2, 0, 2, 2]] * 100
    counts = np.bincount(np.concatenate(walks))
    training = replace(original.Config(), nodes=3, dimension=2, min_count=1, epochs=2)
    config = experiment.Config(training=training)
    baseline = experiment.make_model(walks, config, 0, 7)
    model = experiment.make_model(walks, config, threshold, 7)
    assert experiment.signature(model) == experiment.signature(baseline)
    slots = np.array([model.wv.key_to_index[n] for n in range(3)])
    keep = experiment.keep_probability(counts, threshold)
    np.testing.assert_allclose(
        model.wv.expandos["sample_int"][slots] / (2**32 - 1),
        keep,
        rtol=0,
        atol=1 / (2**32 - 1),
    )
    effective, raw = train_checked(model, walks)
    assert raw == counts.sum() * training.epochs
    if threshold == 0:
        np.testing.assert_array_equal(keep, np.ones(3))
        assert effective == raw
    else:
        assert effective < raw
    result = original.train_word2vec(
        walks, counts, training, threshold, 7 + config.optimizer_seed_offset
    )
    center = model.wv.vectors[slots].astype(np.float64)
    context = model.syn1neg[slots].astype(np.float64)
    np.testing.assert_array_equal(result["raw_norm"], np.linalg.norm(center, axis=1))
    np.testing.assert_array_equal(
        result["balanced_norm"], original.balanced_left_norm(center, context)
    )


def test_uncertainty_weights_graphs_equally_not_nodes() -> None:
    config = experiment.Config(
        training=replace(original.Config(), seeds=2), thresholds=(0,), bins=2
    )
    rows = []
    # First graph has ten times as many nodes. The mean must still weight graphs
    # equally: means 1 and 3 give mean 2 and SE 1 in both degree bins.
    for seed, copies in [(0, 10), (1, 1)]:
        for degree in [1, 2]:
            rows.extend(
                {
                    "seed": seed,
                    "threshold": 0,
                    "degree": degree,
                    "raw_norm": 1 + 2 * seed,
                    "balanced_norm": 1 + 2 * seed,
                }
                for _ in range(copies)
            )
    result = experiment.summarize(rows, config)
    for aggregate in result["across_graphs"]:
        for group in aggregate["bins"]:
            assert group["node_counts_by_seed"] == [10, 1]
            assert group["mean_norm"] == 2
            assert group["standard_error"] == 1
            assert group["graphs"] == 2


def test_rejects_historical_and_unrelated_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected historical"):
        experiment.run(experiment.Config(output=Path("results/gensim_threshold_sweep")))
    (tmp_path / "unrelated.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="Nonempty output"):
        experiment.run(experiment.Config(output=tmp_path))
    assert (tmp_path / "unrelated.txt").read_text() == "preserve me"
