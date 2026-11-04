"""Independent enumeration checks and exact uninterrupted-training bridge tests."""

from __future__ import annotations

import itertools
import sys
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
from gensim.models import Word2Vec  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "training_trajectory")
)
import run as trajectory  # noqa: E402
from gensim_training import require_correct_blas, set_negative_table  # noqa: E402
from metrics import evaluate, expected_pairs, proposal_probabilities  # noqa: E402


def test_expected_pairs_against_exhaustive_deletion_masks() -> None:
    walks = np.array([[0, 1, 0, 2], [2, 2, 1, 0]])
    retention = np.array([0.3, 0.8, 0.6])
    reference = np.zeros((3, 3))
    for walk in walks:
        for mask in itertools.product((False, True), repeat=len(walk)):
            probability = np.prod(
                [
                    retention[n] if kept else 1 - retention[n]
                    for n, kept in zip(walk, mask, strict=True)
                ]
            )
            retained = walk[np.array(mask)]
            for left, right in zip(retained, retained[1:]):
                reference[left, right] += probability
                reference[right, left] += probability
    np.testing.assert_allclose(expected_pairs(walks, retention), reference, atol=1e-15)
    assert reference[0, 0] > 0 and reference[2, 2] > 0
    full = expected_pairs(walks, np.ones(3))
    assert full.sum() == 12


@pytest.mark.parametrize("cdf", [[2, 4, 7], [0, 0, 7], [1, 6, 7]])
def test_proposal_mass_matches_every_small_integer_draw(cdf: list[int]) -> None:
    table = np.array(cdf)
    slots = [
        np.searchsorted(table, draw % table[-1], side="left") for draw in range(16)
    ]
    expected = np.bincount(slots, minlength=3) / 16
    np.testing.assert_array_equal(proposal_probabilities(table, bits=4), expected)


def test_expected_loss_includes_collision_skip_and_row_normalization() -> None:
    positive = np.array([[2.0, 1.0], [1.0, 3.0]])
    q = np.array([0.3, 0.7])
    negatives = 5 * q[None, :] * (positive.sum(axis=1)[:, None] - positive)
    center, context = np.array([[1.2], [-0.6]]), np.array([[0.4], [-0.3]])
    actual, scores = evaluate(center, context, positive, negatives, None)
    rows = np.zeros(2)
    for i, j in itertools.product(range(2), repeat=2):
        cost = np.logaddexp(0, -scores[i, j])
        for n in range(2):
            if n != j:
                cost += 5 * q[n] * np.logaddexp(0, scores[i, n])
        rows[i] += positive[i, j] * cost
    np.testing.assert_allclose(actual["node_loss"], rows / positive.sum(axis=1))
    assert actual["expected_loss"] == pytest.approx(rows.sum() / positive.sum())


def toy_model(scheduled: bool, alpha: float = 0.025, end: float = 0.0001) -> Word2Vec:
    kwargs = dict(
        vector_size=2,
        window=1,
        min_count=1,
        sg=1,
        negative=5,
        ns_exponent=1.0,
        workers=1,
        epochs=5,
        seed=17,
        sample=0.05,
        alpha=alpha,
        min_alpha=end,
        sorted_vocab=1,
        shrink_windows=False,
        batch_words=30,
    )
    if scheduled:
        return trajectory.ScheduledWord2Vec(decay_epochs=5, **kwargs)
    return Word2Vec(**kwargs)


def test_exact_prefix_and_observational_callbacks() -> None:
    require_correct_blas()
    walks = [[2, 0, 2, 1, 2], [2, 0, 2, 0, 2], [1, 2, 0, 2, 2]] * 30
    models = [toy_model(False), toy_model(True), toy_model(True)]
    for model in models:
        model.build_vocab(walks)
        set_negative_table(model)
    original, recorded, unrecorded = models
    original.train(
        walks, total_examples=len(walks), epochs=5, start_alpha=0.025, end_alpha=0.0001
    )
    slots = np.array([original.wv.key_to_index[n] for n in range(3)])
    reference = {
        "input_sha256": trajectory.digest_array(
            original.wv.vectors[slots].astype(float)
        ),
        "output_sha256": trajectory.digest_array(original.syn1neg[slots].astype(float)),
    }
    positive = expected_pairs(np.asarray(walks), np.ones(3))
    negative = np.ones((3, 3))
    callback = trajectory.Recorder(trajectory.Config(), positive, negative, reference)
    for model, callbacks in [(recorded, [callback]), (unrecorded, [])]:
        model.train(
            walks,
            total_examples=len(walks),
            epochs=20,
            start_alpha=0.025,
            end_alpha=0.0001,
            callbacks=callbacks,
        )
    assert [s["epoch"] for s in callback.snapshots] == [0, 1, 2, 5, 10, 20]
    assert callback.snapshots[3]["baseline_bridge_passed"]
    np.testing.assert_array_equal(recorded.wv.vectors, unrecorded.wv.vectors)
    np.testing.assert_array_equal(recorded.syn1neg, unrecorded.syn1neg)
    assert trajectory.baseline.signature(recorded) == trajectory.baseline.signature(
        unrecorded
    )
    assert len(recorded.epoch_counts) == 20


def test_schedule_does_not_depend_on_planned_horizon() -> None:
    model = toy_model(True)
    for epoch in (0, 1, 4, 5, 19):
        model.epochs = 20
        before = model._get_next_alpha(0.3, epoch)
        model.epochs = 100
        assert model._get_next_alpha(0.3, epoch) == before
        if epoch >= 5:
            assert before == 0.0001
    constant = toy_model(True, 0.0025, 0.0025)
    for epoch in range(20):
        assert constant._get_next_alpha(0.75, epoch) == 0.0025


def test_report_uses_graph_not_node_replication() -> None:
    import report

    graphs = [{"degree": [3, 6] * 10}, {"degree": [3, 6]}]
    snapshots = [{"raw_norm": [1.0] * 20}, {"raw_norm": [3.0] * 2}]
    bins = report.aggregate(graphs, snapshots, "raw_norm")
    for group in bins:
        assert group["node_counts_by_seed"] == [10, 1]
        assert group["mean"] == 2
        assert group["standard_error"] == 1
        assert group["graphs"] == 2
