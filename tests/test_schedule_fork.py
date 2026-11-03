"""Same-checkpoint integrity tests independent of the scientific outcomes."""

from __future__ import annotations

import copy
import sys
import zipfile
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
from window_training import schedule_fork as experiment  # noqa: E402

WALKS = [[2, 0, 2, 1, 2], [2, 0, 2, 0, 2], [1, 2, 0, 2, 2]] * 30


def toy_model(sample: float = 0.0) -> experiment.trajectory.ScheduledWord2Vec:
    experiment.trajectory.require_correct_blas()
    model = experiment.trajectory.ScheduledWord2Vec(
        decay_epochs=5,
        vector_size=3,
        window=1,
        min_count=1,
        sg=1,
        negative=5,
        ns_exponent=1.0,
        workers=1,
        epochs=20,
        seed=17,
        sample=sample,
        alpha=0.025,
        min_alpha=0.0001,
        sorted_vocab=1,
        shrink_windows=False,
        batch_words=30,
    )
    model.build_vocab(WALKS)
    experiment.trajectory.set_negative_table(model)
    return model


def recorder(start: int, rate: float | None = None) -> experiment.Recorder:
    positive = experiment.expected_pairs(np.asarray(WALKS), np.ones(3))
    negative = 5 / 3 * (positive.sum(axis=1)[:, None] - positive)
    return experiment.Recorder(positive, negative, (5, 6, 7, 10, 15, 20), start, rate)


@pytest.mark.parametrize("sample", [0.0, 0.05])
def test_split_floor_is_exactly_uninterrupted_and_fork_is_isolated(
    sample: float,
) -> None:
    uninterrupted, prefix = toy_model(sample), toy_model(sample)
    experiment.train(uninterrupted, WALKS, 20, 0.025, 0.0001)
    experiment.train(prefix, WALKS, 5, 0.025, 0.0001)
    original = experiment.state_fingerprint(prefix)
    floor, reheat = experiment.fork_model(prefix), experiment.fork_model(prefix)
    floor_recorder = recorder(5, 0.0001)
    experiment.train(floor, WALKS, 15, 0.0001, 0.0001, floor_recorder)
    experiment.assert_state(prefix, original)
    experiment.assert_state(reheat, original)
    assert experiment.baseline.signature(floor) == experiment.baseline.signature(
        uninterrupted
    )
    assert floor.epoch_counts == uninterrupted.epoch_counts
    assert [snapshot["epoch"] for snapshot in floor_recorder.snapshots] == [
        5,
        6,
        7,
        10,
        15,
        20,
    ]
    reheat_recorder = recorder(5, 0.0025)
    experiment.train(reheat, WALKS, 15, 0.0025, 0.0025, reheat_recorder)
    experiment.assert_state(prefix, original)
    floor_state, reheat_state = (
        experiment.state_fingerprint(floor),
        experiment.state_fingerprint(reheat),
    )
    for key in ("rng_array_sha256", "rng_metadata", "epoch_counts"):
        assert floor_state[key] == reheat_state[key]
    assert (
        floor_recorder.snapshots[-1]["post_fork_effective_tokens"]
        == reheat_recorder.snapshots[-1]["post_fork_effective_tokens"]
    )
    assert reheat_recorder.snapshots[-1][
        "post_fork_alpha_times_effective_tokens"
    ] == pytest.approx(
        25 * floor_recorder.snapshots[-1]["post_fork_alpha_times_effective_tokens"]
    )
    assert floor_recorder.snapshots[0]["post_fork_raw_tokens"] == 0


def test_observational_callbacks_preserve_entire_training_state() -> None:
    prefix = toy_model()
    experiment.train(prefix, WALKS, 5, 0.025, 0.0001)
    recorded, plain = experiment.fork_model(prefix), experiment.fork_model(prefix)
    experiment.train(recorded, WALKS, 15, 0.0025, 0.0025, recorder(5, 0.0025))
    experiment.train(plain, WALKS, 15, 0.0025, 0.0025)
    assert experiment.state_fingerprint(recorded) == experiment.state_fingerprint(plain)


@pytest.mark.parametrize(
    "mutation", ["rng", "noise", "input", "output", "retention", "count", "lockf"]
)
def test_fork_guard_rejects_reset_or_mutation(mutation: str) -> None:
    model = toy_model()
    experiment.train(model, WALKS, 5, 0.025, 0.0001)
    expected = experiment.state_fingerprint(model)
    bad = experiment.fork_model(model)
    if mutation == "rng":
        bad.random = np.random.RandomState(bad.seed)
    elif mutation == "noise":
        bad.cum_table[0] += 1
    elif mutation == "input":
        bad.wv.vectors[0, 0] += 1
    elif mutation == "output":
        bad.syn1neg[0, 0] += 1
    elif mutation == "retention":
        bad.wv.expandos["sample_int"][0] -= 1
    elif mutation == "count":
        bad.wv.expandos["count"][0] += 1
    else:
        bad.wv.vectors_lockf[0] += 1
    with pytest.raises(AssertionError, match="Training state changed"):
        experiment.assert_state(bad, expected)
    experiment.assert_state(model, expected)


def test_diagnostic_guard_rejects_rng_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = toy_model()
    callback = recorder(5, 0.0001)
    original_evaluate = experiment.evaluate

    def consuming_evaluate(*args: object) -> tuple[dict[str, object], np.ndarray]:
        model.random.random_sample()
        return original_evaluate(*args)

    monkeypatch.setattr(experiment, "evaluate", consuming_evaluate)
    with pytest.raises(AssertionError, match="Training state changed"):
        callback.capture(model)


def test_factor_archive_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    center = np.array([[1.0, 2.0], [-3.0, 4.0]])
    context = center / 7
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    experiment.write_factors(first, center, context)
    experiment.write_factors(second, center, context)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["center"], center)
        np.testing.assert_array_equal(data["context"], context)
    with zipfile.ZipFile(first) as archive:
        assert all(
            item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist()
        )


def test_reference_bridge_and_resume_reject_changed_arrays(tmp_path: Path) -> None:
    original = {"epoch": 5, "input_sha256": "a", "output_sha256": "b"}
    experiment.verify_bridge(original, copy.deepcopy(original))
    with pytest.raises(AssertionError, match="bridge failed"):
        experiment.verify_bridge({**original, "input_sha256": "changed"}, original)
    artifact = tmp_path / "checkpoint.json"
    experiment.baseline.write_json(artifact, original)
    experiment.baseline.write_json(
        tmp_path / "completion.json",
        {artifact.name: experiment.baseline.sha256(artifact)},
    )
    experiment.verify_manifest(tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="Artifact changed"):
        experiment.verify_manifest(tmp_path)
