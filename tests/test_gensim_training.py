"""Regression checks against string-token Gensim, with no experiment reruns."""

from __future__ import annotations

import sys
from collections.abc import Iterator
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
from gensim.models import Word2Vec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gensim_training as training  # noqa: E402
import prospective_norm_turnover as prospective  # noqa: E402
import simulate_norm_frequency as simulation  # noqa: E402


@pytest.fixture(autouse=True)
def reset_blas_cache() -> Iterator[None]:
    training.require_correct_blas.cache_clear()
    yield
    training.require_correct_blas.cache_clear()


@pytest.fixture
def walks() -> list[list[int]]:
    return [[2, 0, 2, 1, 2], [2, 0, 2, 0, 2], [1, 2, 0, 2, 2]] * 5


def build_model(
    sentences: list[list[int]] | list[list[str]],
    sample: float = 0.0,
    exponent: float = 1.0,
) -> Word2Vec:
    model = Word2Vec(
        vector_size=4,
        window=1,
        min_count=1,
        sg=1,
        negative=5,
        ns_exponent=exponent,
        workers=1,
        epochs=3,
        seed=77,
        sample=sample,
        alpha=0.025,
        sorted_vocab=1,
        shrink_windows=False,
    )
    model.build_vocab(sentences)
    return model


def string_reference(
    walks: list[list[int]],
    sample: float = 0.0,
    raw_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sentences = [[str(node) for node in walk] for walk in walks]
    model = build_model(sentences, sample)
    if raw_counts is not None:
        original_counts = model.wv.expandos["count"].copy()
        for node, count in enumerate(raw_counts):
            model.wv.set_vecattr(str(node), "count", int(count))
        model.make_cum_table()
        model.wv.expandos["count"][:] = original_counts
    model.train(
        sentences,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=model.epochs,
        start_alpha=model.alpha,
        end_alpha=model.min_alpha,
    )
    nodes = range(len(model.wv))
    center = np.stack([model.wv[str(node)] for node in nodes]).astype(np.float64)
    context = np.stack(
        [model.syn1neg[model.wv.key_to_index[str(node)]] for node in nodes]
    ).astype(np.float64)
    return center, context


def assert_node_ordered_norms(
    result: dict[str, np.ndarray | float],
    center: np.ndarray,
    context: np.ndarray,
) -> None:
    expected_raw = np.linalg.norm(center, axis=1)
    left, singular, _ = np.linalg.svd(center @ context.T, full_matrices=False)
    expected_balanced = np.linalg.norm(left * np.sqrt(singular), axis=1)
    assert np.ptp(expected_raw) > 0
    np.testing.assert_array_equal(result["raw_norm"], expected_raw)
    np.testing.assert_allclose(
        result["balanced_norm"], expected_balanced, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("sample", [0.0, 0.05])
def test_integer_training_matches_string_reference(
    walks: list[list[int]], sample: float
) -> None:
    strings = [[str(node) for node in walk] for walk in walks]
    integer = build_model(walks, sample)
    reference = build_model(strings, sample)
    assert integer.wv.index_to_key == [2, 0, 1]
    assert reference.wv.index_to_key == ["2", "0", "1"]
    initial_center = integer.wv.vectors.copy()
    initial_context = integer.syn1neg.copy()
    counts = integer.wv.expandos["count"].copy()
    retention = integer.wv.expandos["sample_int"].copy()
    rng_state = integer.random.get_state()

    training.set_negative_table(integer)

    np.testing.assert_array_equal(integer.cum_table, reference.cum_table)
    np.testing.assert_array_equal(integer.wv.vectors, initial_center)
    np.testing.assert_array_equal(integer.syn1neg, initial_context)
    np.testing.assert_array_equal(integer.wv.expandos["count"], counts)
    np.testing.assert_array_equal(integer.wv.expandos["sample_int"], retention)
    np.testing.assert_array_equal(retention, reference.wv.expandos["sample_int"])
    new_rng_state = integer.random.get_state()
    assert rng_state[0] == new_rng_state[0]
    np.testing.assert_array_equal(rng_state[1], new_rng_state[1])
    assert rng_state[2:] == new_rng_state[2:]

    actual_counts = training.train_checked(integer, walks)
    reference_counts = reference.train(
        strings,
        total_examples=reference.corpus_count,
        total_words=reference.corpus_total_words,
        epochs=reference.epochs,
        start_alpha=reference.alpha,
        end_alpha=reference.min_alpha,
    )
    assert actual_counts == reference_counts
    np.testing.assert_array_equal(integer.wv.vectors, reference.wv.vectors)
    np.testing.assert_array_equal(integer.syn1neg, reference.syn1neg)
    for node in range(3):
        slot = integer.wv.key_to_index[node]
        reference_slot = reference.wv.key_to_index[str(node)]
        np.testing.assert_array_equal(integer.wv.vectors[slot], reference.wv[str(node)])
        np.testing.assert_array_equal(
            integer.syn1neg[slot], reference.syn1neg[reference_slot]
        )


@pytest.mark.parametrize("exponent", [1.0, 0.75])
@pytest.mark.parametrize(
    "corpus", [[[2] * 8 + [0] * 4 + [1] * 2], [[0] * 8 + [1] * 4 + [2] * 2]]
)
def test_supplied_raw_counts_follow_node_ids(
    corpus: list[list[int]], exponent: float
) -> None:
    raw_counts = np.array([11, 7, 29])
    model = build_model(corpus, exponent=exponent)
    reference = build_model(
        [[str(node) for node in walk] for walk in corpus], exponent=exponent
    )
    for node, count in enumerate(raw_counts):
        reference.wv.set_vecattr(str(node), "count", int(count))
    reference.make_cum_table()

    training.set_negative_table(model, raw_counts)

    np.testing.assert_array_equal(model.cum_table, reference.cum_table)
    masses = np.diff(np.r_[0, model.cum_table.astype(np.int64)])
    by_node = np.empty(3)
    by_node[np.array(model.wv.index_to_key)] = masses / model.cum_table[-1]
    expected = raw_counts.astype(float) ** exponent
    np.testing.assert_allclose(
        by_node, expected / expected.sum(), rtol=0, atol=1 / training.TABLE_DOMAIN
    )


@pytest.mark.parametrize(
    ("total_walks", "walk_length", "walk", "expected"),
    [(1, 3, [0, 1, 0], [2, 1]), (3, 5, [0, 1, 0, 1, 0], [9, 6])],
)
def test_walk_counts_include_every_repeated_visit(
    total_walks: int, walk_length: int, walk: list[int], expected: list[int]
) -> None:
    walks, counts = simulation.sample_walks(
        [np.array([1]), np.array([0])],
        np.array([1.0, 0.0]),
        total_walks=total_walks,
        walk_length=walk_length,
        rng=np.random.default_rng(123),
    )
    assert walks == [walk] * total_walks
    np.testing.assert_array_equal(counts, expected)
    assert counts.sum() == total_walks * walk_length


def test_blas_failure_prevents_training(
    walks: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_model(walks)
    initial = model.wv.vectors.copy()
    called = False

    def broken_blas() -> None:
        raise RuntimeError("invalid score -1")

    def unexpected_train(*args: object, **kwargs: object) -> tuple[int, int]:
        nonlocal called
        called = True
        return 0, 0

    monkeypatch.setattr(training, "blas_check", broken_blas)
    monkeypatch.setattr(model, "train", unexpected_train)
    with pytest.raises(RuntimeError, match="build_control_gensim.py"):
        training.train_checked(model, walks)
    assert not called
    np.testing.assert_array_equal(model.wv.vectors, initial)
    assert training.require_correct_blas.cache_info().currsize == 0


@pytest.mark.parametrize("sample", [0.0, 0.05])
def test_norm_frequency_wrapper_uses_corrected_table(
    walks: list[list[int]], sample: float
) -> None:
    config = replace(simulation.Config(), nodes=3, dimension=4, epochs=3, min_count=1)
    counts = np.bincount(np.concatenate(walks), minlength=3)
    expected_center, expected_context = string_reference(walks, sample)
    result = simulation.train_word2vec(walks, counts, config, sample, 77)
    assert_node_ordered_norms(result, expected_center, expected_context)


@pytest.mark.parametrize("raw_counts", [None, np.array([11, 7, 29])])
def test_prospective_wrapper_uses_requested_counts(
    walks: list[list[int]], raw_counts: np.ndarray | None
) -> None:
    config = replace(prospective.Config(), nodes=3, dimension=4, epochs=3)
    expected_center, expected_context = string_reference(walks, raw_counts=raw_counts)
    result = prospective.train_embedding(
        walks, 3, config, 77, negative_counts=raw_counts
    )
    assert_node_ordered_norms(result, expected_center, expected_context)
