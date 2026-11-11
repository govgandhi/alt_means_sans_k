"""Scientific integrity checks for the negative-sampling diagnostic."""

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
from gensim.models import Word2Vec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import negative_sampling_control as control  # noqa: E402


@pytest.fixture
def walks() -> list[list[int]]:
    return [[2, 0, 2, 1, 2], [2, 0, 2, 0, 2], [1, 2, 0, 2, 2]] * 5


@pytest.fixture
def config() -> control.original.Config:
    return replace(control.original.Config(), nodes=3, dimension=4, epochs=2)


def test_token_counts_include_repeated_visits() -> None:
    np.testing.assert_array_equal(control.token_counts([[0, 1, 0, 1, 0]], 2), [3, 2])
    with pytest.raises(ValueError, match="Every node"):
        control.token_counts([[0, 0]], 2)


def test_blas_negative_one_is_a_valid_score() -> None:
    assert control.blas_check()["passed"] is True


def test_integer_token_table_against_string_reference(walks, config) -> None:
    template = control.make_template(walks, config, 77, 0.0001)
    counts = control.token_counts(walks, 3)
    fixed = control.sampler_arm(template, counts, control.ControlConfig.table_domain)
    reference = Word2Vec(min_count=1, ns_exponent=1, sample=0, sorted_vocab=1)
    reference.build_vocab([[str(node) for node in walk] for walk in walks])
    assert reference.wv.index_to_key == list(map(str, template.wv.index_to_key))
    np.testing.assert_array_equal(fixed.cum_table, reference.cum_table)
    assert not np.array_equal(template.cum_table, fixed.cum_table)
    np.testing.assert_allclose(
        control.table_probabilities(fixed), counts / counts.sum(), atol=1e-9
    )


def test_only_table_changes_and_identical_table_null(walks, config) -> None:
    template = control.make_template(walks, config, 77, 0.0001)
    a = control.sampler_arm(template, None, control.ControlConfig.table_domain)
    b = control.sampler_arm(template, None, control.ControlConfig.table_domain)
    initial = template.wv.vectors.copy()
    corpus_hash = control.digest_walks(walks)
    first = control.train(a, walks)
    second = control.train(b, walks)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a.wv.vectors, b.wv.vectors)
    np.testing.assert_array_equal(a.syn1neg, b.syn1neg)
    np.testing.assert_array_equal(template.wv.vectors, initial)
    assert control.digest_walks(walks) == corpus_hash


def test_corrected_arm_matches_production_training(walks, config) -> None:
    template = control.make_template(walks, config, 77, 0.0001)
    model = control.sampler_arm(
        template,
        control.token_counts(walks, config.nodes),
        control.ControlConfig.table_domain,
    )
    raw, balanced = control.train(model, walks)
    original = control.original.train_embedding(walks, config.nodes, config, 77)
    np.testing.assert_array_equal(raw, original["raw_norm"])
    np.testing.assert_array_equal(balanced, original["balanced_norm"])


def test_raw_probabilities_do_not_depend_on_vocabulary_order(walks, config) -> None:
    a = control.make_template(walks, config, 77, 0.0001)
    b = control.make_template([[0, 0, 0, 2, 1]], config, 77, 0.0001)
    assert a.wv.index_to_key != b.wv.index_to_key
    counts = np.array([11, 7, 29])
    for template in (a, b):
        arm = control.sampler_arm(template, counts, control.ControlConfig.table_domain)
        np.testing.assert_allclose(
            control.table_probabilities(arm), counts / counts.sum(), atol=1e-9
        )


def test_peak_diagnostic_reports_boundary_without_calling_it_turnover() -> None:
    curve = control.curve(np.arange(1, 61), np.arange(1, 61), 6)
    assert curve["peak_bin"] == 6
    assert curve["interior_peak"] is False
    assert sum(curve["node_counts"]) == 60


def test_output_cannot_overwrite_frozen_or_unrelated_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate from frozen"):
        control.run(
            control.ControlConfig(output=Path("results/prospective_validation")), [0]
        )
    marker = tmp_path / "outcomes.jsonl"
    marker.write_text("existing data\n")
    with pytest.raises(FileExistsError, match="nonempty"):
        control.run(control.ControlConfig(output=tmp_path), [0])
    assert marker.read_text() == "existing data\n"
