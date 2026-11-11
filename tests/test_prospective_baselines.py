"""Check leakage, frozen alignment and output safety for post hoc comparisons."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prospective_baselines as audit


def test_predictions_do_not_depend_on_target_outcomes() -> None:
    calibration = [
        {"degree": degree, "threshold": 0.001, "raw_norm": degree**0.5}
        for degree in range(1, 9)
    ]
    targets = [dict(calibration[2]), dict(calibration[5])]
    parameters = {
        "n_estimators": 5,
        "min_samples_leaf": 1,
        "random_state": 123,
        "n_jobs": 1,
    }
    predicted = audit.fit_predict(
        calibration, targets, "degree_threshold", "raw_norm", parameters
    )
    targets[0]["raw_norm"], targets[1]["raw_norm"] = -1000, 1000
    np.testing.assert_array_equal(
        predicted,
        audit.fit_predict(
            calibration, targets, "degree_threshold", "raw_norm", parameters
        ),
    )


def frozen_rows() -> list[dict[str, object]]:
    return [
        {
            "family": "example",
            "replicate": 1,
            "threshold": 0.001,
            "node": node,
            "degree": node + 1,
            "predicted_raw_norm": float(node),
        }
        for node in range(2)
    ]


def test_frozen_alignment_uses_keys_not_row_order() -> None:
    frozen = frozen_rows()
    targets = [dict(row, raw_norm=0.5) for row in reversed(frozen)]
    assert len(audit.verify_target_rows(targets, frozen)) == 2


@pytest.mark.parametrize("field", ["degree", "predicted_raw_norm"])
def test_changed_target_metadata_or_prediction_is_rejected(field: str) -> None:
    frozen = frozen_rows()
    targets = deepcopy(frozen)
    targets[0][field] = 999
    with pytest.raises(ValueError, match="Target differs"):
        audit.verify_target_rows(targets, frozen)


def test_duplicate_frozen_keys_are_rejected() -> None:
    frozen = frozen_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        audit.verify_target_rows(frozen, frozen + [frozen[0]])


@pytest.mark.parametrize(
    "path",
    [
        "results",
        "results/prospective_validation",
        "results/negative_sampling_revalidation/new",
    ],
)
def test_output_cannot_overwrite_sources(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="overlaps"):
        audit.output_directory(audit.Config(root=tmp_path, output=Path(path)))
