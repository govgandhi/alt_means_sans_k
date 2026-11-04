"""Regression checks for preserved history and prediction-freeze integrity."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prospective_norm_turnover as base  # noqa: E402
import reproduce_calibration_control as historical  # noqa: E402
import revalidate_negative_sampling as validation  # noqa: E402


def test_historical_snapshot_restores_every_recorded_input(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    historical.prepare(root, tmp_path)
    plan = base.load_json(root / "results/negative_sampling_control/plan.json")
    for name, expected in plan["source_sha256"].items():
        assert base.sha256_file(tmp_path / name) == expected


def test_revalidation_refuses_historical_output() -> None:
    for path in (
        "results",
        "results/prospective_validation",
        "results/negative_sampling_control/child",
    ):
        with pytest.raises(ValueError, match="separate"):
            validation.initialize(validation.Config(output=Path(path)))


def test_freeze_detects_tampering(tmp_path: Path) -> None:
    predictions = tmp_path / "frozen_predictions.jsonl"
    predictions.write_text('{"prediction":1}\n')
    validation.save_manifest(tmp_path, "freeze_manifest.json", [predictions.name])
    validation.verify_manifest(tmp_path, "freeze_manifest.json")
    predictions.write_text('{"prediction":2}\n')
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validation.verify_manifest(tmp_path, "freeze_manifest.json")


def test_prediction_cannot_be_frozen_after_target_outcomes(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text("")
    validation.save_manifest(tmp_path, "calibration_manifest.json", [calibration.name])
    (tmp_path / "outcomes.jsonl").write_text("existing targets\n")
    with pytest.raises(RuntimeError, match="after corrected target outcomes"):
        validation.predict(validation.Config(), tmp_path, base.Config())


def test_force_cannot_destroy_calibration_after_freeze(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text("preserved calibration\n")
    (tmp_path / "freeze_manifest.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="after predictions are frozen"):
        base.calibrate(base.Config(), tmp_path, force=True)
    assert calibration.read_text() == "preserved calibration\n"
