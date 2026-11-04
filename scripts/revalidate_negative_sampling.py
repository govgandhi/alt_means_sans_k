"""Revalidate corrected sampling on the original calibration and target conditions.

Inputs: the historical frozen protocol, NetScience graph, pinned corrected runtime,
and analysis/negative_sampling_revalidation_protocol.md. Outputs: corrected
calibration, pre-training frozen predictions, paired target outcomes and summaries.
Run: python3 scripts/revalidate_negative_sampling.py [--output NEW_DIRECTORY]
Stages can also be invoked separately. No historical output is overwritten.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import negative_sampling_control as control
import numpy as np
import prospective_norm_turnover as base
from check_control_gensim import blas_check
from gensim_training import set_negative_table, train_checked
from threadpoolctl import threadpool_limits

ARMS = ("retained", "raw")


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    output: Path = Path("results/negative_sampling_revalidation")
    original: Path = Path("results/prospective_validation")
    graph: Path = Path("are_angles_important/netscience/netscience.gml")
    design: Path = Path("analysis/negative_sampling_revalidation_protocol.md")
    human_protocol: Path = Path("analysis/prospective_validation_protocol.md")
    min_alpha: float = 0.0001  # Original Gensim constructor default.
    bins: int = 6  # Original binned_curve default.


def source_hashes(config: Config) -> dict[str, str]:
    paths = [config.graph, config.design, config.human_protocol]
    paths += [
        Path("scripts") / name
        for name in (
            "revalidate_negative_sampling.py",
            "gensim_training.py",
            "negative_sampling_control.py",
            "prospective_norm_turnover.py",
            "simulate_norm_frequency.py",
            "simulate_exact_center_exposure.py",
            "check_control_gensim.py",
            "build_control_gensim.py",
        )
    ]
    paths += [
        Path("analysis") / f"negative_sampling_control{suffix}_requirements.txt"
        for suffix in ("", "_build")
    ]
    paths += [
        path.relative_to(config.root)
        for path in (config.root / config.original).glob("*.json*")
    ]
    return {str(p): base.sha256_file(config.root / p) for p in sorted(set(paths))}


def verify_manifest(output: Path, name: str) -> None:
    for filename, expected in base.load_json(output / name).items():
        if base.sha256_file(output / filename) != expected:
            raise RuntimeError(f"Artifact hash mismatch: {filename}")


def save_manifest(output: Path, name: str, files: list[str]) -> None:
    base.write_json(output / name, {f: base.sha256_file(output / f) for f in files})


def initialize(config: Config) -> tuple[Path, base.Config]:
    output = (config.root / config.output).resolve()
    for relative in (config.original, Path("results/negative_sampling_control")):
        protected = (config.root / relative).resolve()
        if (
            output == protected
            or output in protected.parents
            or protected in output.parents
        ):
            raise ValueError("Output must be separate from historical results.")
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Nonempty output has no matching plan.")
    historical = config.root / config.original
    base.verify_frozen_state(historical, config.root / config.human_protocol)
    protocol = base.load_json(historical / "protocol.json")
    assert protocol["thresholds"] == list(base.THRESHOLDS)
    assert protocol["calibration_seeds"] == list(base.CALIBRATION_SEEDS)
    assert protocol["model"]["parameters"] == base.MODEL_PARAMETERS
    assert protocol["netscience_sha256"] == base.sha256_file(config.root / config.graph)
    assert protocol["success_criteria"] == base.SUCCESS_CRITERIA
    assert protocol["target_replicates"] == {
        k: list(v) for k, v in base.TARGET_REPLICATES.items()
    }
    training = base.Config(**protocol["config"])
    plan = {
        "status": "retrospective revalidation; original targets previously examined",
        "source_sha256": source_hashes(config),
        "training": asdict(training),
        "arms": list(ARMS),
        "min_alpha": config.min_alpha,
        "bins": config.bins,
        "blas": blas_check(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in sorted(
                importlib.metadata.distributions(), key=lambda d: d.metadata["Name"]
            )
        },
    }
    plan = json.loads(json.dumps(plan))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        if base.load_json(output / "plan.json") != plan:
            raise RuntimeError("Plan changed; choose a fresh output directory.")
    else:
        base.write_json(output / "plan.json", plan)
    return output, training


def verify_sources(config: Config, output: Path) -> None:
    plan = base.load_json(output / "plan.json")
    if source_hashes(config) != plan["source_sha256"] or blas_check() != plan["blas"]:
        raise RuntimeError("Source or numerical runtime changed during the run.")


def prepare(
    config: Config, training: base.Config, family: str, replicate: int
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    if family == "calibration":
        graph = base.configuration_graph(training.nodes, training.attachment, replicate)
    else:
        graph = base.validation_graph(
            family, replicate, training, config.root / config.graph
        )
    degree, walks, counts = base.prepare_walks(graph, training, family, replicate)
    historical_file = (
        "calibration.jsonl" if family == "calibration" else "frozen_predictions.jsonl"
    )
    historical_family = "configuration_model" if family == "calibration" else family
    historical = sorted(
        (
            row
            for row in base.load_jsonl(config.root / config.original / historical_file)
            if row["family"] == historical_family
            and row["replicate"] == replicate
            and row["threshold"] == base.THRESHOLDS[0]
        ),
        key=lambda row: row["node"],
    )
    np.testing.assert_array_equal(
        [row["node"] for row in historical], np.arange(len(degree))
    )
    np.testing.assert_array_equal([row["degree"] for row in historical], degree)
    np.testing.assert_array_equal(counts, control.token_counts(walks, len(degree)))
    assert counts.sum() == len(degree) * training.walks_per_node * training.walk_length
    return degree, walks, counts


def condition_rows(
    family: str,
    replicate: int,
    threshold: float,
    degree: np.ndarray,
    counts: np.ndarray,
    pairs: np.ndarray,
    kl: np.ndarray,
    retained: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base.append_rows(
        rows,
        dataset="corrected_calibration" if family == "calibration" else "target",
        family=family,
        replicate=replicate,
        threshold=threshold,
        degree=degree,
        positive_pairs=pairs,
        context_kl=kl,
        retained_tokens=retained,
    )
    for row, count in zip(rows, counts, strict=True):
        row["raw_tokens"] = int(count)
    return rows


def train_pair(
    config: Config,
    training: base.Config,
    rows: list[dict[str, Any]],
    walks: list[list[int]],
    retained: list[list[int]],
    counts: np.ndarray,
    retained_counts: np.ndarray,
) -> dict[str, Any]:
    first = rows[0]
    local_training = replace(training, nodes=len(counts))
    template = control.make_template(
        retained,
        local_training,
        base.optimizer_seed(first["family"], first["replicate"]),
        config.min_alpha,
    )
    corpus_hash = control.digest_walks(retained)
    initial = control.digest_array(template.wv.vectors)
    tables = {}
    for arm, noise in (("retained", retained_counts), ("raw", counts)):
        model = deepcopy(template)
        set_negative_table(model, noise)
        np.testing.assert_array_equal(model.wv.vectors, template.wv.vectors)
        np.testing.assert_array_equal(model.syn1neg, template.syn1neg)
        np.testing.assert_array_equal(
            model.random.get_state()[1], template.random.get_state()[1]
        )
        assert model.random.get_state()[2:] == template.random.get_state()[2:]
        masses = control.table_probabilities(model)
        np.testing.assert_allclose(
            masses,
            noise / noise.sum(),
            rtol=0,
            atol=1 / control.ControlConfig.table_domain,
        )
        tables[arm] = control.digest_array(model.cum_table)
        train_checked(model, retained)
        indices = [model.wv.key_to_index[node] for node in range(len(counts))]
        center = model.wv.vectors[indices].astype(np.float64)
        context = model.syn1neg[indices].astype(np.float64)
        outcomes = {
            "raw_norm": np.linalg.norm(center, axis=1),
            "balanced_norm": base.balanced_left_norm(center, context),
        }
        for response, values in outcomes.items():
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite trained norms.")
            for row, value in zip(rows, values, strict=True):
                row[f"{arm}_{response}"] = float(value)
        for row, mass in zip(rows, masses, strict=True):
            row[f"{arm}_negative_mass"] = float(mass)
        assert control.digest_walks(retained) == corpus_hash
        assert control.digest_array(template.wv.vectors) == initial
        print(
            f"{first['family']}/{first['replicate']} "
            f"t={first['threshold']:g} {arm} complete",
            flush=True,
        )
    return {
        "family": first["family"],
        "replicate": first["replicate"],
        "threshold": first["threshold"],
        "raw_walk_sha256": control.digest_walks(walks),
        "retained_walk_sha256": corpus_hash,
        "initial_center_sha256": initial,
        "initial_context_sha256": control.digest_array(template.syn1neg),
        "vocabulary_sha256": control.digest_array(
            np.asarray(template.wv.index_to_key, dtype="<i8")
        ),
        "raw_tokens": int(counts.sum()),
        "retained_tokens": int(retained_counts.sum()),
        "positive_pairs_per_epoch": sum(row["realized_positive_pairs"] for row in rows),
        "table_sha256": tables,
    }


def arm_rows(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        copy = dict(row)
        for response in base.RESPONSES:
            if f"{arm}_{response}" in row:
                copy[response] = row[f"{arm}_{response}"]
            if f"predicted_{arm}_{response}" in row:
                copy[f"predicted_{response}"] = row[f"predicted_{arm}_{response}"]
        result.append(copy)
    return result


def calibrate(config: Config, output: Path, training: base.Config) -> None:
    if (output / "calibration_manifest.json").exists():
        verify_manifest(output, "calibration_manifest.json")
        return
    rows, checks = [], []
    for seed in base.CALIBRATION_SEEDS:
        degree, walks, counts = prepare(config, training, "calibration", seed)
        for index, threshold in enumerate(base.THRESHOLDS):
            retained, pairs, kl, retained_counts = base.prepare_condition(
                walks,
                counts,
                threshold,
                index,
                "calibration",
                seed,
            )
            local = condition_rows(
                "calibration",
                seed,
                threshold,
                degree,
                counts,
                pairs,
                kl,
                retained_counts,
            )
            checks.append(
                train_pair(
                    config, training, local, walks, retained, counts, retained_counts
                )
            )
            rows.extend(local)
    verify_sources(config, output)
    base.write_jsonl(output / "calibration.jsonl", rows)
    base.write_json(output / "calibration_checks.json", checks)
    base.write_json(
        output / "calibration_cv.json",
        {arm: base.calibration_cross_validation(arm_rows(rows, arm)) for arm in ARMS},
    )
    save_manifest(
        output,
        "calibration_manifest.json",
        [
            "plan.json",
            "calibration.jsonl",
            "calibration_checks.json",
            "calibration_cv.json",
        ],
    )


def predict(config: Config, output: Path, training: base.Config) -> None:
    verify_manifest(output, "calibration_manifest.json")
    if (output / "freeze_manifest.json").exists():
        verify_manifest(output, "freeze_manifest.json")
        return
    if (output / "outcomes.jsonl").exists():
        raise RuntimeError(
            "Cannot freeze predictions after corrected target outcomes exist."
        )
    calibration = base.load_jsonl(output / "calibration.jsonl")
    models = {arm: base.fit_models(arm_rows(calibration, arm)) for arm in ARMS}
    rows, checks = [], []
    for family, replicates in base.TARGET_REPLICATES.items():
        for seed in replicates:
            degree, walks, counts = prepare(config, training, family, seed)
            for index, threshold in enumerate(base.THRESHOLDS):
                retained, pairs, kl, retained_counts = base.prepare_condition(
                    walks, counts, threshold, index, family, seed
                )
                local = condition_rows(
                    family, seed, threshold, degree, counts, pairs, kl, retained_counts
                )
                for arm in ARMS:
                    predicted = base.predict_models(models[arm], pairs, kl)
                    for response, values in predicted.items():
                        for row, value in zip(local, values, strict=True):
                            row[f"predicted_{arm}_{response}"] = float(value)
                checks.append(
                    {
                        "family": family,
                        "replicate": seed,
                        "threshold": threshold,
                        "retained_walk_sha256": control.digest_walks(retained),
                    }
                )
                rows.extend(local)
            print(f"predicted {family}/{seed}", flush=True)
    verify_sources(config, output)
    base.write_jsonl(output / "frozen_predictions.jsonl", rows)
    base.write_json(output / "prediction_checks.json", checks)
    save_manifest(
        output,
        "freeze_manifest.json",
        [
            "plan.json",
            "calibration_manifest.json",
            "calibration.jsonl",
            "frozen_predictions.jsonl",
            "prediction_checks.json",
        ],
    )
    print(
        f"Froze {len(rows)} target rows for both arms before target training.",
        flush=True,
    )


def sampler_summary(rows: list[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    records = []
    keys = sorted({(r["family"], r["replicate"], r["threshold"]) for r in rows})
    for family, replicate, threshold in keys:
        local = [
            r
            for r in rows
            if (r["family"], r["replicate"], r["threshold"])
            == (family, replicate, threshold)
        ]
        degree = np.array([r["degree"] for r in local])
        for response in base.RESPONSES:
            values = {
                arm: np.array([r[f"{arm}_{response}"] for r in local]) for arm in ARMS
            }
            delta = values["raw"] - values["retained"]
            records.append(
                {
                    "family": family,
                    "replicate": replicate,
                    "threshold": threshold,
                    "response": response,
                    "curves": {
                        arm: control.curve(degree, value, bins)
                        for arm, value in values.items()
                    },
                    "raw_minus_retained_rmse": float(np.sqrt(np.mean(delta**2))),
                    "raw_minus_retained_mean": float(delta.mean()),
                    "sampler_spearman": base.safe_spearman(
                        values["raw"], values["retained"]
                    ),
                }
            )
    return records


def evaluate(config: Config, output: Path, training: base.Config) -> None:
    verify_manifest(output, "calibration_manifest.json")
    verify_manifest(output, "freeze_manifest.json")
    if (output / "completion.json").exists():
        verify_manifest(output, "completion.json")
        return
    frozen = {
        base.prediction_key(r): r
        for r in base.load_jsonl(output / "frozen_predictions.jsonl")
    }
    corpus_hashes = {
        (r["family"], r["replicate"], r["threshold"]): r["retained_walk_sha256"]
        for r in base.load_json(output / "prediction_checks.json")
    }
    rows, checks = [], []
    for family, replicates in base.TARGET_REPLICATES.items():
        for seed in replicates:
            degree, walks, counts = prepare(config, training, family, seed)
            for index, threshold in enumerate(base.THRESHOLDS):
                retained, pairs, kl, retained_counts = base.prepare_condition(
                    walks, counts, threshold, index, family, seed
                )
                assert (
                    control.digest_walks(retained)
                    == corpus_hashes[(family, seed, threshold)]
                )
                local = condition_rows(
                    family, seed, threshold, degree, counts, pairs, kl, retained_counts
                )
                for row in local:
                    expected = frozen[base.prediction_key(row)]
                    for field, value in row.items():
                        if expected[field] != value:
                            raise RuntimeError(f"Frozen feature changed: {field}")
                    row.update(expected)
                checks.append(
                    train_pair(
                        config,
                        training,
                        local,
                        walks,
                        retained,
                        counts,
                        retained_counts,
                    )
                )
                rows.extend(local)
    verify_sources(config, output)
    verify_manifest(output, "freeze_manifest.json")
    base.write_jsonl(output / "outcomes.jsonl", rows)
    base.write_json(output / "target_checks.json", checks)
    base.write_json(
        output / "summary.json",
        {
            "predictor": {
                arm: base.evaluation_summary(arm_rows(rows, arm)) for arm in ARMS
            },
            "sampler_comparisons": sampler_summary(rows, config.bins),
            "calibration_sampler_comparisons": sampler_summary(
                base.load_jsonl(output / "calibration.jsonl"), config.bins
            ),
        },
    )
    save_manifest(
        output,
        "completion.json",
        [
            "plan.json",
            "calibration_manifest.json",
            "calibration.jsonl",
            "calibration_checks.json",
            "calibration_cv.json",
            "freeze_manifest.json",
            "frozen_predictions.jsonl",
            "prediction_checks.json",
            "outcomes.jsonl",
            "target_checks.json",
            "summary.json",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    parser.add_argument(
        "--stage",
        choices=("all", "calibrate", "predict", "evaluate", "verify"),
        default="all",
    )
    args = parser.parse_args()
    config = Config(output=args.output)
    with threadpool_limits(limits=1):
        output, training = initialize(config)
        if args.stage in ("all", "calibrate"):
            calibrate(config, output, training)
        if args.stage in ("all", "predict"):
            predict(config, output, training)
        if args.stage in ("all", "evaluate"):
            evaluate(config, output, training)
        if args.stage in ("all", "verify"):
            verify_sources(config, output)
            verify_manifest(output, "completion.json")
            print("Verified complete corrected revalidation.", flush=True)


if __name__ == "__main__":
    main()
