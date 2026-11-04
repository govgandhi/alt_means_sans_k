"""Reproduce issue 40 without changing frozen data or training embeddings.

Use the pinned PR35 environment documented in
analysis/negative_sampling_control_requirements.txt. Run:
python3 scripts/prospective_baselines.py [--output NEW_DIRECTORY]
Inputs: original frozen validation and corrected negative-sampling revalidation.
Outputs: results/prospective_baselines/{summary.json,table.tex,
balanced_table.tex,README.md}, with
all nine curve scores,
45 replicate-specific scores, source hashes and dependency versions. All model
settings come from the original frozen protocol; comparisons are post hoc.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    original: Path = Path("results/prospective_validation")
    corrected: Path = Path("results/negative_sampling_revalidation")
    output: Path = Path("results/prospective_baselines")
    requirements: Path = Path("analysis/negative_sampling_control_requirements.txt")
    protocol: Path = Path("analysis/prospective_validation_protocol.md")
    methods: Path = Path("analysis/prospective_baselines.md")


def features(rows: list[dict[str, Any]], kind: str) -> np.ndarray:
    columns = {
        "pairs_kl": ("realized_positive_pairs", "context_kl"),
        "degree_threshold": ("degree", "threshold"),
        "pairs": ("realized_positive_pairs",),
        "kl": ("context_kl",),
        "degree": ("degree",),
    }[kind]
    return np.column_stack(
        [
            np.asarray([row[column] for row in rows])
            if column == "context_kl"
            else np.log([row[column] for row in rows])
            for column in columns
        ]
    )


def fit_predict(
    calibration: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    kind: str,
    response: str,
    parameters: dict[str, Any],
) -> np.ndarray:
    model = ExtraTreesRegressor(**parameters)
    model.fit(features(calibration, kind), [row[response] for row in calibration])
    return model.predict(features(targets, kind))


def verify_target_rows(
    targets: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> dict[tuple[str, int, float, int], dict[str, Any]]:
    import prospective_norm_turnover as base

    keys = [base.prediction_key(row) for row in targets]
    frozen_by_key = {base.prediction_key(row): row for row in frozen}
    if len(set(keys)) != len(keys) or len(frozen_by_key) != len(frozen):
        raise ValueError("Duplicate target or frozen prediction key")
    if set(keys) != set(frozen_by_key):
        raise ValueError("Target keys differ from frozen predictions")
    for row in targets:
        expected = frozen_by_key[base.prediction_key(row)]
        for field, value in expected.items():
            if field != "dataset" and row.get(field) != value:
                raise ValueError(f"Target differs from frozen record: {field}")
    return frozen_by_key


def output_directory(config: Config) -> Path:
    output = (config.root / config.output).resolve()
    for source in (config.original, config.corrected):
        protected = (config.root / source).resolve()
        if (
            output == protected
            or output in protected.parents
            or protected in output.parents
        ):
            raise ValueError("Output overlaps frozen source data")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    args = parser.parse_args()
    config = Config(output=args.output)
    output = output_directory(config)
    sys.path.insert(0, str(config.root / "scripts"))
    import prospective_norm_turnover as base

    original = config.root / config.original
    corrected = config.root / config.corrected
    packages = {}
    for line in (config.root / config.requirements).read_text().splitlines():
        name, expected = line.split("==")
        version = importlib.metadata.version(name)
        if version != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {version}")
        packages[name] = version
    base.verify_frozen_state(original, config.root / config.protocol)
    historical = base.load_json(original / "summary.json")
    assert (
        base.sha256_file(original / "outcomes.jsonl") == historical["outcomes_sha256"]
    )
    for name, digest in base.load_json(corrected / "completion.json").items():
        assert base.sha256_file(corrected / name) == digest, name
    parameters = base.load_json(original / "protocol.json")["model"]["parameters"]
    assert parameters == base.MODEL_PARAMETERS
    report: dict[str, Any] = {
        "status": (
            "post hoc; original frozen results unchanged; corrected arms retrospective"
        ),
        "packages": packages,
        "model_parameters": parameters,
        "source_sha256": {},
        "datasets": {},
    }

    for label, directory, arm in (
        ("historical", original, None),
        ("corrected_retained_noise", corrected, "retained"),
        ("corrected_raw_noise", corrected, "raw"),
    ):
        calibration = base.load_jsonl(directory / "calibration.jsonl")
        targets = base.load_jsonl(directory / "outcomes.jsonl")
        frozen = base.load_jsonl(directory / "frozen_predictions.jsonl")
        for filename in (
            "calibration.jsonl",
            "outcomes.jsonl",
            "frozen_predictions.jsonl",
        ):
            report["source_sha256"][f"{label}/{filename}"] = base.sha256_file(
                directory / filename
            )
        target_keys = [base.prediction_key(row) for row in targets]
        frozen_by_key = verify_target_rows(targets, frozen)
        assert not set(row["family"] for row in calibration) & set(
            row["family"] for row in targets
        )
        comparisons = {}
        for kind in ("pairs_kl", "degree_threshold", "pairs", "kl", "degree"):
            rows = [dict(row) for row in targets]
            refit_difference = {}
            for response in base.RESPONSES:
                field = f"{arm}_{response}" if arm else response
                predicted_field = f"predicted_{field}"
                prediction = fit_predict(calibration, targets, kind, field, parameters)
                if kind == "pairs_kl":
                    saved = np.array(
                        [frozen_by_key[key][predicted_field] for key in target_keys]
                    )
                    np.testing.assert_array_equal(
                        saved, [row[predicted_field] for row in targets]
                    )
                    refit_difference[response] = float(
                        np.max(np.abs(prediction - saved))
                    )
                    np.testing.assert_array_equal(prediction, saved)
                    prediction = (
                        saved  # Evaluate the original predictions, never replace them.
                    )
                for row, value in zip(rows, prediction, strict=True):
                    row[response] = row[field]
                    row[f"predicted_{response}"] = float(value)
            summary = base.evaluation_summary(rows)
            for response, curve_field in (
                ("raw_norm", "curves"),
                ("balanced_norm", "balanced_norm_curves"),
            ):
                scores = [record["node_spearman"] for record in summary[curve_field]]
                summary[f"{response}_within_curve"] = {
                    "median": float(np.median(scores)),
                    "min": min(scores),
                    "max": max(scores),
                }
                replicates = []
                for family in base.TARGET_REPLICATES:
                    for threshold in base.THRESHOLDS:
                        for replicate in base.TARGET_REPLICATES[family]:
                            local = [
                                r
                                for r in rows
                                if (r["family"], r["threshold"], r["replicate"])
                                == (family, threshold, replicate)
                            ]
                            score = base.safe_spearman(
                                np.array([r[response] for r in local]),
                                np.array([r[f"predicted_{response}"] for r in local]),
                            )
                            replicates.append(
                                {
                                    "family": family,
                                    "threshold": threshold,
                                    "replicate": replicate,
                                    "spearman": score,
                                }
                            )
                summary[f"{response}_replicates"] = replicates
            summary["refit_max_abs_difference_from_frozen"] = refit_difference
            comparisons[kind] = summary
            print(
                label,
                kind,
                json.dumps(
                    {
                        "criteria": summary["criteria"],
                        "all_pass": summary["all_primary_criteria_passed"],
                        "within": summary["raw_norm_within_curve"],
                        "refit_delta": refit_difference,
                    }
                ),
                flush=True,
            )
        report["datasets"][label] = comparisons
    for path in (
        Path(__file__),
        config.root / "scripts/prospective_norm_turnover.py",
        config.root / config.requirements,
        config.root / config.methods,
        config.root / config.protocol,
    ):
        report["source_sha256"][str(path.relative_to(config.root))] = base.sha256_file(
            path
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_tables(report, output)


def write_tables(report: dict[str, Any], output: Path) -> None:
    labels = {
        "historical": "Original",
        "corrected_retained_noise": "Corrected, retained noise",
        "corrected_raw_noise": "Corrected, raw noise",
    }
    inputs = {
        "pairs_kl": "Pairs + KL",
        "degree_threshold": "Degree + threshold",
        "pairs": "Pairs only",
        "kl": "KL only",
        "degree": "Degree only",
    }
    md = [
        "# Post hoc prospective-predictor comparisons",
        "",
        "Generated by `python3 scripts/prospective_baselines.py`. "
        "See `analysis/prospective_baselines.md` for scope and reproduction.",
        "",
        "All additional baselines are post hoc. The original predictor uses its "
        "saved predictions. The corrected evaluations are retrospective.",
        "",
    ]
    for response in ("raw_norm", "balanced_norm"):
        primary = response == "raw_norm"
        caption = (
            "Post hoc comparisons for raw center norms (primary response). "
            "The last column applies the original four criteria, including "
            "the predictor-independent observed-peak property. "
            if primary
            else "Post hoc comparisons for balanced norms "
            "(secondary response, without a separate primary pass/fail test). "
        )
        caption += (
            "Within-curve Spearman is the median [range] across nine "
            "family--threshold groups, each pooling five replicates. "
            "Corrected results are retrospective revalidations."
        )
        label = "prospective-baselines" if primary else "balanced-baselines"
        header = (
            r"Inputs & \shortstack{Median peak\\error (bins)} "
            r"& \shortstack{Peaks within\\one bin} "
        )
        header += (
            r"& Pooled $\rho$ & Within-curve $\rho$ & All four pass \\"
            if primary
            else r"& Within-curve $\rho$ \\"
        )
        tex = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{6pt}",
            r"\caption{" + caption + "}",
            r"\label{tab:" + label + "}",
            r"\begin{tabular}{" + ("lccccc" if primary else "lccc") + "}",
            r"\toprule",
            header,
        ]
        md += [
            f"## {response}",
            "",
            "Within-curve correlations are over nodes, "
            "pooling five replicates per family/threshold. Ranges are not confidence "
            "intervals. Full curve/replicate records are in `summary.json`.",
            "",
            "| Data | Inputs | Median peak error (bins) | Peaks within one bin "
            "| Pooled Spearman | Within-curve median [range] "
            "| All primary criteria pass |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for label, comparisons in report["datasets"].items():
            tex += [
                r"\midrule",
                r"\multicolumn{"
                + ("6" if primary else "4")
                + r"}{l}{\textit{"
                + labels[label]
                + r"}} \\",
            ]
            for kind, summary in comparisons.items():
                curves = summary[
                    "curves" if response == "raw_norm" else "balanced_norm_curves"
                ]
                errors = [c["absolute_peak_bin_error"] for c in curves]
                within = summary[f"{response}_within_curve"]
                interval = (
                    f"{within['median']:.3f} [{within['min']:.3f}, {within['max']:.3f}]"
                )
                peak = f"{sum(e <= 1 for e in errors)}/{len(errors)}"
                median = f"{np.median(errors):g}"
                pooled = (
                    f"{summary['criteria']['pooled_spearman']:.3f}"
                    if response == "raw_norm"
                    else "—"
                )
                passed = (
                    ("Yes" if summary["all_primary_criteria_passed"] else "No")
                    if response == "raw_norm"
                    else "Not a primary test"
                )
                md.append(
                    f"| {labels[label]} | {inputs[kind]} | {median} | {peak} "
                    f"| {pooled} | {interval} | {passed} |"
                )
                row = f"{inputs[kind]} & {median} & {peak} "
                row += (
                    f"& {pooled} & ${interval}$ & {passed} "
                    if primary
                    else f"& ${interval}$ "
                )
                tex.append(row + r"\\")
        md.append("")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        filename = "table.tex" if primary else "balanced_table.tex"
        (output / filename).write_text("\n".join(tex))
    (output / "README.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
