"""Audit and report the corrected sampler/target revalidation without fitting.

Inputs: completed revalidation JSON/JSONL, historical summary, and training log.
Outputs: report.json, README.md, and raw/balanced sampler and prediction figures.
Run: python3 scripts/report_negative_sampling_revalidation.py --training-log LOG
The report recomputes summaries from node records and checks all artifact hashes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import negative_sampling_control as control
import numpy as np
import prospective_norm_turnover as base
import revalidate_negative_sampling as validation
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

LABELS = {
    "barabasi_albert": "Direct BA",
    "powerlaw_cluster": "Clustered power law",
    "netscience": "NetScience",
}
COLORS = {"retained": "#0072B2", "raw": "#D55E00"}


def validate_rows(
    rows: list[dict[str, Any]], expected: int, calibration: bool = False
) -> None:
    if len(rows) != expected or len({base.prediction_key(r) for r in rows}) != expected:
        raise ValueError("Missing or duplicate node-condition records.")
    for row in rows:
        for arm in validation.ARMS:
            for response in base.RESPONSES:
                assert np.isfinite(row[f"{arm}_{response}"])
                if not calibration:
                    assert np.isfinite(row[f"predicted_{arm}_{response}"])


def audit(output: Path, config: validation.Config) -> dict[str, Any]:
    validation.verify_sources(config, output)
    validation.verify_manifest(output, "completion.json")
    calibration = base.load_jsonl(output / "calibration.jsonl")
    rows = base.load_jsonl(output / "outcomes.jsonl")
    validate_rows(calibration, 9000, calibration=True)
    validate_rows(rows, 23685)
    for name, expected in (("calibration_checks.json", 15), ("target_checks.json", 45)):
        records = base.load_json(output / name)
        assert len(records) == expected
        assert (
            len({(r["family"], r["replicate"], r["threshold"]) for r in records})
            == expected
        )
    frozen = base.load_jsonl(output / "frozen_predictions.jsonl")
    frozen_by_key = {base.prediction_key(r): r for r in frozen}
    assert len(frozen_by_key) == len(rows)
    for row in rows:
        for field, value in frozen_by_key[base.prediction_key(row)].items():
            assert row[field] == value, field
    summary = base.load_json(output / "summary.json")
    recomputed = {
        "predictor": {
            arm: base.evaluation_summary(validation.arm_rows(rows, arm))
            for arm in validation.ARMS
        },
        "sampler_comparisons": validation.sampler_summary(rows, config.bins),
        "calibration_sampler_comparisons": validation.sampler_summary(
            calibration, config.bins
        ),
    }
    assert summary == recomputed, "Stored summary differs from the raw node data."
    aggregate = []
    for family in base.TARGET_REPLICATES:
        for threshold in base.THRESHOLDS:
            local = [
                r for r in rows if r["family"] == family and r["threshold"] == threshold
            ]
            degree = np.asarray([r["degree"] for r in local])
            for response in base.RESPONSES:
                aggregate.append(
                    {
                        "family": family,
                        "threshold": threshold,
                        "response": response,
                        "curves": {
                            arm: control.curve(
                                degree,
                                np.asarray([r[f"{arm}_{response}"] for r in local]),
                                config.bins,
                            )
                            for arm in validation.ARMS
                        },
                    }
                )
    sampler_counts = {}
    for dataset, comparisons in (
        ("calibration", summary["calibration_sampler_comparisons"]),
        ("targets", summary["sampler_comparisons"]),
        ("target_family_curves", aggregate),
    ):
        sampler_counts[dataset] = {}
        for response in base.RESPONSES:
            records = [r for r in comparisons if r["response"] == response]
            sampler_counts[dataset][response] = {
                "conditions": len(records),
                "same_peak": sum(
                    r["curves"]["retained"]["peak_bin"]
                    == r["curves"]["raw"]["peak_bin"]
                    for r in records
                ),
                "retained_interior": sum(
                    r["curves"]["retained"]["interior_peak"] for r in records
                ),
                "raw_interior": sum(
                    r["curves"]["raw"]["interior_peak"] for r in records
                ),
                "lost_interior_under_raw": sum(
                    r["curves"]["retained"]["interior_peak"]
                    and not r["curves"]["raw"]["interior_peak"]
                    for r in records
                ),
                "gained_interior_under_raw": sum(
                    not r["curves"]["retained"]["interior_peak"]
                    and r["curves"]["raw"]["interior_peak"]
                    for r in records
                ),
            }
    # Raw noise must stay fixed across thresholds in node coordinates.
    for family, seeds in base.TARGET_REPLICATES.items():
        for seed in seeds:
            groups = [
                sorted(
                    [
                        r
                        for r in rows
                        if r["family"] == family
                        and r["replicate"] == seed
                        and r["threshold"] == t
                    ],
                    key=lambda r: r["node"],
                )
                for t in base.THRESHOLDS
            ]
            for group in groups:
                np.testing.assert_array_equal(
                    [r["raw_tokens"] for r in group],
                    [r["raw_tokens"] for r in groups[0]],
                )
                counts = np.asarray([r["raw_tokens"] for r in group])
                np.testing.assert_allclose(
                    [r["raw_negative_mass"] for r in group],
                    counts / counts.sum(),
                    rtol=0,
                    atol=1 / control.ControlConfig.table_domain,
                )
    repeat = base.load_json(output / "repeat_check.json")
    if (
        not repeat["all_fields_exact"]
        or repeat["outcomes_sha256"] != base.sha256_file(output / "outcomes.jsonl")
        or repeat["plan_sha256"] != base.sha256_file(output / "plan.json")
        or repeat["check_script_sha256"]
        != base.sha256_file(config.root / "scripts/check_revalidation_repeat.py")
    ):
        raise ValueError("Independent repeat does not match these outcomes/sources.")
    family_details = []
    for family in base.TARGET_REPLICATES:
        for threshold in base.THRESHOLDS:
            local = [
                r for r in rows if r["family"] == family and r["threshold"] == threshold
            ]
            individual = [
                r
                for r in summary["sampler_comparisons"]
                if r["family"] == family
                and r["threshold"] == threshold
                and r["response"] == "raw_norm"
            ]
            for arm in validation.ARMS:
                observed = np.asarray([r[f"{arm}_raw_norm"] for r in local])
                predicted = np.asarray([r[f"predicted_{arm}_raw_norm"] for r in local])
                family_details.append(
                    {
                        "family": family,
                        "threshold": threshold,
                        "arm": arm,
                        "node_rows": len(local),
                        "observed_mean": float(observed.mean()),
                        "predicted_mean": float(predicted.mean()),
                        "observed_minus_predicted_mean": float(
                            (observed - predicted).mean()
                        ),
                        "interior_runs": sum(
                            r["curves"][arm]["interior_peak"] for r in individual
                        ),
                        "runs": len(individual),
                    }
                )
    return {
        "reporter_sha256": base.sha256_file(Path(__file__)),
        "independent_repeat": repeat,
        "verified_calibration_rows": len(calibration),
        "verified_target_rows": len(rows),
        "sampler_counts": sampler_counts,
        "family_sampler_curves": aggregate,
        "family_details": family_details,
        "individual_peak_changes": [
            record
            for record in summary["sampler_comparisons"]
            if record["curves"]["retained"]["peak_bin"]
            != record["curves"]["raw"]["peak_bin"]
        ],
        "predictor": summary["predictor"],
        "historical_predictor": base.load_json(
            config.root / config.original / "summary.json"
        ),
    }


def plot_samplers(rows: list[dict[str, Any]], response: str, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.subplots_adjust(
        left=0.065, right=0.99, bottom=0.09, top=0.87, hspace=0.47, wspace=0.24
    )
    for row_index, (family, seeds) in enumerate(base.TARGET_REPLICATES.items()):
        for col_index, threshold in enumerate(base.THRESHOLDS):
            ax = axes[row_index, col_index]
            local = [
                r for r in rows if r["family"] == family and r["threshold"] == threshold
            ]
            for arm in validation.ARMS:
                for seed in seeds:
                    subset = [r for r in local if r["replicate"] == seed]
                    x, y, _ = base.binned_curve(
                        np.array([r["degree"] for r in subset]),
                        np.array([r[f"{arm}_{response}"] for r in subset]),
                    )
                    ax.plot(x, y, color=COLORS[arm], alpha=0.18, linewidth=0.9)
                x, y, _ = base.binned_curve(
                    np.array([r["degree"] for r in local]),
                    np.array([r[f"{arm}_{response}"] for r in local]),
                )
                ax.plot(
                    x,
                    y,
                    color=COLORS[arm],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    linestyle="--" if arm == "retained" else "-",
                )
            ax.set(
                xscale="log",
                xlabel="Degree",
                ylabel=response.replace("_", " "),
                title=f"{LABELS[family]} · t = {threshold:g}",
            )
            ax.grid(alpha=0.15)
            degree_ticks(ax)
    fig.suptitle(
        f"Corrected sampler control: {response.replace('_', ' ')}\n"
        "Thin curves: individual runs; bold curves: pooled degree bins",
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS[a],
                linestyle="--" if a == "retained" else "-",
                label="Retained-token noise"
                if a == "retained"
                else "Fixed original-token noise",
            )
            for a in validation.ARMS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.savefig(output / f"sampler_{response}.png", dpi=160)
    plt.close(fig)


def plot_predictions(rows: list[dict[str, Any]], output: Path) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.subplots_adjust(
        left=0.065, right=0.99, bottom=0.13, top=0.88, hspace=0.39, wspace=0.24
    )
    for i, arm in enumerate(validation.ARMS):
        for j, family in enumerate(base.TARGET_REPLICATES):
            ax = axes[i, j]
            for threshold, color in zip(base.THRESHOLDS, colors, strict=True):
                local = [
                    r
                    for r in rows
                    if r["family"] == family and r["threshold"] == threshold
                ]
                degree = np.array([r["degree"] for r in local])
                for prefix, style in (("", "-"), ("predicted_", "--")):
                    x, y, _ = base.binned_curve(
                        degree, np.array([r[f"{prefix}{arm}_raw_norm"] for r in local])
                    )
                    ax.plot(
                        x,
                        y,
                        color=color,
                        linestyle=style,
                        marker="o" if not prefix else None,
                        markersize=3,
                    )
            ax.set(
                xscale="log",
                xlabel="Degree",
                ylabel="Raw center-vector norm",
                title=f"{LABELS[family]} · {arm} noise",
            )
            ax.grid(alpha=0.15)
            degree_ticks(ax)
    handles = [
        Line2D([0], [0], color=c, label=f"t = {t:g}")
        for t, c in zip(base.THRESHOLDS, colors, strict=True)
    ]
    handles += [
        Line2D([0], [0], color="0.2", linestyle=style, label=label)
        for style, label in (("-", "Observed"), ("--", "Frozen prediction"))
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Retrospective revalidation of the original prediction procedure", fontsize=14
    )
    fig.savefig(output / "predictor_revalidation.png", dpi=160)
    plt.close(fig)


def degree_ticks(ax: Any) -> None:
    lower, upper = ax.get_xlim()
    ax.set_xticks([v for v in (1, 2, 3, 5, 10, 20, 50, 100) if lower <= v <= upper])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def write_report(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Corrected negative-sampling revalidation",
        "",
        "Retrospective revalidation on the originally held-out targets. The old target "
        "results were already known; all corrected predictions were frozen before "
        "corrected target training. No target outcomes were used to fit or tune "
        "a model.",
        "",
        "Verified 30 calibration model fits and 90 target fits: 9,000 calibration "
        "node-condition rows and 23,685 target rows, each containing both arms. "
        "All 15 calibration pairs and 45 target pairs are present, with unique keys "
        "and unchanged frozen features/predictions.",
        "",
        "## Prediction criteria",
        "",
        "The same two features, ExtraTrees settings and four primary criteria were "
        "used separately for each arm. The historical column describes the original "
        "bug-affected pipeline; differences from it combine multiple corrections.",
        "",
        "| Criterion | Required | Historical | Corrected retained noise | "
        "Corrected fixed raw noise |",
        "|---|---:|---:|---:|---:|",
    ]
    results = [
        report["historical_predictor"],
        report["predictor"]["retained"],
        report["predictor"]["raw"],
    ]
    for label, field, requirement in (
        ("Median absolute peak error", "median_absolute_peak_bin_error", "≤1 bin"),
        ("Peaks within one bin", "fraction_peaks_within_one_bin", "≥6/9"),
        ("Pooled node Spearman", "pooled_spearman", "≥0.5"),
        (
            "Families with nondecreasing peaks",
            "families_with_nondecreasing_observed_peaks",
            "≥2/3",
        ),
    ):
        values = []
        for result in results:
            value = result["criteria"][field]
            values.append(
                f"{round(value * 9)}/9"
                if field == "fraction_peaks_within_one_bin"
                else f"{value:.6g}"
            )
        lines.append(f"| {label} | {requirement} | " + " | ".join(values) + " |")
    lines += [
        "",
        "All four primary criteria passed: "
        + "; ".join(
            f"**{arm}: {report['predictor'][arm]['all_primary_criteria_passed']}**"
            for arm in validation.ARMS
        )
        + ".",
        "",
        "## Sampler robustness",
        "",
        "| Scope | Response | Same peak bin | Interior peaks: retained / raw | "
        "Interior peaks lost / gained under raw noise |",
        "|---|---|---:|---:|---:|",
    ]
    for scope, responses in report["sampler_counts"].items():
        for response, counts in responses.items():
            lines.append(
                f"| {scope} | {response} | "
                f"{counts['same_peak']}/{counts['conditions']} | "
                f"{counts['retained_interior']} / {counts['raw_interior']} | "
                f"{counts['lost_interior_under_raw']} / "
                f"{counts['gained_interior_under_raw']} |"
            )
    lines += [
        "",
        "Individual runs with changed peaks (all exceptions):",
        "",
        "| Family | Seed | Threshold | Response | Retained / raw peak | "
        "Interior: retained / raw |",
        "|---|---:|---:|---|---:|---|",
    ]
    for record in report["individual_peak_changes"]:
        retained, raw = (record["curves"][arm] for arm in validation.ARMS)
        lines.append(
            f"| {LABELS[record['family']]} | {record['replicate']} | "
            f"{record['threshold']:g} | {record['response']} | "
            f"{retained['peak_bin']} / {raw['peak_bin']} | "
            f"{retained['interior_peak']} / {raw['interior_peak']} |"
        )
    lines += [
        "",
        "Peak-bin equality is descriptive, not equivalence of the embeddings. "
        "Interior maxima are not significance tests of unimodality. Ties in "
        "degree quantiles reduce the number of occupied bins. NetScience's five "
        "runs repeat walks/optimization on one observed network.",
        "",
        "## Every family and threshold",
        "",
        "Peak indices below are one-based occupied-bin indices. Absolute norm "
        "errors and within-condition rank correlations expose failures hidden by "
        "pooled criteria. Full individual-run curves, endpoints, bin sizes and "
        "paired sampler differences are in summary.json.",
        "",
        "| Family | Threshold | Noise | Observed / predicted peak | "
        "Interior pooled peak | Individual interior peaks | "
        "Node RMSE | Node Spearman |",
        "|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for family in base.TARGET_REPLICATES:
        for threshold in base.THRESHOLDS:
            for arm in validation.ARMS:
                record = next(
                    r
                    for r in report["predictor"][arm]["curves"]
                    if r["family"] == family and r["threshold"] == threshold
                )
                detail = next(
                    r
                    for r in report["family_details"]
                    if r["family"] == family
                    and r["threshold"] == threshold
                    and r["arm"] == arm
                )
                lines.append(
                    f"| {LABELS[family]} | {threshold:g} | {arm} | "
                    f"{record['actual']['peak_bin'] + 1} / "
                    f"{record['predicted']['peak_bin'] + 1} | "
                    f"{record['actual']['inverted_u']} | "
                    f"{detail['interior_runs']}/{detail['runs']} | "
                    f"{record['node_rmse']:.6f} | {record['node_spearman']:.6f} |"
                )
    lines += [
        "",
        "Family pooling can hide different shapes across graphs. In the "
        "clustered family at t=0.0003, the pooled raw-norm peak is interior while "
        "only one of five individual graphs has an interior peak. At t=0.003, "
        "the pooled peak is at the upper boundary while four of five individual "
        "graphs have interior peaks. Both samplers show these classifications.",
        "",
        "## NetScience absolute-scale error",
        "",
        "The predictor continues to overestimate NetScience norms and misses its "
        "strongest-thinning peak by two bins under both samplers. Passing pooled "
        "criteria therefore does not imply accurate absolute norms. Node rows below "
        "pool five stochastic repetitions of the same 379-node graph.",
        "",
        "| Threshold | Noise | Node rows | Observed mean | Predicted mean | "
        "Observed minus predicted |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for detail in report["family_details"]:
        if detail["family"] == "netscience":
            lines.append(
                f"| {detail['threshold']:g} | {detail['arm']} | "
                f"{detail['node_rows']} | "
                f"{detail['observed_mean']:.6f} | {detail['predicted_mean']:.6f} | "
                f"{detail['observed_minus_predicted_mean']:.6f} |"
            )
    lines += [
        "",
        "## Evidence boundary",
        "",
        "Both arms use corrected repeated-visit counts, correctly assigned "
        "negative weights and the checked BLAS build. Raw-noise probabilities "
        "are verified fixed across thresholds. The exact same retained corpus, "
        "initialization and optimizer state are used within each pair. Equal "
        "nominal negative proposals do not imply equal accepted updates because "
        "Gensim skips collisions with positive output tokens.",
        "",
        "Passing the prediction criteria supports this fixed prediction procedure "
        "on these original targets after correction. It does not establish a "
        "complete exposure/context mechanism, a universal response curve, equal "
        "embeddings under different samplers, or new prospective generalization. "
        "Any lost/shifted peak limits the stronger claim that all shapes "
        "are invariant.",
        "",
        "The original prospective results, calibration-only control and first "
        "unpatched control remain historical artifacts. Other experiments using "
        "the old Gensim wrappers, including the internal-subsampling threshold "
        "sweep, require their own revalidation. No manuscript claim is automatically "
        "promoted by this report.",
        "",
        "## Reproduce",
        "",
        "See analysis/negative_sampling_revalidation_protocol.md for the locked "
        "design and setup. Run scripts/revalidate_negative_sampling.py into a "
        "fresh directory, run scripts/check_revalidation_repeat.py, then run "
        "this reporting script with --training-log. "
        "All numerical summaries are recomputed from node records and checked "
        "against saved summaries before this report is written.",
        "",
        "Training log: 120 completed fits; no numerical warnings or exceptions.",
        "",
    ]
    repeat = report["independent_repeat"]
    lines += [
        f"A fresh-process {LABELS[repeat['family']]} rerun "
        f"(seed {repeat['replicate']}, all three thresholds, both arms) reproduced "
        f"every field of {repeat['node_condition_rows']:,} node-condition rows "
        "exactly. See repeat_check.json and scripts/check_revalidation_repeat.py.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=validation.Config.output)
    parser.add_argument("--training-log", type=Path, required=True)
    args = parser.parse_args()
    config = validation.Config(output=args.output)
    output = (config.root / config.output).resolve()
    report = audit(output, config)
    log = args.training_log.read_text()
    if sum(line.endswith(" complete") for line in log.splitlines()) != 120:
        raise ValueError("Log does not contain exactly 120 completed model fits.")
    if any(word in log.lower() for word in ("warning", "exception", "traceback")):
        raise ValueError(
            "Inspect unexpected numerical warnings/errors before reporting."
        )
    if args.training_log.resolve() != output / "training_log.txt":
        shutil.copy2(args.training_log, output / "training_log.txt")
    rows = base.load_jsonl(output / "outcomes.jsonl")
    for response in base.RESPONSES:
        plot_samplers(rows, response, output)
    plot_predictions(rows, output)
    base.write_json(output / "report.json", report)
    write_report(report, output)
    validation.save_manifest(
        output,
        "report_manifest.json",
        [
            "completion.json",
            "repeat_check.json",
            "training_log.txt",
            "report.json",
            "README.md",
            "sampler_raw_norm.png",
            "sampler_balanced_norm.png",
            "predictor_revalidation.png",
        ],
    )
    print(json.dumps(report["sampler_counts"], indent=2))
    print(
        json.dumps(
            {arm: report["predictor"][arm]["criteria"] for arm in validation.ARMS},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
