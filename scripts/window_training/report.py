"""Summarize every frozen window/history run and draw scientific figures.

Input: complete results/window_training_exploration/{schedule,windows}.
Output: report/summary.json, node-independent bin counts, figures and manifest.
Run: python3 scripts/window_training/report.py [--output /path/to/results]
Graph means have equal weight; average starts within graph before calculating SE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_training.expected_window import Config, read_json, sha, write_json

FIELDS = ("raw_norm", "balanced_norm", "node_loss", "score_change_rms_by_node")
COLORS = ("#245e9b", "#bd621a", "#33835b", "#9654a4")


def check_manifest(directory: Path) -> None:
    for name, expected in read_json(directory / "completion.json").items():
        if sha(directory / name) != expected:
            raise ValueError(f"Modified experiment input: {directory / name}")


def bins_for(degrees: list[np.ndarray]) -> list[np.ndarray]:
    edges = np.unique(np.quantile(np.log(np.concatenate(degrees)), np.arange(9) / 8))
    return [
        np.searchsorted(edges[1:-1], np.log(degree), side="left") for degree in degrees
    ]


def shape(values: list[float]) -> dict[str, Any]:
    peak = int(np.argmax(values))
    return {
        "peak_bin_zero_based": peak,
        "interior_peak": 0 < peak < len(values) - 1,
        "high_minus_low": values[-1] - values[0],
        "high_minus_penultimate": values[-1] - values[-2],
    }


def node_curve(groups: np.ndarray, response: np.ndarray) -> dict[str, Any]:
    means = [float(response[groups == label].mean()) for label in np.unique(groups)]
    return {"values": means, **shape(means)}


def summarize_nodes(
    degrees: list[np.ndarray], values: list[np.ndarray]
) -> dict[str, Any]:
    labels = bins_for(degrees)
    occupied = np.unique(np.concatenate(labels))
    curves = []
    for seed, (degree, groups, response) in enumerate(
        zip(degrees, labels, values, strict=True)
    ):
        if response.shape != degree.shape or not np.isfinite(response).all():
            raise ValueError("Invalid node response.")
        if any(not np.any(groups == label) for label in occupied):
            raise ValueError("Shared degree bin missing a graph.")
        curves.append({"seed": seed, **node_curve(groups, response)})
    bins = []
    for index, label in enumerate(occupied):
        by_seed = [curve["values"][index] for curve in curves]
        chosen_degrees = np.concatenate(
            [d[g == label] for d, g in zip(degrees, labels, strict=True)]
        )
        bins.append(
            {
                "degree": float(np.exp(np.log(chosen_degrees).mean())),
                "counts_by_seed": [int(np.sum(groups == label)) for groups in labels],
                "values_by_seed": by_seed,
                "mean": float(np.mean(by_seed)),
                "standard_error": float(
                    np.std(by_seed, ddof=1) / np.sqrt(len(by_seed))
                ),
            }
        )
    return {
        "bins": bins,
        "per_graph": curves,
        "mean_curve": shape([b["mean"] for b in bins]),
    }


def summary(output: Path) -> dict[str, Any]:
    schedule, windows = output / "schedule", output / "windows"
    check_manifest(schedule)
    check_manifest(windows)
    for seed in range(5):
        check_manifest(schedule / f"seed{seed}")
    graphs = [read_json(windows / f"graphs/seed{s}.json") for s in range(5)]
    degrees = [np.array(g["degree"]) for g in graphs]
    schedules = {
        (s, branch): read_json(schedule / f"seed{s}/{branch}.json")
        for s in range(5)
        for branch in ("floor", "reheat")
    }
    expected = [
        read_json(path) for path in sorted((windows / "runs").glob("*.json.gz"))
    ]
    reference = Config.root / read_json(schedule / "plan.json")["config"]["reference"]
    check_manifest(reference)
    history = [
        read_json(reference / f"runs/seed{seed}_t0_constant.json") for seed in range(5)
    ]
    identities = {(r["seed"], r["window"], r["initialization"]) for r in expected}
    if len(expected) != 40 or identities != {
        (s, w, s + offset)
        for s in range(5)
        for w in (1, 2, 5, 10)
        for offset in (100000, 200000)
    }:
        raise ValueError("Incomplete expected-window sweep.")
    for seed, graph in enumerate(graphs):
        metadata = read_json(schedule / f"seed{seed}/metadata.json")
        if (
            metadata["graph_sha256"] != graph["graph_sha256"]
            or metadata["degree"] != graph["degree"]
        ):
            raise ValueError("Graph pairing between lanes differs.")
    results: dict[str, Any] = {
        "schedule": [],
        "constant_from_start_reference": [],
        "windows": [],
        "free_score": [],
        "schedule_progress": [],
        "window_progress": [],
        "initialization_comparison": [],
        "scope": (
            "Five graphs; two starts are averaged within graph for window curves. "
            "Fixed shared log-degree quantile bins; error bars are graph-level SE, "
            "not significance tests."
        ),
    }
    for field in FIELDS:
        results["constant_from_start_reference"].append(
            {
                "field": field,
                "epoch": 20,
                "source": "PR36 constant 0.0025 from start; archived reference",
                **summarize_nodes(
                    degrees,
                    [
                        np.array(
                            next(s for s in run["snapshots"] if s["epoch"] == 20)[field]
                        )
                        for run in history
                    ],
                ),
            }
        )
    for branch in ("floor", "reheat"):
        for epoch in (5, 6, 7, 10, 15, 20):
            rows = [
                next(
                    r for r in schedules[s, branch]["snapshots"] if r["epoch"] == epoch
                )
                for s in range(5)
            ]
            for field in (*FIELDS, "score_change_from_fork_rms_by_node"):
                results["schedule"].append(
                    {
                        "branch": branch,
                        "epoch": epoch,
                        "field": field,
                        **summarize_nodes(
                            degrees, [np.array(row[field]) for row in rows]
                        ),
                    }
                )
            for seed, row in enumerate(rows):
                results["schedule_progress"].append(
                    {
                        "seed": seed,
                        "branch": branch,
                        **{
                            key: value
                            for key, value in row.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
    common_steps = sorted(
        set.intersection(
            *[{row["iteration"] for row in run["snapshots"]} for run in expected]
        )
    )
    for length in (1, 2, 5, 10):
        local = [r for r in expected if r["window"] == length]
        for step in [*common_steps, "final"]:
            snapshots = [
                (
                    run["seed"],
                    run["snapshots"][-1]
                    if step == "final"
                    else next(
                        row for row in run["snapshots"] if row["iteration"] == step
                    ),
                )
                for run in local
            ]
            for field in FIELDS:
                node_values = [
                    np.mean(
                        [row[field] for seed, row in snapshots if seed == s], axis=0
                    )
                    for s in range(5)
                ]
                results["windows"].append(
                    {
                        "window": length,
                        "step": step,
                        "field": field,
                        **summarize_nodes(degrees, node_values),
                    }
                )
        for run in local:
            for row in run["snapshots"]:
                results["window_progress"].append(
                    {
                        "seed": run["seed"],
                        "window": length,
                        "initialization": run["initialization"],
                        "final": row is run["snapshots"][-1],
                        "termination": run["termination"],
                        **{
                            key: value
                            for key, value in row.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
        for seed in range(5):
            pair = sorted(
                [r for r in local if r["seed"] == seed],
                key=lambda r: r["initialization"],
            )
            ends = [run["snapshots"][-1] for run in pair]
            for field in FIELDS[:2]:
                curves = [
                    node_curve(bins_for(degrees)[seed], np.array(end[field]))
                    for end in ends
                ]
                results["initialization_comparison"].append(
                    {
                        "seed": seed,
                        "window": length,
                        "field": field,
                        "initializations": [r["initialization"] for r in pair],
                        "curves": curves,
                        "node_rms_difference": float(
                            np.sqrt(
                                np.mean(
                                    (np.array(ends[0][field]) - ends[1][field]) ** 2
                                )
                            )
                        ),
                        "losses": [e["expected_loss"] for e in ends],
                    }
                )
        if all(
            g["windows"][str(length)]["free_score"]["finite_full_rank_norms_defined"]
            for g in graphs
        ):
            results["free_score"].append(
                {
                    "window": length,
                    "field": "balanced_norm",
                    **summarize_nodes(
                        degrees,
                        [
                            np.array(
                                g["windows"][str(length)]["free_score"]["balanced_norm"]
                            )
                            for g in graphs
                        ],
                    ),
                }
            )
    return results


def find(records: list[dict[str, Any]], **conditions: Any) -> dict[str, Any]:
    return next(
        r for r in records if all(r[key] == value for key, value in conditions.items())
    )


def draw_curve(
    ax: Any, record: dict[str, Any], label: str, color: str, linestyle: str = "-"
) -> None:
    x = [b["degree"] for b in record["bins"]]
    for row in record["per_graph"]:
        ax.plot(x, row["values"], color=color, alpha=0.18, lw=0.7, linestyle=linestyle)
    ax.errorbar(
        x,
        [b["mean"] for b in record["bins"]],
        yerr=[b["standard_error"] for b in record["bins"]],
        color=color,
        marker="o",
        ms=3,
        capsize=2,
        lw=1.6,
        label=label,
        linestyle=linestyle,
    )
    ax.set_xscale("log")
    ax.set_xlim(min(x) * 0.92, max(x) * 1.08)
    ax.set_xticks([3, 4, 6, 8, 12, 16], labels=[3, 4, 6, 8, 12, 16])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Degree (geometric bin mean)")


def finish(fig: Any, path: Path) -> None:
    for ax in fig.axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=190)
    fig.savefig(
        path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def schedule_figure(data: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, field, title in zip(
        axes[0], FIELDS[:2], ("A  Raw norms", "B  Balanced norms"), strict=True
    ):
        draw_curve(
            ax,
            find(data["schedule"], branch="floor", epoch=5, field=field),
            "Shared state at epoch 5",
            "#6b6b6b",
            "--",
        )
        for branch, color in zip(("floor", "reheat"), COLORS[:2], strict=True):
            draw_curve(
                ax,
                find(data["schedule"], branch=branch, epoch=20, field=field),
                f"{branch.capitalize()}, epoch 20",
                color,
            )
        draw_curve(
            ax,
            find(data["constant_from_start_reference"], field=field),
            "0.0025 from start, epoch 20 (PR36)",
            COLORS[2],
            ":",
        )
        ax.set_title(title, loc="left")
        ax.set_ylabel(field.replace("_", " ").capitalize())
    axes[0, 0].legend(fontsize=8)
    for branch, color in zip(("floor", "reheat"), COLORS[:2], strict=True):
        rows = [
            r
            for r in data["schedule"]
            if r["branch"] == branch and r["field"] == "balanced_norm"
        ]
        gaps = np.array(
            [[g["high_minus_penultimate"] for g in r["per_graph"]] for r in rows]
        )
        for seed in range(5):
            axes[1, 0].plot(
                [r["epoch"] for r in rows],
                gaps[:, seed],
                color=color,
                alpha=0.2,
                lw=0.7,
            )
        axes[1, 0].errorbar(
            [r["epoch"] for r in rows],
            gaps.mean(axis=1),
            yerr=gaps.std(axis=1, ddof=1) / np.sqrt(5),
            color=color,
            marker="o",
            capsize=2,
            label=branch.capitalize(),
        )
        for seed in range(5):
            progress = [
                r
                for r in data["schedule_progress"]
                if r["branch"] == branch and r["seed"] == seed
            ]
            axes[1, 1].plot(
                [r["epoch"] for r in progress],
                [r["expected_loss"] for r in progress],
                color=color,
                alpha=0.3,
                lw=0.8,
            )
        means = [
            np.mean(
                [
                    r["expected_loss"]
                    for r in data["schedule_progress"]
                    if r["branch"] == branch and r["epoch"] == epoch
                ]
            )
            for epoch in (5, 6, 7, 10, 15, 20)
        ]
        axes[1, 1].plot(
            (5, 6, 7, 10, 15, 20),
            means,
            color=color,
            marker="o",
            label=branch.capitalize(),
        )
    axes[1, 0].axhline(0, color="black", lw=0.6, ls=":")
    axes[1, 0].set_title("C  Upper-degree change in balanced norm", loc="left")
    axes[1, 0].set_ylabel("Highest bin minus preceding bin")
    axes[1, 1].set_title("D  Idealized expected-loss diagnostic", loc="left")
    axes[1, 1].set_ylabel("Expected loss per positive pair")
    for ax in axes[1]:
        ax.set_xlabel("Absolute training epoch")
    fig.suptitle(
        "Identical training through epoch 5; late rate 0.0001 vs 0.0025\n"
        "Five graphs; means ± graph SE; faint lines show every graph",
        fontsize=11,
    )
    finish(fig, destination / "schedule_fork")


def window_figure(data: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, field, title in zip(
        axes[0],
        FIELDS[:2],
        ("A  Raw norms at final endpoint", "B  Balanced norms at final endpoint"),
        strict=True,
    ):
        for length, color in zip((1, 2, 5, 10), COLORS, strict=True):
            draw_curve(
                ax,
                find(data["windows"], window=length, step="final", field=field),
                f"Window {length}",
                color,
            )
        ax.set_title(title, loc="left")
        ax.set_ylabel(field.replace("_", " ").capitalize())
    axes[0, 0].legend(fontsize=8)
    for length, color in zip((1, 2, 5, 10), COLORS, strict=True):
        rows = [
            r
            for r in data["windows"]
            if r["window"] == length
            and r["field"] == "balanced_norm"
            and isinstance(r["step"], int)
            and r["step"] > 0
        ]
        gaps = np.array(
            [[g["high_minus_penultimate"] for g in r["per_graph"]] for r in rows]
        )
        axes[1, 0].errorbar(
            [r["step"] for r in rows],
            gaps.mean(axis=1),
            yerr=gaps.std(axis=1, ddof=1) / np.sqrt(5),
            color=color,
            marker="o",
            capsize=2,
        )
        for seed in range(5):
            for start in (100000, 200000):
                progress = [
                    r
                    for r in data["window_progress"]
                    if r["window"] == length
                    and r["seed"] == seed
                    and r["initialization"] == seed + start
                    and r["iteration"] > 0
                ]
                axes[1, 1].loglog(
                    [r["iteration"] for r in progress],
                    [r["gradient_max"] for r in progress],
                    color=color,
                    alpha=0.35,
                    lw=0.8,
                )
    axes[1, 0].set_xscale("log")
    axes[1, 0].axhline(0, color="black", ls=":", lw=0.6)
    axes[1, 0].set_title("C  Upper-degree change during optimization", loc="left")
    axes[1, 0].set_ylabel("Highest minus preceding bin, balanced norm")
    axes[1, 1].set_title("D  Every fit's factor-gradient residual", loc="left")
    axes[1, 1].set_ylabel("Maximum absolute gradient component")
    for ax in axes[1]:
        ax.set_xlabel("L-BFGS iteration (not SGD epochs)")
    fig.suptitle(
        "Exact windows: fixed exposure and noise, rank 24; unconverged endpoints\n"
        "Five graphs × two starts; starts averaged within graph; means ± graph SE",
        fontsize=11,
    )
    finish(fig, destination / "expected_windows")
    if data["free_score"]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for row in data["free_score"]:
            draw_curve(
                ax,
                row,
                f"Window {row['window']}: exact full-rank benchmark",
                "#9654a4",
                "--",
            )
            draw_curve(
                ax,
                find(
                    data["windows"],
                    window=row["window"],
                    step="final",
                    field="balanced_norm",
                ),
                f"Window {row['window']}: rank-24 fit",
                "#245e9b",
            )
        ax.set_ylabel("Balanced norm")
        ax.set_title("Full-rank free scores are a distinct benchmark")
        ax.legend(fontsize=8)
        finish(fig, destination / "full_rank_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/window_training_exploration")
    )
    args = parser.parse_args()
    output = Config.root / args.output
    destination = output / "report"
    destination.mkdir(exist_ok=True)
    data = summary(output)
    write_json(destination / "summary.json", data)
    schedule_figure(data, destination)
    window_figure(data, destination)
    node_figure(output, destination)
    write_tables(data, destination)
    write_json(
        destination / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "input_completion_sha256": {
                lane: sha(output / lane / "completion.json")
                for lane in ("schedule", "windows")
            },
            "reference_completion_sha256": sha(
                Config.root
                / read_json(output / "schedule/plan.json")["config"]["reference"]
                / "completion.json"
            ),
            "artifact_sha256": {
                str(p.relative_to(destination)): sha(p)
                for p in sorted(destination.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    print(f"Summarized every run and graph: {destination}")


def node_figure(output: Path, destination: Path) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(14, 12), sharex=True, sharey="col")
    for seed in range(5):
        graph = read_json(output / "windows" / f"graphs/seed{seed}.json")
        degree = np.array(graph["degree"])
        for col, length in enumerate((1, 2, 5, 10)):
            ax = axes[seed, col]
            for start, color in zip((100000, 200000), COLORS[:2], strict=True):
                run = read_json(
                    output
                    / "windows/runs"
                    / f"seed{seed}_L{length}_init{start}.json.gz"
                )
                values = np.array(run["snapshots"][-1]["balanced_norm"])
                ax.scatter(
                    degree,
                    values,
                    s=4,
                    alpha=0.22,
                    color=color,
                    rasterized=True,
                    label=f"Start {start // 100000}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"Graph {seed}, window {length}", fontsize=9)
            if col == 0:
                ax.set_ylabel("Balanced norm")
            if seed == 4:
                ax.set_xlabel("Node degree")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Every node and both starts at the 500-iteration endpoint\n"
        "Short-window means hide large node-level dispersion; no convergence claim",
        fontsize=12,
    )
    finish(fig, destination / "all_window_nodes")


def write_tables(data: dict[str, Any], destination: Path) -> None:
    lines = [
        "# Generated endpoint tables",
        "",
        "Regenerated by scripts/window_training/report.py from every raw run.",
        "Bins depend only on pooled graph degrees; starts are averaged within graph.",
        "SE is across five graphs, and descriptive maxima are not significance tests.",
        "",
        "## Training history",
        "",
        "| Condition | Interior balanced maxima | Upper-bin gap, mean ± SE | "
        "High minus low |",
        "|---|---:|---:|---:|",
    ]
    rows = [
        (
            "Shared epoch 5",
            find(data["schedule"], branch="floor", epoch=5, field="balanced_norm"),
        ),
        (
            "Floor, epoch 20",
            find(data["schedule"], branch="floor", epoch=20, field="balanced_norm"),
        ),
        (
            "Reheat, epoch 20",
            find(data["schedule"], branch="reheat", epoch=20, field="balanced_norm"),
        ),
        (
            "Constant from start, epoch 20 (PR36)",
            find(data["constant_from_start_reference"], field="balanced_norm"),
        ),
    ]
    for label, row in rows:
        gaps = np.array([g["high_minus_penultimate"] for g in row["per_graph"]])
        count = sum(g["interior_peak"] for g in row["per_graph"])
        lines.append(
            f"| {label} | {count}/5 | {gaps.mean():.6f} ± "
            f"{gaps.std(ddof=1) / np.sqrt(5):.6f} | "
            f"{row['mean_curve']['high_minus_low']:.6f} |"
        )
    lines += [
        "",
        "## Windows: bounded, unconverged fits",
        "",
        "| Window | Upper-bin balanced gap, mean ± SE | High minus low | "
        "Iteration-cap stops | Final gradient max range |",
        "|---|---:|---:|---:|---:|",
    ]
    for length in (1, 2, 5, 10):
        row = find(data["windows"], window=length, step="final", field="balanced_norm")
        gaps = np.array([g["high_minus_penultimate"] for g in row["per_graph"]])
        ends = [
            r for r in data["window_progress"] if r["window"] == length and r["final"]
        ]
        grads = [r["gradient_max"] for r in ends]
        stopped = sum(r["termination"]["status"] == 1 for r in ends)
        lines.append(
            f"| {length} | {gaps.mean():.6f} ± "
            f"{gaps.std(ddof=1) / np.sqrt(5):.6f} | "
            f"{row['mean_curve']['high_minus_low']:.6f} | {stopped}/10 | "
            f"{min(grads):.6g}–{max(grads):.6g} |"
        )
    lines += [
        "",
        "## Every endpoint: initialization and optimization disagreement",
        "",
        "Loss compares starts within a window; windows have different objectives.",
        "Gradients depend on factor gauge; RMS differences compare the same nodes.",
        "",
        "| Graph | Window | Loss, start 1 | Loss, start 2 | "
        "Node balanced-norm RMS difference | Upper gap, start 1 | Upper gap, start 2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in data["initialization_comparison"]:
        if row["field"] != "balanced_norm":
            continue
        a, b = row["curves"]
        lines.append(
            f"| {row['seed']} | {row['window']} | {row['losses'][0]:.6f} | "
            f"{row['losses'][1]:.6f} | {row['node_rms_difference']:.6f} | "
            f"{a['high_minus_penultimate']:.6f} | "
            f"{b['high_minus_penultimate']:.6f} |"
        )
    (destination / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
