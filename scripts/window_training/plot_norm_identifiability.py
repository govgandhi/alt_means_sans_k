"""Plot the exact construction and read-only audit of existing normalized profiles.

Run after exact_norm_profiles.py and normalized_profiles.py:
  python3 scripts/window_training/plot_norm_identifiability.py [--root DIRECTORY]
Reads their JSON evidence and verified manifests; writes one composite figure,
an all-node plot, and a deterministic provenance manifest. No fitting occurs.
Dependencies are pinned in analysis/window_training_exploration_requirements.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    results: Path = Path("results/norm_identifiability")
    figure_tolerance: float = 1e-6
    colors: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: plt.Figure, path: Path) -> list[str]:
    names = []
    for suffix in ("png", "pdf"):
        target = path.with_suffix(f".{suffix}")
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(target, dpi=220, metadata=metadata)
        names.append(target.name)
    plt.close(fig)
    return names


def plot(config: Config) -> None:
    directory = config.root / config.results
    sources = {}
    for folder, key in (("exact", "outputs_sha256"), ("empirical", "artifact_sha256")):
        manifest = read(directory / folder / "manifest.json")
        for name, expected in manifest[key].items():
            path = directory / folder / name
            if sha(path) != expected:
                raise ValueError(f"Evidence checksum mismatch: {path}")
            sources[f"{folder}/{name}"] = expected
        sources[f"{folder}/manifest.json"] = sha(directory / folder / "manifest.json")
    exact = read(directory / "exact/exact.json")
    empirical = read(directory / "empirical/summary.json")
    nodes = read(directory / "empirical/nodes.json")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "legend.fontsize": 11.5,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
    chosen = next(
        row for row in exact["runs"] if row["tolerance"] == config.figure_tolerance
    )
    degree = np.array(exact["degree"])
    for example, color in zip(chosen["examples"], config.colors, strict=False):
        radius = np.array(example["normalized_radius"])
        means = [float(radius[degree == d].mean()) for d in (1, 2, 3)]
        axes[0].plot(
            (1, 2, 3),
            means,
            marker="o",
            color=color,
            label=example["profile"].capitalize(),
        )
    axes[0].set(
        title="A  Same graph and objective",
        xlabel="Degree",
        ylabel="Balanced radius / node RMS",
        xticks=[1, 2, 3],
        ylim=(0, 1.6),
    )
    axes[0].text(
        0.04,
        0.04,
        "Each loss gap ≤ $10^{-6}$\nEqual RMS across profiles",
        transform=axes[0].transAxes,
        fontsize=11.5,
    )
    axes[0].legend(
        loc="upper center",
        ncol=3,
        fontsize=11,
        frameon=False,
        handlelength=0.8,
        columnspacing=0.5,
    )

    graph_nodes = [
        row
        for row in nodes
        if row["window"] == 5 and row["initialization"] - row["seed"] == 100000
    ]
    degrees = np.concatenate([row["degree"] for row in graph_nodes])
    bins = np.concatenate([row["bin"] for row in graph_nodes])
    centers = np.array(
        [np.exp(np.mean(np.log(degrees[bins == group]))) for group in np.unique(bins)]
    )
    for window, color in zip((5, 10), config.colors[2:], strict=True):
        row = next(row for row in empirical["windows"] if row["window"] == window)
        curve = row["normalized_curve"]
        for graph_curve in curve["values_by_graph"]:
            axes[1].plot(centers, graph_curve, color=color, alpha=0.25, lw=0.7)
        axes[1].errorbar(
            centers,
            curve["mean"],
            yerr=curve["se"],
            color=color,
            marker="o",
            capsize=2,
            label=f"Window {window}",
        )
    axes[1].set(
        title="B  Existing fits, scale removed",
        xlabel="Degree-bin center",
        ylabel="Balanced radius / node RMS",
        xscale="log",
    )
    axes[1].legend(frameon=False)
    axes[1].set_xticks([3, 5, 10, 15], labels=["3", "5", "10", "15"])
    axes[1].xaxis.set_minor_locator(NullLocator())

    for x, (window, color) in enumerate(zip((1, 2, 5, 10), config.colors, strict=True)):
        pairs = [row for row in empirical["paired_starts"] if row["window"] == window]
        values = [row["normalized_profile"]["node_rms"] for row in pairs]
        axes[2].scatter(
            x + np.linspace(-0.12, 0.12, len(values)),
            values,
            color=color,
            s=30,
            zorder=3,
        )
    axes[2].axhline(np.sqrt(2), color="0.65", linestyle=":", linewidth=1)
    axes[2].set(
        title="C  Saved-start disagreement",
        xlabel="Context window",
        ylabel="Normalized node RMS difference",
        yscale="log",
        xticks=range(4),
        xticklabels=["1", "2", "5", "10"],
        ylim=(8e-7, 4),
    )
    axes[2].text(
        0.04,
        0.78,
        "One point per graph\nAgreement does not\nprove uniqueness",
        transform=axes[2].transAxes,
        va="top",
        fontsize=11.5,
    )
    outputs = save(fig, directory / "norm_identifiability")

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), layout="constrained", sharex=True, sharey=True
    )
    seed_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555")
    for ax, window in zip(axes.flat, (1, 2, 5, 10), strict=True):
        for row in nodes:
            if row["window"] != window:
                continue
            first = row["initialization"] - row["seed"] == 100000
            ax.scatter(
                row["degree"],
                row["normalized_radius"],
                s=7,
                alpha=0.3,
                marker="o" if first else "x",
                color=seed_colors[row["seed"]],
                rasterized=True,
            )
        ax.set(
            title=f"Window {window}: all 6,000 node readouts",
            xlabel="Degree",
            ylabel="Balanced radius / node RMS",
            xscale="log",
            yscale="log",
        )
    fig.suptitle(
        "All nodes retained (log scales); colors identify graphs, "
        "circles/crosses identify starts",
        fontsize=12,
    )
    outputs += save(fig, directory / "normalized_all_nodes")
    manifest = {
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(config.root)): sha(Path(__file__))
        },
        "input_sha256": sources,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "figure_tolerance": config.figure_tolerance,
        "output_sha256": {name: sha(directory / name) for name in outputs},
    }
    (directory / "figure_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({"figures": outputs}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Config.root)
    args = parser.parse_args()
    plot(Config(root=args.root))
