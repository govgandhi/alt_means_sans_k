"""Step-0 problem figure: published curves reproduced (top row) and the 3 x 2 grid.

Inputs: results/step0/runs/*.npz (train.py), results/step0/theory.json (theory.py), and
  the PNAS
anchor inputs (see anchor_pnas.py). Bins: log-spaced, 4 per decade; per-seed bin means,
then
mean +/- SE across the 5 seeds, keeping bins with >= 10 tokens in every seed; grey dots
show every token (seed 0).
Run: python scripts/step0/figure.py --runs results/step0/runs --anchor-factors ...
  --anchor-sizes ...
     --out results/step0/step0_problem.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_pnas import read_sizes  # noqa: E402

INK, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#ecebe7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MIN_PER_BIN = 10
ROWS = [
    ("real", "PNAS organization network, weighted walks", "node strength"),
    ("null", "Same strengths, random wiring (null)", "node strength"),
    ("text", "hep-th abstracts (Schakel & Wilson corpus)", "word frequency"),
]
COLS = [("A", "A: Gensim defaults (skip-gram)"), ("B", "B: PNAS-reported settings")]


def edges_for(x: np.ndarray) -> np.ndarray:
    lo, hi = np.floor(np.log10(x.min()) * 4) / 4, np.ceil(np.log10(x.max()) * 4) / 4
    return 10 ** np.arange(lo, hi + 0.25, 0.25)


def seed_curves(paths: list[Path], key: str):
    data = [np.load(p) for p in paths]
    allx = np.concatenate([d["x"] for d in data])
    edges = edges_for(allx[allx > 0])
    per_seed = []
    for d in data:
        idx = np.digitize(d["x"], edges)
        per_seed.append(
            {b: (d[key][idx == b].mean(), (idx == b).sum()) for b in np.unique(idx)}
        )
    keep = [
        b
        for b in per_seed[0]
        if all(b in s and s[b][1] >= MIN_PER_BIN for s in per_seed)
    ]
    keep = sorted(keep)
    centers = np.sqrt(edges[np.array(keep) - 1] * edges[np.array(keep)])
    values = np.array([[s[b][0] for b in keep] for s in per_seed])
    return (
        centers,
        values.mean(0),
        values.std(0, ddof=1) / np.sqrt(len(per_seed)),
        min(d["x"].min() for d in data),
    )


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel("vector norm", fontsize=8.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(ax, paths, theory=None, shade=True):
    first = np.load(paths[0])
    ax.scatter(
        first["x"],
        first["raw_norm"],
        s=2,
        color=MUTED,
        alpha=0.12,
        linewidths=0,
        zorder=1,
    )
    for key, color, width, label in (
        ("raw_norm", BLUE, 2, "raw norm (what users read)"),
        ("balanced_norm", ORANGE, 1.4, "balanced norm"),
    ):
        x, m, se, xmin = seed_curves(paths, key)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.2, linewidth=0)
        ax.plot(
            x, m, color=color, linewidth=width, marker="o", markersize=3, label=label
        )
    if shade:
        ax.set_xlim(left=1)
        ax.axvspan(1, xmin, color=SHADE, zorder=0)
    if theory is not None:
        ax.axhline(
            theory,
            color=MUTED,
            linestyle=":",
            linewidth=1.6,
            label="population optimum (theory, balanced)",
        )


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("runs", "anchor_factors", "anchor_sizes", "out"):
        p.add_argument(
            f"--{name.replace('_', '-')}", dest=name, type=Path, required=True
        )
    a = p.parse_args()
    theory = json.loads((a.runs.parent / "theory.json").read_text())
    plt.rcParams.update(
        {"text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK}
    )
    fig = plt.figure(figsize=(11, 13.5), facecolor=SURFACE)
    grid = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.22)

    ax = fig.add_subplot(grid[0, 0])
    sizes = read_sizes(a.anchor_sizes)
    with a.anchor_factors.open() as f:
        pairs = [
            (sizes[r["cwts_org_no"]], float(r["l2norm"]))
            for r in csv.DictReader(f)
            if r["cwts_org_no"] in sizes
        ]
    sx, sy = np.array(pairs).T
    ax.scatter(sx, sy, s=3, color=MUTED, alpha=0.15, linewidths=0)
    edges = edges_for(sx)
    idx = np.digitize(sx, edges)
    bins = [b for b in np.unique(idx) if (idx == b).sum() >= 20]
    cx = np.sqrt(edges[np.array(bins) - 1] * edges[np.array(bins)])
    ax.plot(
        cx,
        [sy[idx == b].mean() for b in bins],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=3,
    )
    style(
        ax,
        "Published: PNAS vectors (authors' own, no retraining)",
        "organization size (mean yearly authors)",
    )

    ax = fig.add_subplot(grid[0, 1])
    draw(ax, sorted(a.runs.glob("text_SW_s*.npz")), shade=False)
    style(
        ax,
        "Published settings reproduced: Schakel & Wilson on hep-th",
        "word frequency",
    )

    for r, (corpus, row_title, xlabel) in enumerate(ROWS, start=1):
        for c, (column, col_title) in enumerate(COLS):
            ax = fig.add_subplot(grid[r, c])
            level = theory[column]["level_mean"] if corpus == "null" else None
            draw(
                ax,
                sorted(a.runs.glob(f"{corpus}_{column}_s*.npz")),
                theory=level,
                shade=corpus != "text",
            )
            style(ax, f"{row_title}\n{col_title}", xlabel)
    handles = [
        plt.Line2D(
            [],
            [],
            color=BLUE,
            lw=2,
            marker="o",
            ms=3,
            label="raw norm (what users read)",
        ),
        plt.Line2D(
            [], [], color=ORANGE, lw=1.4, marker="o", ms=3, label="balanced norm"
        ),
        plt.Line2D(
            [], [], color=MUTED, lw=1.6, ls=":", label="population optimum (theory)"
        ),
        plt.Rectangle((0, 0), 1, 1, color=SHADE, label="removed by vocabulary cutoff"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.text(
        0.5,
        0.005,
        "Bins: 4 per decade; mean ± SE over 5 seeds (bins with ≥10 tokens in every "
        "seed); grey dots: every token, raw norm, seed 0. "
        "Walks: 10 per node, 80 steps, started ∝ strength, next node ∝ edge weight.",
        ha="center",
        fontsize=7.5,
        color=MUTED,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
