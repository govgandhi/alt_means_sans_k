"""Paper version of the step-0 figure (PRE double-column width, vector PDF).

Panels: (a) PNAS published vectors vs organization size; (b) hep-th abstracts; (c)
random walks on BlogCatalog; (d) walks on the expected graph of a Barabasi-Albert graph
(each token an independent draw in proportion to degree) with the exact theory line.
Series in (b)-(d): Gensim defaults, PNAS settings, Schakel & Wilson settings. y = raw
norm / mean raw norm of the run, so the exact optimum (identical vectors for every node,
in any gauge, under every setting) is exactly 1. Bins: 4 per decade; each point at the
mean log x of its tokens; per-seed bin means, then mean +/- SE over 5 seeds as error
bars (bins with >= 10 tokens in every seed). Markers carry identity as well as colour
(palette validated for colour-blind separation).
Inputs: results/step0/runs/*.npz, PNAS factors table and sizes file.
Run: python scripts/step0/figure_paper.py --runs results/step0/runs --anchor-factors ...
     --anchor-sizes ... --out results/step0/fig1_problem.pdf   (also writes a 600-dpi
     PNG)
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_pnas import read_sizes  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {
    "A": "#2a78d6",
    "B": "#eb6834",
    "SW": "#1baf7a",
}  # validated: CVD all-pairs dE 9.2
MARKERS = {
    "A": "o",
    "B": "s",
    "SW": "^",
    "published": "D",
}  # identity never by color alone
LABELS = {
    "A": "Gensim defaults",
    "B": "PNAS settings",
    "SW": "Schakel & Wilson settings",
}
MIN_PER_BIN = 10
THEORY = "exact optimum, all settings (theory)"
SW_LABEL_AT = {  # axes position and alignment of the direct label, next to the curve
    "text": (0.02, 0.97, "left", "top"),
    "blog": (0.02, 0.97, "left", "top"),
    "ba": (0.97, 0.15, "right", "bottom"),
}


def edges_for(x: np.ndarray) -> np.ndarray:
    lo, hi = np.floor(np.log10(x.min()) * 4) / 4, np.ceil(np.log10(x.max()) * 4) / 4
    return 10 ** np.arange(lo, hi + 0.25, 0.25)


def curve(paths: list[Path]):
    runs = [np.load(p) for p in paths]
    values = [(r["x"], r["raw_norm"] / r["raw_norm"].mean()) for r in runs]
    edges = edges_for(np.concatenate([x for x, _ in values]))
    per_seed = []
    for x, y in values:
        idx = np.digitize(x, edges)
        per_seed.append(
            {
                b: (y[idx == b].mean(), (idx == b).sum(), np.log(x[idx == b]).mean())
                for b in np.unique(idx)
            }
        )
    keep = sorted(
        b
        for b in per_seed[0]
        if all(b in s and s[b][1] >= MIN_PER_BIN for s in per_seed)
    )
    m = np.array([[s[b][0] for b in keep] for s in per_seed])
    position = np.exp(
        np.array([[s[b][2] for b in keep] for s in per_seed]).mean(0)
    )  # mean log x in bin
    return position, m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m))


def plot(ax, x, m, se, color, label, marker="o"):
    ax.errorbar(
        x,
        m,
        yerr=se,
        color=color,
        linewidth=1.1,
        marker=marker,
        markersize=2.6,
        elinewidth=0.7,
        capsize=1.4,
        capthick=0.6,
        label=label,
    )


def style(ax, letter, title, xlabel):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("norm / mean norm")
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)
    ax.text(
        -0.17,
        1.04,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
    )
    ax.grid(True, which="major", color=GRID, linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("runs", "anchor_factors", "anchor_sizes", "out"):
        p.add_argument(
            f"--{name.replace('_', '-')}", dest=name, type=Path, required=True
        )
    a = p.parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "axes.edgecolor": MUTED,
            "pdf.fonttype": 42,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), constrained_layout=True)
    (a_ax, b_ax), (c_ax, d_ax) = axes

    sizes = read_sizes(a.anchor_sizes)
    with a.anchor_factors.open() as f:
        pairs = np.array(
            [
                (sizes[r["cwts_org_no"]], float(r["l2norm"]))
                for r in csv.DictReader(f)
                if r["cwts_org_no"] in sizes
            ]
        )
    sx, sy = pairs[:, 0], pairs[:, 1] / pairs[:, 1].mean()
    edges = edges_for(sx)
    idx = np.digitize(sx, edges)
    bins = [b for b in np.unique(idx) if (idx == b).sum() >= MIN_PER_BIN]
    cx = np.array(
        [np.exp(np.log(sx[idx == b]).mean()) for b in bins]
    )  # mean log size in bin
    m = np.array([sy[idx == b].mean() for b in bins])
    se = np.array([sy[idx == b].std(ddof=1) / np.sqrt((idx == b).sum()) for b in bins])
    plot(
        a_ax, cx, m, se, COLORS["B"], "PNAS published vectors", MARKERS["published"]
    )  # trained with PNAS settings
    style(
        a_ax,
        "a",
        "Seen in published vectors (PNAS, Murray et al. 2023)",
        "organization size (mean yearly authors)",
    )

    for ax, corpus, letter, title, xlabel in (
        (
            b_ax,
            "text",
            "b",
            "Also in text; the peak moves with training settings",
            "word frequency",
        ),
        (
            c_ax,
            "blog",
            "c",
            "Also in a real network (BlogCatalog)",
            "node degree",
        ),
        (
            d_ax,
            "ba",
            "d",
            "Even where the exact optimum is flat (expected BA graph)",
            "node degree",
        ),
    ):
        columns = ("A", "B", "SW")
        for column in columns:
            x, m, se = curve(sorted(a.runs.glob(f"{corpus}_{column}_s*.npz")))
            plot(ax, x, m, se, COLORS[column], LABELS[column], MARKERS[column])
            if column == "SW":  # direct label: aqua is below 3:1 contrast on white
                x0, y0, ha, va = SW_LABEL_AT[corpus]
                ax.text(
                    x0,
                    y0,
                    "Schakel & Wilson\nsettings",
                    transform=ax.transAxes,
                    fontsize=6.3,
                    color=INK,
                    ha=ha,
                    va=va,
                )
        if corpus == "ba":
            ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.1, label=THEORY)
        style(ax, letter, title, xlabel)
    handles = [
        plt.Line2D(
            [], [], color=COLORS[c], lw=1.1, marker=MARKERS[c], ms=2.6, label=LABELS[c]
        )
        for c in ("A", "B")
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            color=COLORS["SW"],
            lw=1.1,
            marker=MARKERS["SW"],
            ms=2.6,
            label=LABELS["SW"],
        ),
        plt.Line2D([], [], color=MUTED, lw=1.1, ls=":", label=THEORY),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=1.8,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix(".png"), dpi=600)
    print(f"saved {a.out} and {a.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
