"""Redraw PNAS's norm-size curve (Murray et al. 2023, Fig. 5) from their deposited data.

Inputs (from Figshare 13072790, see data/external/pnas_murray2023/PROVENANCE.md):
  --factors  Derived/Embeddings/precedence/factors/d300_ws1_precedence_w2vfactors.csv
             (cwts_org_no, l2norm: their trained in-vector norms; no training here)
  --sizes    Additional/2008-2019_inst_sizes.txt (person_count per organization per
  year)
  --gml      Derived/Network/co_occur_network.gml (node label = cwts_org_no, ISO
  country)
Size = mean yearly number of unique authors over the years an organization appears
(our reading of their S11 definition). Output: --out PNG and a JSON summary beside it.
Run: python scripts/step0/anchor_pnas.py --factors ... --sizes ... --gml ... --out
  results/step0/anchor_pnas.png
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def read_sizes(path: Path) -> dict[str, float]:
    yearly = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            yearly[row["cwts_org_no"]].append(float(row["person_count"]))
    return {org: float(np.mean(v)) for org, v in yearly.items()}


def read_countries(path: Path) -> dict[str, str]:
    text = path.read_text()
    return dict(re.findall(r'label "([^"]+)"\s+ISO "([^"]*)"', text))


def binned(x: np.ndarray, y: np.ndarray, per_decade: int = 4, min_count: int = 20):
    edges = 10 ** np.arange(
        np.floor(np.log10(x.min())), np.log10(x.max()) + 1 / per_decade, 1 / per_decade
    )
    idx = np.digitize(x, edges)
    rows = []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= min_count:
            rows.append(
                (
                    float(np.sqrt(edges[b - 1] * edges[b])),
                    float(y[sel].mean()),
                    float(y[sel].std(ddof=1) / np.sqrt(sel.sum())),
                    int(sel.sum()),
                )
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("factors", "sizes", "gml", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    sizes, iso = read_sizes(a.sizes), read_countries(a.gml)
    with a.factors.open() as f:
        rows = [(r["cwts_org_no"], float(r["l2norm"])) for r in csv.DictReader(f)]
    matched = [
        (org, sizes[org], norm) for org, norm in rows if org in sizes and sizes[org] > 0
    ]
    size = np.array([m[1] for m in matched])
    norm = np.array([m[2] for m in matched])
    usa = np.array([iso.get(m[0]) == "USA" for m in matched])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(
        size,
        norm,
        s=4,
        color=MUTED,
        alpha=0.18,
        linewidths=0,
        label=f"organizations (n={len(size):,})",
    )
    summary = {
        "matched_organizations": int(len(size)),
        "factors_rows": len(rows),
        "unmatched": len(rows) - len(matched),
        "usa_organizations": int(usa.sum()),
    }
    for sel, color, name in (
        (np.ones_like(usa), BLUE, "all organizations"),
        (usa, ORANGE, "U.S. organizations"),
    ):
        b = binned(size[sel], norm[sel])
        x, m, se = (np.array([r[i] for r in b]) for i in range(3))
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x,
            m,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{name}, binned mean ± SE",
        )
        peak = int(np.argmax(m))
        summary[name] = {"peak_size": x[peak], "peak_norm": m[peak], "bins": b}
    ax.set_xscale("log")
    ax.set_xlabel("organization size (mean yearly unique authors, log scale)")
    ax.set_ylabel("L2 norm of organization vector")
    ax.set_title(
        "PNAS curve redrawn from the authors' deposited vectors (no retraining)",
        loc="left",
        fontsize=11,
    )
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {
                k: v
                if not isinstance(v, dict)
                else {kk: vv for kk, vv in v.items() if kk != "bins"}
                for k, v in summary.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
