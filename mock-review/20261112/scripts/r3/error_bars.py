"""R3 check: node-pooled SE (what the plotting code draws) vs seed-level SE (what
the Fig. 1 and Fig. 6 captions claim).

Usage:
  python error_bars.py fig6 <outcomes.jsonl>
  python error_bars.py fig1 <node_results.csv>   (scratch rerun of simulate_turnover_surface.py)
Bins: quantiles of pooled log degree, as in the repository's binned_curve; bin means
per replicate/seed; SE = sd(replicate means)/sqrt(R). Reports the ratio seed-SE/node-SE.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np


def bins(log_degree: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(log_degree, np.linspace(0.0, 1.0, n_bins + 1)))
    return np.clip(np.digitize(log_degree, edges[1:-1], right=True), 0, len(edges) - 2)


def compare(groups: dict, n_bins: int, label: str) -> None:
    ratios = []
    for key, rows in sorted(groups.items()):
        deg = np.array([r[0] for r in rows])
        val = np.array([r[1] for r in rows])
        rep = np.array([r[2] for r in rows])
        lab = bins(np.log(deg), n_bins)
        out = []
        for b in np.unique(lab):
            m = lab == b
            node_se = float(np.std(val[m]) / math.sqrt(m.sum()))
            reps = sorted(set(rep[m]))
            means = [float(np.mean(val[m & (rep == r)])) for r in reps]
            seed_se = (
                float(np.std(means, ddof=1) / math.sqrt(len(means)))
                if len(means) > 1
                else float("nan")
            )
            out.append((node_se, seed_se))
            ratios.append(seed_se / node_se if node_se > 0 else float("nan"))
        print(
            f"{label} {key}: node-SE "
            + " ".join(f"{a:.4f}" for a, _ in out)
            + " | seed-SE "
            + " ".join(f"{b:.4f}" for _, b in out)
        )
    r = np.array([x for x in ratios if np.isfinite(x)])
    print(
        f"{label}: seed-SE / node-SE ratio median {np.median(r):.2f}, range {r.min():.2f}-{r.max():.2f}"
    )


def main() -> None:
    kind, path = sys.argv[1], sys.argv[2]
    groups: dict = defaultdict(list)
    if kind == "fig6":
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                groups[(row["family"], row["threshold"])].append(
                    (row["degree"], row["raw_norm"], row["replicate"])
                )
        compare(groups, 6, "Fig6")
    else:
        keep = {
            "beta_000_lambda_000",
            "beta_033_lambda_000",
            "beta_067_lambda_000",
            "beta_100_lambda_000",
            "beta_000_lambda_100",
            "holdout_t3e-4",
        }
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["condition"] in keep:
                    groups[row["condition"]].append(
                        (float(row["degree"]), float(row["raw_norm"]), int(row["seed"]))
                    )
        compare(groups, 8, "Fig1")


if __name__ == "__main__":
    main()
