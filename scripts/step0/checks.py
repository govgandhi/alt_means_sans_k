"""Background checks for step 0; prints only problems unless --verbose.

1. Corpus: node token counts track strength (walks started in proportion to strength).
2. Window-1 transitions on the real network match edge weights.
3. Gensim's update range: share of positive mass whose window-1 target score exceeds +6
(column B).
4. Vocabulary cutoff: share of nodes or words each run keeps.
5. Replay: retrain real/B/seed 0 and compare norms byte for byte (needs --replay-dir).
Run: python scripts/step0/checks.py --data data/derived/step0 --runs results/step0/runs
  --replay-dir DIR
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    report, issues = {}, []

    for name in ("real", "null"):
        z = np.load(a.data / f"{name}_walks_s0.npz")
        walks, strength = z["walks"], z["strength"]
        counts = np.bincount(walks.ravel(), minlength=len(strength))
        expected = walks.size * strength / strength.sum()
        big = expected >= 200
        ratio = counts[big] / expected[big]
        report[f"{name}_count_ratio"] = [
            float(ratio.min()),
            float(np.median(ratio)),
            float(ratio.max()),
        ]
        if abs(np.median(ratio) - 1) > 0.05 or ratio.min() < 0.7 or ratio.max() > 1.3:
            issues.append(
                f"{name}: token counts do not track strength "
                f"{report[f'{name}_count_ratio']}"
            )

    z = np.load(a.data / "real_walks_s0.npz")
    walks, strength = z["walks"], z["strength"]
    n = len(strength)
    pairs = np.concatenate(
        [walks[:, :-1].ravel().astype(np.int64) * n + walks[:, 1:].ravel()]
    )
    keys, observed = np.unique(pairs, return_counts=True)
    from corpora import load_edges  # noqa: E402

    ids, src, dst, w = load_edges(
        Path(
            "data/external/pnas_murray2023/extracted/derived/Derived/Network/precedence_flows_edgelist.csv"
        )
    )
    weight = dict(zip((src * n + dst).tolist(), w.tolist()))
    top = np.argsort(observed)[::-1][:2000]
    expect = np.array(
        [len(pairs) * weight.get(int(k), 0) / strength.sum() for k in keys[top]]
    )
    rel = observed[top] / np.maximum(expect, 1e-12)
    report["window1_transition_ratio_top2000"] = [
        float(rel.min()),
        float(np.median(rel)),
        float(rel.max()),
    ]
    if (expect == 0).any() or abs(np.median(rel) - 1) > 0.05:
        issues.append(
            "window-1 transitions do not match weights "
            f"{report['window1_transition_ratio_top2000']}"
        )

    target = np.log(w * strength.sum() / (5 * strength[src] * strength[dst]))
    report["share_positive_mass_target_above_6"] = float(w[target > 6].sum() / w.sum())
    report["share_positive_mass_target_below_minus6"] = float(
        w[target < -6].sum() / w.sum()
    )

    for path in sorted(a.runs.glob("*.npz")):
        r = np.load(path)
        report.setdefault("kept", {})[path.stem] = int(len(r["tokens"]))

    a.replay_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/step0/train.py",
            "real",
            "B",
            "0",
            "--data",
            str(a.data),
            "--out",
            str(a.replay_dir),
        ],
        check=True,
        capture_output=True,
    )
    original, replay = (
        np.load(a.runs / "real_B_s0.npz"),
        np.load(a.replay_dir / "real_B_s0.npz"),
    )
    same = all(
        np.array_equal(original[k], replay[k])
        for k in ("tokens", "raw_norm", "balanced_norm")
    )
    report["replay_identical"] = same
    if not same:
        issues.append("replay of real/B/seed 0 is not byte-identical")

    (a.runs.parent / "checks.json").write_text(json.dumps(report, indent=1))
    print("ISSUES:" if issues else "no issues", *issues, sep="\n")
    if a.verbose:
        print(json.dumps({k: v for k, v in report.items() if k != "kept"}, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
