"""R3 check: verify every SHA-256 recorded in the repo's freeze/completion manifests.

Usage: python check_manifests.py <repo_root>
Reads only. For manifests that do not name their files (prospective freeze),
matches each recorded digest against candidate files and reports which file, if any,
has that digest at the reviewed commit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root: Path) -> None:
    # 1. Prospective freeze manifest: keys name roles, not files.
    freeze = json.loads(
        (root / "results/prospective_validation/freeze_manifest.json").read_text()
    )
    candidates = {
        "protocol_sha256": ["results/prospective_validation/protocol.json"],
        "human_protocol_sha256": ["analysis/prospective_validation_protocol.md"],
        "calibration_sha256": [
            "results/prospective_validation/calibration.jsonl",
            "results/prospective_validation/calibration_summary.json",
        ],
        "predictions_sha256": [
            "results/prospective_validation/frozen_predictions.jsonl",
            "results/prospective_validation/frozen_prediction_summary.json",
        ],
    }
    print("== results/prospective_validation/freeze_manifest.json")
    for key, recorded in freeze.items():
        if not key.endswith("sha256"):
            print(f"  {key}: {recorded}")
            continue
        hits = [
            c
            for c in candidates.get(key, [])
            if (root / c).exists() and sha(root / c) == recorded
        ]
        status = (
            "MATCH " + ", ".join(hits)
            if hits
            else "NO MATCH among " + ", ".join(candidates.get(key, []))
        )
        print(f"  {key}: {recorded[:12]}... {status}")
        if not hits:
            for c in candidates.get(key, []):
                if (root / c).exists():
                    print(f"      actual {c}: {sha(root / c)[:12]}...")

    # 2. Generic manifests that map relative filenames to digests.
    generic = [
        "results/negative_sampling_revalidation/freeze_manifest.json",
        "results/negative_sampling_revalidation/completion.json",
        "results/negative_sampling_revalidation/calibration_manifest.json",
        "results/negative_sampling_revalidation/report_manifest.json",
        "results/subsampling_origin/completion.json",
        "results/training_trajectory/completion.json",
        "results/training_trajectory/report_manifest.json",
        "results/negative_sampling_control/completion.json",
        "results/window_training_continuation/completion.json",
        "results/window_training_continuation/report/manifest.json",
        "results/norm_identifiability/figure_manifest.json",
        "results/norm_identifiability/empirical/manifest.json",
        "results/norm_identifiability/exact/manifest.json",
        "results/window_training_exploration/report/manifest.json",
        "results/realized_graph_audit/provenance.json",
    ]
    for rel in generic:
        path = root / rel
        if not path.exists():
            print(f"== {rel}: MISSING")
            continue
        data = json.loads(path.read_text())
        checked = mismatched = missing = 0
        problems: list[str] = []

        def walk(obj: object, prefix: str = "") -> None:
            nonlocal checked, mismatched, missing
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if (
                        isinstance(v, str)
                        and len(v) == 64
                        and all(ch in "0123456789abcdef" for ch in v)
                    ):
                        # resolve k as a path relative to the manifest dir or to repo root
                        for base in (path.parent, root):
                            target = base / k
                            if target.is_file():
                                checked += 1
                                if sha(target) != v:
                                    mismatched += 1
                                    problems.append(
                                        f"MISMATCH {target.relative_to(root)}"
                                    )
                                break
                        else:
                            if "/" in k or "." in k:
                                missing += 1
                                problems.append(f"unresolved key {prefix}{k}")
                    else:
                        walk(v, prefix + k + "/")
            elif isinstance(obj, list):
                for v in obj:
                    walk(v, prefix)

        walk(data)
        print(
            f"== {rel}: {checked} file digests checked, {mismatched} mismatched, {missing} unresolved"
        )
        for p in problems[:12]:
            print("   ", p)
        if len(problems) > 12:
            print(f"    ... {len(problems) - 12} more")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
