"""R3 check: numerically compare two JSON summaries (committed vs scratch rerun).

Usage: python compare_json.py <committed.json> <rerun.json> [n_worst]
Reports the number of numeric leaves, how many differ, the largest absolute and
relative differences with their JSON paths, and any structural/string differences.
"""

from __future__ import annotations

import json
import sys


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def main() -> None:
    a = dict(leaves(json.load(open(sys.argv[1]))))
    b = dict(leaves(json.load(open(sys.argv[2]))))
    n_worst = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    numeric, differ, other = 0, [], []
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            if x != y:
                other.append((key, x, y))
            continue
        numeric += 1
        if x != y:
            rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
            differ.append((rel, abs(x - y), key, x, y))
    differ.sort(reverse=True)
    print(
        f"numeric leaves {numeric}; differing {len(differ)}; non-numeric differences {len(other)}; "
        f"keys only in committed {len(only_a)}, only in rerun {len(only_b)}"
    )
    for rel, ab, key, x, y in differ[:n_worst]:
        print(f"  rel {rel:.3g} abs {ab:.3g}  {key}: {x} -> {y}")
    for key, x, y in other[:n_worst]:
        print(f"  NONNUMERIC {key}: {x!r} -> {y!r}")


if __name__ == "__main__":
    main()
