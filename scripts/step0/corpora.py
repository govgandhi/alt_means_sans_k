"""Build step-0 corpora: weighted walks on the PNAS network, its strength-preserving
null, hep-th text.

Inputs:  --edgelist  PNAS Derived/Network/precedence_flows_edgelist.csv (each pair
  listed twice)
         --abstracts KDD Cup 2003 hep-th extracted/ folder (one .abs file per paper)
Outputs (in --out, git-ignored data/derived/step0):
  real_walks_s{k}.npz        walks on the PNAS network, seeds 0-4
  null_graph_s{k}.npz        weighted configuration-model realization k (same strengths)
  null_walks_s{k}.npz        walks on null realization k
  hepth_tokens.txt           one paper (title + abstract) per line, tokenized
Walks: 10 x n walks of 80 steps, starts proportional to strength, next node proportional
to
edge weight (PNAS S16 rule; p = q = 1). Null: stub matching on integer strengths
(Serrano-Boguna weighted configuration model); self-loops removed and reported.
Run: python scripts/step0/corpora.py --edgelist ... --abstracts ... --out
  data/derived/step0
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

WALKS_PER_NODE, LENGTH, SEEDS = 10, 80, range(5)


def load_edges(path: Path):
    src, dst, w = np.loadtxt(
        path, delimiter=",", skiprows=1, dtype=np.int64, unpack=True
    )
    ids = np.unique(src)
    return (
        ids,
        np.searchsorted(ids, src),
        np.searchsorted(ids, dst),
        w.astype(np.float64),
    )


def csr(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    strength = np.bincount(src, weights=w, minlength=n)
    cumulative = np.cumsum(w)
    first = np.searchsorted(src, np.arange(n))
    base = np.where(first > 0, cumulative[np.maximum(first - 1, 0)], 0.0)
    return dst.astype(np.int32), cumulative, base, strength


def walks(graph, rng: np.random.Generator) -> np.ndarray:
    nbr, cumulative, base, strength = graph
    n = len(strength)
    out = np.empty((WALKS_PER_NODE * n, LENGTH), dtype=np.int32)
    out[:, 0] = rng.choice(n, size=len(out), p=strength / strength.sum())
    for t in range(1, LENGTH):
        cur = out[:, t - 1]
        target = base[cur] + rng.random(len(out)) * strength[cur]
        out[:, t] = nbr[
            np.minimum(np.searchsorted(cumulative, target, side="right"), len(nbr) - 1)
        ]
    return out


def null_edges(strength: np.ndarray, rng: np.random.Generator):
    stubs = np.repeat(
        np.arange(len(strength), dtype=np.int32), strength.astype(np.int64)
    )
    rng.shuffle(stubs)
    a, b = stubs[0::2].astype(np.int64), stubs[1::2].astype(np.int64)
    loops = a == b
    lo, hi = np.minimum(a[~loops], b[~loops]), np.maximum(a[~loops], b[~loops])
    keys, counts = np.unique(lo * len(strength) + hi, return_counts=True)
    i, j = keys // len(strength), keys % len(strength)
    return (
        np.concatenate([i, j]),
        np.concatenate([j, i]),
        np.concatenate([counts, counts]).astype(np.float64),
        int(loops.sum()),
    )


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\$[^$]*\$", " ", text)  # drop inline math
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)  # drop TeX commands
    text = re.sub(r"[{}\\^_~]", " ", text.lower())
    return re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[.,;:!?()\[\]\"]", text)


def paper_text(raw: str) -> str:
    parts = raw.split("\\\\")
    header, body = (parts[1], parts[2]) if len(parts) >= 3 else ("", raw)
    title = re.search(r"^Title:(.*(?:\n[ \t].*)*)", header, flags=re.M)
    return (title.group(1) if title else "") + " " + body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edgelist", type=Path, required=True)
    p.add_argument("--abstracts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids, src, dst, w = load_edges(a.edgelist)
    n = len(ids)
    real = csr(n, src, dst, w)
    report = {
        "nodes": n,
        "total_strength": float(real[3].sum()),
        "walks": WALKS_PER_NODE * n,
        "length": LENGTH,
    }
    for k in SEEDS:
        np.savez_compressed(
            a.out / f"real_walks_s{k}.npz",
            walks=walks(real, np.random.default_rng(k)),
            strength=real[3],
            ids=ids,
        )
        ns, nd, nw, loops = null_edges(real[3], np.random.default_rng(1000 + k))
        graph = csr(n, ns, nd, nw)
        np.savez_compressed(
            a.out / f"null_graph_s{k}.npz", src=ns, dst=nd, w=nw, strength=graph[3]
        )
        np.savez_compressed(
            a.out / f"null_walks_s{k}.npz",
            walks=walks(graph, np.random.default_rng(2000 + k)),
            strength=graph[3],
            ids=ids,
        )
        lost = 1 - graph[3] / np.maximum(real[3], 1)
        report[f"null_s{k}"] = {
            "self_loop_stubs_pairs_removed": loops,
            "edges": int(len(ns) // 2),
            "max_strength_loss": float(lost.max()),
            "mean_strength_loss": float(lost.mean()),
            "isolated_nodes": int((graph[3] == 0).sum()),
        }
        print(f"seed {k} done", flush=True)
    papers = sorted(a.abstracts.rglob("*.abs"))
    with (a.out / "hepth_tokens.txt").open("w") as f:
        counts = [0, 0]
        for path in papers:
            tokens = tokenize(paper_text(path.read_text(errors="replace")))
            if tokens:
                f.write(" ".join(tokens) + "\n")
                counts[0] += 1
                counts[1] += len(tokens)
    report["hepth"] = {"papers": counts[0], "tokens": counts[1]}
    (a.out / "corpora_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
