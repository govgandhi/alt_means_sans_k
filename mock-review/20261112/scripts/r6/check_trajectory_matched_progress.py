"""R6 check: compare saved training-trajectory curves at matched optimisation progress.

Reads results/training_trajectory (read-only). For every graph/threshold/schedule and
checkpoint prints the expected loss, the binned raw-norm curve (8 requested log-degree
quantile bins), the peak bin, and the share of positive/negative weighted mass whose
score lies outside Gensim's (-6, 6) update window.
Run: r6/.venv/bin/python r6/check_trajectory_matched_progress.py
"""
import json
import numpy as np
D = "<scratch>/round/wt/results/training_trajectory/"

def curve(deg, vals):
    ld = np.log(deg); e = np.unique(np.quantile(ld, np.linspace(0, 1, 9)))
    lab = np.clip(np.digitize(ld, e[1:-1], right=True), 0, len(e) - 2)
    return np.array([vals[lab == b].mean() for b in range(len(e) - 1) if (lab == b).any()])

for t in ["0", "0.003", "0.001", "0.0003"]:
    for sched in ["baseline_then_floor", "constant"]:
        print("== t", t, sched)
        for seed in range(5):
            r = json.load(open(D + f"runs/seed{seed}_t{t}_{sched}.json"))
            g = json.load(open(D + r["graph_file"]))
            deg = np.array(g["degree"])
            line = []
            for s in r["snapshots"]:
                if s["epoch"] == 0:
                    continue
                m = curve(deg, np.array(s["raw_norm"]))
                pk = int(np.argmax(m))
                line.append(f"e{s['epoch']}: L={s['expected_loss']:.4f} pk{pk} out(+/-)={s['positive_mass_outside_update_window']:.3f}/{s['negative_mass_outside_update_window']:.2f} [{' '.join(f'{x:.2f}' for x in m)}]")
            if seed < 2 or t == "0":
                print(f" seed{seed}: " + " | ".join(line))
