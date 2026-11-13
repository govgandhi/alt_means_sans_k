"""Report text font sizes in each figure PDF, scaled to print size in main.pdf."""

import sys
from collections import defaultdict

import fitz  # PyMuPDF

ROOT = sys.argv[1]
FIGS = {
    "Fig1 turnover_surface_qualified": "results/realized_graph_audit/turnover_surface_qualified.pdf",
    "Fig2 wide_window_comparison": "results/window_training_continuation/report/wide_window_comparison.pdf",
    "Fig3 norm_identifiability": "results/norm_identifiability/norm_identifiability.pdf",
    "Fig4 gensim_threshold_sweep": "results/gensim_threshold_sweep/gensim_threshold_sweep.pdf",
    "Fig5 unfiltered_schedule_comparison": "results/training_trajectory/unfiltered_schedule_comparison.pdf",
    "Fig6 prospective_validation": "results/prospective_validation/prospective_validation.pdf",
}
PRINT_WIDTH = 510.0  # pt, \textwidth in revtex reprint

for name, rel in FIGS.items():
    doc = fitz.open(f"{ROOT}/{rel}")
    page = doc[0]
    w = page.rect.width
    scale = PRINT_WIDTH / w
    sizes = defaultdict(list)
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                txt = span["text"].strip()
                if txt:
                    sizes[round(span["size"], 2)].append(txt)
    print(f"== {name}: native width {w:.1f} pt, scale {scale:.3f}")
    for s in sorted(sizes):
        ex = sizes[s]
        uniq = []
        for t in ex:
            if t not in uniq:
                uniq.append(t)
        print(
            f"  native {s:5.2f} pt -> print {s * scale:4.2f} pt  n={len(ex):3d}  e.g. {uniq[:8]}"
        )
