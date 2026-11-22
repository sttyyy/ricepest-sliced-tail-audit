# -*- coding: utf-8 -*-
"""Fig. same-dump read-out and rescoring methods vs the frozen SAHI baseline (audit set).

Each row is one method run on the identical candidate dump; each panel shows the paired
cluster-bootstrap difference in recall@0.5 against the baseline (percentage points, 95% CI over
audit groups) at a global threshold that matches the baseline's total false-positive count.
Panels (left to right): tail-small objects, all small objects, all objects.
Data: results/methods/R5R9-methods-yolo.json (paired_vs_baseline)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _figbase import load, out, plt, save_fig, set_paper_placement, NP, COLORS

RUN = os.environ.get("MH_RUN_METHODS", "R5R9-methods-yolo")
d = load(f"methods/{RUN}.json")
P = d["paired_vs_baseline"]
names = {"norcal": "NorCal prior", "cls_size_calib": "Class$\\times$size thr.",
         "na_agnostic": "Na rescore (agn.)", "na_perclass": "Na rescore (cls)",
         "ml_greedynmm_calib": "ML + SAHI merge", "ccpr_std_evidence": "Peak read-out (std)",
         "ccpr_nocalib": "CCPR w/o thr.", "ccpr": "CCPR"}
order = ["norcal", "cls_size_calib", "na_agnostic", "na_perclass", "ml_greedynmm_calib", "ccpr_std_evidence",
         "ccpr_nocalib", "ccpr"]
fam = {k: (NP["teal"] if k in ("norcal", "cls_size_calib", "na_agnostic", "na_perclass") else NP["violet"]) for k in order}
panels = [("tail_small", "tail-small"), ("small", "small"), ("all", "all")]
fig, axs = plt.subplots(1, 3, figsize=(5.15, 2.7), sharey=True)
y = np.arange(len(order))[::-1]
for ax, (key, lab) in zip(axs, panels):
    ax.axvline(0, color=COLORS["ref_line"], lw=0.8, ls="--", zorder=1)
    for yy, m in zip(y, order):
        r = P[m][f"{key}_vs_baseline_matched_fp"]
        if r is None:
            continue
        lo, hi = 100 * r["ci95"][0], 100 * r["ci95"][1]
        ax.hlines(yy, lo, hi, color=fam[m], lw=1.5, zorder=2)
        ax.scatter(100 * r["diff"], yy, s=24, marker="D", color=fam[m], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("$\\Delta$ " + lab + " (pp)")
axs[0].set_yticks(y, [names[m] for m in order])
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_methods.pdf"))
