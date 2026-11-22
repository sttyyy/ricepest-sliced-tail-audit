# -*- coding: utf-8 -*-
"""Fig. YST native-resolution source re-check (held-out test images; not a long-tail test).

Left: outcome of every GT object per class under the frozen pipeline (SAHI GREEDYNMM at the floor, cap 100)
      and under NMS read-out with a 300-box cap; 100% stacked bars; n in tick labels.
Right: stage survival per class under the frozen pipeline (V full tile view, C candidate, M merged,
      >=tau above threshold, D emitted); hollow marker = oracle full-view read-out.
Data: results/ledger/R8-ledger-yst-T640.json, results/ledger/S-ledger-yst-T640-nms-cap300.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.patches import Patch
from _figbase import load, out, plt, save_fig, set_paper_placement, CATS, CAT_LABEL, CAT_COLOR, NP

A = load("ledger/R8-ledger-yst-T640.json")["summary"]["audit"]
B = load("ledger/S-ledger-yst-T640-nms-cap300.json")["summary"]["audit"]
cls = [("WF", NP["teal"], "s"), ("MR", NP["red_strong"], "^"), ("NC", NP["violet"], "D")]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.5), layout="constrained")
rows = []
for c, _, _ in cls:
    rows.append((f"{c} frozen", A[c + "|all"]))
    rows.append((f"{c} NMS", B[c + "|all"]))
y = -(np.arange(len(rows)) + 0.35 * (np.arange(len(rows)) // 2))
for yy, (lab, t) in zip(y, rows):
    n = t["n"]; left = 0.0
    for k in CATS:
        v = t.get("cat_" + k, 0) / n
        ax.barh(yy, v, 0.72, left=left, color=CAT_COLOR[k], edgecolor="white", linewidth=0.4, zorder=2)
        left += v
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(0, 1)
ax.set_xlabel("Share of GT objects")
stages = ["V", "C", "M", "S", "D"]
for c, col, mk in cls:
    t = A[f"{c}|all"]; n = t["n"]
    yv = [t.get(s, 0) / n for s in stages]
    bx.plot(range(5), yv, color=col, marker=mk, ms=3.6, lw=1.2, label=c, zorder=3)
    bx.scatter([4], [t.get("oracle", 0) / n], s=22, facecolor="white", edgecolor=col, linewidth=1.1, marker=mk, zorder=4)
bx.set_xticks(range(5), ["View", "Cand.", "Merged", "$\\geq\\tau$", "Emitted"], rotation=90, ha="center")
bx.set_ylim(0, 1.03)
bx.set_ylabel("Surviving")
bx.legend(frameon=False, loc="lower left", handlelength=1.3, borderaxespad=0.2)
fig.legend(handles=[Patch(color=CAT_COLOR[k], label=CAT_LABEL[k]) for k in CATS], loc="outside upper center",
           ncol=4, frameon=False, handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_yst.pdf"))
