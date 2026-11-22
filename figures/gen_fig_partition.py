# -*- coding: utf-8 -*-
"""Fig. recovery-ceiling partition on the group-disjoint audit set (frozen detector, SAHI T=320).

Every GT object's outcome (detected / no candidate / pre-suppression only / post-hoc recoverable)
as 100% stacked bars per frequency stratum x size bin; object counts in the tick labels.
Rows with no objects are omitted.
Data: results/ledger/<RUN>.json (summary.audit)
Placement: full text width.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.patches import Patch
from _figbase import load, out, plt, save_fig, set_paper_placement, STRATA, CATS, CAT_LABEL, CAT_COLOR, COLORS

RUN = os.environ.get("MH_RUN", "R3-ledger-yolo-T320")
OUTNAME = os.environ.get("MH_OUT", "fig_partition.pdf")
led = load(f"ledger/{RUN}.json")["summary"]["audit"]

rows, gaps = [], []
for gi, s in enumerate(STRATA):
    for z, zl in (("tiny", "<16$^2$"), ("s16_32", "16$^2$–32$^2$"), ("notsmall", "$\\geq$32$^2$")):
        t = led.get(f"{s}|{z}", {})
        if t.get("n", 0):
            rows.append((f"{s.capitalize()} {zl} (n={t['n']})", t)); gaps.append(gi)
fig, ax = plt.subplots(figsize=(5.15, 0.25 * len(rows) + 1.0))
y = -(np.arange(len(rows)) + 0.45 * np.array(gaps))
for yy, (lab, t) in zip(y, rows):
    n = t["n"]; left = 0.0
    for c in CATS:
        v = t.get("cat_" + c, 0) / n
        ax.barh(yy, v, 0.76, left=left, color=CAT_COLOR[c], edgecolor="white", linewidth=0.4, zorder=2)
        left += v
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(0, 1.0)
ax.set_xlabel("Share of GT objects")
handles = [Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in CATS]
ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.4, 1.0),
          handlelength=0.9, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.2)
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out(OUTNAME))
