# -*- coding: utf-8 -*-
"""Fig. cross-detector check of the recovery-ceiling partition (audit set).

Rows: detector x merge read-out (frozen SAHI GREEDYNMM-IOS at the 0.01 floor, and NMS-IoU 0.5),
for tail-small (left, n=38) and head-small (right, n=1686) objects; 100% stacked outcome shares.
Data: results/ledger/{R3-ledger-yolo-T320,S-ledger-yolo-T320-nms,R7-ledger-fcos-T320,S-ledger-fcos-T320-nms_iou}.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.patches import Patch
from _figbase import load, out, plt, save_fig, set_paper_placement, CATS, CAT_LABEL, CAT_COLOR

runs = [("YOLO11n\nSAHI merge", "R3-ledger-yolo-T320"), ("YOLO11n\nNMS", "S-ledger-yolo-T320-nms"),
        ("FCOS\nSAHI merge", "R7-ledger-fcos-T320"), ("FCOS\nNMS", "S-ledger-fcos-T320-nms_iou")]
fig, axs = plt.subplots(1, 2, figsize=(5.15, 2.3), sharey=True, layout="constrained")
y = np.arange(len(runs))[::-1]
for ax, key, lab in zip(axs, ("tail|small", "head|small"), ("tail-small", "head-small")):
    for yy, (name, run) in zip(y, runs):
        t = load(f"ledger/{run}.json")["summary"]["audit"][key]
        n = t["n"]; left = 0.0
        for c in CATS:
            v = t.get("cat_" + c, 0) / n
            ax.barh(yy, v, 0.72, left=left, color=CAT_COLOR[c], edgecolor="white", linewidth=0.4, zorder=2)
            left += v
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Share of {lab} objects")
axs[0].set_yticks(y, [r[0] for r in runs])
fig.legend(handles=[Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in CATS], loc="outside upper center",
           ncol=4, frameon=False, handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_detectors.pdf"))
