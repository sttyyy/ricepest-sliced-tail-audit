# -*- coding: utf-8 -*-
"""Fig. sensitivity of the ledger to the output cap and to the read-out / tiling (audit set, YOLO11n).

Top: recall@0.5 of small objects per frequency stratum vs the global per-image cap (10-300 boxes);
      dashed = frozen cap 100.
Bottom: composition of small-object misses (no candidate / pre-suppression only / post-hoc recoverable)
      under the frozen pipeline and under alternative read-outs and tilings; n misses in tick labels.
Data: results/ledger/R3-ledger-yolo-T320.json and results/ledger/S-ledger-*.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.patches import Patch
from _figbase import load, out, plt, save_fig, set_paper_placement, STRATA, STRATUM_COLOR, CATS, CAT_LABEL, CAT_COLOR, COLORS

caps = [(10, "S-ledger-yolo-T320-cap10"), (20, "S-ledger-yolo-T320-cap20"), (50, "S-ledger-yolo-T320-cap50"),
        (100, "R3-ledger-yolo-T320"), (300, "S-ledger-yolo-T320-cap300")]
fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.15, 4.3), gridspec_kw={"height_ratios": [1, 1.25]})
marks = {"head": "o", "mid": "s", "tail": "^"}
for st in STRATA:
    xs, ys = [], []
    for c, run in caps:
        t = load(f"ledger/{run}.json")["summary"]["audit"].get(f"{st}|small", {})
        if t.get("n"):
            xs.append(c); ys.append(t["D"] / t["n"])
    ax.plot(xs, ys, marker=marks[st], color=STRATUM_COLOR[st], lw=1.2, ms=3.6, label=st.capitalize(), zorder=3)
ax.axvline(100, color=COLORS["ref_line"], lw=0.8, ls="--", zorder=1)
ax.set_xscale("log")
ax.set_xticks([10, 20, 50, 100, 300], ["10", "20", "50", "100", "300"])
ax.set_xlabel("Global output cap (boxes per image)")
ax.set_ylabel("Recall, small objects")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="center right", handlelength=1.3, borderaxespad=0.2)

variants = [("Frozen", "R3-ledger-yolo-T320"), ("NMS merge", "S-ledger-yolo-T320-nms"),
            ("Agnostic", "S-ledger-yolo-T320-agn"), ("Thr. first", "S-ledger-yolo-T320-prethr"),
            ("No slicing", "S-ledger-yolo-fullimage"), ("T=256", "S-ledger-yolo-T256"),
            ("T=480", "S-ledger-yolo-T480"), ("Overlap 0", "S-ledger-yolo-T320-ov0")]
labels = []
y = np.arange(len(variants))[::-1]
for yy, (lab, run) in zip(y, variants):
    t = load(f"ledger/{run}.json")["summary"]["audit"]["all|small"]
    miss = t["n"] - t["D"]
    left = 0.0
    for c in CATS[1:]:
        v = t.get("cat_" + c, 0) / miss if miss else 0
        bx.barh(yy, v, 0.74, left=left, color=CAT_COLOR[c], edgecolor="white", linewidth=0.4, zorder=2)
        left += v
    labels.append(f"{lab} ({miss})")
bx.set_yticks(y, labels)
bx.set_xlim(0, 1)
bx.set_xlabel("Share of small-object misses")
bx.legend(handles=[Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in CATS[1:]], loc="lower center",
          bbox_to_anchor=(0.35, 1.0), ncol=3, frameon=False, handlelength=0.9, columnspacing=0.6, handletextpad=0.3,
          borderaxespad=0.2)
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_sensitivity.pdf"))
