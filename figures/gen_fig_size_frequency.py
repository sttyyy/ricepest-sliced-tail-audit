# -*- coding: utf-8 -*-
"""Fig. size-frequency dissociation (annotation level, no detector).

(a) one point per RicePest-30 class: train instances (log x) vs fraction of its objects that are
    small (<32^2 in the 640-long-edge frame); colour = frequency stratum; point area grows with the
    number of small objects of that class in the group-disjoint audit set (sqrt scaling).
(b) size composition of each stratum, train vs audit set (horizontal stacked shares).
Data: results/audit/class_stats.json (code/s1b_dataset_stats.py).
Placement: full text width (5.15 in), ~2.5 in tall.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.lines import Line2D
from _figbase import load, out, plt, save_fig, set_paper_placement, STRATA, STRATUM_COLOR, NP, COLORS, _lighten

d = load("audit/class_stats.json")
rows = d["classes"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
for st in STRATA:
    rr = [r for r in rows if r["stratum"] == st]
    x = np.array([r["train_n"] for r in rr]); y = np.array([r["train_small_frac"] for r in rr])
    s = 9 + 5.0 * np.sqrt([r["audit_small"] for r in rr])
    ax.scatter(x, y, s=s, color=STRATUM_COLOR[st], alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
for a, b in (("head", "mid"), ("mid", "tail")):
    lo = min(r["train_count_published"] for r in rows if r["stratum"] == a)
    hi = max(r["train_count_published"] for r in rows if r["stratum"] == b)
    ax.axvline(np.sqrt(lo * hi), color=COLORS["ref_line"], lw=0.7, ls="--", zorder=1)
ax.set_xscale("log")
ax.set_xlabel("Train instances per class")
ax.set_ylabel("Fraction small (<32$^2$)")
ax.set_ylim(-0.04, 1.06)
handles = [Line2D([], [], marker="o", ls="", color=STRATUM_COLOR[s], markersize=5, label=s.capitalize()) for s in STRATA]
ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)

bins = ["tiny", "s16_32", "medium", "large"]
labels = ["<16$^2$", "16$^2$–32$^2$", "32$^2$–96$^2$", "$\\geq$96$^2$"]
cols = [NP["red_strong"], _lighten(NP["red_strong"], 0.5), _lighten(NP["blue_main"], 0.6), NP["blue_main"]]
comp = d["size_composition"]
ypos, yt = [], []
k = 0.0
for st in STRATA:
    for s in ("train", "audit"):
        c = comp.get(f"{s}|{st}", {})
        tot = sum(c.get(b, 0) for b in bins)
        left = 0.0
        for b, col in zip(bins, cols):
            v = c.get(b, 0) / tot if tot else 0
            bx.barh(-k, v, 0.78, left=left, color=col, edgecolor="white", linewidth=0.4,
                    label=labels[bins.index(b)] if k == 0 else None)
            left += v
        ypos.append(-k); yt.append(f"{st.capitalize()} {s}")
        k += 1
    k += 0.4
bx.set_yticks(ypos, yt)
bx.set_xlabel("Share of objects")
bx.set_xlim(0, 1)
bx.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, handlelength=0.8,
          columnspacing=0.5, handletextpad=0.3, borderaxespad=0.1)
fig.text(0.005, 0.985, "a", fontweight="bold", va="top")
fig.text(0.505, 0.985, "b", fontweight="bold", va="top")
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_size_frequency.pdf"))
