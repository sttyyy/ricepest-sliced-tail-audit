# -*- coding: utf-8 -*-
"""Fig. stage-survival profile of the per-object ledger + oracle full-view bound (audit set).

x = ledger stage: V (some tile holds a full view), C (same-class candidate in the per-view dump at
the 0.01 floor), M (matched in the merged list at the floor), S (that merged box >= threshold 0.25),
D (emitted within the 100-box cap = detected). Hollow marker at D = oracle full-view bound (a
same-class candidate >= 0.25 exists in a view that fully contains the object). One line per stratum;
left panel: small objects (<32^2), right panel: objects >=32^2; legend column gives n (small / not small).
Data: results/ledger/<RUN>.json (summary.audit)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.lines import Line2D
from _figbase import load, out, plt, save_fig, set_paper_placement, STRATA, STRATUM_COLOR, NP, COLORS

RUN = os.environ.get("MH_RUN", "R3-ledger-yolo-T320")
OUTNAME = os.environ.get("MH_OUT", "fig_stage_survival.pdf")
A = load(f"ledger/{RUN}.json")["summary"]["audit"]
stages = ["V", "C", "M", "S", "D"]
xl = ["View", "Cand.", "Merged", "$\\geq\\tau$", "Emitted"]
marks = {"head": "o", "mid": "s", "tail": "^"}
fig, axs3 = plt.subplots(1, 3, figsize=(5.15, 2.3), gridspec_kw={"width_ratios": [1, 1, 0.62]}, layout="constrained")
axs = axs3[:2]
for ax, (z, zl) in zip(axs, (("small", "<32$^2$"), ("notsmall", "$\\geq$32$^2$"))):
    for st in STRATA:
        t = A.get(f"{st}|{z}", {})
        n = t.get("n", 0)
        if not n:
            continue
        y = [t.get(s, 0) / n for s in stages]
        ax.plot(range(5), y, color=STRATUM_COLOR[st], marker=marks[st], ms=3.6, lw=1.2, zorder=3)
        orc = t.get("oracle", 0) / n
        ax.plot([4, 4], [y[-1], orc], color=STRATUM_COLOR[st], lw=0.8, ls=":", zorder=2)
        ax.scatter([4.0], [orc], s=22, facecolor="white", edgecolor=STRATUM_COLOR[st], linewidth=1.1, zorder=4,
                   marker=marks[st])
    ax.set_xticks(range(5), xl, rotation=90, ha="center")
    ax.set_xlim(-0.3, 4.4)
    ax.set_ylim(0, 1.03)
    ax.text(0.04, 0.04, zl, transform=ax.transAxes, color=COLORS["text"])
axs[1].tick_params(labelleft=False)
axs[0].set_ylabel("Surviving")
handles = []
for s in STRATA:
    ns, nl = A.get(f"{s}|small", {}).get("n", 0), A.get(f"{s}|notsmall", {}).get("n", 0)
    handles.append(Line2D([], [], color=STRATUM_COLOR[s], marker=marks[s], ms=3.6, lw=1.2,
                          label=s.capitalize() + "\n" + f"(n={ns} / {nl})"))
handles.append(Line2D([], [], color=NP["neutral_mid"], marker="o", ms=4, lw=0, markerfacecolor="white",
                      label="Oracle full-" + "\n" + "view bound"))
axs3[2].axis("off")
axs3[2].legend(handles=handles, loc="center left", frameon=False, handlelength=1.4, handletextpad=0.3,
               borderaxespad=0.0, labelspacing=0.9)
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out(OUTNAME))
