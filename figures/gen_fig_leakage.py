# -*- coding: utf-8 -*-
"""Fig. train/val leakage in the published split and its effect.

Left: distance (64-bit pHash Hamming) from every validation image to its nearest training image;
      dashed line = frozen grouping threshold tau = 8 (red bars: <= tau).
Middle: share of frame-consistent validation objects kept in the group-disjoint audit set as tau varies
      (head-small, tail-small, all objects; frozen tau dashed).
Right: recall@0.5 of the frozen YOLO11n sliced pipeline on audit images (hollow) vs contaminated validation
      images (filled; near-duplicates of training images), per frequency stratum.
Data: results/audit/grouping_sensitivity.json, results/audit/truncation.json,
      results/ledger/R3-ledger-yolo-T320.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.lines import Line2D
from _figbase import load, out, plt, save_fig, set_paper_placement, STRATA, STRATUM_COLOR, NP, COLORS, _lighten

g = load("audit/grouping_sensitivity.json")
tr = load("audit/truncation.json")
led = load("ledger/R3-ledger-yolo-T320.json")
tau_primary = 8

fig, axs = plt.subplots(1, 3, figsize=(5.15, 2.6), gridspec_kw={"width_ratios": [1, 1, 0.9]}, layout="constrained")
ax = axs[0]
h = g["val_nearest_train_phash_hist"]
xs = np.array(sorted(int(k) for k in h)); ys = np.array([h[str(k)] for k in xs])
ax.bar(xs, ys, width=1.6, color=[NP["red_strong"] if x <= tau_primary else _lighten(NP["blue_main"], 0.45) for x in xs],
       edgecolor="white", linewidth=0.4, zorder=2)
ax.axvline(tau_primary + 1, color=COLORS["dark"], lw=0.8, ls="--", zorder=3)
ax.set_xticks([0, 8, 16])
ax.set_xlabel("pHash distance")
ax.set_ylabel("Val images")

bx = axs[1]
val_tot = {}
for r in tr:
    if r["T"] == 320 and r["overlap"] == 0.25 and r["set"] == "val":
        val_tot[(r["stratum"], r["size"])] = r["n"]
taus = [row["tau"] for row in g["sensitivity"]]
series = [("Head small", "head|small", val_tot[("head", "small")], NP["blue_main"], "o"),
          ("Tail small", "tail|small", val_tot[("tail", "small")], NP["violet"], "s")]
for lab, key, tot, col, mk in series:
    y = [row["counts"][key] / tot for row in g["sensitivity"]]
    bx.plot(taus, y, marker=mk, color=col, lw=1.1, ms=3.5, label=lab, zorder=3)
allv = val_tot[("all", "all")]
y = [sum(row["counts"].values()) / allv for row in g["sensitivity"]]
bx.plot(taus, y, marker="^", color=NP["neutral_mid"], lw=1.1, ms=3.5, label="All", zorder=3)
bx.vlines(tau_primary, 0, 1.0, color=COLORS["dark"], lw=0.8, ls="--", zorder=1)
bx.set_xticks([0, 4, 8, 12])
bx.set_xlabel("Threshold $\\tau$")
bx.set_ylabel("Share kept")
bx.set_ylim(0, 1.5)
bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
bx.legend(frameon=False, loc="upper center", handlelength=1.0, borderaxespad=0.1, labelspacing=0.25, handletextpad=0.3)

cx = axs[2]
A = led["summary"]["audit"]; Cn = led["summary"]["contaminated"]
for i, st in enumerate(STRATA):
    ra = A[f"{st}|all"]["D"] / A[f"{st}|all"]["n"]; rc = Cn[f"{st}|all"]["D"] / Cn[f"{st}|all"]["n"]
    cx.plot([i, i], [ra, rc], color=COLORS["grid"], lw=2.2, zorder=1, solid_capstyle="round")
    cx.scatter([i], [ra], s=26, color="white", edgecolor=STRATUM_COLOR[st], linewidth=1.2, zorder=3)
    cx.scatter([i], [rc], s=26, color=STRATUM_COLOR[st], zorder=3)
handles = [Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor="white", markeredgecolor=NP["neutral_dark"], label="Audit"),
           Line2D([], [], marker="o", ls="", markersize=5, color=NP["neutral_dark"], label="Contaminated")]
cx.set_xticks(range(3), [s.capitalize() for s in STRATA])
cx.set_xlim(-0.5, 2.5)
cx.set_ylabel("Recall@0.5")
cx.set_ylim(0, 1)
cx.legend(handles=handles, frameon=False, loc="lower left", borderaxespad=0.2, handletextpad=0.2)
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_leakage.pdf"))
