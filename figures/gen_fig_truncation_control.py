# -*- coding: utf-8 -*-
"""Fig. literal truncation (no full view in any SAHI tile) as a function of tile overlap:
negative control at the frozen overlap 0.25, positive control at overlap 0.

Lines: RicePest-30 small objects in the audit set (T=320 in the 640 frame), and YST objects per class
(native T=640). y = % of objects with no full view (log-free linear scale); markers = measured settings.
Data: results/audit/truncation.json, results/yst/truncation.json
Placement: ~0.62 text width.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figbase import load, out, plt, save_fig, set_paper_placement, NP, COLORS

rt = load("audit/truncation.json"); yt = load("yst/truncation.json")
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ovs = [0.0, 0.1, 0.25]
series = []
y = []
for ov in ovs:
    r = next(r for r in rt if r["T"] == 320 and r["overlap"] == ov and r["set"] == "audit" and r["stratum"] == "all" and r["size"] == "small")
    y.append(100 * r["truncated"] / r["n"])
series.append(("RicePest small (T=320)", y, NP["blue_main"], "o"))
for c, col, mk in (("WF", NP["teal"], "s"), ("MR", NP["red_strong"], "^"), ("NC", NP["violet"], "D")):
    y = [100 * next(r for r in yt if r["T"] == 640 and r["overlap"] == ov and r["set"] == "all" and r["cls"] == c)["rate"] for ov in ovs]
    series.append((f"YST {c} (T=640)", y, col, mk))
for lab, y, col, mk in series:
    ax.plot(ovs, y, marker=mk, color=col, lw=1.2, ms=4, label=lab, zorder=3)
ax.axvline(0.25, color=COLORS["ref_line"], lw=0.8, ls="--", zorder=1)
ax.set_xticks(ovs, ["0", "0.10", "0.25"])
ax.set_xlabel("Tile overlap ratio")
ax.set_ylabel("No full view in any tile (%)")
ax.set_ylim(bottom=-1)
ax.legend(frameon=False, loc="upper right", handlelength=1.3, borderaxespad=0.2)
set_paper_placement(fig, width_fraction=0.65)
save_fig(fig, out("fig_truncation_control.pdf"))
