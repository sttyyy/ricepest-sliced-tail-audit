# -*- coding: utf-8 -*-
"""Fig. size-matched frequency contrast (primary K3 test, audit set).

For each outcome, the share among tail-small objects minus the share among head-small objects
re-weighted to the tail sqrt-area mix (bins 0-12-16-24-32 px), in percentage points; bars = 95%
cluster-bootstrap CI (2,000 resamples of audit groups); text = Holm-adjusted class-label permutation
p for the three miss categories.
Data: results/stats/<RUN>_stats.json (size_matched_frequency_contrast)
Placement: ~0.85 text width.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _figbase import load, out, plt, save_fig, set_paper_placement, CATS, CAT_LABEL, CAT_COLOR, NP, COLORS

RUN = os.environ.get("MH_RUN", "R3-ledger-yolo-T320")
con = load(f"stats/{RUN}_stats.json")["size_matched_frequency_contrast"]
d, ci = con["diff"], con["ci95"]
fig, bx = plt.subplots(figsize=(4.4, 2.0))
yb = np.arange(len(CATS))[::-1].astype(float)
bx.axvline(0, color=COLORS["ref_line"], lw=0.8, ls="--", zorder=1)
for yy, c in zip(yb, CATS):
    lo, hi = ci[c]
    col = CAT_COLOR[c] if c != "detected" else NP["blue_secondary"]
    bx.hlines(yy, 100 * lo, 100 * hi, color=col, lw=1.6, zorder=2)
    bx.scatter(100 * d[c], yy, s=30, color=col, edgecolor="white", linewidth=0.5, zorder=3, marker="D")
labels = []
for c in CATS:
    p = con["holm_p_3cats"].get(c)
    labels.append(CAT_LABEL[c] if p is None else CAT_LABEL[c] + "\n" + f"(Holm p={p:.3f})")
bx.set_yticks(yb, labels)
bx.set_ylim(-0.6, len(CATS) - 0.4)
lim = max(abs(100 * v) for c in CATS for v in ci[c]) * 1.1
bx.set_xlim(-lim, lim)
bx.set_xlabel("Tail minus head share (pp)")
set_paper_placement(fig, width_fraction=0.85)
save_fig(fig, out("fig_contrast.pdf"))
