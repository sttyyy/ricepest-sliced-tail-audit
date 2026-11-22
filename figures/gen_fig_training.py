# -*- coding: utf-8 -*-
"""Fig. training losses of the two frozen RicePest-30 detectors (no validation was used for selection).

Left: YOLO11n per-epoch mean training losses (box, cls, dfl) from Ultralytics results.csv.
Right: FCOS-ResNet50-FPN total training loss, running mean over each block of 200 iterations.
Data: results/runs/yolo11n_rice_s0_train_log.csv, results/runs/fcos_rice_s0_train_log.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from _figbase import RES, load, out, plt, save_fig, set_paper_placement, NP

with open(os.path.join(RES, "runs", "yolo11n_rice_s0_train_log.csv"), encoding="utf-8") as f:
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.1))
ep = [int(float(r["epoch"])) for r in rows]
for key, lab, col in (("train/box_loss", "Box", NP["blue_main"]), ("train/cls_loss", "Class", NP["red_strong"]),
                      ("train/dfl_loss", "DFL", NP["teal"])):
    ax.plot(ep, [float(r[key]) for r in rows], color=col, lw=1.2, label=lab)
ax.set_xlabel("Epoch (YOLO11n)")
ax.set_ylabel("Training loss")
ax.legend(frameon=False, loc="upper right")
lg = load("runs/fcos_rice_s0_train_log.json")
rec = load("runs/fcos_rice_s0_train.json")
ipe = rec["iterations"] // rec["epochs"]          # iterations per epoch
# the training script resets its running sum at each epoch start, so a 200-iteration window that crosses an
# epoch boundary holds fewer than 200 losses but is divided by 200; those logged points are dropped here
keep = [r for r in lg if (r["iter"] - 200) // ipe == (r["iter"] - 1) // ipe]
bx.plot([r["iter"] for r in keep], [r["loss"] for r in keep], color=NP["violet"], lw=1.2)
bx.set_xlabel("Iteration (FCOS)")
bx.set_ylabel("Total loss")
set_paper_placement(fig, width_fraction=0.98)
save_fig(fig, out("fig_training.pdf"))
