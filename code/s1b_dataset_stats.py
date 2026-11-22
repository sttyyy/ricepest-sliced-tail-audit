# -*- coding: utf-8 -*-
"""Per-class frequency x size statistics (annotation level, no detector) for the dissociation figure.
Output: results/audit/class_stats.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict

from common import RESULTS, WORK, load_json, load_rice, gt_to_ref, area_xyxy, size_bin, save_json, rice_categories


def main():
    audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
    strata = {int(k): v for k, v in audit["strata"].items()}
    counts = {int(k): v for k, v in audit["train_counts"].items()}
    names = rice_categories()
    meta = load_json(os.path.join(WORK, "img_meta.json"))
    frames = {(m["split"], m["image_id"]): m["frame"] for m in meta}
    aset = set(audit["audit_image_ids"])
    per = defaultdict(Counter)
    comp = defaultdict(Counter)
    for split in ("train", "val"):
        d = load_rice(split)
        for a in d["annotations"]:
            fr = frames[(split, a["image_id"])]
            if not fr["consistent"]:
                continue
            sb = size_bin(area_xyxy(gt_to_ref(a["bbox"], fr)))
            c = a["category_id"]
            sets = [split] + (["audit"] if split == "val" and a["image_id"] in aset else [])
            for s in sets:
                per[c][f"{s}_n"] += 1
                per[c][f"{s}_{sb}"] += 1
                comp[(s, strata[c])][sb] += 1
    rows = []
    for c in sorted(names):
        p = per[c]
        n = p["train_n"]
        rows.append({"cls": c, "name": names[c], "stratum": strata[c], "train_count_published": counts[c],
                     "train_n": n, "train_small_frac": (p["train_tiny"] + p["train_s16_32"]) / n if n else None,
                     "train_tiny_frac": p["train_tiny"] / n if n else None,
                     "val_n": p["val_n"], "audit_n": p["audit_n"],
                     "audit_small": p["audit_tiny"] + p["audit_s16_32"], "audit_tiny": p["audit_tiny"]})
    composition = {f"{s}|{st}": dict(v) for (s, st), v in sorted(comp.items())}
    save_json({"classes": rows, "size_composition": composition,
               "note": "frame-consistent images only; sizes in the 640-long-edge reference frame"},
              os.path.join(RESULTS, "audit", "class_stats.json"))
    for r in rows:
        print(r["cls"], r["stratum"], r["train_n"], round(r["train_small_frac"], 3), r["audit_n"], r["audit_small"])


if __name__ == "__main__":
    main()
