# -*- coding: utf-8 -*-
"""Build the Ultralytics YOLO view of RicePest-30 (hard links + normalized labels).

Frame-inconsistent images (JSON frame aspect != JPEG aspect, single-pest close-ups whose
boxes cannot be mapped) are excluded from training and evaluation; the list is in
results/audit/audit_set.json["frame_inconsistent"].
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RICE, WORK, RESULTS, load_rice, load_json, rice_categories

OUT = os.path.join(WORK, "yolo_rice")


def main():
    audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
    bad = {s: set(v) for s, v in audit["frame_inconsistent"].items()}
    cats = rice_categories()
    for split in ["train", "val"]:
        d = load_rice(split)
        idir = os.path.join(OUT, "images", split); ldir = os.path.join(OUT, "labels", split)
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        n_img = n_box = 0
        for im in d["images"]:
            if im["id"] in bad[split]:
                continue
            src = os.path.join(RICE, split, im["file_name"])
            dst = os.path.join(idir, im["file_name"])
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            W, H = im["width"], im["height"]
            lines = []
            for a in by.get(im["id"], []):
                x, y, w, h = a["bbox"]
                x0, y0 = max(0.0, x), max(0.0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 - x0 <= 1 or y1 - y0 <= 1:
                    continue
                xc, yc = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                lines.append(f"{a['category_id'] - 1} {xc:.6f} {yc:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
                n_box += 1
            with open(os.path.join(ldir, os.path.splitext(im["file_name"])[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
        print(split, "images", n_img, "boxes", n_box)
    # Ultralytics always validates at the end of training; point that at 200 TRAIN images
    # ("trainstub") so the validation/audit split is never used during training.
    import random
    imgs = sorted(os.listdir(os.path.join(OUT, "images", "train")))
    random.Random(0).shuffle(imgs)
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(OUT, sub, "trainstub"), exist_ok=True)
    for f in imgs[:200]:
        dst = os.path.join(OUT, "images", "trainstub", f)
        if not os.path.exists(dst):
            os.link(os.path.join(OUT, "images", "train", f), dst)
        lf = os.path.splitext(f)[0] + ".txt"
        shutil.copy2(os.path.join(OUT, "labels", "train", lf), os.path.join(OUT, "labels", "trainstub", lf))
    names = "\n".join(f"  {cid - 1}: '{cats[cid]}'" for cid in sorted(cats))
    yaml = f"path: {OUT.replace(os.sep, '/')}\ntrain: images/train\nval: images/trainstub\nnames:\n{names}\n"
    with open(os.path.join(OUT, "rice30.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    print("wrote", os.path.join(OUT, "rice30.yaml"))


if __name__ == "__main__":
    main()
