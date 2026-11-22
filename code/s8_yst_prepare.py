# -*- coding: utf-8 -*-
"""YST native-resolution source re-check: group-disjoint split + sliced training tiles.

Split: 64-bit pHash near-duplicate components (Hamming <= 8, as for RicePest), then a seeded
70/30 split of components. Training tiles: native T=640, overlap 0.25 (SAHI geometry);
annotations clipped to the tile and kept if >= 10% of the box area is inside (SAHI slice_coco
default min_area_ratio); all tiles with >= 1 object plus 5% of empty tiles.
YST is never used for a long-tail claim (3 classes, 8.4x imbalance).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env  # noqa: F401
import glob
import random
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from common import YST, WORK, RESULTS, save_json
from s1_groups_audit import phash, hamming_matrix, UF
from ledger_lib import parse_yst_xml, YST_CLASSES

OUT = os.path.join(WORK, "yolo_yst")


def main():
    from sahi.slicing import get_slice_bboxes
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(YST, "annotations", "*.xml")))
    H = []
    for n in names:
        with Image.open(os.path.join(YST, "images", f"{n}.jpg")) as I:
            H.append(phash(I))
    D = hamming_matrix(np.stack(H))
    uf = UF(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if D[i, j] <= 8:
                uf.union(i, j)
    comp = [uf.find(i) for i in range(len(names))]
    comps = sorted(set(comp))
    random.Random(0).shuffle(comps)
    n_test = int(round(0.3 * len(names)))
    test_c, acc = set(), 0
    for c in comps:
        if acc >= n_test:
            break
        test_c.add(c); acc += comp.count(c)
    split = {"train": [n for n, c in zip(names, comp) if c not in test_c],
             "test": [n for n, c in zip(names, comp) if c in test_c],
             "group": {n: int(c) for n, c in zip(names, comp)},
             "rule": "pHash<=8 components, seeded (0) 70/30 component split", "n_components": len(comps)}
    cnt = {s: Counter(nm for n in split[s] for nm, _ in parse_yst_xml(n)) for s in ("train", "test")}
    split["class_counts"] = {s: dict(v) for s, v in cnt.items()}
    os.makedirs(os.path.join(RESULTS, "yst"), exist_ok=True)
    save_json(split, os.path.join(RESULTS, "yst", "split.json"))
    print("YST split", len(split["train"]), len(split["test"]), split["class_counts"], "components", len(comps))
    rng = random.Random(0)
    for sub in ("images/train", "labels/train", "images/trainstub", "labels/trainstub"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    n_tiles = 0; n_boxes = 0
    stub = []
    for n in split["train"]:
        img = cv2.imread(os.path.join(YST, "images", f"{n}.jpg"))
        h, w = img.shape[:2]
        objs = parse_yst_xml(n)
        for k, (x0, y0, x1, y1) in enumerate(get_slice_bboxes(h, w, 640, 640, overlap_height_ratio=0.25, overlap_width_ratio=0.25)):
            lines = []
            for nm, (a, b, c, d) in objs:
                ca, cb, cc, cd = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
                if cc <= ca or cd <= cb:
                    continue
                if (cc - ca) * (cd - cb) < 0.1 * (c - a) * (d - b):
                    continue
                tw, th = x1 - x0, y1 - y0
                lines.append(f"{YST_CLASSES.index(nm)} {((ca+cc)/2-x0)/tw:.6f} {((cb+cd)/2-y0)/th:.6f} {(cc-ca)/tw:.6f} {(cd-cb)/th:.6f}")
            if not lines and rng.random() > 0.05:
                continue
            fn = f"{n}_{k:03d}"
            cv2.imwrite(os.path.join(OUT, "images", "train", fn + ".jpg"), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(OUT, "labels", "train", fn + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_tiles += 1; n_boxes += len(lines)
            if lines and len(stub) < 100 and rng.random() < 0.05:
                stub.append(fn)
    import shutil
    for fn in stub:
        shutil.copy2(os.path.join(OUT, "images", "train", fn + ".jpg"), os.path.join(OUT, "images", "trainstub", fn + ".jpg"))
        shutil.copy2(os.path.join(OUT, "labels", "train", fn + ".txt"), os.path.join(OUT, "labels", "trainstub", fn + ".txt"))
    with open(os.path.join(OUT, "yst.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {OUT.replace(os.sep, '/')}\ntrain: images/train\nval: images/trainstub\nnames:\n"
                + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(YST_CLASSES)))
    print("tiles", n_tiles, "boxes", n_boxes)


if __name__ == "__main__":
    main()
