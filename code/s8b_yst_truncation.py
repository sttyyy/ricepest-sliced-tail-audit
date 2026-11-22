# -*- coding: utf-8 -*-
"""YST native-geometry truncation audit: share of objects with no full view in any SAHI tile,
per class, for several tile sizes / overlaps (all 284 images and the held-out test split).
Output: results/yst/truncation.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env  # noqa: F401
import glob
from collections import defaultdict

from PIL import Image

from common import YST, RESULTS, load_json, save_json, contains, intersects
from ledger_lib import parse_yst_xml

TILINGS = [(640, 0.25), (640, 0.10), (640, 0.0), (512, 0.25), (1024, 0.25)]


def main():
    from sahi.slicing import get_slice_bboxes
    sp = load_json(os.path.join(RESULTS, "yst", "split.json"))
    test = set(sp["test"])
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(YST, "annotations", "*.xml")))
    out = []
    for T, ov in TILINGS:
        tab = defaultdict(lambda: [0, 0, 0])
        for n in names:
            with Image.open(os.path.join(YST, "images", f"{n}.jpg")) as I:
                W, H = I.size
            tiles = get_slice_bboxes(H, W, T, T, overlap_height_ratio=ov, overlap_width_ratio=ov)
            for cls, b in parse_yst_xml(n):
                full = any(contains(t, b) for t in tiles)
                cut = any(intersects(t, b) and not contains(t, b) for t in tiles)
                for s in ["all"] + (["test"] if n in test else []):
                    for c in (cls, "all"):
                        v = tab[(s, c)]; v[0] += 1; v[1] += int(not full); v[2] += int(cut)
        for (s, c), (nn, tr, cut) in sorted(tab.items()):
            out.append({"T": T, "overlap": ov, "set": s, "cls": c, "n": nn, "truncated": tr, "boundary_cut": cut,
                        "rate": tr / nn if nn else None})
        print(T, ov, {k: v for k, v in tab.items() if k[0] == "all"})
    save_json(out, os.path.join(RESULTS, "yst", "truncation.json"))


if __name__ == "__main__":
    main()
