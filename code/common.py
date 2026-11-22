# -*- coding: utf-8 -*-
"""Shared definitions for the sliced long-tail audit (RicePest-30 + YST).

Everything that the pre-registration freezes (reference frame, size bins, frequency
strata, frame-consistency rule) is defined here once and imported by every stage.
"""
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RICE = os.path.join(ROOT, "user_data", "RicePest-30-Dataset")
YST = os.path.join(ROOT, "user_data", "Yellow-Sticky-Traps-Dataset")
RESULTS = os.path.join(ROOT, "results")
WORK = os.path.join(ROOT, "work")
CONFIGS = os.path.join(ROOT, "configs")

REF_LONG = 640.0          # 640-long-edge reference frame
TINY = 16 * 16            # area < 256  -> tiny
SMALL = 32 * 32           # area < 1024 -> small (pooled "small" = tiny + 16-32)
MEDIUM = 96 * 96
SIZE_BINS = ["tiny", "s16_32", "medium", "large"]
FRAME_TOL = 0.01          # relative tolerance between x- and y-scale JSON->JPEG


def size_bin(area):
    if area < TINY:
        return "tiny"
    if area < SMALL:
        return "s16_32"
    if area < MEDIUM:
        return "medium"
    return "large"


def is_small(area):
    return area < SMALL


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_rice(split):
    fn = {"train": "filtered_train_translated.json", "val": "filtered_val_translated.json"}[split]
    return load_json(os.path.join(RICE, split, fn))


def rice_categories():
    d = load_rice("train")
    return {c["id"]: c["name"] for c in d["categories"]}


def frequency_strata():
    """Head/mid/tail tertiles by TRAIN instance count (all train annotations as published).

    Identical rule to the data-level pilot: sort classes by count desc (ties by id),
    10 head / 10 mid / 10 tail.
    """
    d = load_rice("train")
    cnt = Counter(a["category_id"] for a in d["annotations"])
    for c in d["categories"]:
        cnt.setdefault(c["id"], 0)
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    k = len(items) // 3
    strata = {}
    for i, (cid, _) in enumerate(items):
        strata[cid] = "head" if i < k else ("mid" if i < 2 * k else "tail")
    return strata, dict(cnt)


def image_frame(im, jpeg_size):
    """Return dict with JSON->JPEG scales, frame consistency, and JPEG->ref scale."""
    W, H = im["width"], im["height"]
    aw, ah = jpeg_size
    rx, ry = aw / W, ah / H
    consistent = abs(rx - ry) <= FRAME_TOL * max(rx, ry)
    r_ref = REF_LONG / max(aw, ah)
    return {"W": W, "H": H, "aw": aw, "ah": ah, "rx": rx, "ry": ry,
            "consistent": bool(consistent), "r_ref": r_ref,
            "ref_w": aw * r_ref, "ref_h": ah * r_ref}


def gt_to_ref(bbox, fr):
    """COCO xywh in JSON frame -> xyxy in the 640-long-edge reference frame (clipped)."""
    x, y, w, h = bbox
    sx, sy = fr["rx"] * fr["r_ref"], fr["ry"] * fr["r_ref"]
    x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
    x0 = min(max(0.0, x0), fr["ref_w"]); x1 = min(max(0.0, x1), fr["ref_w"])
    y0 = min(max(0.0, y0), fr["ref_h"]); y1 = min(max(0.0, y1), fr["ref_h"])
    return [x0, y0, x1, y1]


def sahi_slice_starts(L, T, overlap):
    """1-D start positions reproducing sahi.slicing.get_slice_bboxes (overlap ratio)."""
    step = int(T * (1 - overlap)) if overlap < 1 else int(T - overlap)
    starts = []
    y = 0
    while y < L:
        y_max = y + T
        if y_max > L:
            starts.append(max(0, int(L) - T) if L >= T else 0)
            break
        starts.append(y)
        if y_max == L:
            break
        y += step
    return sorted(set(starts))


def tile_grid(W, H, T, overlap):
    """List of tiles (x0,y0,x1,y1) in the given frame, SAHI geometry (integer frame)."""
    Wi, Hi = int(round(W)), int(round(H))
    xs = sahi_slice_starts(Wi, T, overlap)
    ys = sahi_slice_starts(Hi, T, overlap)
    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((x0, y0, min(x0 + T, Wi), min(y0 + T, Hi)))
    return tiles


def contains(tile, box, eps=1e-6):
    return (box[0] >= tile[0] - eps and box[1] >= tile[1] - eps and
            box[2] <= tile[2] + eps and box[3] <= tile[3] + eps)


def intersects(tile, box):
    return min(box[2], tile[2]) - max(box[0], tile[0]) > 0 and min(box[3], tile[3]) - max(box[1], tile[1]) > 0


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area_xyxy(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
