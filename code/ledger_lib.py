# -*- coding: utf-8 -*-
"""Core bookkeeping for the per-object stage ledger: read-outs, matching, operating point.

All functions are pure numpy on one image at a time. Box format: xyxy in the reference frame.
The GREEDYNMM read-out reuses sahi.postprocess.combine.(batched_)greedy_nmm for the keep->merge
lists and replays SAHI's merge rule (union box, max score, sequential re-check of the match on
the growing box), so the merged list equals sahi.get_sliced_prediction's output on the same dump.
"""
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env  # noqa: F401  (sahi lives in .pydeps_gpu)
from common import (RESULTS, WORK, YST, load_json, load_rice, gt_to_ref, area_xyxy, size_bin, frequency_strata,
                    contains)


# ----------------------------------------------------------------------------- geometry
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def contains_matrix(views, boxes, eps=1e-6):
    """views (V,4), boxes (N,4) -> (V,N) bool: view fully contains box."""
    v = np.asarray(views, dtype=float)
    if len(boxes) == 0:
        return np.zeros((len(v), 0), bool)
    return ((boxes[None, :, 0] >= v[:, None, 0] - eps) & (boxes[None, :, 1] >= v[:, None, 1] - eps) &
            (boxes[None, :, 2] <= v[:, None, 2] + eps) & (boxes[None, :, 3] <= v[:, None, 3] + eps))


# ----------------------------------------------------------------------------- read-outs
def _ios(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    sa = (a[2] - a[0]) * (a[3] - a[1]); sb = (b[2] - b[0]) * (b[3] - b[1])
    sm = min(sa, sb)
    return inter / sm if sm > 0 else 0.0


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def readout_sahi(boxes, scores, cls, kind="GREEDYNMM", metric="IOS", thr=0.5, class_agnostic=False):
    """Return merged (boxes, scores, cls, members) with SAHI semantics."""
    from sahi.postprocess.combine import greedy_nmm, batched_greedy_nmm, nms, batched_nms
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), []
    P = np.column_stack([boxes, scores, cls]).astype(np.float64)
    if kind == "NMS":
        keep = (nms if class_agnostic else batched_nms)(P, match_metric=metric, match_threshold=thr)
        keep = list(keep)
        # attribute suppressed boxes to the keeper with the largest overlap (bookkeeping only)
        members = [[k] for k in keep]
        return boxes[keep], scores[keep], cls[keep].astype(int), members
    k2m = (greedy_nmm if class_agnostic else batched_greedy_nmm)(P, match_metric=metric, match_threshold=thr)
    ob, os_, oc, mem = [], [], [], []
    f = _ios if metric == "IOS" else _iou
    for keep, mlist in k2m.items():
        kb = boxes[keep].astype(float).copy(); ks = float(scores[keep]); kc = int(cls[keep]); members = [keep]
        for m in mlist:
            if f(kb, boxes[m]) >= thr:
                kb = np.array([min(kb[0], boxes[m][0]), min(kb[1], boxes[m][1]),
                               max(kb[2], boxes[m][2]), max(kb[3], boxes[m][3])])
                if scores[m] > ks:  # SAHI: merged category/score from the higher-scored prediction
                    kc = int(cls[m])
                ks = max(ks, float(scores[m]))
                members.append(m)
        ob.append(kb); os_.append(ks); oc.append(kc); mem.append(members)
    return np.array(ob), np.array(os_), np.array(oc, int), mem


def readout_wbf(boxes, scores, cls, views, thr=0.55):
    """Class-conditional peak read-out: per class, greedy peak clustering at IoU>=thr to the peak,
    score-weighted mean box (no union growth), peak score kept, support = #distinct views."""
    ob, os_, oc, mem, sup = [], [], [], [], []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        B = boxes[idx]
        I = iou_matrix(B, B)
        free = np.ones(len(idx), bool)
        for a in range(len(idx)):
            if not free[a]:
                continue
            grp = np.where(free & (I[a] >= thr))[0]
            free[grp] = False
            w = scores[idx[grp]]
            ob.append((B[grp] * w[:, None]).sum(0) / w.sum()); os_.append(float(scores[idx[a]])); oc.append(int(c))
            mem.append(idx[grp].tolist()); sup.append(len(set(views[idx[grp]].tolist())))
    if not ob:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), [], np.zeros(0)
    return np.array(ob), np.array(os_), np.array(oc, int), mem, np.array(sup)


# ----------------------------------------------------------------------------- matching / operating point
def match(pb, ps, pc, gb, gc, thr=0.5):
    """COCO-style greedy one-to-one matching per class. Returns gt->pred index (-1) and pred TP flags."""
    gt_m = np.full(len(gb), -1, int); tp = np.zeros(len(pb), bool)
    if len(gb) == 0 or len(pb) == 0:
        return gt_m, tp
    ious = iou_matrix(pb, gb)
    ious[pc[:, None] != gc[None, :]] = 0.0
    for i in np.argsort(-ps, kind="stable"):
        row = ious[i]
        if row.max() < thr:
            continue
        row = np.where(gt_m >= 0, 0.0, row)
        j = int(row.argmax())
        if row[j] >= thr:
            gt_m[j] = i; tp[i] = True
    return gt_m, tp


def emit(scores, tau, cap, cls=None, per_class_cap=None):
    idx = np.where(scores >= tau)[0]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    if per_class_cap is not None and cls is not None and len(idx):
        cnt = defaultdict(int); keep = []
        for i in idx:
            if cnt[cls[i]] < per_class_cap:
                keep.append(i); cnt[cls[i]] += 1
        idx = np.array(keep, int)
    return idx[:cap] if cap is not None else idx


# ----------------------------------------------------------------------------- data loading
def _full_idx(meta, im):
    """The dump writes the full-image view first when meta['full_view'] is true; verify it explicitly."""
    if not meta.get("full_view", True):
        return None
    v0 = im["views"][0]
    assert v0[0] == 0 and v0[1] == 0 and v0[2] == im["W"] and v0[3] == im["H"], "view 0 is not the full image"
    return 0


def load_eval_rice(tag):
    """Per-image evaluation records for a RicePest dump tag (val split)."""
    meta = load_json(os.path.join(WORK, "dumps", f"{tag}_meta.json"))
    dump = np.load(os.path.join(WORK, "dumps", f"{tag}.npz"))
    audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
    audit_ids = set(audit["audit_image_ids"]); groups = audit["group_of_val_image"]
    strata = {int(k): v for k, v in audit["strata"].items()}
    frames = {m["image_id"]: m["frame"] for m in load_json(os.path.join(WORK, "img_meta.json")) if m["split"] == "val"}
    d = load_rice("val")
    by = defaultdict(list)
    for a in d["annotations"]:
        by[a["image_id"]].append(a)
    recs = {}
    for key in ("std", "ml"):
        arr = dump[key]
        order = np.argsort(arr[:, 0], kind="stable"); arr = arr[order]
        cuts = np.searchsorted(arr[:, 0], np.arange(meta["n_images"] + 1))
        for sid, im in meta["images"].items():
            ii = im["idx"]
            r = recs.setdefault(sid, {"image_id": int(sid), "W": im["W"], "H": im["H"], "views": np.array(im["views"], float),
                                      "full_idx": _full_idx(meta, im)})
            a = arr[cuts[ii]:cuts[ii + 1]]
            r[key] = {"view": a[:, 1].astype(int), "boxes": a[:, 2:6], "scores": a[:, 6], "cls": a[:, 7].astype(int)}
    for sid, r in recs.items():
        iid = int(sid); fr = frames[iid]
        gb, gc = [], []
        for a in by[iid]:
            b = gt_to_ref(a["bbox"], fr)
            if area_xyxy(b) <= 0:
                continue
            gb.append(b); gc.append(a["category_id"] - 1)
        r["gt_boxes"] = np.array(gb, float).reshape(-1, 4); r["gt_cls"] = np.array(gc, int)
        r["gt_strata"] = [strata[c + 1] for c in gc]
        r["audit"] = iid in audit_ids; r["group"] = int(groups[str(iid)])
    return meta, [recs[k] for k in sorted(recs, key=lambda s: recs[s]["image_id"])]


def parse_yst_xml(name):
    t = ET.parse(os.path.join(YST, "annotations", f"{name}.xml")).getroot()
    out = []
    for o in t.findall("object"):
        bb = o.find("bndbox")
        out.append((o.find("name").text, [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]))
    return out


YST_CLASSES = ["WF", "MR", "NC"]


def load_eval_yst(tag):
    meta = load_json(os.path.join(WORK, "dumps", f"{tag}_meta.json"))
    dump = np.load(os.path.join(WORK, "dumps", f"{tag}.npz"))
    sp = load_json(os.path.join(RESULTS, "yst", "split.json"))
    recs = {}
    for key in ("std", "ml"):
        arr = dump[key]
        order = np.argsort(arr[:, 0], kind="stable"); arr = arr[order]
        cuts = np.searchsorted(arr[:, 0], np.arange(meta["n_images"] + 1))
        for sid, im in meta["images"].items():
            ii = im["idx"]
            r = recs.setdefault(sid, {"image_id": sid, "W": im["W"], "H": im["H"], "views": np.array(im["views"], float),
                                      "full_idx": _full_idx(meta, im)})
            a = arr[cuts[ii]:cuts[ii + 1]]
            r[key] = {"view": a[:, 1].astype(int), "boxes": a[:, 2:6], "scores": a[:, 6], "cls": a[:, 7].astype(int)}
    for sid, r in recs.items():
        s = meta["images"][sid]["scale"]
        objs = parse_yst_xml(sid)
        r["gt_boxes"] = np.array([[v * s for v in b] for _, b in objs], float).reshape(-1, 4)
        r["gt_cls"] = np.array([YST_CLASSES.index(n) for n, _ in objs], int)
        r["gt_strata"] = [n for n, _ in objs]
        r["audit"] = True; r["group"] = int(sp["group"][sid])
    return meta, [recs[k] for k in sorted(recs)]
