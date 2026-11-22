# -*- coding: utf-8 -*-
"""Per-object stage ledger + recovery-ceiling partition for one dump x read-out x operating point.

Stages per GT object (reference frame):
  V  some tile (not the full-image view) fully contains the object (literal truncation if 0)
  C  a same-class candidate with IoU>=0.5 exists in the standard per-view dump at the floor
  M  the object is matched (one-to-one, IoU>=0.5) in the merged scored list (read-out at floor)
  S  that matched merged box scores >= the frozen threshold
  D  the object is matched in the emitted list (threshold + cap)  -> detected
Recovery-ceiling category of a miss (D=0), assigned with priority M > C:
post_hoc_recoverable (M=1) / pre_suppression_only (M=0, C=1) / no_candidate (M=0, C=0).
(M=1 with C=0 happens when a merged union box matches although no single candidate does.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np

from common import RESULTS, size_bin, save_json
from ledger_lib import (iou_matrix, contains_matrix, readout_sahi, match, emit, load_eval_rice, load_eval_yst)

READOUTS = {
    "greedynmm_ios": dict(kind="GREEDYNMM", metric="IOS", thr=0.5, class_agnostic=False),
    "nms_iou": dict(kind="NMS", metric="IOU", thr=0.5, class_agnostic=False),
    "greedynmm_ios_agn": dict(kind="GREEDYNMM", metric="IOS", thr=0.5, class_agnostic=True),
    "fullimage": dict(kind="GREEDYNMM", metric="IOS", thr=0.5, class_agnostic=False),  # view 0 only
    # SAHI deployment order (post-freeze sensitivity): per-view confidence threshold BEFORE merging,
    # so threshold-removed candidates count as pre-suppression-only
    "greedynmm_ios_prethr": dict(kind="GREEDYNMM", metric="IOS", thr=0.5, class_agnostic=False),
}


def valid(c):
    b = c["boxes"]
    # drop degenerate boxes and torchvision's background index (FCOS label 0 -> cls -1)
    ok = (b[:, 2] - b[:, 0] > 1e-3) & (b[:, 3] - b[:, 1] > 1e-3) & (c["cls"] >= 0)
    return {k: v[ok] for k, v in c.items()}


def merged_list(r, readout, tau=None):
    std = valid(r["std"])
    if readout == "fullimage":
        sel = std["view"] == 0
        std = {k: v[sel] for k, v in std.items()}
    src = std
    if readout.endswith("_prethr"):
        sel = std["scores"] >= tau
        src = {k: v[sel] for k, v in std.items()}
    L = readout_sahi(src["boxes"], src["scores"], src["cls"], **READOUTS[readout])
    return std, L


def ledger_image(r, readout, tau, cap, per_class_cap=None):
    std, (Lb, Ls, Lc, Lm) = merged_list(r, readout, tau)
    ml = valid(r["ml"])
    gb, gc = r["gt_boxes"], r["gt_cls"]
    n = len(gb)
    views = r["views"]
    assert len(views) > 0, "no views stored for image"
    fi = r.get("full_idx")                                 # index of the full-image view (None if absent)
    tile_idx = np.array([v for v in range(len(views)) if v != fi], int)
    tiles = views[tile_idx] if len(tile_idx) else views
    cont = contains_matrix(tiles, gb)                      # tiles x gt
    inter = np.zeros_like(cont)
    if n:
        ix = np.minimum(gb[None, :, 2], tiles[:, None, 2]) - np.maximum(gb[None, :, 0], tiles[:, None, 0])
        iy = np.minimum(gb[None, :, 3], tiles[:, None, 3]) - np.maximum(gb[None, :, 1], tiles[:, None, 1])
        inter = (ix > 0) & (iy > 0)
    I_std = iou_matrix(std["boxes"], gb)
    same = std["cls"][:, None] == gc[None, :]
    I_ml = iou_matrix(ml["boxes"], gb); same_ml = ml["cls"][:, None] == gc[None, :]
    full_by_view = contains_matrix(views, gb)              # all views x gt
    gm_full, _ = match(Lb, Ls, Lc, gb, gc)
    e = emit(Ls, tau, cap, Lc, per_class_cap)
    gm_emit_local, tp_e = match(Lb[e], Ls[e], Lc[e], gb, gc)
    rank = np.empty(len(Ls), int); rank[np.argsort(-Ls, kind="stable")] = np.arange(len(Ls))
    rows = []
    emitted_set = set(e.tolist())
    for j in range(n):
        hit = (I_std[:, j] >= 0.5) & same[:, j]
        C = bool(hit.any())
        best = float(std["scores"][hit].max()) if C else 0.0
        best_full = False; fv_score = 0.0; fv_tile_score = 0.0; C_full = False; C_tile = False
        if C:
            hv = std["view"][hit]; hs = std["scores"][hit]
            fv = full_by_view[hv, j]
            best_full = bool(fv[np.argmax(hs)])
            fv_score = float(hs[fv].max()) if fv.any() else 0.0            # best same-class score in a full view
            tl = fv & (hv != (fi if fi is not None else -1))
            fv_tile_score = float(hs[tl].max()) if tl.any() else 0.0
            C_full = bool((hv == fi).any()) if fi is not None else False
            C_tile = bool((hv != (fi if fi is not None else -1)).any())
        C_any = bool((I_std[:, j] >= 0.5).any())
        C_ml = bool(((I_ml[:, j] >= 0.5) & same_ml[:, j]).any())
        M = gm_full[j] >= 0
        m_score = float(Ls[gm_full[j]]) if M else float("nan")
        S = bool(M and m_score >= tau)
        D = gm_emit_local[j] >= 0
        # oracle full-view read-out: keep every baseline detection and additionally emit any same-class
        # candidate that lies in a view holding the full object and scores >= tau (fixed scores);
        # oracle_floor: same at the 0.01 floor (what re-scoring of full-view candidates could reach)
        oracle = bool(D or fv_score >= tau)
        oracle_floor = bool(D or fv_score > 0)
        oracle_tile = bool(D or fv_tile_score >= tau)
        # priority D > M > C (a merged union box can match even when no single candidate does)
        if D:
            cat = "detected"
        elif M:
            cat = "post_hoc_recoverable"
        elif C:
            cat = "pre_suppression_only"
        else:
            cat = "no_candidate"
        reason = ""
        if cat == "post_hoc_recoverable":
            k = gm_full[j]
            if not S:
                reason = "threshold"
            elif per_class_cap is not None and int((Ls[Lc == Lc[k]] > Ls[k]).sum()) >= per_class_cap:
                reason = "per_class_cap"
            elif rank[k] >= (cap or 10 ** 9) or k not in emitted_set:
                reason = "cap"
            else:
                reason = "conflict"
            if not C:
                reason += "|union_only"
        elif cat == "no_candidate":
            reason = "ml_evidence" if C_ml else ("wrong_class_box" if C_any else "no_box")
        elif cat == "pre_suppression_only":
            # was the same-class candidate absorbed/suppressed by a same-class merged box?
            reason = "merge"
        x0, y0, x1, y1 = gb[j]
        area = (x1 - x0) * (y1 - y0)
        rows.append({"image_id": r["image_id"], "group": r["group"], "audit": int(r["audit"]), "cls": int(gc[j]),
                     "stratum": r["gt_strata"][j], "area": round(area, 2), "sqrt_area": round(math.sqrt(area), 3),
                     "size": size_bin(area), "small": int(area < 1024),
                     "V": int(cont[:, j].any()), "boundary_cut": int((inter[:, j] & ~cont[:, j]).any()),
                     "n_full_tiles": int(cont[:, j].sum()), "C_any": int(C_any), "C": int(C), "C_ml": int(C_ml),
                     "C_tile": int(C_tile), "C_full": int(C_full),
                     "c_best_score": round(best, 4), "c_best_in_fullview": int(best_full),
                     "fullview_best_score": round(fv_score, 4), "oracle": int(oracle), "oracle_floor": int(oracle_floor),
                     "oracle_tile": int(oracle_tile), "M": int(M), "m_score": round(m_score, 4) if M else "",
                     "S": int(S), "D": int(D), "category": cat, "reason": reason})
    preds = {"cls": Lc[e], "tp": tp_e, "score": Ls[e], "area": (Lb[e, 2] - Lb[e, 0]) * (Lb[e, 3] - Lb[e, 1]) if len(e) else np.zeros(0)}
    return rows, preds, {"n_std": len(std["scores"]), "n_L": len(Ls), "n_emit": len(e)}


def summarize(rows, preds_all, strata_of_cls):
    out = {}
    for setname, sel in (("audit", lambda r: r["audit"] == 1), ("contaminated", lambda r: r["audit"] == 0),
                         ("all", lambda r: True)):
        tab = defaultdict(Counter)
        for r in rows:
            if not sel(r):
                continue
            pooled = "small" if r["small"] else "notsmall"
            for key in [(r["stratum"], r["size"]), (r["stratum"], pooled), (r["stratum"], "all"), ("all", pooled), ("all", "all")]:
                t = tab["|".join(key)]
                t["n"] += 1
                for s in ("V", "boundary_cut", "C_any", "C", "C_ml", "C_tile", "C_full", "M", "S", "D", "oracle", "oracle_floor", "oracle_tile"):
                    t[s] += r[s]
                t["cat_" + r["category"]] += 1
                if r["reason"]:
                    t["reason_" + r["reason"]] += 1
        out[setname] = {k: dict(v) for k, v in sorted(tab.items())}
    # precision of emitted predictions per stratum (audit images only)
    prec = defaultdict(lambda: [0, 0])
    for p in preds_all:
        if not p["audit"]:
            continue
        for c, t in zip(p["cls"], p["tp"]):
            for key in (strata_of_cls(c), "all"):
                prec[key][0] += int(t); prec[key][1] += 1
    out["audit_precision"] = {k: {"tp": v[0], "n_pred": v[1], "precision": v[0] / v[1] if v[1] else None} for k, v in prec.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="rice", choices=["rice", "yst"])
    ap.add_argument("--readout", default="greedynmm_ios", choices=list(READOUTS))
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--per_class_cap", type=int, default=None)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--write_objects", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    meta, recs = (load_eval_rice if args.dataset == "rice" else load_eval_yst)(args.tag)
    rows, preds_all = [], []
    cls2stratum = {}
    tot = Counter()
    for r in recs:
        rw, pr, st = ledger_image(r, args.readout, args.tau, args.cap, args.per_class_cap)
        rows += rw
        pr["audit"] = r["audit"]; preds_all.append(pr)
        for x in rw:
            cls2stratum[x["cls"]] = x["stratum"]
        tot.update(st)
    if args.dataset == "rice":
        from common import load_json
        cls2stratum = {int(k) - 1: v for k, v in load_json(os.path.join(RESULTS, "audit", "audit_set.json"))["strata"].items()}
    else:
        from ledger_lib import YST_CLASSES
        cls2stratum = dict(enumerate(YST_CLASSES))
    summ = summarize(rows, preds_all, lambda c: cls2stratum.get(int(c), "unk"))
    cfg = {"run_id": args.run_id, "tag": args.tag, "dataset": args.dataset, "readout": args.readout,
           "readout_cfg": READOUTS[args.readout], "tau": args.tau, "cap": args.cap, "per_class_cap": args.per_class_cap,
           "dump": {k: meta[k] for k in ("weights", "T", "overlap", "floor", "nms_iou", "max_det", "full_view", "ref_long")},
           "n_images": len(recs), "n_objects": len(rows), "totals": dict(tot), "seconds": round(time.time() - t0, 1)}
    odir = os.path.join(RESULTS, "ledger"); os.makedirs(odir, exist_ok=True)
    save_json({"config": cfg, "summary": summ}, os.path.join(odir, f"{args.run_id}.json"))
    if args.write_objects and rows:
        with open(os.path.join(odir, f"{args.run_id}_objects.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    a = summ["audit"]
    for key in ("all|all", "tail|small", "head|small", "mid|small"):
        t = a.get(key, {})
        n = t.get("n", 0)
        if n:
            print(f"{args.run_id} audit {key}: n={n} V={t.get('V',0)} C={t.get('C',0)} M={t.get('M',0)} D={t.get('D',0)} "
                  f"recall={t.get('D',0)/n:.3f} nocand={t.get('cat_no_candidate',0)} presupp={t.get('cat_pre_suppression_only',0)} "
                  f"posthoc={t.get('cat_post_hoc_recoverable',0)}")
    print(json.dumps(cfg["totals"]), f"{cfg['seconds']}s")


if __name__ == "__main__":
    main()
