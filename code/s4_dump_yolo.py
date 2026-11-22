# -*- coding: utf-8 -*-
"""Sliced-inference candidate dump for a frozen Ultralytics detector (RicePest-30 val or YST).

For each image in the 640-long-edge reference frame we build the SAHI view set
(full image = view 0, plus sahi.slicing.get_slice_bboxes tiles), run ONE forward pass per
view, and keep two read-outs of the same raw head output:
  std : the detector's standard per-view read-out (Ultralytics NMS, class-aware, IoU 0.7,
        single label per anchor, max_det 300) at the low score floor  -> frozen candidate dump
  ml  : multi-label read-out of the same forward pass (every class score >= floor per anchor,
        class-aware NMS)                                              -> evidence for CCPR
Every candidate keeps its view id (provenance). Boxes are mapped back to reference-frame coords.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env  # noqa: F401
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, RICE, YST, WORK, load_json, load_rice, save_json


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top = (size - nh) // 2; left = (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = im
    return out, r, left, top


def build_items(dataset, split):
    items = []
    if dataset == "rice":
        audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
        bad = set(audit["frame_inconsistent"][split])
        d = load_rice(split)
        for im in d["images"]:
            if im["id"] in bad:
                continue
            items.append({"image_id": im["id"], "path": os.path.join(RICE, split, im["file_name"])})
    else:
        sp = load_json(os.path.join(RESULTS, "yst", "split.json"))
        for name in sp[split]:
            items.append({"image_id": name, "path": os.path.join(YST, "images", f"{name}.jpg")})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="rice", choices=["rice", "yst"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--T", type=int, default=320)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--ref_long", type=float, default=640.0, help="reference long edge; <=0 keeps native")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--nms_iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no_full_view", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from sahi.slicing import get_slice_bboxes

    yolo = YOLO(args.weights)
    net = yolo.model.to("cuda").eval().float()
    nc = len(yolo.names)
    items = build_items(args.dataset, args.split)
    out = {"std": [], "ml": []}
    views_meta = {}
    t0 = time.time()
    queue = []  # (item_idx, view_id, tensor, r, left, top, x0, y0)

    def flush():
        if not queue:
            return
        x = torch.from_numpy(np.stack([q[2] for q in queue])).to("cuda").permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            pred = net(x)
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        for key, ml in (("std", False), ("ml", True)):
            dets = non_max_suppression(pred.clone(), conf_thres=args.floor, iou_thres=args.nms_iou, multi_label=ml,
                                       max_det=args.max_det, nc=nc, max_time_img=10.0)
            for q, d in zip(queue, dets):
                if d is None or len(d) == 0:
                    continue
                d = d.cpu().numpy().astype(np.float64)
                ii, vid, _, r, left, top, ox, oy, vw, vh = q
                bx = d[:, :4].copy()
                bx[:, [0, 2]] = (bx[:, [0, 2]] - left) / r
                bx[:, [1, 3]] = (bx[:, [1, 3]] - top) / r
                bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0, vw) + ox
                bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0, vh) + oy
                n = len(d)
                out[key].append(np.column_stack([np.full(n, ii), np.full(n, vid), bx, d[:, 4], d[:, 5]]))
        queue.clear()

    for ii, it in enumerate(items):
        img = cv2.imread(it["path"], cv2.IMREAD_COLOR)
        h0, w0 = img.shape[:2]
        if args.ref_long > 0:
            s = args.ref_long / max(h0, w0)
            img = cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_LINEAR) if abs(s - 1) > 1e-9 else img
        else:
            s = 1.0
        H, W = img.shape[:2]
        rgb = img[:, :, ::-1]
        views = [] if args.no_full_view else [[0, 0, W, H]]
        views += get_slice_bboxes(image_height=H, image_width=W, slice_height=args.T, slice_width=args.T,
                                  overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap)
        views_meta[str(it["image_id"])] = {"idx": ii, "W": W, "H": H, "scale": s, "views": views}
        for vid, (x0, y0, x1, y1) in enumerate(views):
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            lb, r, left, top = letterbox(crop, args.imgsz)
            queue.append((ii, vid, lb, r, left, top, x0, y0, x1 - x0, y1 - y0))
            if len(queue) >= args.batch:
                flush()
        if (ii + 1) % 100 == 0:
            print(f"  {ii+1}/{len(items)} images  {time.time()-t0:.0f}s", flush=True)
    flush()
    odir = os.path.join(WORK, "dumps"); os.makedirs(odir, exist_ok=True)
    arrs = {k: (np.concatenate(v) if v else np.zeros((0, 8))) for k, v in out.items()}
    np.savez_compressed(os.path.join(odir, f"{args.tag}.npz"), std=arrs["std"], ml=arrs["ml"])
    meta = {"tag": args.tag, "weights": os.path.basename(args.weights), "dataset": args.dataset, "split": args.split,
            "T": args.T, "overlap": args.overlap, "ref_long": args.ref_long, "imgsz": args.imgsz, "floor": args.floor,
            "nms_iou": args.nms_iou, "max_det": args.max_det, "full_view": not args.no_full_view,
            "n_images": len(items), "n_std": int(len(arrs["std"])), "n_ml": int(len(arrs["ml"])),
            "seconds": round(time.time() - t0, 1), "columns": ["img_idx", "view", "x1", "y1", "x2", "y2", "score", "cls"],
            "images": views_meta}
    save_json(meta, os.path.join(odir, f"{args.tag}_meta.json"))
    print(json.dumps({k: v for k, v in meta.items() if k != "images"}, indent=1))


if __name__ == "__main__":
    main()
