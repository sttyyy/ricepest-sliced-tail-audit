# -*- coding: utf-8 -*-
"""Frozen detector B: torchvision FCOS-ResNet50-FPN fine-tuned from COCO on RicePest-30 train,
plus its sliced-inference candidate dump (same view set and output format as s4_dump_yolo.py).

  train: python code/s3b_fcos.py train --epochs 6
  dump : python code/s3b_fcos.py dump --weights work/weights/fcos_rice_s0_last.pth --tag fcos_T320
torchvision FCOS scores every (location, class) pair, so its standard read-out is already
multi-label; the dump stores it under both keys ("std" and "ml").
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env  # noqa: F401
import argparse
import hashlib
import json
import math
import random
import time

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fcos_resnet50_fpn

from common import RICE, WORK, RESULTS, load_json, load_rice, save_json

NUM_CLASSES = 31  # 30 pests + background index 0 (torchvision convention)


def build_model():
    m = fcos_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91, min_size=640, max_size=640,
                          score_thresh=0.01, nms_thresh=0.6, detections_per_img=300, topk_candidates=1000)
    sd = torch.load(os.path.join(WORK, "weights", "fcos_resnet50_fpn_coco-99b0c9b7.pth"), map_location="cpu")
    m.load_state_dict(sd)
    head = m.head.classification_head
    na = head.num_anchors
    head.cls_logits = torch.nn.Conv2d(256, na * NUM_CLASSES, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(head.cls_logits.weight, std=0.01)
    torch.nn.init.constant_(head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    head.num_classes = NUM_CLASSES
    return m


class RiceDet(torch.utils.data.Dataset):
    def __init__(self, train=True):
        audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
        bad = set(audit["frame_inconsistent"]["train"])
        d = load_rice("train")
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        self.items = []
        for im in d["images"]:
            if im["id"] in bad:
                continue
            self.items.append((os.path.join(RICE, "train", im["file_name"]), im["width"], im["height"], by.get(im["id"], [])))
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, W, H, anns = self.items[i]
        img = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
        ah, aw = img.shape[:2]
        rx, ry = aw / W, ah / H
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            x0, y0 = max(0, x) * rx, max(0, y) * ry
            x1, y1 = min(W, x + w) * rx, min(H, y + h) * ry
            if x1 - x0 > 1 and y1 - y0 > 1:
                boxes.append([x0, y0, x1, y1]); labels.append(a["category_id"])
        boxes = np.array(boxes, np.float32).reshape(-1, 4)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            boxes = boxes.copy(); boxes[:, [0, 2]] = aw - boxes[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t, {"boxes": torch.from_numpy(boxes), "labels": torch.tensor(labels, dtype=torch.int64)}


def collate(b):
    return tuple(zip(*b))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    m = build_model().to(dev)
    ds = RiceDet(train=True)
    g = torch.Generator(); g.manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, generator=g, persistent_workers=True, drop_last=True)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(dl); warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * it / total)))
    scaler = torch.amp.GradScaler("cuda")
    log = []
    t0 = time.time(); it = 0
    m.train()
    for ep in range(args.epochs):
        run = 0.0
        for imgs, tg in dl:
            imgs = [x.to(dev, non_blocking=True) for x in imgs]
            tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
            with torch.autocast("cuda", dtype=torch.float16):
                losses = m(imgs, tg)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss); it += 1
            if it % 200 == 0:
                print(f"ep {ep+1} it {it}/{total} loss {run/200:.3f} lr {sched.get_last_lr()[0]:.4f} {time.time()-t0:.0f}s", flush=True)
                log.append({"epoch": ep + 1, "iter": it, "loss": run / 200, **{k: float(v) for k, v in losses.items()}})
                run = 0.0
    out = os.path.join(WORK, "weights", f"{args.name}_last.pth")
    torch.save(m.state_dict(), out)
    rec = {"run_id": f"R6-{args.name}", "detector": "FCOS-ResNet50-FPN", "init": "fcos_resnet50_fpn_coco-99b0c9b7.pth",
           "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed, "min_size": 640, "max_size": 640,
           "amp": True, "checkpoint": "last", "frozen_weights": f"work/weights/{args.name}_last.pth",
           "frozen_sha256": sha256(out), "train_seconds": round(time.time() - t0, 1), "iterations": it}
    save_json(rec, os.path.join(RESULTS, "runs", f"{args.name}_train.json"))
    save_json(log, os.path.join(RESULTS, "runs", f"{args.name}_train_log.json"))
    print(json.dumps(rec, indent=1))


def dump(args):
    from sahi.slicing import get_slice_bboxes
    m = build_model()
    m.load_state_dict(torch.load(args.weights, map_location="cpu"))
    m = m.to("cuda").eval()
    audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
    bad = set(audit["frame_inconsistent"]["val"])
    d = load_rice("val")
    items = [im for im in d["images"] if im["id"] not in bad]
    out = []; views_meta = {}; t0 = time.time()
    for ii, im in enumerate(items):
        img = cv2.imread(os.path.join(RICE, "val", im["file_name"]), cv2.IMREAD_COLOR)
        h0, w0 = img.shape[:2]
        s = 640.0 / max(h0, w0)
        if abs(s - 1) > 1e-9:
            img = cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_LINEAR)
        H, W = img.shape[:2]
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        views = [[0, 0, W, H]] + get_slice_bboxes(image_height=H, image_width=W, slice_height=args.T, slice_width=args.T,
                                                    overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap)
        views_meta[str(im["id"])] = {"idx": ii, "W": W, "H": H, "scale": s, "views": views}
        tens = [torch.from_numpy(np.ascontiguousarray(rgb[y0:y1, x0:x1])).permute(2, 0, 1).float().div(255).cuda()
                for (x0, y0, x1, y1) in views]
        with torch.no_grad():
            res = []
            for k in range(0, len(tens), 8):
                res += m(tens[k:k + 8])
        for vid, (r, (x0, y0, x1, y1)) in enumerate(zip(res, views)):
            b = r["boxes"].cpu().numpy().astype(np.float64)
            if len(b) == 0:
                continue
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, x1 - x0) + x0
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, y1 - y0) + y0
            sc = r["scores"].cpu().numpy(); lb = r["labels"].cpu().numpy() - 1
            keep = sc >= args.floor
            n = int(keep.sum())
            out.append(np.column_stack([np.full(n, ii), np.full(n, vid), b[keep], sc[keep], lb[keep]]))
        if (ii + 1) % 100 == 0:
            print(f"  {ii+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    arr = np.concatenate(out) if out else np.zeros((0, 8))
    odir = os.path.join(WORK, "dumps"); os.makedirs(odir, exist_ok=True)
    np.savez_compressed(os.path.join(odir, f"{args.tag}.npz"), std=arr, ml=arr)
    meta = {"tag": args.tag, "weights": os.path.basename(args.weights), "dataset": "rice", "split": "val", "T": args.T,
            "overlap": args.overlap, "ref_long": 640.0, "imgsz": 640, "floor": args.floor, "nms_iou": 0.6, "max_det": 300,
            "full_view": True, "n_images": len(items), "n_std": int(len(arr)), "n_ml": int(len(arr)),
            "seconds": round(time.time() - t0, 1), "images": views_meta,
            "note": "torchvision FCOS read-out is multi-label by construction; std == ml"}
    save_json(meta, os.path.join(odir, f"{args.tag}_meta.json"))
    print({k: v for k, v in meta.items() if k != "images"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "dump"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="fcos_rice_s0")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tag", default="fcos_T320")
    ap.add_argument("--T", type=int, default=320)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--floor", type=float, default=0.01)
    args = ap.parse_args()
    train(args) if args.mode == "train" else dump(args)


if __name__ == "__main__":
    main()
