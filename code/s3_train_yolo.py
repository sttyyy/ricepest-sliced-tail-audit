# -*- coding: utf-8 -*-
"""Train the frozen detector A: YOLO11n on RicePest-30 train (frame-consistent images).

No model selection on the validation split: training runs with val=False and the
LAST checkpoint is frozen (so the audit set is never used for early stopping).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env  # noqa: F401  (must precede torch import; also inherited by spawn workers)
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, RESULTS, save_json


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(WORK, "weights", "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="yolo11n_rice_s0")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--data", default=os.path.join(WORK, "yolo_rice", "rice30.yaml"))
    args = ap.parse_args()
    gpu_env.configure_ultralytics()
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(args.model)
    data = args.data
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                seed=args.seed, deterministic=True, val=False, plots=False, project=os.path.join(WORK, "runs"),
                name=args.name, exist_ok=True, device=0, cache=False, fraction=args.fraction, amp=True,
                verbose=True)
    run_dir = os.path.join(WORK, "runs", args.name)
    last = os.path.join(run_dir, "weights", "last.pt")
    frozen = os.path.join(WORK, "weights", f"{args.name}_last.pt")
    shutil.copy2(last, frozen)
    rec = {"run_id": f"R2-{args.name}", "detector": "YOLO11n", "init": os.path.basename(args.model),
           "init_sha256": sha256(args.model), "data": os.path.basename(args.data), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
           "seed": args.seed, "deterministic": True, "val_during_training": False, "checkpoint": "last",
           "frozen_weights": os.path.relpath(frozen, os.path.dirname(WORK)).replace(os.sep, "/"),
           "frozen_sha256": sha256(frozen), "train_seconds": round(time.time() - t0, 1)}
    csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(csv):
        os.makedirs(os.path.join(RESULTS, "runs"), exist_ok=True)
        shutil.copy2(csv, os.path.join(RESULTS, "runs", f"{args.name}_train_log.csv"))
    save_json(rec, os.path.join(RESULTS, "runs", f"{args.name}_train.json"))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
