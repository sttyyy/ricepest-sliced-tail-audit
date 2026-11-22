# -*- coding: utf-8 -*-
"""Context metric: COCO-style AP@0.5 (101-point interpolation) per class on the audit set, for a dump and
read-out, using the merged list at the floor capped at 100 boxes per image (the list AP evaluation scores).
Reported as mean AP50 over the classes of each frequency stratum. Output: results/ledger/AP-<run_id>.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from collections import defaultdict

import numpy as np

from common import RESULTS, load_json, save_json
from ledger_lib import match, load_eval_rice
from s5_ledger import merged_list


def ap101(scores, tps, n_gt):
    if n_gt == 0:
        return None
    if not len(scores):
        return 0.0
    o = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tps, float)[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-12)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="yolo_T320")
    ap.add_argument("--readout", default="greedynmm_ios")
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()
    audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
    strata = {int(k) - 1: v for k, v in audit["strata"].items()}
    meta, recs = load_eval_rice(args.tag)
    S = defaultdict(list); T = defaultdict(list); NG = defaultdict(int)
    for r in recs:
        if not r["audit"]:
            continue
        _, (Lb, Ls, Lc, _) = merged_list(r, args.readout, 0.25)
        o = np.argsort(-Ls, kind="stable")[:100]
        gm, tp = match(Lb[o], Ls[o], Lc[o], r["gt_boxes"], r["gt_cls"])
        for c, s, t in zip(Lc[o], Ls[o], tp):
            S[int(c)].append(float(s)); T[int(c)].append(bool(t))
        for c in r["gt_cls"]:
            NG[int(c)] += 1
    per = {c: ap101(S[c], T[c], NG[c]) for c in range(30)}
    out = {"run_id": args.run_id, "tag": args.tag, "readout": args.readout, "per_class_AP50": per,
           "n_gt": dict(NG)}
    for st in ("head", "mid", "tail"):
        v = [per[c] for c in range(30) if strata[c] == st and per[c] is not None]
        out[f"mAP50|{st}"] = float(np.mean(v)) if v else None
    v = [x for x in per.values() if x is not None]
    out["mAP50|all"] = float(np.mean(v))
    save_json(out, os.path.join(RESULTS, "ledger", f"AP-{args.run_id}.json"))
    print({k: round(v, 3) for k, v in out.items() if k.startswith("mAP50")})


if __name__ == "__main__":
    main()
