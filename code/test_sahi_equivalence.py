# -*- coding: utf-8 -*-
"""Check that ledger_lib.readout_sahi reproduces SAHI's own GreedyNMMPostprocess / NMSPostprocess
on the same candidate list (uses any dump in work/dumps). Usage: python code/test_sahi_equivalence.py <tag>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import gpu_env  # noqa: F401
from common import WORK, save_json, RESULTS
from ledger_lib import readout_sahi
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import GreedyNMMPostprocess, NMSPostprocess


def to_ops(b, s, c, W, H):
    return [ObjectPrediction(bbox=[float(x) for x in b[i]], category_id=int(c[i]), category_name=str(int(c[i])),
                             score=float(s[i]), full_shape=[int(H), int(W)]) for i in range(len(b))]


def main():
    tag = sys.argv[1]
    d = np.load(os.path.join(WORK, "dumps", f"{tag}.npz"))["std"]
    rng = np.random.default_rng(0)
    imgs = rng.choice(np.unique(d[:, 0]).astype(int), size=min(40, len(np.unique(d[:, 0]))), replace=False)
    worst = 0.0; worst_s = 0.0; cls_mis = 0; n_checked = 0; mismatched = 0
    for ii in imgs:
        a = d[d[:, 0] == ii]
        b, s, c = a[:, 2:6], a[:, 6], a[:, 7].astype(int)
        ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]); b, s, c = b[ok], s[ok], c[ok]
        W, H = float(b[:, 2].max()) + 1, float(b[:, 3].max()) + 1
        for kind, post in (("GREEDYNMM", GreedyNMMPostprocess(match_threshold=0.5, match_metric="IOS", class_agnostic=False)),
                           ("NMS", NMSPostprocess(match_threshold=0.5, match_metric="IOU", class_agnostic=False))):
            ref = post(to_ops(b, s, c, W, H))
            rb = np.array([o.bbox.to_xyxy() for o in ref]).reshape(-1, 4); rs = np.array([o.score.value for o in ref])
            rc = np.array([o.category.id for o in ref])
            mb, ms, mc, _ = readout_sahi(b, s, c, kind=kind, metric="IOS" if kind == "GREEDYNMM" else "IOU", thr=0.5)
            n_checked += 1
            if len(rb) != len(mb):
                mismatched += 1; continue
            o1 = np.lexsort((rb[:, 0], -rs)); o2 = np.lexsort((mb[:, 0], -ms))
            worst = max(worst, float(np.abs(rb[o1] - mb[o2]).max()) if len(rb) else 0.0)
            worst_s = max(worst_s, float(np.abs(rs[o1] - ms[o2]).max()) if len(rs) else 0.0)
            cls_mis += int((rc[o1] != mc[o2]).sum())
    res = {"dump": tag, "lists_checked": n_checked, "count_mismatches": mismatched, "max_abs_coord_diff": worst,
           "max_abs_score_diff": worst_s, "class_mismatches": cls_mis}
    save_json(res, os.path.join(RESULTS, "audit", "sahi_equivalence_check.json"))
    print(res)


if __name__ == "__main__":
    main()
