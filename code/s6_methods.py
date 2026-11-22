# -*- coding: utf-8 -*-
"""Runs 5 and 9: same-dump post-hoc baselines (NorCal-style prior correction, class x size threshold
calibration, Na-style cross-view agreement rescoring, class-agnostic and per-class) and the
class-conditional peak read-out (CCPR) with ablations, all on the identical frozen candidate dump.

Parameters are fit by 5-fold GROUP cross-fitting on the audit set (never on the fold being scored).
Every method is reported at (a) its own fitted operating point (threshold 0.25 on transformed scores)
and (b) a global threshold that matches the baseline's total false-positive count.
Recoveries (baseline miss -> method hit) are attributed to the baseline's recovery-ceiling category.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import csv
import json
import time
from collections import Counter, defaultdict

import numpy as np

from common import RESULTS, size_bin, save_json, load_json
from ledger_lib import iou_matrix, contains_matrix, readout_sahi, readout_wbf, match, emit, load_eval_rice
from s5_ledger import valid, READOUTS

TAU0 = 0.25
CAP = 100
SIZES = ["tiny", "s16_32", "medium", "large"]


def area_bins(b):
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.array([SIZES.index(size_bin(x)) for x in a], int) if len(a) else np.zeros(0, int)


def prepare(recs, readout):
    out = []
    for r in recs:
        if not r["audit"]:
            continue
        std = valid(r["std"]); ml = valid(r["ml"])
        Lb, Ls, Lc, Lm = readout_sahi(std["boxes"], std["scores"], std["cls"], **READOUTS[readout])
        # Na-style re-detection rate over the effective view set E(b) = views fully containing b
        V = r["views"]
        cont = contains_matrix(V, Lb)                                  # views x L
        I = iou_matrix(Lb, std["boxes"])                               # L x std
        hit_any = np.zeros((len(V), len(Lb)), bool); hit_cls = np.zeros_like(hit_any)
        for v in range(len(V)):
            sel = std["view"] == v
            if sel.any() and len(Lb):
                Iv = I[:, sel] >= 0.5
                hit_any[v] = Iv.any(1)
                hit_cls[v] = (Iv & (Lc[:, None] == std["cls"][sel][None, :])).any(1)
        nE = np.maximum(cont.sum(0), 1)
        r_any = (hit_any & cont).sum(0) / nE; r_cls = (hit_cls & cont).sum(0) / nE
        Wb, Ws, Wc, Wm, Wsup = readout_wbf(ml["boxes"], ml["scores"], ml["cls"], ml["view"])
        Sb, Ss, Sc, Sm, Ssup = readout_wbf(std["boxes"], std["scores"], std["cls"], std["view"])
        Gb, Gs, Gc, Gm = readout_sahi(ml["boxes"], ml["scores"], ml["cls"], **READOUTS[readout])
        out.append({"image_id": r["image_id"], "group": r["group"], "gb": r["gt_boxes"], "gc": r["gt_cls"],
                    "gs": r["gt_strata"], "gbin": area_bins(r["gt_boxes"]),
                    "lists": {"L": (Lb, Ls, Lc), "WBF_ml": (Wb, Ws, Wc), "WBF_std": (Sb, Ss, Sc), "GNMM_ml": (Gb, Gs, Gc)},
                    "r_any": r_any, "r_cls": r_cls})
    return out


# ----------------------------------------------------------------------------- score transforms
def t_identity(img, key, p):
    return img["lists"][key][1]


def t_norcal(img, key, p, counts=None):
    s = img["lists"][key][1]; c = img["lists"][key][2]
    return s * np.power(p["Nref"] / p["N"][c], p["gamma"]) if len(s) else s


def t_na(img, key, p):
    s = img["lists"][key][1]
    r = img["r_cls"] if p["per_class"] else img["r_any"]
    return s * np.power(np.clip(r, 1e-6, 1), p["alpha"]) if len(s) else s


def t_cell(img, key, p):
    b, s, c = img["lists"][key]
    if not len(s):
        return s
    tb = area_bins(b)
    th = np.array([p["thr"].get((int(ci), int(bi)), TAU0) for ci, bi in zip(c, tb)])
    return s * TAU0 / th


# ----------------------------------------------------------------------------- evaluation helpers
def eval_lists(imgs, key, scores_fn, tau=TAU0, cap=CAP):
    """Exact re-matching on emitted lists. Returns per-image (gt detected flags, pred tp flags, pred cls)."""
    res = []
    for img in imgs:
        b, s0, c = img["lists"][key]
        s = scores_fn(img)
        e = emit(s, tau, cap)
        gm, tp = match(b[e], s[e], c[e], img["gb"], img["gc"])
        res.append((gm >= 0, tp, c[e], s[e]))
    return res


def macro_f1(imgs, res, nc=30):
    tp = np.zeros(nc); fp = np.zeros(nc); ng = np.zeros(nc)
    for img, (det, ptp, pc, _) in zip(imgs, res):
        np.add.at(ng, img["gc"], 1)
        np.add.at(tp, pc[ptp], 1); np.add.at(fp, pc[~ptp], 1)
    valid_c = ng > 0
    f1 = np.where(valid_c, 2 * tp / np.maximum(2 * tp + fp + (ng - tp), 1), 0)
    return float(f1[valid_c].mean())


def fit_cell_thresholds(imgs, key, strata_of_cls, min_gt=10):
    """Per (class, size-bin) threshold maximizing cell F1 using full-list matching flags; fallback to
    (stratum, size-bin), then to TAU0."""
    grid = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7])
    # exact re-matching at every grid threshold (TP status can change once higher boxes are removed);
    # TP/pred counts per (class, predicted-box size bin), GT counts per (class, GT size bin)
    # cell = (class, size bin). TP and FP are counted in the bin of the PREDICTED box (that is where the
    # threshold acts at inference); FN = unmatched GT counted in the bin of the GT. F1 = 2TP / (2TP + FP + FN).
    tp_c = np.zeros((len(grid), 30, 4)); np_c = np.zeros((len(grid), 30, 4)); fn_c = np.zeros((len(grid), 30, 4))
    ng = np.zeros((30, 4))
    for img in imgs:
        b, s, c = img["lists"][key]
        pb = area_bins(b)
        np.add.at(ng, (img["gc"], img["gbin"]), 1)
        for gi, g in enumerate(grid):
            e = np.where(s >= g)[0]
            e = e[np.argsort(-s[e], kind="stable")][:CAP]
            gm, tp = match(b[e], s[e], c[e], img["gb"], img["gc"])
            if len(e):
                np.add.at(np_c[gi], (c[e], pb[e]), 1)
                np.add.at(tp_c[gi], (c[e][tp], pb[e][tp]), 1)
            un = gm < 0
            np.add.at(fn_c[gi], (img["gc"][un], img["gbin"][un]), 1)

    def best(tp, npred, fn):
        f = 2 * tp / np.maximum(2 * tp + (npred - tp) + fn, 1)
        return float(grid[int(np.argmax(f))])
    thr = {}
    strata_ids = defaultdict(list)
    for ci in range(30):
        strata_ids[strata_of_cls.get(ci, "tail")].append(ci)
    for ci in range(30):
        for bi in range(4):
            if ng[ci, bi] >= min_gt:
                thr[(ci, bi)] = best(tp_c[:, ci, bi], np_c[:, ci, bi], fn_c[:, ci, bi])
            else:
                ids = strata_ids[strata_of_cls.get(ci, "tail")]
                n_pool = ng[ids, bi].sum()
                thr[(ci, bi)] = (best(tp_c[:, ids, bi].sum(1), np_c[:, ids, bi].sum(1), fn_c[:, ids, bi].sum(1))
                                 if n_pool >= min_gt else TAU0)
    return thr


METHODS = [  # name, candidate list, transform, fixed params
    ("baseline", "L", "identity", {}),
    ("norcal", "L", "norcal", {}),
    ("cls_size_calib", "L", "cell", {}),
    ("na_agnostic", "L", "na", {"per_class": False}),
    ("na_perclass", "L", "na", {"per_class": True}),
    ("ccpr", "WBF_ml", "cell", {}),
    ("ccpr_nocalib", "WBF_ml", "identity", {}),
    ("ccpr_std_evidence", "WBF_std", "cell", {}),
    ("ml_greedynmm_calib", "GNMM_ml", "cell", {}),
]


def fit_and_score(name, key, kind, fixed, train, test, counts, strata_of_cls):
    if kind == "identity":
        return {}, [t_identity(im, key, {}) for im in test]
    if kind == "norcal":
        N = np.array([max(1, counts.get(c + 1, 1)) for c in range(30)], float)
        base = {"N": N, "Nref": float(np.median(N))}
        best = None
        for g in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            p = dict(base, gamma=g)
            f = macro_f1(train, eval_lists(train, key, lambda im: t_norcal(im, key, p)))
            if best is None or f > best[0]:
                best = (f, p)
        p = best[1]
        return {"gamma": p["gamma"]}, [t_norcal(im, key, p) for im in test]
    if kind == "na":
        best = None
        for a in [0.0, 0.25, 0.5, 1.0, 2.0]:
            p = dict(fixed, alpha=a)
            f = macro_f1(train, eval_lists(train, key, lambda im: t_na(im, key, p)))
            if best is None or f > best[0]:
                best = (f, p)
        p = best[1]
        return {"alpha": p["alpha"]}, [t_na(im, key, p) for im in test]
    if kind == "cell":
        thr = fit_cell_thresholds(train, key, strata_of_cls)
        p = {"thr": thr}
        return {"n_cells_nondefault": int(sum(1 for v in thr.values() if v != TAU0))}, [t_cell(im, key, p) for im in test]
    raise ValueError(kind)


def metrics(imgs, res):
    agg = Counter()
    per_cls = defaultdict(lambda: [0, 0])
    for img, (det, ptp, pc, _) in zip(imgs, res):
        for j, (c, st, bi) in enumerate(zip(img["gc"], img["gs"], img["gbin"])):
            small = bi <= 1
            for k in (st, "all", f"{st}|{'small' if small else 'notsmall'}"):
                agg[f"n|{k}"] += 1; agg[f"tp|{k}"] += int(det[j])
            per_cls[int(c)][0] += 1; per_cls[int(c)][1] += int(det[j])
        for c, t in zip(pc, ptp):
            st = STRATA[int(c)]
            agg[f"pred|{st}"] += 1; agg[f"ptp|{st}"] += int(t); agg["pred|all"] += 1; agg["ptp|all"] += int(t)
    out = {"FP": agg["pred|all"] - agg["ptp|all"]}
    for k in ("all", "head", "mid", "tail", "head|small", "mid|small", "tail|small", "tail|notsmall", "head|notsmall"):
        n = agg[f"n|{k}"]
        out[f"recall|{k}"] = agg[f"tp|{k}"] / n if n else None; out[f"n|{k}"] = n; out[f"tp|{k}"] = agg[f"tp|{k}"]
    for k in ("all", "head", "mid", "tail"):
        n = agg[f"pred|{k}"]
        out[f"precision|{k}"] = agg[f"ptp|{k}"] / n if n else None
    tails = [v[1] / v[0] for c, v in per_cls.items() if STRATA[c] == "tail" and v[0] > 0]
    out["tail_macro_recall"] = float(np.mean(tails)) if tails else None
    return out


def global_threshold_for_fp(imgs, key, scores, target_fp):
    lo, hi = 1e-4, max(1.0, max((float(s.max()) for s in scores if len(s)), default=1.0))
    res_best = None
    for _ in range(28):
        mid = np.sqrt(lo * hi)
        res = eval_lists(imgs, key, lambda im, _s=scores: _s[im["_i"]], tau=mid)
        fp = sum(int((~r[1]).sum()) for r in res)
        if fp > target_fp:
            lo = mid
        else:
            hi = mid; res_best = (mid, res)
    if res_best is None:
        res_best = (hi, eval_lists(imgs, key, lambda im, _s=scores: _s[im["_i"]], tau=hi))
    return res_best


def cluster_boot(imgs, flags_a, flags_b, sel_fn, B=2000, seed=0):
    """Paired cluster percentile bootstrap of recall(a) - recall(b) over audit groups for GT subset sel_fn."""
    groups = sorted({im["group"] for im in imgs})
    gi = {g: i for i, g in enumerate(groups)}
    na = np.zeros(len(groups)); nb = np.zeros(len(groups)); nn = np.zeros(len(groups))
    for im, fa, fb in zip(imgs, flags_a, flags_b):
        m = np.array([sel_fn(c, st, bi) for c, st, bi in zip(im["gc"], im["gs"], im["gbin"])], bool)
        if not m.any():
            continue
        k = gi[im["group"]]
        na[k] += fa[m].sum(); nb[k] += fb[m].sum(); nn[k] += m.sum()
    used = nn > 0
    na, nb, nn = na[used], nb[used], nn[used]
    rng = np.random.default_rng(seed)
    G = len(nn)
    if G == 0:
        return None
    idx = rng.integers(0, G, size=(B, G))
    d = (na[idx].sum(1) - nb[idx].sum(1)) / np.maximum(nn[idx].sum(1), 1)
    est = (na.sum() - nb.sum()) / nn.sum()
    return {"diff": float(est), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "lower_one_sided_95": float(np.percentile(d, 5)), "groups": int(G), "events": int(nn.sum()),
            "extra_tp": int(na.sum() - nb.sum())}


STRATA = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="yolo_T320")
    ap.add_argument("--readout", default="greedynmm_ios")
    ap.add_argument("--run_id", default="R5R9-methods-yolo")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    audit = load_json(os.path.join(RESULTS, "audit", "audit_set.json"))
    for k, v in audit["strata"].items():
        STRATA[int(k) - 1] = v
    counts = {int(k): v for k, v in audit["train_counts"].items()}
    meta, recs = load_eval_rice(args.tag)
    imgs = prepare(recs, args.readout)
    # baseline categories per GT (C from std dump, M from full merged list)
    cat = []
    for r in [r for r in recs if r["audit"]]:
        std = valid(r["std"])
        I = iou_matrix(std["boxes"], r["gt_boxes"]); same = std["cls"][:, None] == r["gt_cls"][None, :]
        C = ((I >= 0.5) & same).any(0) if len(std["scores"]) else np.zeros(len(r["gt_cls"]), bool)
        cat.append(C)
    for i, im in enumerate(imgs):
        im["_i"] = i
    # stratified group folds: groups sorted by tail-small count (ties shuffled with seed 0), dealt round-robin
    groups = sorted({im["group"] for im in imgs})
    ts_cnt = Counter()
    for im in imgs:
        ts_cnt[im["group"]] += int(sum(1 for st, bi in zip(im["gs"], im["gbin"]) if st == "tail" and bi <= 1))
    rng = np.random.default_rng(0)
    tie = dict(zip(groups, rng.permutation(len(groups))))
    ordered = sorted(groups, key=lambda g: (-ts_cnt[g], tie[g]))
    fold_of = {g: i % args.folds for i, g in enumerate(ordered)}
    fold_ts = Counter()
    for g in groups:
        fold_ts[fold_of[g]] += ts_cnt[g]
    results = {}; flags = {}; params_log = {}
    for name, key, kind, fixed in METHODS:
        scores = [None] * len(imgs); plog = []
        for k in range(args.folds):
            tr = [im for im in imgs if fold_of[im["group"]] != k]
            te_idx = [i for i, im in enumerate(imgs) if fold_of[im["group"]] == k]
            p, sc = fit_and_score(name, key, kind, fixed, tr, [imgs[i] for i in te_idx], counts, STRATA)
            for i, s in zip(te_idx, sc):
                scores[i] = s
            plog.append(p)
        params_log[name] = plog
        own = eval_lists(imgs, key, lambda im, _s=scores: _s[im["_i"]])
        results[name] = {"own": metrics(imgs, own)}
        flags[name] = {"own": [r[0] for r in own]}
        print(f"{name:22s} own: R_all={results[name]['own']['recall|all']:.3f} R_tailsmall={results[name]['own']['recall|tail|small']:.3f} "
              f"P_head={results[name]['own']['precision|head']:.3f} FP={results[name]['own']['FP']}  {time.time()-t0:.0f}s", flush=True)
        results[name]["_scores"] = scores; results[name]["_key"] = key
    target = results["baseline"]["own"]["FP"]
    for name, key, kind, fixed in METHODS:
        tau, res = global_threshold_for_fp(imgs, key, results[name]["_scores"], target)
        results[name]["matched_fp"] = dict(metrics(imgs, res), tau_global=float(tau))
        flags[name]["matched_fp"] = [r[0] for r in res]
    # matched head precision for CCPR vs class x size calibration
    hp_target = results["cls_size_calib"]["own"]["precision|head"]
    sc = results["ccpr"]["_scores"]
    lo, hi = 1e-3, 1.01 * max((float(x.max()) for x in sc if len(x)), default=1.0); best = None
    for _ in range(28):
        mid = np.sqrt(lo * hi)
        res = eval_lists(imgs, "WBF_ml", lambda im: sc[im["_i"]], tau=mid)
        hp = metrics(imgs, res)["precision|head"] or 0
        if hp < hp_target:
            lo = mid
        else:
            hi = mid; best = (mid, res)
    if best is None:
        best = (hi, eval_lists(imgs, "WBF_ml", lambda im: sc[im["_i"]], tau=hi))
    results["ccpr"]["matched_head_precision"] = dict(metrics(imgs, best[1]), tau_global=float(best[0]),
                                                     head_precision_target=hp_target)
    results["ccpr"]["matched_head_precision"]["precision_matched"] = bool(
        abs((results["ccpr"]["matched_head_precision"]["precision|head"] or 0) - hp_target) <= 0.01)
    flags["ccpr"]["matched_head_precision"] = [r[0] for r in best[1]]
    # recoveries vs baseline attributed to baseline recovery-ceiling category
    Mfull = []
    for im in imgs:
        b, s, c = im["lists"]["L"]
        gm, _ = match(b, s, c, im["gb"], im["gc"]); Mfull.append(gm >= 0)
    attrib = {}
    for name in results:
        for op, fl in flags[name].items():
            base = flags["baseline"].get(op, flags["baseline"]["own"])
            rec = Counter(); lost = Counter()
            for im, fb, fm, C, M in zip(imgs, base, fl, cat, Mfull):
                for j in range(len(fb)):
                    ts = im["gs"][j] == "tail" and im["gbin"][j] <= 1
                    if fm[j] and not fb[j]:
                        k = "post_hoc_recoverable" if M[j] else ("pre_suppression_only" if C[j] else "no_candidate")
                        rec[k] += 1
                        if ts:
                            rec["tailsmall|" + k] += 1
                    if fb[j] and not fm[j]:
                        lost["all"] += 1
                        if ts:
                            lost["tailsmall"] += 1
            attrib[f"{name}|{op}"] = {"recovered": dict(rec), "lost": dict(lost)}
    # CCPR go/no-go (pre-registered): vs class x size calibration
    sel_ts = lambda c, st, bi: st == "tail" and bi <= 1
    sel_head = lambda c, st, bi: st == "head"
    go = {}
    for op_c, op_k in (("own", "own"), ("matched_head_precision", "own"), ("matched_fp", "matched_fp")):
        ts = cluster_boot(imgs, flags["ccpr"][op_c], flags["cls_size_calib"][op_k], sel_ts)
        hd = cluster_boot(imgs, flags["ccpr"][op_c], flags["cls_size_calib"][op_k], sel_head)
        delta = 0.01 if (hd and hd["groups"] >= 20 and hd["events"] >= 200) else (0.02 if hd and hd["groups"] >= 10 else None)
        crit = {"extra_tail_small_tp>=5": bool(ts and ts["extra_tp"] >= 5),
                "head_NI_pass": bool(delta is not None and hd["lower_one_sided_95"] > -delta),
                "tail_small_CI_excludes_0": bool(ts and ts["ci95"][0] > 0)}
        go[f"ccpr[{op_c}]_vs_calib[{op_k}]"] = {"tail_small": ts, "head": hd, "NI_margin": delta, "criteria": crit,
                                               "GO": all(crit.values())}
    # paired comparisons of each method vs baseline (tail-small) at matched FP
    paired = {}
    for name in results:
        if name == "baseline":
            continue
        paired[name] = {}
        for lab, fn in (("tail_small", sel_ts), ("head", sel_head), ("tail", lambda c, st, bi: st == "tail"),
                        ("small", lambda c, st, bi: bi <= 1), ("all", lambda c, st, bi: True)):
            paired[name][f"{lab}_vs_baseline_matched_fp"] = cluster_boot(imgs, flags[name]["matched_fp"], flags["baseline"]["matched_fp"], fn)
            paired[name][f"{lab}_vs_baseline_own"] = cluster_boot(imgs, flags[name]["own"], flags["baseline"]["own"], fn)
    for name in results:
        results[name].pop("_scores"); results[name].pop("_key")
    out = {"run_id": args.run_id, "tag": args.tag, "readout": args.readout, "tau0": TAU0, "cap": CAP, "folds": args.folds,
           "n_audit_images": len(imgs), "n_groups": len(groups), "tail_small_per_fold": dict(fold_ts),
           "methods": results, "attribution": attrib,
           "ccpr_go_no_go": go, "paired_vs_baseline": paired,
           "fitted_params": {k: [{kk: (vv if not isinstance(vv, dict) else len(vv)) for kk, vv in p.items()} for p in v] for k, v in params_log.items()},
           "seconds": round(time.time() - t0, 1)}
    save_json(out, os.path.join(RESULTS, "methods", f"{args.run_id}.json"))
    print(json.dumps(go, indent=1))


if __name__ == "__main__":
    main()
