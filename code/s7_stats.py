# -*- coding: utf-8 -*-
"""Run 3 (primary, confirmatory) statistics on a ledger objects CSV (audit set only).

K2: recovery-ceiling partition of pooled tail-small misses: counts, proportions, cluster percentile
    bootstrap CIs over audit groups (2,000 resamples) + Bonferroni-simultaneous 98.3% CIs.
    CIs need >=10 groups that contain misses; 10-19 such groups also get a leave-one-group-out range.
K3: size-matched frequency contrast: tail-small vs head-small re-weighted to the tail sqrt-area bin
    mix (bins populated in both groups only); per category share difference with cluster-bootstrap CI
    and class-label permutation p (the k tail classes that have small objects vs the head classes that
    have small objects; exact enumeration of all splits when feasible); Holm over the 3 miss categories.
Exploratory: tail vs head share differences per size bin x category (bootstrap CI, class-label
    permutation p, BH-adjusted).
Minimum-cell rule: <10 events or <5 groups -> counts only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from common import RESULTS, save_json

CATS = ["no_candidate", "pre_suppression_only", "post_hoc_recoverable"]
ALLCATS = ["detected"] + CATS
MATCH_BINS = [0, 12, 16, 24, 32]


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("audit", "cls", "group", "small", "D", "C", "M", "V", "C_ml", "C_any", "C_tile", "C_full",
                  "oracle", "oracle_floor", "oracle_tile"):
            r[k] = int(r[k])
        r["sqrt_area"] = float(r["sqrt_area"])
    return rows


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k]); adj[k] = min(1.0, run)
    return adj.tolist()


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1):
        k = order[i]; run = min(run, ps[k] * m / (i + 1)); adj[k] = run
    return adj.tolist()


def partition(rows, B, rng, label):
    misses = [r for r in rows if r["category"] != "detected"]
    groups = sorted({r["group"] for r in rows})
    gm = sorted({r["group"] for r in misses})
    cnt = Counter(r["category"] for r in rows)
    out = {"label": label, "n_objects": len(rows), "n_groups": len(groups), "n_detected": cnt["detected"],
           "n_misses": len(misses), "n_miss_groups": len(gm), "counts": {c: cnt[c] for c in CATS}}
    out["recall"] = cnt["detected"] / len(rows) if rows else None
    if len(misses) < 10 or len(gm) < 5:
        out["rule"] = "below minimum-cell rule: counts only"
        return out
    by = defaultdict(lambda: np.zeros(4))
    for r in rows:
        by[r["group"]][ALLCATS.index(r["category"])] += 1
    G = np.array([by[g] for g in groups])
    out["prop_of_misses"] = {c: cnt[c] / len(misses) for c in CATS}
    out["prop_of_objects"] = {c: cnt[c] / len(rows) for c in ALLCATS}
    if len(gm) < 10:
        out["rule"] = "<10 groups with misses: no CI"
        return out
    idx = rng.integers(0, len(groups), size=(B, len(groups)))
    S = G[idx].sum(1)                                  # B x 4
    miss = S[:, 1:].sum(1)
    ok = miss > 0
    out["n_valid_resamples"] = int(ok.sum())
    P = S[ok, 1:] / miss[ok, None]
    out["ci95_prop_of_misses"] = {c: [float(np.percentile(P[:, i], 2.5)), float(np.percentile(P[:, i], 97.5))] for i, c in enumerate(CATS)}
    a = 0.05 / 3
    out["ci_simult_98.3_prop_of_misses"] = {c: [float(np.percentile(P[:, i], 100 * a / 2)), float(np.percentile(P[:, i], 100 * (1 - a / 2)))] for i, c in enumerate(CATS)}
    nz = S.sum(1) > 0
    R = S[nz, 0] / S[nz].sum(1)
    out["ci95_recall"] = [float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))]
    if len(gm) < 20:
        # leave-one-group-out jackknife summary as required for 10-19 groups
        jk = []
        for g in range(len(groups)):
            s = G.sum(0) - G[g]
            jk.append((s[1:] / max(1, s[1:].sum())).tolist())
        out["jackknife_range_prop_of_misses"] = {c: [float(min(j[i] for j in jk)), float(max(j[i] for j in jk))] for i, c in enumerate(CATS)}
    dom = max(CATS, key=lambda c: cnt[c])
    out["dominant_category"] = dom
    return out


def std_shares(rows_t, rows_h, bins=None):
    """Category shares (of all objects) for tail and size-standardized head (weights = tail bin mix).
    Only size bins populated in BOTH groups are used, for tail and head alike (no silent renormalization)."""
    def binof(x):
        for i in range(len(MATCH_BINS) - 1):
            if MATCH_BINS[i] <= x < MATCH_BINS[i + 1]:
                return i
        return len(MATCH_BINS) - 2
    tb = defaultdict(list); hb = defaultdict(list)
    for r in rows_t:
        tb[binof(r["sqrt_area"])].append(r)
    for r in rows_h:
        hb[binof(r["sqrt_area"])].append(r)
    common = [b for b in tb if hb.get(b)] if bins is None else [b for b in bins if tb.get(b) and hb.get(b)]
    nt = sum(len(tb[b]) for b in common)
    share_t = np.zeros(len(ALLCATS)); share_h = np.zeros(len(ALLCATS))
    for b in common:
        w = len(tb[b]) / max(nt, 1)
        share_t += np.array([sum(r["category"] == c for r in tb[b]) for c in ALLCATS], float) / max(nt, 1)
        share_h += w * np.array([sum(r["category"] == c for r in hb[b]) for c in ALLCATS], float) / len(hb[b])
    return share_t, share_h


def common_bins(rows_t, rows_h):
    def binof(x):
        for i in range(len(MATCH_BINS) - 1):
            if MATCH_BINS[i] <= x < MATCH_BINS[i + 1]:
                return i
        return len(MATCH_BINS) - 2
    return sorted({binof(r["sqrt_area"]) for r in rows_t} & {binof(r["sqrt_area"]) for r in rows_h})


def contrast(rows, B, rng, n_perm=10000):
    small = [r for r in rows if r["small"] == 1 and r["stratum"] in ("head", "tail")]
    t = [r for r in small if r["stratum"] == "tail"]; h = [r for r in small if r["stratum"] == "head"]
    bins = common_bins(t, h)                     # standardization domain fixed to the observed common bins
    st, sh = std_shares(t, h, bins)
    d = st - sh
    # cluster bootstrap over groups containing head- or tail-small objects
    groups = sorted({r["group"] for r in small})
    gi = defaultdict(list)
    for r in small:
        gi[r["group"]].append(r)
    boots = []
    for b in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        rr = [x for k in pick for x in gi[groups[k]]]
        tt = [x for x in rr if x["stratum"] == "tail"]; hh = [x for x in rr if x["stratum"] == "head"]
        if not tt or not hh:
            continue
        a, c = std_shares(tt, hh, bins); boots.append(a - c)
    boots = np.array(boots).reshape(-1, len(ALLCATS))
    # class-label permutation restricted to classes that HAVE small objects in the audit set;
    # exact enumeration of all splits when C(n, k) <= 20000, else 10,000 random splits
    from itertools import combinations
    from math import comb
    classes = sorted({r["cls"] for r in small})
    tail_cls = sorted({r["cls"] for r in small if r["stratum"] == "tail"})
    k = len(tail_cls)
    ge = np.zeros(len(ALLCATS)); done = 0
    if comb(len(classes), k) <= 20000:
        splits = (set(c) for c in combinations(classes, k)); exact = True
    else:
        prng = np.random.default_rng(1)
        splits = (set(prng.choice(classes, size=k, replace=False).tolist()) for _ in range(n_perm)); exact = False
    for pt in splits:
        tt = [x for x in small if x["cls"] in pt]; hh = [x for x in small if x["cls"] not in pt]
        if not tt or not hh:
            continue
        a, c = std_shares(tt, hh, common_bins(tt, hh))
        ge += (np.abs(a - c) >= np.abs(d) - 1e-12); done += 1
    p = ge / done if exact else (1 + ge) / (1 + done)
    p_holm = holm([p[1], p[2], p[3]])
    out = {"n_tail_small": len(t), "n_head_small": len(h), "n_groups": len(groups), "match_bins_sqrt_area": MATCH_BINS,
           "tail_share": dict(zip(ALLCATS, st.tolist())), "head_std_share": dict(zip(ALLCATS, sh.tolist())),
           "diff": dict(zip(ALLCATS, d.tolist())),
           "common_bins": bins, "n_boot_valid": int(len(boots)),
           "ci95": ({c: [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))] for i, c in enumerate(ALLCATS)}
                    if len(boots) >= 100 else None),
           "perm_p": dict(zip(ALLCATS, p.tolist())), "perm_draws": done, "perm_exact": exact,
           "perm_classes": classes, "perm_tail_classes": tail_cls,
           "holm_p_3cats": dict(zip(CATS, p_holm))}
    return out


def exploratory(rows, B, rng):
    tests = []
    for sz in ("tiny", "s16_32", "medium", "large"):
        sub = [r for r in rows if r["size"] == sz and r["stratum"] in ("head", "tail")]
        groups = sorted({r["group"] for r in sub}); gi = defaultdict(list)
        for r in sub:
            gi[r["group"]].append(r)
        t = [r for r in sub if r["stratum"] == "tail"]; h = [r for r in sub if r["stratum"] == "head"]
        classes = sorted({r["cls"] for r in sub}); tail_cls = {r["cls"] for r in t}
        for c in CATS:
            row = {"size": sz, "category": c, "n_tail": len(t), "n_head": len(h),
                   "tail_share": (sum(r["category"] == c for r in t) / len(t)) if t else None,
                   "head_share": (sum(r["category"] == c for r in h) / len(h)) if h else None,
                   "tail_groups": len({r["group"] for r in t}), "head_groups": len({r["group"] for r in h}),
                   "tail_classes": len(tail_cls)}
            if len(t) < 10 or row["tail_groups"] < 5 or len(h) < 10 or row["head_groups"] < 5:
                row["p"] = None; tests.append(row); continue
            ds = []
            for b in range(B // 4):
                pick = rng.integers(0, len(groups), size=len(groups))
                rr = [x for k in pick for x in gi[groups[k]]]
                tt = [x for x in rr if x["stratum"] == "tail"]; hh = [x for x in rr if x["stratum"] == "head"]
                if tt and hh:
                    ds.append(np.mean([x["category"] == c for x in tt]) - np.mean([x["category"] == c for x in hh]))
            ds = np.array(ds)
            row["diff"] = row["tail_share"] - row["head_share"]
            row["ci95"] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if len(ds) else None
            # p from class-label permutation within this size bin (same exchangeable unit as the primary test)
            prng = np.random.default_rng(2); k = len(tail_cls); ge = 0; done = 0
            for _ in range(2000):
                pt = set(prng.choice(classes, size=k, replace=False).tolist())
                tt = [x for x in sub if x["cls"] in pt]; hh = [x for x in sub if x["cls"] not in pt]
                if tt and hh:
                    dd = np.mean([x["category"] == c for x in tt]) - np.mean([x["category"] == c for x in hh])
                    ge += abs(dd) >= abs(row["diff"]) - 1e-12; done += 1
            row["p"] = float((1 + ge) / (1 + done))
            tests.append(row)
    ps = [t["p"] for t in tests if t["p"] is not None]
    adj = bh(ps) if ps else []
    it = iter(adj)
    for t in tests:
        t["p_bh"] = next(it) if t["p"] is not None else None
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="R3-ledger-yolo-T320")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rows = [r for r in load_rows(os.path.join(RESULTS, "ledger", f"{args.run_id}_objects.csv")) if r["audit"] == 1]
    rng = np.random.default_rng(0)
    out = {"run_id": args.run_id, "B": args.B, "partitions": {}}
    for st in ("tail", "mid", "head"):
        for sm, lab in ((1, "small"), (0, "notsmall")):
            sub = [r for r in rows if r["stratum"] == st and r["small"] == sm]
            out["partitions"][f"{st}|{lab}"] = partition(sub, args.B, rng, f"{st}|{lab}")
        for sz in ("tiny", "s16_32"):
            sub = [r for r in rows if r["stratum"] == st and r["size"] == sz]
            out["partitions"][f"{st}|{sz}"] = partition(sub, args.B, rng, f"{st}|{sz}")
    out["partitions"]["all|all"] = partition(rows, args.B, rng, "all|all")
    out["size_matched_frequency_contrast"] = contrast(rows, args.B, rng)
    out["exploratory_size_by_category"] = exploratory(rows, args.B, rng)
    # no-candidate sub-split and post-hoc reasons (descriptive)
    sub = Counter((r["stratum"], "small" if r["small"] else "notsmall", r["category"], r["reason"]) for r in rows)
    out["reasons"] = {"|".join(k): v for k, v in sorted(sub.items())}
    save_json(out, os.path.join(RESULTS, "stats", f"{args.run_id}_stats.json"))
    p = out["partitions"]["tail|small"]
    print(json.dumps(p, indent=1))
    print(json.dumps(out["size_matched_frequency_contrast"], indent=1))


if __name__ == "__main__":
    main()
