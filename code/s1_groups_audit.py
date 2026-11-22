# -*- coding: utf-8 -*-
"""Run 1 (MUST, CPU): acquisition/dedup grouping, train<->val leakage scan,
group-disjoint validation audit set, and the literal-truncation re-audit (negative control).

Outputs
  work/img_meta.json                 per-image frame, md5, perceptual hashes
  results/audit/near_dup_edges.json  all cross/within-split pHash pairs with Hamming <= 12
  results/audit/groups_tau{t}.json   group id per image for each tau (sensitivity)
  results/audit/audit_set.json       FROZEN audit set (tau = PRIMARY_TAU)
  results/audit/grouping_sensitivity.json
  results/audit/truncation.json      no-full-view truncation by split x stratum x size x tiling
No detector output is read here: the audit set is fixed blind to detection results.
"""
import argparse
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.fft import dct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RICE, RESULTS, WORK, frequency_strata, image_frame, gt_to_ref, load_rice,
                    size_bin, area_xyxy, tile_grid, contains, intersects, save_json, load_json)

TAUS = [0, 4, 6, 8, 10, 12]
PRIMARY_TAU = 8          # frozen before any detector output is produced
TILINGS = [(320, 0.25), (320, 0.10), (320, 0.0), (256, 0.25), (480, 0.25), (640, 0.25)]


def phash(img):
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    med = np.median(d.ravel()[1:])
    bits = (d > med).ravel()
    return np.packbits(bits)


def dhash(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.float64)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return np.packbits(bits)


def build_meta():
    p = os.path.join(WORK, "img_meta.json")
    if os.path.exists(p):
        return load_json(p)
    meta = []
    t0 = time.time()
    for split in ["train", "val"]:
        d = load_rice(split)
        cls_by_img = defaultdict(set)
        for a in d["annotations"]:
            cls_by_img[a["image_id"]].add(a["category_id"])
        for i, im in enumerate(d["images"]):
            fp = os.path.join(RICE, split, im["file_name"])
            with open(fp, "rb") as f:
                raw = f.read()
            with Image.open(fp) as I:
                I.load()
                fr = image_frame(im, I.size)
                ph = phash(I); dh = dhash(I)
            meta.append({"split": split, "image_id": im["id"], "file_name": im["file_name"],
                         "md5": hashlib.md5(raw).hexdigest(), "phash": ph.tobytes().hex(),
                         "dhash": dh.tobytes().hex(), "classes": sorted(cls_by_img[im["id"]]),
                         "frame": fr})
            if (i + 1) % 1000 == 0:
                print(f"  {split} {i+1}/{len(d['images'])}  {time.time()-t0:.0f}s", flush=True)
    save_json(meta, p)
    return meta


def hamming_matrix(H):
    """H: (n, 8) uint8. Return (n, n) uint8 Hamming distances, block-wise."""
    n = H.shape[0]
    bits = np.unpackbits(H, axis=1).astype(np.uint8)  # n x 64
    D = np.zeros((n, n), dtype=np.uint8)
    B = 256
    for s in range(0, n, B):
        blk = bits[s:s + B]
        D[s:s + B] = (blk[:, None, :] != bits[None, :, :]).sum(axis=2).astype(np.uint8)
    return D


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary_tau", type=int, default=PRIMARY_TAU)
    args = ap.parse_args()
    os.makedirs(os.path.join(RESULTS, "audit"), exist_ok=True)
    meta = build_meta()
    n = len(meta)
    print(f"images: {n}", flush=True)
    strata, train_cnt = frequency_strata()

    PH = np.stack([np.frombuffer(bytes.fromhex(m["phash"]), dtype=np.uint8) for m in meta])
    DH = np.stack([np.frombuffer(bytes.fromhex(m["dhash"]), dtype=np.uint8) for m in meta])
    print("pairwise pHash ...", flush=True)
    Dp = hamming_matrix(PH)
    Dd = hamming_matrix(DH)
    iu = np.triu_indices(n, 1)
    sel = Dp[iu] <= max(TAUS)
    I, J = iu[0][sel], iu[1][sel]
    md5 = [m["md5"] for m in meta]
    md5_groups = defaultdict(list)
    for k, h in enumerate(md5):
        md5_groups[h].append(k)
    edges = []
    for a, b in zip(I.tolist(), J.tolist()):
        shared = bool(set(meta[a]["classes"]) & set(meta[b]["classes"]))
        edges.append({"a": a, "b": b, "dp": int(Dp[a, b]), "dd": int(Dd[a, b]), "shared_class": shared,
                      "cross": meta[a]["split"] != meta[b]["split"], "md5_equal": md5[a] == md5[b]})
    for h, ks in md5_groups.items():
        for x in ks[1:]:
            if Dp[ks[0], x] > max(TAUS):
                edges.append({"a": ks[0], "b": x, "dp": int(Dp[ks[0], x]), "dd": int(Dd[ks[0], x]),
                              "shared_class": True, "cross": meta[ks[0]]["split"] != meta[x]["split"],
                              "md5_equal": True})
    save_json(edges, os.path.join(RESULTS, "audit", "near_dup_edges.json"))
    # nearest-train distance for every val image (leakage scan, histogram)
    is_tr = np.array([m["split"] == "train" for m in meta])
    val_idx = np.where(~is_tr)[0]; tr_idx = np.where(is_tr)[0]
    nn_val = Dp[np.ix_(val_idx, tr_idx)].min(axis=1)
    hist = Counter(int(v) for v in nn_val)
    exact_cross = sum(1 for h, ks in md5_groups.items()
                      if len({meta[k]["split"] for k in ks}) > 1)

    # GT objects per image in the reference frame
    gts = {}
    for split in ["train", "val"]:
        d = load_rice(split)
        by = defaultdict(list)
        for a in d["annotations"]:
            by[a["image_id"]].append(a)
        gts[split] = by
    sens = []
    groups_primary = None
    for tau in TAUS:
        uf = UF(n)
        for e in edges:
            if e["md5_equal"] or (e["dp"] <= tau and e["shared_class"]):
                uf.union(e["a"], e["b"])
        comp = [uf.find(k) for k in range(n)]
        comp_has_train = defaultdict(bool)
        for k in range(n):
            if meta[k]["split"] == "train":
                comp_has_train[comp[k]] = True
        audit, contaminated = [], []
        for k in val_idx.tolist():
            m = meta[k]
            if not m["frame"]["consistent"]:
                continue
            (contaminated if comp_has_train[comp[k]] else audit).append(k)
        # tail-small objects and their groups in the audit set
        cnt = Counter(); grp = defaultdict(set)
        for k in audit:
            m = meta[k]
            for a in gts["val"][m["image_id"]]:
                b = gt_to_ref(a["bbox"], m["frame"])
                sb = size_bin(area_xyxy(b))
                key = (strata[a["category_id"]], "small" if sb in ("tiny", "s16_32") else "notsmall")
                cnt[key] += 1; grp[key].add(comp[k])
        row = {"tau": tau, "n_components": len(set(comp)), "val_consistent": len(audit) + len(contaminated),
               "val_contaminated": len(contaminated), "audit_images": len(audit),
               "audit_groups": len({comp[k] for k in audit}),
               "counts": {f"{s}|{z}": cnt[(s, z)] for s in ["head", "mid", "tail"] for z in ["small", "notsmall"]},
               "groups": {f"{s}|{z}": len(grp[(s, z)]) for s in ["head", "mid", "tail"] for z in ["small", "notsmall"]}}
        sens.append(row)
        save_json({"tau": tau, "group": comp}, os.path.join(RESULTS, "audit", f"groups_tau{tau}.json"))
        print(f"tau={tau}: comps={row['n_components']} val_contaminated={row['val_contaminated']} "
              f"audit_imgs={row['audit_images']} tail-small={row['counts']['tail|small']} "
              f"(groups {row['groups']['tail|small']})", flush=True)
        if tau == args.primary_tau:
            groups_primary = comp
            audit_primary = audit
            contaminated_primary = contaminated
    save_json({"sensitivity": sens, "val_nearest_train_phash_hist": dict(sorted(hist.items())),
               "exact_md5_cross_split_groups": exact_cross,
               "n_edges_le12": len(edges)},
              os.path.join(RESULTS, "audit", "grouping_sensitivity.json"))
    excluded_frame = [k for k in range(n) if not meta[k]["frame"]["consistent"]]
    audit_obj = {"primary_tau": args.primary_tau,
                 "rule": "val image in audit set iff frame-consistent and its near-duplicate component "
                         "(pHash Hamming <= tau AND shared class label, or identical md5) contains no train image",
                 "audit_image_ids": [meta[k]["image_id"] for k in audit_primary],
                 "contaminated_val_image_ids": [meta[k]["image_id"] for k in contaminated_primary],
                 "group_of_val_image": {str(meta[k]["image_id"]): int(groups_primary[k]) for k in val_idx.tolist()},
                 "frame_inconsistent": {"train": [meta[k]["image_id"] for k in excluded_frame if meta[k]["split"] == "train"],
                                        "val": [meta[k]["image_id"] for k in excluded_frame if meta[k]["split"] == "val"]},
                 "strata": {str(k): v for k, v in strata.items()}, "train_counts": {str(k): v for k, v in train_cnt.items()}}
    save_json(audit_obj, os.path.join(RESULTS, "audit", "audit_set.json"))

    # ---------------- truncation re-audit (negative + positive controls) ----------------
    audit_set = set(audit_primary)
    out = []
    for (T, ov) in TILINGS:
        tab = defaultdict(lambda: [0, 0, 0])   # key -> [n, truncated(no full view), boundary-cut]
        for k, m in enumerate(meta):
            fr = m["frame"]
            if not fr["consistent"]:
                continue
            tiles = tile_grid(fr["ref_w"], fr["ref_h"], T, ov)
            sets = [m["split"]] + (["audit"] if k in audit_set else [])
            for a in gts[m["split"]][m["image_id"]]:
                b = gt_to_ref(a["bbox"], fr)
                if area_xyxy(b) <= 0:
                    continue
                full = any(contains(t, b) for t in tiles)
                cut = any(intersects(t, b) and not contains(t, b) for t in tiles)
                sb = size_bin(area_xyxy(b))
                pooled = "small" if sb in ("tiny", "s16_32") else "notsmall"
                for s in sets:
                    for key in [(s, strata[a["category_id"]], sb), (s, strata[a["category_id"]], pooled),
                                (s, "all", pooled), (s, "all", "all")]:
                        v = tab[key]; v[0] += 1; v[1] += int(not full); v[2] += int(cut)
        for key, (nn, tr, cut) in sorted(tab.items()):
            out.append({"T": T, "overlap": ov, "stride": int(T * (1 - ov)), "set": key[0], "stratum": key[1],
                        "size": key[2], "n": nn, "truncated": tr, "boundary_cut": cut})
        sm = tab[("audit", "tail", "small")]
        print(f"tiling T={T} ov={ov}: audit tail-small n={sm[0]} truncated={sm[1]} cut={sm[2]}", flush=True)
    save_json(out, os.path.join(RESULTS, "audit", "truncation.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
