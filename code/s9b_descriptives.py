# -*- coding: utf-8 -*-
"""Descriptive companions to the primary ledger (no new inference):
  - per-class outcome counts for tail classes (audit set), all sizes and small only
  - recall on audit vs contaminated validation images per stratum (leakage effect), with counts
  - YOLO vs FCOS agreement on the tail-small category of each object (if both ledgers exist)
Output: results/stats/descriptives.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from collections import Counter, defaultdict

from common import RESULTS, load_json, save_json, rice_categories


def rows(run):
    p = os.path.join(RESULTS, "ledger", f"{run}_objects.csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    names = rice_categories()
    out = {}
    R = rows("R3-ledger-yolo-T320")
    per = defaultdict(Counter)
    for r in R:
        if r["audit"] != "1" or r["stratum"] != "tail":
            continue
        c = int(r["cls"]) + 1
        per[c]["n"] += 1; per[c][r["category"]] += 1
        if r["small"] == "1":
            per[c]["small_n"] += 1; per[c]["small_" + r["category"]] += 1
    out["tail_classes_audit"] = {f"{c} {names[c]}": dict(v) for c, v in sorted(per.items(), key=lambda kv: -kv[1]["small_n"])}
    led = load_json(os.path.join(RESULTS, "ledger", "R3-ledger-yolo-T320.json"))["summary"]
    leak = {}
    for st in ("head", "mid", "tail", "all"):
        a = led["audit"][f"{st}|all"]; c = led["contaminated"][f"{st}|all"]
        leak[st] = {"audit_n": a["n"], "audit_recall": a["D"] / a["n"], "contaminated_n": c["n"],
                    "contaminated_recall": c["D"] / c["n"]}
        for z in ("small",):
            a2 = led["audit"].get(f"{st}|{z}", {}); c2 = led["contaminated"].get(f"{st}|{z}", {})
            if a2.get("n") and c2.get("n"):
                leak[st][f"{z}_audit"] = f"{a2['D']}/{a2['n']}"; leak[st][f"{z}_contaminated"] = f"{c2['D']}/{c2['n']}"
    out["leakage_effect"] = leak
    F = rows("R7-ledger-fcos-T320")
    if F:
        key = lambda r: (r["image_id"], r["cls"], r["area"])
        fy = {key(r): r for r in F}
        agree = Counter()
        for r in R:
            if r["audit"] == "1" and r["stratum"] == "tail" and r["small"] == "1":
                f = fy.get(key(r))
                agree[(r["category"], f["category"] if f else "missing")] += 1
        out["yolo_vs_fcos_tail_small_category"] = {f"{a}|{b}": v for (a, b), v in sorted(agree.items())}
    save_json(out, os.path.join(RESULTS, "stats", "descriptives.json"))
    import json
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
