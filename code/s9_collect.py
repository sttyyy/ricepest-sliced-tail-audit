# -*- coding: utf-8 -*-
"""Consolidate every results/**/*.json|csv summary into figures/experiment_data.json and write the
run registry results/run_registry.json (run id -> config, output files, sha256).
Large per-object CSVs are referenced by path + sha256 instead of being inlined."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import glob
import hashlib
import json

from common import RESULTS, ROOT, save_json


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    data, registry = {}, {}
    skip_inline = ("near_dup_edges.json", "groups_tau")
    for p in sorted(glob.glob(os.path.join(RESULTS, "**", "*.*"), recursive=True)):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        if os.path.basename(p).startswith("debug") or "sanity" in os.path.basename(p):
            continue
        key = os.path.relpath(p, RESULTS).replace(os.sep, "/")
        registry[rel] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
        if p.endswith(".json") and not any(s in p for s in skip_inline):
            with open(p, encoding="utf-8") as f:
                data[key] = json.load(f)
        elif p.endswith(".csv") and "_objects" not in p:
            with open(p, encoding="utf-8") as f:
                data[key] = list(csv.DictReader(f))
        else:
            data[key] = {"file": rel, "sha256": registry[rel]["sha256"], "note": "not inlined (large)"}
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    with open(os.path.join(ROOT, "figures", "experiment_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    save_json(registry, os.path.join(RESULTS, "run_registry.json"))
    # code freeze record (stand-in until the team tags a git commit)
    files = sorted(glob.glob(os.path.join(ROOT, "code", "*.py")) + glob.glob(os.path.join(ROOT, "code", "*.sh")) +
                   glob.glob(os.path.join(ROOT, "figures", "gen_fig_*.py")) + [os.path.join(ROOT, "figures", "_figbase.py")] +
                   glob.glob(os.path.join(ROOT, "configs", "*.json")) + [os.path.join(ROOT, "run_all.sh")])
    with open(os.path.join(ROOT, "CODE_MANIFEST.sha256"), "w", encoding="utf-8", newline="\n") as f:
        for p in files:
            if os.path.exists(p):
                f.write(f"{sha(p)}  {os.path.relpath(p, ROOT).replace(os.sep, '/')}\n")
    print(f"Consolidated {len(data)} result files -> figures/experiment_data.json; manifest of {len(files)} code files")


if __name__ == "__main__":
    main()
