# Slice boundaries are a secondary contributor to small-object misses in tail classes

Supporting code and frozen results for a **preregistered, group-disjoint per-object recovery-ceiling audit of sliced (SAHI-style) inference** on long-tailed small-object detection, with RicePest-30 as the long-tail experiment and a native-resolution replication on the Yellow Sticky Traps (YST) dataset.

> *Slice boundaries are a secondary contributor to small-object misses in tail classes: a preregistered recovery-ceiling audit of sliced inference for rice pest detection*

The analysis answers one question per annotated object: **when a sliced detector misses a small object in a tail class, is it because the tile boundary cut the object, or because the detector never produced a usable candidate?** Every number below and every figure in `figures/` is read from a file in `results/`; nothing is typed in by hand.

---

## 1. What the pipeline measures

Each annotated object is pushed through five stages, in this order:

| Stage | Meaning |
|---|---|
| **V** | some tile view fully contains the object (if none does, it is *literal truncation* — used as a negative control) |
| **C** | at the low score floor, a same-class candidate (IoU ≥ 0.5) exists in some view |
| **M** | the object is matched one-to-one in the merged candidate list |
| **S** | the matched merged box scores ≥ the frozen threshold (0.25) |
| **D** | the object is matched in the emitted list (threshold + ≤100 boxes per image) — i.e. *detected* |

Misses are then attributed to one of three recovery classes (priority M > C):

- `no_candidate` — no candidate at all; only the detector/training can reach these;
- `pre_suppression_only` — a candidate existed but never entered the merged list (read-out / suppression);
- `post_hoc_recoverable` — entered the merged list but was not emitted; rescoring or calibration can reach these.

An **oracle full-view read-out** (baseline detections plus each same-class candidate scoring ≥ 0.25 in a view that holds the whole object) gives the recovery ceiling at fixed detector evidence.

Detectors are frozen (YOLO11n and torchvision FCOS-ResNet50-FPN, COCO-initialised then fine-tuned; no model selection on the validation data). The audit set is fixed *before* any detector output is inspected: near-duplicate images are grouped by pHash/MD5 (`τ = 8`), and only validation images whose group contains no training image are audited.

## 2. Headline results

These are quoted from `experiment_results.md`, which cites the `results/` file behind each number.

- **Audit set**: 523 images in 508 groups, 4,429 objects. The primary endpoint, pooled tail-class small objects (<32² px in the 640 px reference frame), has 38 objects in 27 groups.
- **Tail-small misses are mostly detector-side.** Tail-small recall at the operating point is 28.9% [95% CI 12.5–48.9]; of the 27 misses, 16 (59%) have **no same-class candidate at all** in any view at the 0.01 floor. Head-small misses are the opposite: 88% have a candidate and are lost in the merge or below the score threshold.
- **Size alone does not explain it.** Tail-small versus head-small re-weighted to the tail size mix: the no-candidate share is +33.3 pp higher for tail (Holm-adjusted p = 0.013) and the pre-suppression-only share is −11.9 pp lower (p = 0.026). Literal truncation is 0% in both strata.
- **A better read-out has little headroom for this endpoint.** The oracle full-view read-out adds one tail-small object at fixed scores; perfect rescoring of full-view candidates could reach 57.9%, and the remaining 42.1% (16 objects) have no candidate.
- **Published splits overstate tail recall.** 285 of the 933 published validation images have a training image at pHash distance 0; on contaminated images the same frozen pipeline reaches 81.7% tail recall against 53.6% on the audit set.

## 3. Repository layout

```
.
├── README.md                  this file
├── run_all.sh                 full pipeline, in paper order
├── REPRODUCE.md               environment, data prep, per-figure command → data map
├── DATA_SOURCES.md            datasets, pretrained weights, software: source + licence
├── experiment_results.md      every reported number, with the results/ file behind it
├── CODE_MANIFEST.sha256       sha256 of every file that produced the paper numbers
├── requirements.txt           CPU/analysis dependencies
├── requirements-gpu.txt       GPU stack (installed into .pydeps_gpu/, see §4)
├── LICENSE                    AGPL-3.0
├── code/                      the pipeline scripts (see note below)
├── configs/prereg_frozen.json preregistered frozen quantities (written before detector output)
├── results/                   frozen run outputs: audit set, per-object ledgers, stats, methods, YST
├── figures/                   figure PDFs, LaTeX tables, experiment_data.json, gen_fig_*.py
├── paper/                     LaTeX source and compiled PDF of the manuscript
└── _utils/plot_utils.py       shared plotting/style helper used by figures/gen_fig_*.py
```

**Note on the code directory name.** The pipeline scripts live in `code/` rather than `src/`. Every command in `REPRODUCE.md`, `run_all.sh` and the frozen `CODE_MANIFEST.sha256` refers to `code/`, and the pipeline resolves the repository root as the *parent* of its own directory (`ROOT` in `code/common.py`), so `code/` is load-bearing: renaming it would invalidate the frozen hash manifest and every documented command. Files, by role:

| File | Role |
|---|---|
| `code/common.py` | frozen shared definitions: 640 px reference frame, size bins, frequency strata, frame-consistency rule, SAHI tile geometry |
| `code/ledger_lib.py` | ledger core: IoU/containment, SAHI merge replay, one-to-one matching, emit, data loading |
| `code/gpu_env.py` | puts `.pydeps_gpu/` first on `sys.path` and pins Ultralytics config to `work/` |
| `code/s1*`, `s2*` | grouping + leakage audit + audit set + literal truncation; dataset view |
| `code/s3*`, `s4*` | the two frozen detectors and the candidate dumps |
| `code/s5*`, `s7*` | per-object stage ledger, recovery classes, AP; confirmatory statistics |
| `code/s6*` | post-hoc baselines and CCPR (category-conditional peak read-out) |
| `code/s8*` | YST preparation, native-resolution truncation |
| `code/s9*` | collection, LaTeX tables, descriptives, figure includes |
| `code/test_sahi_equivalence.py` | self-check: merge replay against SAHI's own postprocessors |

A detailed, file-by-file guide with every CLI flag is in [`code/README.md`](code/README.md) (Chinese).

## 4. Environment

The paper's runs used Windows 11, Python 3.11.9 and one RTX 5060 Laptop (8 GB). CPU packages go into the interpreter; the CUDA build of torch is kept in a separate `.pydeps_gpu/` directory so the base interpreter is never modified.

```bash
pip install -r requirements.txt

# CUDA 12.8 wheels, installed into a private directory:
python -m pip install --no-deps --target .pydeps_gpu \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1+cu128-cp311-cp311-win_amd64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.24.1+cu128-cp311-cp311-win_amd64.whl
python -m pip install --no-deps --target .pydeps_gpu -r requirements-gpu.txt
```

For another CUDA version, take the matching wheel from the official PyTorch index instead; `code/gpu_env.py` discovers `.pydeps_gpu/` automatically. The pinned versions are the ones recorded in `DATA_SOURCES.md`; only the pinned versions were used for the paper.

## 5. Data

Neither dataset is redistributed here (size and licence terms); `results/` contains every derived artefact needed to check the numbers.

| Dataset | Place at | Licence | Notes |
|---|---|---|---|
| RicePest-30 | `user_data/RicePest-30-Dataset/{train,val}` | CC BY-NC 4.0 | COCO JSON + JPEG; address in `DATA_SOURCES.md` |
| Yellow Sticky Traps | `user_data/Yellow-Sticky-Traps-Dataset/{images,annotations}` | CC0 1.0 | 284 images, 5184×3456, Pascal VOC XML |

Initial weights go to `work/weights/`: Ultralytics `yolo11n.pt` and torchvision `fcos_resnet50_fpn_coco-99b0c9b7.pth` (sources and sha256 in `DATA_SOURCES.md`).

## 6. Running

`results/` is committed, so every figure and table can be regenerated without re-running detection:

```bash
python figures/gen_fig_size_frequency.py     # any figures/gen_fig_*.py
python code/s9_tables.py                     # LaTeX tables
```

The complete pipeline (training included, ~2.5 h of GPU time on the reference machine) runs with:

```bash
bash run_all.sh                              # see REPRODUCE.md for the per-step commands and timings
```

Small intermediate data (`work/`, `.pydeps_gpu/`) is git-ignored. `CODE_MANIFEST.sha256` pins every file that produced the paper's numbers — verify a checkout with any `sha256sum -c`-compatible check, e.g.:

```bash
python -c "import hashlib,pathlib,sys; \
bad=[l for l in open('CODE_MANIFEST.sha256') if l.strip() and \
     hashlib.sha256(pathlib.Path(l.split('  ')[1].strip()).read_bytes()).hexdigest()!=l.split()[0]]; \
print('mismatches:', len(bad))"
```

## 7. Reproducibility status and limits

- **Frozen version:** tag [`v1.0.0`](../../releases/tag/v1.0.0) (commit `94a7abb`) is the state that produced every number in `experiment_results.md` and every figure; `CODE_MANIFEST.sha256` pins the individual files.
- The committed `results/` are the frozen outputs behind the paper. A later from-scratch re-run from the frozen weights reproduced the candidate dumps byte-identically and every regenerated result summary exactly.
- The ledger, statistics, methods and table/figure generation are **deterministic CPU code**: given the same frozen weights and data, they reproduce exactly.
- GPU training is **not** bit-exact across GPUs or driver versions even with `deterministic=True`; frozen-weight sha256 values are recorded in `results/runs/*_train.json`.
- The pHash grouping is computed from the JPEG bytes as published; a re-encoded copy of the dataset would change the hashes and possibly the audit set.
- The pooled tail-small endpoint (38 objects / 27 groups) passes the preregistered minimum-cell rule but only just, and is dominated by one leafhopper species (*Jacobiasca formosana*, 25 of 38). It is reported as a limit rather than hidden by pooling.
- YST has only 3 classes and an 8.4× imbalance, so it is used **only** for a native-resolution source/resolution replication, never for the long-tail conclusion.

## 8. Licence

Code in this repository is released under **AGPL-3.0** (see `LICENSE`), because it imports Ultralytics (AGPL-3.0). Other dependencies keep their own licences (torchvision BSD-3-Clause, SAHI MIT, PyMuPDF AGPL, …); datasets are not covered by this licence — see `DATA_SOURCES.md` for the item-by-item terms.

## 9. Citation

```bibtex
@article{ricepest_sliced_tail_audit,
  title  = {Slice boundaries are a secondary contributor to small-object misses in tail
            classes: a preregistered recovery-ceiling audit of sliced inference for rice
            pest detection},
  author = {{Author list to be completed}},
  year   = {2026},
  note   = {Code: https://github.com/sttyyy/ricepest-sliced-tail-audit}
}
```

> **Note.** The author list, affiliation and venue are still placeholders in the manuscript source (`paper/main.tex`); they must be filled in before this entry is used.
