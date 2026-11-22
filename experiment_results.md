# Experiment Results

**Date**: 2026-11-21 · **Step**: experiment-bridge (6/10)
**Plan**: `refine-logs/EXPERIMENT_PLAN.md` · **Frozen config**: `configs/prereg_frozen.json` (written before any detector output existed)
**Hardware**: one RTX 5060 Laptop GPU (8 GB). GPU time actually used: about 2.5 h (YOLO11n 65 min, FCOS ~37 min, YST detector ~20 min, dumps ~10 min), against the 8–11 GPU-days estimated in the plan; the scope reductions this forced are listed in §9.

Every number below is read from a file in `results/` (paths given per section) and is consolidated in `figures/experiment_data.json`. Run IDs follow the plan's run numbers (R1 … R9). "Audit set" always means the group-disjoint validation audit set of §1.

## 1. Run 1 — grouping, leakage and the audit set (MUST, CPU) · `results/audit/`

- **Frame-inconsistent images.** 343 RicePest-30 images (300 train, 43 val) are single-pest close-ups whose JSON frame (e.g. 4080×3060) has a different aspect ratio from the JPEG (e.g. 117×180), so their boxes cannot be mapped. They are excluded from training and evaluation (`audit_set.json["frame_inconsistent"]`). Every other image is published at ≤640 px long edge while boxes are stored in the original frame; boxes are rescaled per axis.
- **Leakage in the published split.** 285 of the 933 validation images have a training image at pHash distance 0, and 279 byte-identical (MD5) groups span train and val (`grouping_sensitivity.json`).
- **Frozen grouping** (τ = 8, pHash Hamming ≤ 8 with a shared class, or identical MD5; connected components): 367 of 890 frame-consistent validation images fall in a component that contains a training image. The **audit set keeps 523 images in 508 groups with 4,429 objects**. The pooled tail-small endpoint has **38 objects in 27 groups** (head-small 1,686; mid-small 114), which passes the pre-registered minimum-cell rule (≥10 events, ≥5 groups) but only just.
- Sensitivity of the audit set to τ (τ = 0/4/6/8/10/12): 641/616/575/523/446/335 images; tail-small 43/43/42/38/28/22 objects (Table `TABLE_leakage.tex`).
- **Leakage effect.** On contaminated validation images the same frozen pipeline reaches recall 81.7% for tail classes (93 objects) against 53.6% on audit images (248 objects); head 85.7% vs 82.4%; all objects 84.1% vs 79.3% (`results/stats/descriptives.json`). Evaluating on the published validation split would overstate tail recall by about 28 points.
- **Composition of the endpoint.** The 38 tail-small audit objects come from five classes: *Jacobiasca formosana* 25, *Scirpophaga incertulas* 9, *Spodoptera exigua* 2, *S. litura* 1, *Agrotis ipsilon* 1. The pooled endpoint is therefore dominated by one leafhopper species; this is reported as a limit, not hidden by pooling.

## 2. Run 1 / E2 — literal truncation (negative control) · `results/audit/truncation.json`, `results/yst/truncation.json`

| Tiling (640 frame) | Train small | Val small | Audit small | Audit tail-small | Seam-cut (audit small) |
|---|---|---|---|---|---|
| T=320, stride 240 (frozen) | 0/21,048 | 0/2,680 | 0/1,838 | 0/38 | 26.3% |
| T=320, overlap 0.10 | 67/21,048 | 10/2,680 | 6/1,838 | 0/38 | 26.2% |
| T=320, overlap 0 (positive control) | 1,614/21,048 | 214/2,680 | 138/1,838 | 3/38 | 19.3% |
| T=256 / T=480, overlap 0.25 | 0 | 0 | 0 | 0 | 41.0% / 12.5% |

- Validity condition 1 holds: truncation is negligible on the **validation/audit** split, not only on train. A quarter of small objects are crossed by a tile border, but every one of them has a full view in a neighbouring tile.
- The positive control works: removing the overlap creates truncation (7.5% of audit small objects), so the measurement can detect it.
- YST (native 5184×3456, T=640, stride 480): 31/8,114 objects truncated (WF 0, MR 23, NC 8), reproducing the pilot; 1,639/8,114 at overlap 0.
- The pilot's train count (1 of 21,017) differs slightly from this run (0 of 21,048) because this run uses SAHI's exact tile geometry in the JPEG-derived 640 frame and drops the 300 frame-inconsistent images.

## 3. Run 2 — frozen detectors and candidate dumps · `results/runs/`, `results/audit/sahi_equivalence_check.json`

| Run | Detector | Training | Frozen weights (sha256 prefix) |
|---|---|---|---|
| R2-yolo11n_rice_s0 | YOLO11n, COCO init | 60 epochs, 640, batch 16, seed 0, `val=False`, last checkpoint, 3,874 s | `bcde46c946b0` |
| R6-fcos_rice_s0 | FCOS-ResNet50-FPN, COCO init | 6 epochs, 640, batch 4, SGD 0.01 cosine, AMP, last checkpoint | see `results/runs/fcos_rice_s0_train.json` |

- Ultralytics' end-of-training validation was pointed at 200 training images (`trainstub`), so no validation image touched training or checkpoint choice.
- Primary dump `yolo_T320`: 890 validation images, full-image view + SAHI tiles (T=320, overlap 0.25), 113,489 standard candidates and 142,855 multi-label candidates at the 0.01 floor; 43,603 boxes in the merged list, 11,310 emitted.
- Gate G3 (dump sanity): candidate recall at the floor (94.8% of audit objects) ≥ merged recall at the floor (89.1%) ≥ emitted recall (79.3%). Passed.
- Merge replay: on 80 real candidate lists, `ledger_lib.readout_sahi` reproduces SAHI 0.14.0's own `GreedyNMMPostprocess` and `NMSPostprocess` exactly (0 count, class or score mismatches; max coordinate difference 0.0).
- Context (AP@0.5, 101-point, audit set, `results/ledger/AP-*.json`): sliced SAHI mAP50 0.570 (head 0.703, mid 0.511, tail 0.496); full image only 0.607 (head 0.743, mid 0.535, tail 0.543); sliced with NMS merge 0.549. On these downscaled ≤640-px images slicing raises small-object recall at the operating point (below) but does not raise AP.

## 4. Run 3 — primary recovery-ceiling partition (confirmatory) · `results/ledger/R3-ledger-yolo-T320*`, `results/stats/R3-ledger-yolo-T320_stats.json`

Operating point: score ≥ 0.25, ≤100 boxes per image, IoU 0.5 one-to-one matching. Miss categories with priority M > C (a merged union box can match although no single candidate does).

| Stratum, size | N (groups) | Recall [95% CI] | Misses | No candidate | Pre-suppression only | Post-hoc recoverable |
|---|---|---|---|---|---|---|
| **Tail <32² (primary)** | 38 (27) | 28.9% [12.5, 48.9] | 27 (19 groups) | **16 (59% [41, 78])** | 2 (7% [0, 18]) | 9 (33% [17, 55]) |
| Head <32² | 1,686 (257) | 76.1% [73.5, 78.8] | 403 | 47 (12% [8, 16]) | 159 (39% [32, 46]) | 197 (49% [42, 56]) |
| Mid <32² | 114 (29) | 77.2% [63.5, 85.4] | 26 | 7 (27%) | 5 (19%) | 14 (54%) |
| Tail ≥32² | 210 (156) | 58.1% | 88 | 40 (45%) | 5 (6%) | 43 (49%) |
| Head ≥32² | 1,827 (332) | 88.2% | 215 | 50 (23%) | 65 (30%) | 100 (47%) |
| All objects | 4,429 (508) | 79.3% [77.4, 80.9] | 919 | 227 (25%) | 255 (28%) | 437 (48%) |

- Gate G1 (minimum cell) passed: 38 events in 27 groups. The tail-small misses sit in 19 groups (10–19), so a leave-one-group-out range is reported next to the bootstrap CI, as pre-registered (`jackknife_range_prop_of_misses` in the stats file).
- **Dominant category for tail-small = no candidate.** 16 of the 27 tail-small misses have no same-class candidate with IoU ≥ 0.5 in any view at the 0.01 floor. Only the detector/training can reach them. Head-small misses are the opposite: 88% have a candidate and are lost in the merge (39%) or below the score threshold (49%).
- Sub-split of the 16 tail-small no-candidate misses (descriptive): 7 have no box at all, 5 have a box of the wrong class, 4 have same-class evidence that only the multi-label read-out of the same forward pass exposes.
- Within tail-small, the post-hoc-recoverable misses (9) are all threshold losses; none is a cap loss.

## 5. K3 — size-matched frequency contrast (confirmatory) · same stats file

Tail-small vs head-small re-weighted to the tail size mix (sqrt-area bins 12–16–24–32 px; bin <12 px dropped because head has no object there, leaving 34 tail-small objects). p from exact enumeration of all 462 splits of the 11 head/tail classes that have small objects (5 relabelled tail), Holm over the three miss categories.

| Outcome | Tail-small share | Size-matched head share | Difference [95% CI] | Holm p |
|---|---|---|---|---|
| Detected | 32.4% | 63.4% | −31.0 pp [−50.8, −12.0] | (perm p 0.004) |
| No candidate | 38.2% | 4.9% | **+33.3 pp [16.7, 50.0]** | **0.013** |
| Pre-suppression only | 2.9% | 14.8% | **−11.9 pp [−18.0, −5.2]** | **0.026** |
| Post-hoc recoverable | 26.5% | 16.8% | +9.6 pp [−3.7, 25.1] | 0.149 |

- K2/K3 allowed claim: the tail-small loss profile differs from the head-small profile at matched size; tail-small objects are lost before any candidate exists, head-small objects after. Size alone does not explain it, and the truncation share is 0 in both strata.
- Exploratory (BH-adjusted, within-bin class permutation): the no-candidate excess of tail classes also appears for 32²–96² objects (+23.9 pp, p_BH 0.004), and post-hoc-recoverable shares are higher for tail in the 16²–32² (+18.4 pp, p_BH 0.004) and 32²–96² bins (+25.5 pp, p_BH 0.004). The <16² bin holds a single tail class and gives no test.

## 6. E3 — oracle full-view bound · ledger summary

| Subset | Emitted (D) | Oracle read-out at τ=0.25 | Oracle with rescoring (floor) | Same-class candidate (C) |
|---|---|---|---|---|
| Tail small | 28.9% | 31.6% | 57.9% | 57.9% |
| Head small | 76.1% | 84.4% | 97.1% | 97.2% |
| All objects | 79.3% | 84.4% | 94.5% | 94.8% |

The oracle read-out keeps every baseline detection and adds each same-class candidate that scores ≥ 0.25 in a view holding the whole object. For tail-small it adds one object: a better read-out at fixed scores has almost no headroom. Perfect rescoring of full-view candidates could reach 57.9%; the remaining 42.1% (16 objects) have no candidate. For head-small, a better read-out alone is worth up to 8.3 points.

## 7. Runs 5 and 9 — same-dump baselines and CCPR (exploratory) · `results/methods/R5R9-methods-yolo.json`

All methods re-read the identical candidate dump. Parameters (NorCal γ, Na α, class×size thresholds) are fit by 5-fold group cross-fitting on the audit set, with folds stratified by tail-small count (10/9/7/7/5 tail-small objects per fold). "Matched FP" = one global threshold per method so that its false positives do not exceed the baseline's 3,671.

| Method | R all (own) | Tail-small (own) | P head (own) | R all (matched FP) | R head-small (matched FP) | Tail-small (matched FP) | Recovered P/H/N | Lost |
|---|---|---|---|---|---|---|---|---|
| SAHI baseline | 79.3 | 11/38 | 54.6 | 79.3 | 76.1 | 11/38 | – | – |
| NorCal-style prior | 79.1 | 11/38 | 55.1 | 79.1 | 75.9 | 11/38 | 0/2/0 | 9 |
| Class×size thresholds | 69.1 | 11/38 | 68.9 | 78.5 | 75.7 | 14/38 | 0/31/0 | 64 |
| Na-style rescoring, agnostic | 77.5 | 10/38 | 61.3 | 80.3 | 77.9 | 11/38 | 0/103/0 | 57 |
| Na-style rescoring, per class | 74.9 | 9/38 | 65.2 | 80.2 | 78.4 | 11/38 | 0/143/0 | 102 |
| Multi-label evidence + SAHI merge + thr. | 69.2 | 12/38 | 68.9 | 78.6 | 75.2 | 14/38 | 2/36/20 | 85 |
| Peak read-out on std evidence + thr. | 72.3 | 10/38 | 67.1 | 80.6 | 80.1 | 14/38 | 201/13/0 | 153 |
| CCPR without thresholds | 83.4 | 11/38 | 51.4 | 81.0 | 81.1 | 8/38 | 207/3/1 | 132 |
| CCPR (full) | 72.6 | 11/38 | 67.1 | 80.7 | 79.6 | 15/38 | 200/20/16 | 170 |

(R, P in %. Recovered P/H/N = baseline misses recovered at matched FP by the baseline's category: pre-suppression only / post-hoc recoverable / no candidate.)

- **K6 (attribution).** Every rescoring or calibration method recovers only post-hoc-recoverable misses, as the partition predicts (e.g. Na-style per-class: 143 recovered, all post-hoc). Methods that change the read-out (peak read-out, CCPR) recover pre-suppression-only misses (≈200, almost all head-small objects that SAHI's GREEDYNMM union merge had absorbed).
- **K7 (CCPR go/no-go) — NO-GO, negative increment.** Against class×size calibration: own operating points 0 extra tail-small TPs (Δ 0.0 pp [−11.1, 9.1]); matched head precision −1 (−2.6 pp [−13.8, 4.8], head precision matched within 0.01); matched FP +1 (+2.6 pp [−6.1, 10.8]). Head non-inferiority passes in all three (margin 1 pp), but criteria 1 and 3 fail. As pre-registered, CCPR is reported as a negative increment: class×size calibration absorbs the tail-small gain.
- Read-out matters for head-small: at matched FP the peak read-out raises head-small recall from 76.1% to 80.1% and all-object recall from 79.3% to 80.6%, with the same gain whether it reads standard or multi-label evidence, so the gain comes from replacing the union merge, not from new evidence.
- Exploratory, not pre-registered: multi-label evidence raises tail recall over all sizes at matched FP (60.5% for CCPR and for multi-label + SAHI merge vs 53.6% baseline; `paired_vs_baseline` in the methods file holds the paired CIs).

## 8. E8 — sensitivity of the ledger · `results/ledger/S-ledger-*.json`

| Configuration | Head-small recall | Tail-small detected | Tail-small misses N/P/H |
|---|---|---|---|
| Frozen (GREEDYNMM-IOS at floor, τ 0.25, cap 100) | 76.1% | 11/38 | 16/2/9 |
| NMS-IoU 0.5 merge | 83.3% | 10/38 | 16/2/10 |
| Class-agnostic GREEDYNMM | 71.8% | 10/38 | 16/6/6 |
| Threshold before merge (SAHI deployment order) | 82.1% | 11/38 | 16/11/0 |
| Full image only (no slicing) | 70.2% | 5/38 | 20/1/12 |
| T=256 / T=480 | 72.6% / 75.2% | 11/38 / 12/38 | 13/2/12 / 15/5/6 |
| Overlap 0 (positive control) | 76.6% | 11/38 | 18/1/8 |
| τ = 0.10 / 0.40 | 82.0% / 68.7% | 16/38 / 6/38 | 16/2/4 / 16/2/14 |
| Global cap 10 / 20 / 50 / 300 | 36.5 / 53.9 / 71.9 / 76.2% | 9 / 10 / 11 / 11 of 38 | – |
| Per-class cap 5 / 10 / 20 | 39.0 / 53.3 / 66.5% | 11/38 each | – |

- The tail-small no-candidate count stays at 13–20 of 38 under every read-out, tiling, threshold and cap. The dominant-category finding does not depend on the frozen configuration.
- The head-small profile does depend on the read-out: SAHI's GREEDYNMM union merge applied to the floor-level list (the configuration used for AP evaluation) creates most head-small pre-suppression losses (159); NMS removes 142 of them, and thresholding before the merge also avoids them.
- The output cap is not a tail-small bottleneck at cap ≥ 50, and per-class caps do not affect tail-small at all; caps ≤ 50 cut head-small recall on dense trap images (Dave et al.'s cap effect appears for head classes here).
- Slicing matters for tail-small detection at the operating point: full-image inference finds 5/38 against 11/38 with slicing.

## 9. Run 7 — second detector (FCOS-ResNet50-FPN) · `results/ledger/R7-*`, `S-ledger-fcos-*`, `results/stats/R7-*`, `results/methods/R5R9-methods-fcos*.json`

| FCOS read-out | R all | AP50 | Head-small R | Tail-small det. | Tail-small misses N/P/H | Head-small misses N/P/H |
|---|---|---|---|---|---|---|
| Frozen (GREEDYNMM-IOS at floor) | 48.9% | 0.152 | 52.7% | 4/38 | 9/18/7 | 92/610/95 |
| NMS-IoU 0.5 | 89.9% | 0.512 | 86.1% | 15/38 | 10/4/9 | 92/16/126 |
| Threshold before merge | 85.7% | – | 80.1% | 19/38 | 10/7/2 | 91/163/82 |

- Under the frozen pipeline FCOS collapses. It writes the full 300 candidates per view at the 0.01 floor (1.85 M candidates on 890 images, against 113 k for YOLO11n), and SAHI's union merge fuses them into oversized boxes, so the merge stage dominates for every stratum. This is a property of the frozen read-out interacting with a detector that emits dense low-score boxes, not of the tail.
- **Detector-side loss is detector-dependent.** Tail-small objects without any same-class candidate: 9–10 of 38 for FCOS under all three read-outs, against 16 of 38 for YOLO11n. Per object, 8 of YOLO11n's 16 no-candidate tail-small objects are also no-candidate for FCOS (`results/stats/descriptives.json`).
- **The direction of the size-matched contrast replicates; its size does not.** Tail-small minus size-matched head-small no-candidate share: FCOS frozen +11.1 pp [−3.8, 26.7], exact class-permutation Holm p = 0.006; FCOS-NMS +14.1 pp [−1.3, 30.2], Holm p = 0.006; YOLO11n frozen +33.3 pp [16.7, 50.0], Holm p = 0.013; YOLO11n-NMS +33.3 pp [16.7, 50.0], Holm p = 0.013. For FCOS the group bootstrap CI includes 0 while the class permutation does not.
- CCPR on FCOS: against class×size calibration computed on the collapsed SAHI-merged list, CCPR passes the go criteria at matched FP (+19 tail-small TPs) only because the comparator inherits the merge collapse (the head-precision match also failed there). Against an NMS-based comparator CCPR is NO-GO (extra tail-small TPs +2/+1/+3; head NI fails at two of the three operating points). On YOLO11n with an NMS-based comparator CCPR is also NO-GO (+2/+1/+1). Exploratory; it does not change the pre-registered NO-GO on the primary detector.

## 10. Run 8 — YST native-resolution source re-check (not a long-tail test) · `results/ledger/R8-*`, `S-ledger-yst-*`, `results/yst/`

85 held-out test images (pHash-component split, 199/85), 2,723 objects; YOLO11n trained on 6,398 native 640-px tiles for 20 epochs (1,192 s); SAHI T=640, overlap 0.25 (77 tiles + full view per image).

| Class | N | No full view | Recall, frozen | Misses N/P/H, frozen | Recall, NMS + cap 300 | Misses N/P/H, NMS + cap 300 |
|---|---|---|---|---|---|---|
| WF (*T. vaporariorum*) | 1,882 | 0 | 38.8% | 15/955/182 | 92.0% | 15/56/80 |
| MR (*M. pygmaeus*) | 584 | 8 | 66.1% | 7/155/36 | 89.2% | 8/12/43 |
| NC (*N. tenuis*) | 257 | 1 | 81.7% | 2/45/0 | 96.9% | 3/4/1 |
| All | 2,723 | 9 | 48.7% | 24/1,155/218 | 91.8% | 26/72/124 |

- On native high-resolution sticky traps, candidate formation is almost never the problem: 24–26 of 2,723 objects (≤1%) have no candidate under any read-out, and 9 of 2,723 test objects have no full tile view (source-specific truncation exists, as in the pilot, but is negligible).
- Losses there sit in post-processing: the floor-level union merge on densely packed whiteflies (955 WF pre-suppression misses) and the 100-box cap (dense traps hold more than 100 insects). NMS with a 300-box cap raises recall from 48.7% to 91.8%.
- K8 allowed claim: the stage profile is source-dependent. On native-resolution traps the loss is in the read-out and cap; on downscaled RicePest-30 the tail-small loss is detector-side. No long-tail claim is drawn from YST.

## 11. Summary against the claims (`refine-logs/FINAL_PROPOSAL.md` §7)

| Claim | Status | Evidence |
|---|---|---|
| K1 [N] literal truncation negligible on the val/audit split | **Supported** | 0/2,680 val small, 0/38 audit tail-small; positive control 7.5% at overlap 0 |
| K2 [C] tail-small miss profile | **Supported**: no candidate dominates (16/27, 59% [41, 78]) | Table `TABLE_partition.tex`, Fig. `fig_partition.pdf` |
| K3 [C] not attributable to size alone | **Supported for YOLO11n** (no-candidate +33.3 pp, Holm p 0.013; pre-suppression −11.9 pp, Holm p 0.026); **direction only for FCOS** (+11 to +14 pp, permutation p 0.006, bootstrap CI includes 0) | `fig_contrast.pdf`, §9 |
| K4 [D] dominant stage | Candidate formation for tail-small with YOLO11n; merge/threshold for head-small; detector- and source-dependent (FCOS, YST) | §4, §8–10 |
| K5 [D] oracle bound | Read-out headroom for tail-small is 1 object; rescoring headroom 11 objects; 16 have no candidate | §6 |
| K6 [E] Na-style baselines reach only post-hoc-recoverable misses | **Supported** | §7 attribution |
| K7 [E] CCPR increment | **Negative increment (NO-GO)** on the primary detector, confirmed with an NMS comparator on both detectors | `TABLE_ccpr.tex`, §7, §9 |
| K8 [D] YST re-check | Source-specific: native traps lose objects in the read-out and cap, not in candidate formation | §10 |

Main result in one sentence: under standard overlapping tiling no small object of the audit set lacks a full tile view; tail-small rice pests are lost mostly before any candidate exists (16 of 27 misses with YOLO11n), whereas head-small pests of the same size are lost in the merge and at the score threshold, and same-dump rescoring or a class-conditional read-out cannot recover the tail-small loss.

## 12. Deviations from `EXPERIMENT_PLAN.md` (all decided before the final runs unless stated)

1. **Compute.** About 2.5 GPU-hours instead of 8–11 GPU-days: YOLO11n 60 epochs, FCOS 6 epochs, YST detector 20 epochs; no repeated training seeds. Seed-to-seed variance of the partition is therefore not measured.
2. **Frame-inconsistent images** (343) excluded; not foreseen in the plan.
3. **Grouping threshold τ = 8** chosen by the AI (author confirmation open, OPEN_QUESTIONS 3); τ sensitivity 0–12 reported.
4. **Category operationalisation.** The read-out is applied to the floor-level dump, and the score threshold and cap form the operating point (`configs/prereg_frozen.json`). After the external code review the miss categories are assigned with priority M > C. That review came after the code had been debugged on an epoch-16 checkpoint, before the final model existed.
5. **Permutation unit.** Class-label permutation (exact enumeration over the 11 head/tail classes with small objects) instead of group-level permutation: a frequency label is a class property, so group-level relabelling is undefined. The cluster bootstrap over groups is reported beside it.
6. **Oracle definition** made explicit (baseline detections plus full-view candidates ≥ τ; a floor variant for the rescoring bound).
7. **Tiling sweep.** T=640 equals the whole image in the 640 frame; replaced by T=256 and an overlap-0 positive control. T=480 kept.
8. **Post-hoc parameters** fit by 5-fold group cross-fitting on the audit set, with folds stratified by tail-small count (not specified in the plan).
9. **Post-freeze additions (exploratory, labelled as such):** threshold-before-merge read-out, NMS-based comparator runs for the methods, YST NMS/cap variants, AP50 context metric, per-object YOLO/FCOS agreement.

## 13. Output files

- `figures/experiment_data.json`: consolidated results; `results/run_registry.json`: sha256 of every result file; `CODE_MANIFEST.sha256`: code freeze record.
- Figures (PDF, vector): `fig_size_frequency`, `fig_leakage`, `fig_truncation_control`, `fig_stage_survival`, `fig_partition`, `fig_contrast`, `fig_methods`, `fig_sensitivity`, `fig_detectors`, `fig_yst`, `fig_training` (supplementary: `fig_partition_fcos`, `fig_stage_survival_fcos`).
- Tables (booktabs): `TABLE_config`, `TABLE_leakage`, `TABLE_truncation`, `TABLE_partition`, `TABLE_partition_fcos`, `TABLE_methods`, `TABLE_methods_fcos`, `TABLE_ccpr`, `TABLE_sensitivity`, `TABLE_yst`; LaTeX snippets in `figures/latex_includes.tex` (test-compiled with pdflatex, 0 overfull boxes).
- Code review record: `refine-logs/CODE_REVIEW_round1.md`, `refine-logs/CODE_REVIEW_round2.md`.

## Summary

- 9/9 planned runs completed (Runs 1–5 and 7–9 in full; Run 6 FCOS at a reduced 6-epoch budget).
- Main result: **positive for the diagnostic claims (K1–K3 on the primary detector), negative for the method increment (K7)**, with detector- and source-dependence made explicit (K4, K8).

