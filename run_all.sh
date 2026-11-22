#!/usr/bin/env bash
# Full reproduction pipeline, in the order used for the paper (per-step timings in REPRODUCE.md).
#
#     bash run_all.sh
#
# Set PY=/path/to/python to use a different interpreter. Stages write to results/, figures/
# and work/; work/ holds large intermediates and is git-ignored. Expect ~2.5 h of GPU time on
# the reference machine (RTX 5060 Laptop), most of it the two detector trainings.
#
# Prerequisites: the datasets must already be unpacked (REPRODUCE.md section 2) and the two
# pretrained initialisations must be in work/weights/ (REPRODUCE.md section 1).
set -euo pipefail
PY="${PY:-python}"
export PY
cd "$(dirname "$0")"
step() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- Run 1: audit design (CPU)
step "Run 1  grouping, leakage scan, frozen audit set, literal truncation"
$PY code/s1_groups_audit.py
$PY code/s1b_dataset_stats.py

# ---------------------------------------------------------------- dataset view + detectors
step "YOLO dataset view for RicePest-30"
$PY code/s2_prepare_yolo.py
step "Run 2  detector A, YOLO11n (freeze the last checkpoint)"
$PY code/s3_train_yolo.py --epochs 60 --name yolo11n_rice_s0
step "Run 6  detector B, FCOS-ResNet50-FPN (freeze the last checkpoint)"
$PY code/s3b_fcos.py train --epochs 6 --name fcos_rice_s0

# ---------------------------------------------------------------- candidate dumps
step "candidate dumps: T=320 (primary), T=480, T=256, overlap 0"
$PY code/s4_dump_yolo.py --weights work/weights/yolo11n_rice_s0_last.pt --tag yolo_T320
$PY code/s4_dump_yolo.py --weights work/weights/yolo11n_rice_s0_last.pt --tag yolo_T480 --T 480
$PY code/s4_dump_yolo.py --weights work/weights/yolo11n_rice_s0_last.pt --tag yolo_T256 --T 256
$PY code/s4_dump_yolo.py --weights work/weights/yolo11n_rice_s0_last.pt --tag yolo_T320_ov0 --overlap 0.0
$PY code/s3b_fcos.py dump --weights work/weights/fcos_rice_s0_last.pth --tag fcos_T320
step "merge-replay self-check against SAHI's own postprocessors"
$PY code/test_sahi_equivalence.py yolo_T320

# ---------------------------------------------------------------- Run 3 / Run 7: primary ledgers
step "Run 3  primary per-object ledger + confirmatory statistics (YOLO11n)"
$PY code/s5_ledger.py --tag yolo_T320 --run_id R3-ledger-yolo-T320 --write_objects
$PY code/s7_stats.py --run_id R3-ledger-yolo-T320
step "Run 7  per-object ledger + statistics (FCOS)"
$PY code/s5_ledger.py --tag fcos_T320 --run_id R7-ledger-fcos-T320 --write_objects
$PY code/s7_stats.py --run_id R7-ledger-fcos-T320

# ---------------------------------------------------------------- context metric (AP@0.5)
step "AP@0.5 on the audit set"
$PY code/s5b_ap.py --tag yolo_T320 --run_id yolo-T320
$PY code/s5b_ap.py --tag yolo_T320 --readout nms_iou --run_id yolo-T320-nms
$PY code/s5b_ap.py --tag yolo_T320 --readout fullimage --run_id yolo-fullimage
$PY code/s5b_ap.py --tag fcos_T320 --run_id fcos-T320
$PY code/s5b_ap.py --tag fcos_T320 --readout nms_iou --run_id fcos-T320-nms

# ---------------------------------------------------------------- sensitivity
step "sensitivity ledgers: read-out, cap, per-class cap, tau, tiling (YOLO11n)"
bash code/run_sensitivity.sh
step "sensitivity ledgers that also emit the per-object CSV used by s7_stats"
$PY code/s5_ledger.py --tag yolo_T320 --readout nms_iou --run_id S-ledger-yolo-T320-nms --write_objects
$PY code/s7_stats.py --run_id S-ledger-yolo-T320-nms
$PY code/s5_ledger.py --tag fcos_T320 --readout nms_iou --run_id S-ledger-fcos-T320-nms_iou --write_objects
$PY code/s7_stats.py --run_id S-ledger-fcos-T320-nms_iou
$PY code/s5_ledger.py --tag fcos_T320 --readout greedynmm_ios_prethr --run_id S-ledger-fcos-T320-greedynmm_ios_prethr --write_objects

# ---------------------------------------------------------------- Run 5 / Run 9: methods + CCPR
step "Runs 5/9  post-hoc baselines, CCPR and ablations on the same frozen dumps"
$PY code/s6_methods.py --tag yolo_T320 --run_id R5R9-methods-yolo
$PY code/s6_methods.py --tag yolo_T320 --readout nms_iou --run_id R5R9-methods-yolo-nms
$PY code/s6_methods.py --tag fcos_T320 --run_id R5R9-methods-fcos
$PY code/s6_methods.py --tag fcos_T320 --readout nms_iou --run_id R5R9-methods-fcos-nms

# ---------------------------------------------------------------- Run 8: YST
step "Run 8  YST native-resolution replication: split, train, dump, ledger, truncation"
$PY code/s8_yst_prepare.py
$PY code/s3_train_yolo.py --epochs 20 --name yolo11n_yst_s0 --data work/yolo_yst/yst.yaml
$PY code/s4_dump_yolo.py --weights work/weights/yolo11n_yst_s0_last.pt --tag yst_T640 --dataset yst --split test --T 640 --ref_long 0
$PY code/s5_ledger.py --tag yst_T640 --dataset yst --run_id R8-ledger-yst-T640 --write_objects
$PY code/s8b_yst_truncation.py
step "YST sensitivity ledgers"
$PY code/s5_ledger.py --tag yst_T640 --dataset yst --readout nms_iou --run_id S-ledger-yst-T640-nms
$PY code/s5_ledger.py --tag yst_T640 --dataset yst --cap 300 --run_id S-ledger-yst-T640-cap300
$PY code/s5_ledger.py --tag yst_T640 --dataset yst --readout nms_iou --cap 300 --run_id S-ledger-yst-T640-nms-cap300
$PY code/s5_ledger.py --tag yst_T640 --dataset yst --readout greedynmm_ios_prethr --run_id S-ledger-yst-T640-prethr

# ---------------------------------------------------------------- collect, tables, figures
step "collect the run registry and the summary data table"
$PY code/s9_collect.py
step "descriptives and LaTeX tables"
$PY code/s9b_descriptives.py
$PY code/s9_tables.py
step "all figures"
for f in figures/gen_fig_*.py; do $PY "$f"; done
step "supplementary FCOS panels of the partition and stage-survival figures"
MH_RUN=R7-ledger-fcos-T320 MH_OUT=fig_partition_fcos.pdf      $PY figures/gen_fig_partition.py
MH_RUN=R7-ledger-fcos-T320 MH_OUT=fig_stage_survival_fcos.pdf $PY figures/gen_fig_stage_survival.py
step "figure include list for the manuscript"
$PY code/s9c_latex_includes.py

printf '\nDone. Outputs are in results/ and figures/.\n'
