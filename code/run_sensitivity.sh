#!/bin/bash
# Sensitivity ledgers on the primary YOLO11n dumps (read-out, operating point, cap, tiling).
PY="${PY:-python}"
cd "$(dirname "$0")/.."
$PY code/s5_ledger.py --tag yolo_T320 --readout nms_iou --run_id S-ledger-yolo-T320-nms
$PY code/s5_ledger.py --tag yolo_T320 --readout greedynmm_ios_agn --run_id S-ledger-yolo-T320-agn
$PY code/s5_ledger.py --tag yolo_T320 --readout greedynmm_ios_prethr --run_id S-ledger-yolo-T320-prethr
$PY code/s5_ledger.py --tag yolo_T320 --readout fullimage --run_id S-ledger-yolo-fullimage
for cap in 10 20 50 300; do $PY code/s5_ledger.py --tag yolo_T320 --cap $cap --run_id S-ledger-yolo-T320-cap$cap; done
for pc in 5 10 20; do $PY code/s5_ledger.py --tag yolo_T320 --per_class_cap $pc --run_id S-ledger-yolo-T320-pcc$pc; done
for tau in 0.1 0.4; do $PY code/s5_ledger.py --tag yolo_T320 --tau $tau --run_id S-ledger-yolo-T320-tau$tau; done
$PY code/s5_ledger.py --tag yolo_T480 --run_id S-ledger-yolo-T480
$PY code/s5_ledger.py --tag yolo_T256 --run_id S-ledger-yolo-T256
$PY code/s5_ledger.py --tag yolo_T320_ov0 --run_id S-ledger-yolo-T320-ov0
