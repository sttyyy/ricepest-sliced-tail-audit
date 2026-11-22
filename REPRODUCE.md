# Reproduction guide

All numbers in `experiment_results.md`, the figures and the tables come from the commands below. Seeds are fixed (training seed 0, bootstrap seed 0, permutation seed 1/2, fold seed 0); the 2026-11-21 runs used the hardware and versions listed here.

## 1. Environment

| Item | Version used |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB, driver 596.56 (CUDA 13.2 runtime support) |
| Python | 3.11.9 (embedded runtime; ignores `PYTHONPATH`, hence `code/gpu_env.py`) |
| CPU packages | numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, opencv 4.10/4.13, pillow 12.1.1, matplotlib 3.10.8, PyMuPDF |
| GPU packages (`.pydeps_gpu/`) | torch 2.9.1+cu128, torchvision 0.24.1+cu128, ultralytics 8.3.240, ultralytics-thop 2.2.5, sahi 0.14.0, pybboxes 0.2.0, fire 0.7.1, termcolor 3.3.0, terminaltables 3.1.10, polars 2.1.0 |

Install the GPU packages without touching the base interpreter:

```
python -m pip install --no-deps --target .pydeps_gpu \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1+cu128-cp311-cp311-win_amd64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.24.1+cu128-cp311-cp311-win_amd64.whl
python -m pip install --no-deps --target .pydeps_gpu ultralytics==8.3.240 ultralytics-thop sahi pybboxes fire termcolor terminaltables polars
```

Weights: `work/weights/yolo11n.pt` (Ultralytics release asset, sha256 `0ebbc80d…44ee1`) and `work/weights/fcos_resnet50_fpn_coco-99b0c9b7.pth` (torchvision model zoo). Ultralytics settings are kept in `work/ultralytics_cfg/` through `YOLO_CONFIG_DIR`.

## 2. Data preparation

Unpack RicePest-30 to `user_data/RicePest-30-Dataset/{train,val}` (COCO JSON + JPEGs) and YST to `user_data/Yellow-Sticky-Traps-Dataset/{images,annotations}`. Every script reads the complete datasets; nothing is subsampled.

## 3. Run order and timings (one RTX 5060 Laptop)

| Step | Command | Time | Main outputs |
|---|---|---|---|
| Run 1 grouping + audit set + truncation | `python code/s1_groups_audit.py` | ~2 min CPU | `results/audit/{audit_set,grouping_sensitivity,truncation}.json` |
| class statistics | `python code/s1b_dataset_stats.py` | <1 min | `results/audit/class_stats.json` |
| YOLO view | `python code/s2_prepare_yolo.py` | <1 min | `work/yolo_rice/` |
| Run 2 detector A | `python code/s3_train_yolo.py --epochs 60 --name yolo11n_rice_s0` | 65 min | `work/weights/yolo11n_rice_s0_last.pt`, `results/runs/yolo11n_rice_s0_train*.{json,csv}` |
| detector B | `python code/s3b_fcos.py train --epochs 6 --name fcos_rice_s0` | 35 min (2,084 s) | `work/weights/fcos_rice_s0_last.pth`, `results/runs/fcos_rice_s0_train*.json` |
| dumps | `python code/s4_dump_yolo.py --weights work/weights/yolo11n_rice_s0_last.pt --tag yolo_T320` (+ `--T 480`, `--T 256`, `--overlap 0.0`); `python code/s3b_fcos.py dump --weights work/weights/fcos_rice_s0_last.pth --tag fcos_T320` | ~1 min each | `work/dumps/*.npz` |
| merge-replay check | `python code/test_sahi_equivalence.py yolo_T320` | <1 min | `results/audit/sahi_equivalence_check.json` |
| Run 3 primary ledger + statistics | `python code/s5_ledger.py --tag yolo_T320 --run_id R3-ledger-yolo-T320 --write_objects`; `python code/s7_stats.py --run_id R3-ledger-yolo-T320` | ~3 min | `results/ledger/R3-*`, `results/stats/R3-*_stats.json` |
| sensitivity ledgers | `bash code/run_sensitivity.sh` | ~8 min | `results/ledger/S-ledger-*.json` |
| Runs 5/9 methods + CCPR | `python code/s6_methods.py --tag yolo_T320 --run_id R5R9-methods-yolo` (and `--tag fcos_T320 --run_id R5R9-methods-fcos`) | ~1 min each | `results/methods/*.json` |
| Run 7 FCOS ledger | `python code/s5_ledger.py --tag fcos_T320 --run_id R7-ledger-fcos-T320 --write_objects`; `python code/s7_stats.py --run_id R7-ledger-fcos-T320` | ~3 min | `results/ledger/R7-*`, `results/stats/R7-*` |
| Run 8 YST | `python code/s8_yst_prepare.py`; `python code/s3_train_yolo.py --epochs 20 --name yolo11n_yst_s0 --data work/yolo_yst/yst.yaml`; `python code/s4_dump_yolo.py --weights work/weights/yolo11n_yst_s0_last.pt --tag yst_T640 --dataset yst --split test --T 640 --ref_long 0`; `python code/s5_ledger.py --tag yst_T640 --dataset yst --run_id R8-ledger-yst-T640 --write_objects`; `python code/s8b_yst_truncation.py` | ~25 min (training 1,192 s) | `results/yst/`, `results/ledger/R8-*` |
| collect, tables, figures | `python code/s9_collect.py`; `python code/s9_tables.py`; `python code/s9b_descriptives.py`; `for f in figures/gen_fig_*.py; do python "$f"; done` | ~3 min | `figures/experiment_data.json`, `figures/TABLE_*.tex`, `figures/fig_*.pdf` |

`run_all.sh` runs everything in this order.

## 4. Figure / table → command → data

| Item | Script | Reads |
|---|---|---|
| `fig_size_frequency.pdf` | `figures/gen_fig_size_frequency.py` | `results/audit/class_stats.json` |
| `fig_leakage.pdf` | `figures/gen_fig_leakage.py` | `results/audit/grouping_sensitivity.json`, `truncation.json`, `results/ledger/R3-ledger-yolo-T320.json` |
| `fig_truncation_control.pdf` | `figures/gen_fig_truncation_control.py` | `results/audit/truncation.json`, `results/yst/truncation.json` |
| `fig_partition.pdf`, `fig_stage_survival.pdf` | `gen_fig_partition.py`, `gen_fig_stage_survival.py` | `results/ledger/R3-ledger-yolo-T320.json` |
| `fig_detectors.pdf` | `gen_fig_detectors.py` | `results/ledger/{R3-ledger-yolo-T320,S-ledger-yolo-T320-nms,R7-ledger-fcos-T320,S-ledger-fcos-T320-nms_iou}.json` |
| `fig_partition_fcos.pdf`, `fig_stage_survival_fcos.pdf` (supplementary) | same scripts with `MH_RUN=R7-ledger-fcos-T320 MH_OUT=...` | `results/ledger/R7-ledger-fcos-T320.json` |
| `fig_contrast.pdf` | `gen_fig_contrast.py` | `results/stats/R3-ledger-yolo-T320_stats.json` |
| `fig_methods.pdf` | `gen_fig_methods.py` | `results/methods/R5R9-methods-yolo.json` |
| `fig_sensitivity.pdf` | `gen_fig_sensitivity.py` | `results/ledger/S-ledger-*.json` |
| `fig_yst.pdf` | `gen_fig_yst.py` | `results/ledger/R8-ledger-yst-T640.json`, `results/ledger/S-ledger-yst-T640-nms-cap300.json` |
| `fig_training.pdf` | `gen_fig_training.py` | `results/runs/*_train_log.*` |
| `TABLE_*.tex` | `code/s9_tables.py` | the JSON files named in each table function |

## 5. Known limits of reproduction

- GPU training is not bit-exact across GPUs/driver versions even with `deterministic=True`; the frozen weights' sha256 values are recorded in `results/runs/*_train.json`, and every downstream number is reproducible exactly from those weights (the ledger, statistics and methods are deterministic CPU code).
- The pHash grouping is computed from the JPEG bytes shipped with the dataset; a re-encoded copy of the dataset would change the hashes and possibly the audit set.
- RicePest-30 images are not redistributed (CC BY-NC 4.0 allows it, but the package stays small); obtain them from the dataset address in `DATA_SOURCES.md`.
