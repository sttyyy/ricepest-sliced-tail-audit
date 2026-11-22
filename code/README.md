# 代码说明（中文）

本目录是论文 **《Slice boundaries are a secondary contributor to small-object misses in tail classes: a preregistered recovery-ceiling audit of sliced inference for rice pest detection》** 的可复现代码。

这份 README 只是说明文档：新增它不会改动 `code/` 下任何 `.py` / `.sh` 文件。代码与论文中报告结果所用版本一致，文件哈希见仓库根目录的 `CODE_MANIFEST.sha256`。仓库根目录另有一份英文 `README.md`，内容与本文对应。

---

## 1. 这套代码在做什么

对“切片推理”（sliced inference，SAHI 式：先切片、逐片检测、再合并）下的长尾小目标做一次**预注册、按采集组留出**的逐目标审计。

流程把每一个标注目标依次经过五个阶段：

| 阶段 | 含义 |
|---|---|
| **V** | 某个切片视图完整包含该目标（若没有任何视图完整包含，就是“字面截断”，用作负对照） |
| **C** | 低分下限时，标准逐视图输出里存在同类候选（IoU ≥ 0.5） |
| **M** | 该目标在合并后的候选列表里被一对一匹配到 |
| **S** | 匹配到的合并框分数 ≥ 冻结阈值（0.25） |
| **D** | 在最终输出列表（阈值 + 每图上限 100）里被匹配到，即“检出” |

漏检目标按“怎样才能救回来”分成三类（优先级 M > C）：

- `no_candidate`：连候选都没有，只能靠检测器/训练去救；
- `pre_suppression_only`：有候选但没进合并列表，属于读出口径/抑制环节的问题；
- `post_hoc_recoverable`：进过合并列表但没被最终输出，重打分/校准可救。

另外给出“完整视图可达”的 oracle 上限（基线检出 + 任一完整包含该目标的视图里同类候选分数 ≥ 0.25），用来估计固定检测证据下还剩多少空间。

数据分工：RicePest-30 承担长尾主实验；Yellow Sticky Traps（YST）只做原生分辨率的来源/分辨率复验，不做长尾结论（YST 只有 3 类、8.4 倍不平衡）。检测器为 YOLO11n 与 torchvision FCOS-ResNet50-FPN，权重冻结，不因验证集做模型选择。

---

## 2. 文件说明

| 文件 | 作用 | 主要用法 / 参数 | 主要输出 |
|---|---|---|---|
| `common.py` | 全流程共享的冻结定义：640 长边参考帧、尺寸分箱（tiny < 16²；s16_32；medium < 96²；large）、按训练实例数排序的 head/mid/tail 三分位（各 10 类）、JSON→JPEG 帧一致性判定、SAHI 切片几何 | 被各脚本 import | — |
| `gpu_env.py` | 把 `.pydeps_gpu`（CUDA 版 torch 等）放到 `sys.path` 最前；把 Ultralytics 的权重/运行/数据目录固定到 `work/`（`YOLO_CONFIG_DIR`）。凡是用 torch 的脚本，都在 import torch 之前先 import 它 | 被各 GPU 脚本 import | — |
| `ledger_lib.py` | 台账核心库：IoU/包含矩阵、SAHI 合并读出回放（GREEDYNMM / NMS）、一对一匹配、按阈值+上限发射、Rice/YST 数据加载、YST XML 解析 | 被 s5 / s6 / s7 等调用 | — |
| `s1_groups_audit.py` | Run 1：pHash + MD5 近重复分组（τ = 0/4/6/8/10/12，主阈值 τ=8）、训练↔验证泄漏扫描、组不相交的冻结审计集、字面截断复核。**不读任何检测器输出**，审计集在看到结果前固定 | `python code/s1_groups_audit.py` | `work/img_meta.json`；`results/audit/{near_dup_edges,groups_tau*,audit_set,grouping_sensitivity,truncation}.json` |
| `s1b_dataset_stats.py` | 每类“频率 × 尺寸”统计（标注级，不用检测器） | `python code/s1b_dataset_stats.py` | `results/audit/class_stats.json` |
| `s2_prepare_yolo.py` | 生成 Ultralytics 的 RicePest-30 数据集视图（硬链接 + 归一化标签），剔除帧不一致的图像 | `python code/s2_prepare_yolo.py` | `work/yolo_rice/` |
| `s3_train_yolo.py` | 检测器 A：YOLO11n。训练时 `val=False`，冻结**最后一个** checkpoint | `--epochs`（默认 40；本项目用 60）、`--seed 0`、`--name`、`--data`、`--batch 16`、`--imgsz 640`、`--fraction 1.0` | `work/weights/yolo11n_rice_s0_last.pt`；`results/runs/*_train*.{json,csv}` |
| `s3b_fcos.py` | 检测器 B：torchvision FCOS-ResNet50-FPN（COCO 预训练 → RicePest-30 微调），以及它的切片候选转储 | `train --epochs 6 --name fcos_rice_s0 --seed 0`；`dump --weights … --tag fcos_T320` | `work/weights/`、`work/dumps/`、`results/runs/` |
| `s4_dump_yolo.py` | 切片候选转储：参考帧（`--ref_long 640`，全图 = 视图 0）+ SAHI 切片，每个视图一次前向；同一次前向保存两种读出：`std`（标准单标签读出：类内 NMS、IoU 0.7、max_det 300）与 `ml`（多标签读出，供 CCPR 用）。每个候选保留视图编号（来源可追溯） | `--weights`（必填）、`--tag`（必填）、`--dataset rice\|yst`、`--split val`、`--T 320`、`--overlap 0.25`、`--ref_long 640`、`--imgsz 640`、`--floor 0.01`、`--nms_iou 0.7`、`--max_det 300` | `work/dumps/<tag>.npz` |
| `s5_ledger.py` | 逐目标阶段台账 + 恢复上限三分类（V/C/M/S/D 与 no_candidate / pre_suppression_only / post_hoc_recoverable） | `--tag`、`--run_id`（都必填）、`--dataset`、`--readout`（`greedynmm_ios`（默认）/ `nms_iou` / `greedynmm_ios_agn` / `fullimage` / `greedynmm_ios_prethr`）、`--tau 0.25`、`--cap 100`、`--per_class_cap`、`--write_objects` | `results/ledger/<run_id>.json`；加 `--write_objects` 时另出 `_objects.csv` |
| `s5b_ap.py` | 上下文指标：审计集上按类的 COCO 风格 AP@0.5（101 点插值；合并列表，每图最多 100 框），按频率层报平均 | `--tag`、`--readout`、`--run_id` | `results/ledger/AP-<run_id>.json` |
| `s6_methods.py` | Run 5/9：同一冻结转储上的事后基线（NorCal 式先验校正、类别×尺寸阈值校准、Na 式跨视图一致性重打分）与 CCPR（类别条件峰值读出）+ 消融。参数用 5 折**按组交叉拟合**（不在被打分的折上拟合）；每个方法报两套工作点（自身拟合点 / 与基线假阳性总数匹配的全局阈值）；恢复项归因到基线漏检的恢复上限类别；给出 CCPR go/no-go | `--tag`、`--run_id`、`--readout`、`--folds` | `results/methods/<run_id>.json` |
| `s7_stats.py` | Run 3 主统计（确证性）：K2 恢复上限分区的计数/比例、按审计组聚类的 bootstrap CI（2000 次重采样）与 Bonferroni 同时 98.3% CI；K3 尺寸匹配的频率对比（尾类小目标 vs 重加权到头类小目标尺寸结构后的头类小目标），逐类别份额差 + 聚类 bootstrap CI + 类别标签精确置换 p（可行时穷举全部分组），三类漏检上用 Holm；探索性（尺寸×类别）用 BH；最小格规则：<10 个事件或 <5 个组只报计数 | `--run_id` | `results/stats/<run_id>_stats.json` |
| `s8_yst_prepare.py` | YST：64 位 pHash 近重复组（Hamming ≤ 8）→ 按组做 70/30 带种子划分；训练切片用原生 T=640、重叠 0.25，标注裁到切片内、框面积 ≥10% 才保留（SAHI `slice_coco` 的 `min_area_ratio` 默认），保留含 ≥1 目标的切片 + 5% 空切片 | `python code/s8_yst_prepare.py` | `work/yolo_yst/`、`results/yst/split.json` |
| `s8b_yst_truncation.py` | YST 原生几何的截断审计：若干切片尺寸/重叠下，无任何切片完整包含的目标占比（逐类；全部 284 张与留出测试集） | `python code/s8b_yst_truncation.py` | `results/yst/truncation.json` |
| `s9_collect.py` | 汇总 `results/` 下全部 JSON/CSV 摘要到一张表；写运行登记表（运行编号 → 配置、输出文件、sha256）；大的逐目标 CSV 只记路径与 sha256，不内联 | `python code/s9_collect.py` | `figures/experiment_data.json`、`results/run_registry.json` |
| `s9_tables.py` | 生成论文用的 LaTeX 表格 | `python code/s9_tables.py` | `figures/TABLE_*.tex` |
| `s9b_descriptives.py` | 描述性补充（不做新推断）：尾类逐类结果计数（全部尺寸 / 仅小目标）、审计图 vs 受污染验证图的逐层召回、YOLO 与 FCOS 在每个尾类小目标对象上的一致性 | `python code/s9b_descriptives.py` | `results/stats/descriptives.json` |
| `s9c_latex_includes.py` | 生成 `figures/latex_includes.tex`：每张数据图一个 figure 块、每张表一个 `\input` | `python code/s9c_latex_includes.py` | `figures/latex_includes.tex` |
| `test_sahi_equivalence.py` | 一致性自检：在真实候选列表上核对 `ledger_lib` 的合并回放与 SAHI 自带的 `GreedyNMMPostprocess` / `NMSPostprocess` 是否一致 | `python code/test_sahi_equivalence.py yolo_T320` | `results/audit/sahi_equivalence_check.json` |
| `run_sensitivity.sh` | 主 YOLO 转储上的敏感性台账：读出方式（NMS-IoU / 类别无关 GreedyNMM / 抑制前阈值 / 全图）、输出上限 10/20/50/300、每类上限 5/10/20、τ=0.1/0.4、T=480 / T=256 / 重叠 0 | `bash code/run_sensitivity.sh` | `results/ledger/S-ledger-*.json` |

---

## 3. 环境与依赖

| 项目 | 本论文使用的版本 |
|---|---|
| 操作系统 | Windows 11 Home（10.0.26200） |
| GPU | NVIDIA GeForce RTX 5060 Laptop，8 GB（驱动 596.56，支持 CUDA 13.2 运行时） |
| Python | 3.11.9（内嵌运行时；它会忽略 `PYTHONPATH`，所以要用 `code/gpu_env.py`） |
| CPU 包 | numpy 2.4.6、scipy 1.17.1、pandas 2.3.3、opencv 4.10/4.13、pillow 12.1.1、matplotlib 3.10.8、PyMuPDF |
| GPU 包（装在 `.pydeps_gpu/`） | torch 2.9.1+cu128、torchvision 0.24.1+cu128、ultralytics 8.3.240、ultralytics-thop 2.2.5、sahi 0.14.0、pybboxes 0.2.0、fire 0.7.1、termcolor 3.3.0、terminaltables 3.1.10、polars 2.1.0 |

GPU 包装进独立目录、不动基础解释器：

```bash
python -m pip install --no-deps --target .pydeps_gpu \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1+cu128-cp311-cp311-win_amd64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.24.1+cu128-cp311-cp311-win_amd64.whl
python -m pip install --no-deps --target .pydeps_gpu ultralytics==8.3.240 ultralytics-thop sahi pybboxes fire termcolor terminaltables polars
```

（Windows 的 cmd 里把行尾的 `\` 换成 `^`。）

初始权重（不随包分发，从官方渠道获取）：

- `work/weights/yolo11n.pt`：Ultralytics 发布的 YOLO11n 权重（sha256 `0ebbc80d…44ee1`）；
- `work/weights/fcos_resnet50_fpn_coco-99b0c9b7.pth`：torchvision 模型库的 COCO FCOS 权重。

Ultralytics 的配置目录通过 `YOLO_CONFIG_DIR` 固定在 `work/ultralytics_cfg/`，不会写到用户目录。

---

## 4. 数据准备

- RicePest-30 解压到 `user_data/RicePest-30-Dataset/{train,val}`（COCO JSON + JPEG）；
- YST 解压到 `user_data/Yellow-Sticky-Traps-Dataset/{images,annotations}`（JPEG + Pascal VOC XML）。

所有脚本都读完整数据集，不做抽样。数据与许可的逐项登记见 `DATA_SOURCES.md`（RicePest-30：CC BY-NC 4.0；YST：CC0 1.0）。RicePest-30 的图像不随代码包分发。

---

## 5. 运行顺序与耗时（单张 RTX 5060 Laptop）

| 步骤 | 命令 | 耗时 | 主要产物 |
|---|---|---|---|
| Run 1：分组 + 审计集 + 截断 | `python code/s1_groups_audit.py` | 约 2 分钟（CPU） | `results/audit/{audit_set,grouping_sensitivity,truncation}.json` |
| 类别统计 | `python code/s1b_dataset_stats.py` | <1 分钟 | `results/audit/class_stats.json` |
| YOLO 数据视图 | `python code/s2_prepare_yolo.py` | <1 分钟 | `work/yolo_rice/` |
| Run 2：检测器 A | `python code/s3_train_yolo.py --epochs 60 --name yolo11n_rice_s0` | 65 分钟 | `work/weights/yolo11n_rice_s0_last.pt`、`results/runs/yolo11n_rice_s0_train*.{json,csv}` |
| 检测器 B | `python code/s3b_fcos.py train --epochs 6 --name fcos_rice_s0` | 35 分钟（2,084 秒） | `work/weights/fcos_rice_s0_last.pth`、`results/runs/fcos_rice_s0_train*.json` |
| 候选转储 | `python code/s4_dump_yolo.py --weights work/weights/yolo11n_rice_s0_last.pt --tag yolo_T320`（另加 `--T 480`、`--T 256`、`--overlap 0.0` 三份敏感性转储）；`python code/s3b_fcos.py dump --weights work/weights/fcos_rice_s0_last.pth --tag fcos_T320` | 每份约 1 分钟 | `work/dumps/*.npz` |
| 合并回放自检 | `python code/test_sahi_equivalence.py yolo_T320` | <1 分钟 | `results/audit/sahi_equivalence_check.json` |
| Run 3：主台账 + 统计 | `python code/s5_ledger.py --tag yolo_T320 --run_id R3-ledger-yolo-T320 --write_objects`；`python code/s7_stats.py --run_id R3-ledger-yolo-T320` | 约 3 分钟 | `results/ledger/R3-*`、`results/stats/R3-*_stats.json` |
| 敏感性台账 | `bash code/run_sensitivity.sh` | 约 8 分钟 | `results/ledger/S-ledger-*.json` |
| Run 5/9：方法 + CCPR | `python code/s6_methods.py --tag yolo_T320 --run_id R5R9-methods-yolo`（FCOS 同样一条，把 tag/run_id 换成 `fcos_T320` / `R5R9-methods-fcos`） | 每个约 1 分钟 | `results/methods/*.json` |
| Run 7：FCOS 台账 | `python code/s5_ledger.py --tag fcos_T320 --run_id R7-ledger-fcos-T320 --write_objects`；`python code/s7_stats.py --run_id R7-ledger-fcos-T320` | 约 3 分钟 | `results/ledger/R7-*`、`results/stats/R7-*` |
| Run 8：YST | `python code/s8_yst_prepare.py`；`python code/s3_train_yolo.py --epochs 20 --name yolo11n_yst_s0 --data work/yolo_yst/yst.yaml`；`python code/s4_dump_yolo.py --weights work/weights/yolo11n_yst_s0_last.pt --tag yst_T640 --dataset yst --split test --T 640 --ref_long 0`；`python code/s5_ledger.py --tag yst_T640 --dataset yst --run_id R8-ledger-yst-T640 --write_objects`；`python code/s8b_yst_truncation.py` | 约 25 分钟（训练 1,192 秒） | `results/yst/`、`results/ledger/R8-*` |
| 汇总、表格、出图 | `python code/s9_collect.py`；`python code/s9_tables.py`；`python code/s9b_descriptives.py`；`for f in figures/gen_fig_*.py; do python "$f"; done` | 约 3 分钟 | `figures/experiment_data.json`、`figures/TABLE_*.tex`、`figures/fig_*.pdf` |

`run_all.sh` 就是按上面的顺序一次跑完（另含 AP、FCOS 敏感性台账、`latex_includes` 等收尾步骤）。

### 单独重算某一项

- 只换读出方式 / 工作点 / 输出上限：直接给 `s5_ledger.py` 加 `--readout`、`--tau`、`--cap`、`--per_class_cap`，换一个 `--run_id` 即可；
- 只要 AP：`s5b_ap.py`；
- 只要某张图：`figures/gen_fig_*.py` 读 `results/`，不重跑检测。

---

## 6. 输出在哪里

| 目录 | 内容 |
|---|---|
| `results/` | 全部运行产物：审计集与分组（`audit/`）、逐目标台账（`ledger/`）、统计（`stats/`）、方法与 CCPR（`methods/`）、YST（`yst/`）、训练日志（`runs/`）、运行登记表 `run_registry.json` |
| `figures/` | 汇总数据 `experiment_data.json`、LaTeX 表 `TABLE_*.tex`、图 `fig_*.pdf`、插图清单 `latex_includes.tex` |
| `work/` | 大中间件：YOLO 数据视图、候选转储 `dumps/*.npz`、权重 `weights/`、Ultralytics 配置；**不进提交包** |

---

## 7. 图 / 表 → 脚本 → 输入数据

| 图表 | 脚本 | 读什么 |
|---|---|---|
| `fig_size_frequency.pdf` | `figures/gen_fig_size_frequency.py` | `results/audit/class_stats.json` |
| `fig_leakage.pdf` | `figures/gen_fig_leakage.py` | `results/audit/grouping_sensitivity.json`、`truncation.json`、`results/ledger/R3-ledger-yolo-T320.json` |
| `fig_truncation_control.pdf` | `figures/gen_fig_truncation_control.py` | `results/audit/truncation.json`、`results/yst/truncation.json` |
| `fig_partition.pdf`、`fig_stage_survival.pdf` | `gen_fig_partition.py`、`gen_fig_stage_survival.py` | `results/ledger/R3-ledger-yolo-T320.json` |
| `fig_detectors.pdf` | `gen_fig_detectors.py` | `results/ledger/{R3-ledger-yolo-T320, S-ledger-yolo-T320-nms, R7-ledger-fcos-T320, S-ledger-fcos-T320-nms_iou}.json` |
| `fig_partition_fcos.pdf`、`fig_stage_survival_fcos.pdf`（补充材料） | 同上面两个脚本，环境变量 `MH_RUN=R7-ledger-fcos-T320 MH_OUT=…` | `results/ledger/R7-ledger-fcos-T320.json` |
| `fig_contrast.pdf` | `gen_fig_contrast.py` | `results/stats/R3-ledger-yolo-T320_stats.json` |
| `fig_methods.pdf` | `gen_fig_methods.py` | `results/methods/R5R9-methods-yolo.json` |
| `fig_sensitivity.pdf` | `gen_fig_sensitivity.py` | `results/ledger/S-ledger-*.json` |
| `fig_yst.pdf` | `gen_fig_yst.py` | `results/ledger/R8-ledger-yst-T640.json`、`results/ledger/S-ledger-yst-T640-nms-cap300.json` |
| `fig_training.pdf` | `gen_fig_training.py` | `results/runs/*_train_log.*` |
| `TABLE_*.tex` | `code/s9_tables.py` | 各表函数里点名的 JSON |

---

## 8. 随机种子与确定性

- 训练种子 0；bootstrap 种子 0；置换检验种子 1 / 2；交叉拟合折种子 0。
- 台账（`s5_ledger.py`）、统计（`s7_stats.py`）、方法（`s6_methods.py`）是**确定性 CPU 代码**：给定同一份冻结权重和同一份数据，结果可精确复现。
- 训练本身受 GPU / 驱动影响，不保证逐位一致；冻结权重的 sha256 记录在 `results/runs/*_train.json` 里。

---

## 9. 复现的边界

- **GPU 训练非逐位一致**：换 GPU 或驱动版本，即使 `deterministic=True` 也可能有细微差别；下游所有数字都可以从冻结权重精确复现。
- **pHash 依赖 JPEG 字节**：分组是用数据集原样发布的 JPEG 计算的；重新编码过的副本会改变哈希，可能改变审计集。
- **数据不分发**：RicePest-30 图像不随包分发（许可允许，但为控制包体积），请按 `DATA_SOURCES.md` 里的地址获取。
- **YST 的定位**：只有 3 类、8.4 倍不平衡，只用于原生分辨率的来源/分辨率复验，不做长尾结论。

---

## 10. 许可

- 代码依赖 Ultralytics（AGPL-3.0），因此按 AGPL-3.0 兼容条款分发；另依赖 torchvision、SAHI 等开源包，许可见各自仓库。
- 数据：RicePest-30 为 CC BY-NC 4.0；YST 为 CC0 1.0。逐项登记见 `DATA_SOURCES.md`。

---

## 11. 仓库入口与冻结版本

- 入口：`run_all.sh`（完整运行顺序）；细节与耗时见 `REPRODUCE.md`。
- 冻结版本：仓库 tag `v1.0.0`（commit `94a7abb`）。`CODE_MANIFEST.sha256` 列出了产出论文数字时 `code/`、`figures/gen_fig_*.py` 与 `configs/` 下每个文件的 sha256。

---

## 12. 配套文件（代码目录之外）

| 文件 | 内容 |
|---|---|
| `README.md`（仓库根目录） | 英文总览：入口、目录、依赖、许可 |
| `REPRODUCE.md` | 环境、数据准备、运行顺序、每张图对应的命令与数据、复现限制 |
| `DATA_SOURCES.md` | 数据与标注的发布方、版本、许可、获取方式；外部代码与权重来源 |
| `experiment_results.md` | 结果汇总；原始输出在 `results/`，汇总在 `figures/experiment_data.json` |
| `results/run_registry.json` | 每个运行编号对应的配置、输出文件与 sha256 |
| `configs/prereg_frozen.json` | 预注册冻结量（在任何检测器输出之前写入） |
