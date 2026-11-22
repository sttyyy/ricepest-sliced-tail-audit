# Data, model and code sources and licences

Every third-party input used by the experiments, with its source, version, licence, how it was obtained and what may be redistributed. Licences of software packages were read from the installed package metadata (`.pydeps_gpu/*.dist-info/METADATA`) on 2026-11-21. Dataset licences were read from the files shipped with each dataset.

## Datasets

| Item | Publisher / source | Version used | Licence (evidence) | How obtained | Redistribution in this package |
|---|---|---|---|---|---|
| RicePest-30 images + COCO annotations (`filtered_train_translated.json`, `filtered_val_translated.json`) | RicePest-30 authors; stated public address https://github.com/kkb20334-lang/RicePest-30 | Copy uploaded by the team (`user_data/RicePest-30-Dataset.zip`); 7,915 train / 933 val images, 62,227 boxes | CC BY-NC 4.0 (badge and footer in the dataset `README.md`) | Uploaded by the team; the dataset README describes UV light-trap images from Hunan Province, web images and lab close-ups | Non-commercial redistribution with attribution is allowed by CC BY-NC 4.0. To keep the package small we ship no images, only derived artefacts (hashes, group ids, per-object ledger rows). Reviewers obtain the images from the address above. |
| Yellow Sticky Traps (relabelled) images + PASCAL VOC XML | M. Deserno and A. Briassouli (relabelled version, MetroAgriFor 2021, doi:10.1109/MetroAgriFor52389.2021.9628708), based on A.T. Nieuwenhuizen et al., 4TU.ResearchData, doi:10.4121/uuid:8b8ba63a-1010-4de7-a7fb-6f9e3baf128e | Copy uploaded by the team (`user_data/Yellow-Sticky-Traps-Dataset.rar`); 284 images 5184×3456, 8,114 boxes | CC0 1.0 (`LICENSE` file in the dataset folder) | Uploaded by the team | CC0 allows redistribution; we still ship only derived artefacts and the split file `results/yst/split.json`. |

Known annotation issues found by our scripts (reported, not corrected): 343 RicePest-30 images (300 train, 43 val) are single-pest close-ups whose JSON frame size has a different aspect ratio from the JPEG, so their boxes cannot be mapped; they are excluded and listed in `results/audit/audit_set.json["frame_inconsistent"]`. Two class ids (9 and 19) carry the same English name *Anomala corpulenta* in the JSON; we keep them as separate classes, as published.

## Pretrained weights

| Weights | Source | Licence | File / sha256 |
|---|---|---|---|
| `yolo11n.pt` (COCO-pretrained YOLO11n) | Ultralytics release asset, downloaded through the Hugging Face mirror `hf-mirror.com/Ultralytics/YOLO11` | AGPL-3.0 (Ultralytics) | `work/weights/yolo11n.pt`, sha256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `fcos_resnet50_fpn_coco-99b0c9b7.pth` (COCO FCOS-ResNet50-FPN) | `download.pytorch.org/models/` (torchvision model zoo) | BSD-3-Clause (torchvision); trained on COCO | `work/weights/fcos_resnet50_fpn_coco-99b0c9b7.pth`, sha256 prefix `99b0c9b7` matches the file name |

Fine-tuned weights produced by us (`work/weights/*_last.pt|pth`) inherit the licence terms of their initialisation (AGPL-3.0 for YOLO11n). Their sha256 values are in `results/runs/*_train.json`.

## Software

| Package | Version | Licence (package metadata) | Use |
|---|---|---|---|
| PyTorch (CUDA 12.8 build) | 2.9.1+cu128 | BSD-3-Clause | training / inference |
| torchvision | 0.24.1+cu128 | BSD | FCOS detector |
| Ultralytics | 8.3.240 | AGPL-3.0 | YOLO11n training, NMS read-out |
| SAHI | 0.14.0 | MIT | slice geometry (`get_slice_bboxes`) and merge (`greedy_nmm`, `nms`) |
| NumPy, SciPy, OpenCV, Pillow, Matplotlib, PyMuPDF | bundled runtime | BSD / MIT-style / HPND / AGPL (PyMuPDF, previews only) | analysis and figures |

Because the code imports Ultralytics (AGPL-3.0), the code in `code/` is released under AGPL-3.0-compatible terms.

## Third-party services

The experiment scripts send nothing to external services. The cross-model reviewer called in the workflow steps receives text prompts only (the full exchanges are in `_reviewer_thread.json`); no image or annotation file is sent.
