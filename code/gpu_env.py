# -*- coding: utf-8 -*-
"""Put the CUDA build of torch (installed with `pip install --target .pydeps_gpu`) first on sys.path.

The bundled embedded Python ignores PYTHONPATH (python311._pth), so GPU scripts import this
module before torch. Import it at module top level so Windows spawn workers inherit it.
"""
import os
import sys

_DEPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".pydeps_gpu")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)
_WORK = os.path.join(os.path.dirname(_DEPS), "work")
# keep Ultralytics settings (weights_dir, runs_dir) inside the workspace, not in the user profile
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_WORK, "ultralytics_cfg"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)


def configure_ultralytics():
    from ultralytics import settings
    settings.update({"weights_dir": os.path.join(_WORK, "weights"),
                     "runs_dir": os.path.join(_WORK, "runs"),
                     "datasets_dir": _WORK, "sync": False})
