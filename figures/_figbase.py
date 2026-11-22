# -*- coding: utf-8 -*-
"""Shared helpers for every figures/gen_fig_*.py script.

All numbers come from results/ (JSON/CSV written by code/s1..s8); nothing is typed in by hand.
Colour roles are taken from the workspace Nature palette (nature_palette()), never hard-coded.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten, nature_palette, set_paper_placement  # noqa: E402,F401

setup_style()
import matplotlib.pyplot as plt  # noqa: E402

NP = nature_palette()
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

CATS = ["detected", "no_candidate", "pre_suppression_only", "post_hoc_recoverable"]
CAT_LABEL = {"detected": "Detected", "no_candidate": "No candidate",
             "pre_suppression_only": "Pre-suppression only", "post_hoc_recoverable": "Post-hoc recoverable"}
CAT_COLOR = {"detected": _lighten(NP["blue_secondary"], 0.55), "no_candidate": NP["red_strong"],
             "pre_suppression_only": NP["gold"], "post_hoc_recoverable": NP["teal"]}
STRATA = ["head", "mid", "tail"]
STRATUM_COLOR = {"head": NP["blue_main"], "mid": NP["teal"], "tail": NP["violet"]}
SIZE_LABEL = {"tiny": "<16$^2$", "s16_32": "16$^2$–32$^2$", "notsmall": "$\\geq$32$^2$", "small": "<32$^2$"}


def load(rel):
    with open(os.path.join(RES, rel), encoding="utf-8") as f:
        return json.load(f)


def out(name):
    return os.path.join(FIG, name)
