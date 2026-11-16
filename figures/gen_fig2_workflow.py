"""
gen_fig2_workflow.py
====================
Generate Fig 2 for Paper 2: clean adversarial robustness evaluation workflow.

Output files
------------
  documents_v2/paper2/figures/Fig2_workflow.pdf  (vector, for LaTeX)
  documents_v2/paper2/figures/Fig2_workflow.tif  (600 DPI, for Springer submission)

Design constraints (Springer Nature two-column)
------------------------------------------------
  Width : 174 mm  (double-column)
  Height: <= 234 mm
  DPI   : 600
  Fonts : Helvetica/Arial, 8-12 pt inside boxes
  Style : white background, grey component boxes, blue flow arrows, black text
  Rule  : NO result numbers, NO "Supported"/"Not supported"/"Mixed"/"Inconclusive",
          NO verdict outcomes anywhere in the figure.

Workflow stages (from configs/, scripts/06_robustness.py)
---------------------------------------------------------
  1. CICIoMT2024 Wi-Fi+MQTT (45 features, 19 classes, seeds 42-46)
  2. Train/Val/Test split (file-level, 80/15/20 of captures)
  3. Train four models: TRACE-MoE, Transformer, BiLSTM-DNN, TF-SAGE-lite
  4. PCRP fitting: Gaussian profiles per protocol-category on clean training windows
  5. Adversarial fine-tuning: 5-step PGD, epsilon~Uniform[0,0.25], 3 epochs
  6. Robustness evaluation: clean model + adv-FT model, PGD at epsilon in {0,0.05,0.10,0.25,0.50}
     18 mutable features; H6 uses 5-step PGD on 1% subsample
  7. Hypothesis tests H1-H6 (labels only, no verdicts)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---- Output paths -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIG_DIR, "Fig2_workflow.pdf")
OUT_TIF = os.path.join(FIG_DIR, "Fig2_workflow.tif")

# ---- Page geometry ------------------------------------------------------
FIG_WIDTH_INCH = 174 / 25.4
FIG_HEIGHT_INCH = 110 / 25.4
DPI = 600

# ---- Colours ------------------------------------------------------------
COL_DATA   = "#D0E4F7"   # blue: data
COL_TRAIN  = "#E8E8E8"   # grey: training
COL_EVAL   = "#E8F5E9"   # green: evaluation
COL_HYP    = "#FFF3E0"   # orange: hypothesis tests
COL_PCRP   = "#F3E5F5"   # purple: PCRP
COL_ARROW  = "#2255AA"   # blue arrow
COL_EDGE   = "#555555"
FONT_SIZE  = 7

fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=DPI)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(ax, x, y, w, h, label, sublabel=None, color=COL_TRAIN,
        fontsize=FONT_SIZE):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01",
        linewidth=0.6,
        edgecolor=COL_EDGE,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + h * 0.14, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                fontfamily="DejaVu Sans", zorder=3)
        ax.text(x, y - h * 0.18, sublabel, ha="center", va="center",
                fontsize=fontsize - 1, style="italic",
                fontfamily="DejaVu Sans", zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                fontfamily="DejaVu Sans", zorder=3)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COL_ARROW,
            lw=0.9,
            mutation_scale=6,
        ),
        zorder=4,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.01, my, label, ha="left", va="center",
                fontsize=FONT_SIZE - 1, color="#444444",
                fontfamily="DejaVu Sans", zorder=5)


# ---- Workflow: top-to-bottom with parallel branches ---------------------

# Stage 1: Dataset
box(ax, 0.5, 0.90, 0.46, 0.12,
    "CICIoMT2024 (Wi-Fi + MQTT)",
    "45 features · 19 classes · seeds 42 to 46 · 40 IoMT devices",
    color=COL_DATA)

# Arrow down
arrow(ax, 0.5, 0.84, 0.5, 0.78)
ax.text(0.52, 0.81, "file-level split", ha="left", va="center",
        fontsize=FONT_SIZE - 1, color="#444444", fontfamily="DejaVu Sans")

# Stage 2: Split
box(ax, 0.5, 0.74, 0.70, 0.09,
    "Train / Validation / Test split",
    "80% capture files (85% rows train, 15% rows val), 20% capture files test; dedup",
    color=COL_DATA)

# Two arrows: left branch (Train models + PCRP), right stays single
arrow(ax, 0.33, 0.70, 0.22, 0.60)
arrow(ax, 0.5, 0.70, 0.5, 0.60)   # straight down for PCRP
arrow(ax, 0.67, 0.70, 0.78, 0.60)

ax.text(0.20, 0.66, "train\nwindows", ha="center", va="center",
        fontsize=FONT_SIZE - 1, color="#444444", fontfamily="DejaVu Sans")
ax.text(0.56, 0.65, "train\nwindows", ha="center", va="center",
        fontsize=FONT_SIZE - 1, color="#444444", fontfamily="DejaVu Sans")
ax.text(0.84, 0.65, "test\nwindows", ha="center", va="center",
        fontsize=FONT_SIZE - 1, color="#444444", fontfamily="DejaVu Sans")

# Stage 3L: Train four architectures
box(ax, 0.22, 0.54, 0.34, 0.10,
    "Train four architectures",
    "TRACE-MoE · Transformer · BiLSTM-DNN · TF-SAGE-lite",
    color=COL_TRAIN)

# Stage 3M: PCRP fitting
box(ax, 0.5, 0.54, 0.16, 0.12,
    "PCRP fitting",
    "Gaussian profiles\n6 protocol-categories",
    color=COL_PCRP)

# Stage 3R: Test set (held out)
box(ax, 0.78, 0.54, 0.22, 0.12,
    "Test set (held out)",
    "4,905 windows\n(class-balanced)",
    color=COL_DATA)

# Arrow from train to adv fine-tuning
arrow(ax, 0.22, 0.485, 0.22, 0.435)

# Stage 4: Adv fine-tuning
box(ax, 0.22, 0.365, 0.34, 0.12,
    "Adversarial fine-tuning (fine-tuned model)",
    "3 epochs · $\\varepsilon{\\sim}\\mathrm{Uniform}[0,0.25]$\n5-step PGD · 18 mutable features",
    color=COL_TRAIN)

# Clean model + fine-tuned model both feed evaluation
# Arrow from clean train box to evaluation
arrow(ax, 0.33, 0.49, 0.50, 0.26)
ax.text(0.455, 0.42, "clean\nmodel", ha="center", va="center",
        fontsize=FONT_SIZE - 1, color="#444444", fontfamily="DejaVu Sans",
        bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)

# Arrow from fine-tuned to evaluation
arrow(ax, 0.30, 0.295, 0.36, 0.262)

# Arrow from test set to evaluation
arrow(ax, 0.78, 0.49, 0.62, 0.26)

# Arrow from PCRP to evaluation
arrow(ax, 0.5, 0.49, 0.5, 0.26)

# Stage 5: Robustness evaluation
box(ax, 0.5, 0.21, 0.56, 0.10,
    "Robustness evaluation",
    "$\\varepsilon \\in \\{0, 0.05, 0.10, 0.25, 0.50\\}$ · macro-F1 curves · PCRP discrepancy"
    "  ·  H6: 5-step PGD, 1% subsample",
    color=COL_EVAL)

# Arrow to hypothesis tests
arrow(ax, 0.5, 0.16, 0.5, 0.09)

# Stage 6: Hypothesis tests
box(ax, 0.5, 0.05, 0.56, 0.08,
    "Hypothesis tests  H1 · H2 · H3 · H4 · H5 · H6",
    color=COL_HYP)

# ---- Title --------------------------------------------------------------
ax.text(0.5, 1.0, "Adversarial Robustness Evaluation Workflow",
        ha="center", va="bottom", fontsize=FONT_SIZE + 1,
        fontweight="bold", fontfamily="DejaVu Sans")

# ---- Legend -------------------------------------------------------------
legend_items = [
    mpatches.Patch(facecolor=COL_DATA,  edgecolor=COL_EDGE, label="Data"),
    mpatches.Patch(facecolor=COL_TRAIN, edgecolor=COL_EDGE, label="Model training"),
    mpatches.Patch(facecolor=COL_PCRP,  edgecolor=COL_EDGE, label="PCRP fingerprint"),
    mpatches.Patch(facecolor=COL_EVAL,  edgecolor=COL_EDGE, label="Evaluation"),
    mpatches.Patch(facecolor=COL_HYP,   edgecolor=COL_EDGE, label="Hypothesis tests"),
]
ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(1.0, 0.0),
          fontsize=FONT_SIZE - 1, framealpha=0.9, edgecolor=COL_EDGE, ncol=1)

plt.tight_layout(pad=0.1)
fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", format="pdf")
fig.savefig(OUT_TIF, dpi=DPI, bbox_inches="tight", format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_TIF}")
