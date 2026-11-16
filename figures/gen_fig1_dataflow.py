"""
gen_fig1_dataflow.py
====================
Generate Fig 1 for Paper 2: TRACE-MoE data-flow architecture diagram.

Output files
------------
  documents_v2/paper2/figures/Fig1_dataflow.pdf  (vector, for LaTeX)
  documents_v2/paper2/figures/Fig1_dataflow.tif  (600 DPI, for Springer submission)

Design constraints (Springer Nature two-column)
------------------------------------------------
  Width : 174 mm  (double-column)
  Height: <= 234 mm
  DPI   : 600
  Fonts : Helvetica/Arial, 8-12 pt inside boxes
  Style : white background, grey component boxes, blue flow arrows, black text
  Rule  : NO results, verdicts, file names, or hypothesis outcomes anywhere

Architecture details (from src/trace_iomt/models/networks.py and configs/)
---------------------------------------------------------------------------
  T=16 flows, F=45 features, N_P=5 protocol tokens
  FeatureTokenizer: d_t=16 per feature  (source: configs/default_v2.yaml: token_dim=16)
  AttentionPool over F=45 tokens        (source: networks.py TRACEEncoder)
  Step embedding: concat(pooled, raw) -> Linear(61,128)+LN+GELU
  Protocol Embedding: vocab=5, dim=128; for router dim=32
  Sequence: BiGRU(128->64 bidir) -> Transformer(2 layers, H=4, ff=256, pre-LN, learned PE)
  Final pool: AttentionPool(T) + last step -> Linear(256,128)+GELU -> h in R^128
  Router: concat(h:128, e_cT:32) -> Linear(160,4) -> softmax over E=4 experts
  Top-1 selection: k* = argmax g; z = h + g_{k*} * E_{k*}(h)
  Expert E_k: LN -> Linear(128,256) -> GELU -> Dropout(0.2) -> Linear(256,128)
  Heads: Linear(128,2), Linear(128,6), Linear(128,19)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---- Output paths -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIG_DIR, "Fig1_dataflow.pdf")
OUT_TIF = os.path.join(FIG_DIR, "Fig1_dataflow.tif")

# ---- Page geometry ------------------------------------------------------
FIG_WIDTH_INCH = 174 / 25.4   # 174 mm in inches
FIG_HEIGHT_INCH = 130 / 25.4  # taller to fit parallel expert row
DPI = 600

# ---- Colour palette (accessible, greyscale-readable) --------------------
COL_BOX   = "#E8E8E8"   # light grey box fill
COL_BOX2  = "#D0E4F7"   # light blue for special boxes (router/experts)
COL_ARROW = "#2255AA"   # medium blue for arrows
COL_HEAD  = "#F5E6CC"   # warm cream for output heads
COL_EDGE  = "#555555"   # dark grey box edge
FONT_SIZE = 7           # pt — 7 pt is within the 8-12 pt Springer band scaled to inches

fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=DPI)
ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 1)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(ax, x, y, w, h, label, sublabel=None, color=COL_BOX, fontsize=FONT_SIZE):
    """Draw a rounded rectangle with centred text."""
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
        ax.text(x, y + h * 0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                fontfamily="DejaVu Sans", zorder=3)
        ax.text(x, y - h * 0.18, sublabel, ha="center", va="center",
                fontsize=fontsize - 1, style="italic",
                fontfamily="DejaVu Sans", zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                fontfamily="DejaVu Sans", zorder=3)


def arrow(ax, x1, y1, x2, y2):
    """Draw a simple arrow from (x1,y1) to (x2,y2)."""
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


def label_between(ax, x, y, text, fontsize=FONT_SIZE - 1):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color="#444444",
            fontfamily="DejaVu Sans", zorder=5)


# ---- Layout: left-to-right pipeline with two levels --------------------
# Row 1 (y=0.80): Input -> Tokenizer -> AttentionPool -> StepEmbed
# Row 2 (y=0.50): ProtocolEmbed, BiGRU, Transformer, FinalPool -> h
# Row 3 (y=0.20): Router -> E=4 experts (top-1) -> Heads (3 outputs)

BH = 0.15   # box height
BW = 0.14   # box width (default)

# ---- ROW 1: Input representation ----------------------------------------
Y1 = 0.85
positions_r1 = [0.17, 0.34, 0.51, 0.68, 0.86]
labels_r1 = [
    ("Input window", "$T{=}16$ flows\n$F{=}45$ features"),
    ("FeatureTokenizer", "per-feature affine\n$d_t{=}16$"),
    ("AttentionPool", "over $F{=}45$ tokens"),
    ("Step Projection", "concat+Linear(61,128)\n+LN+GELU"),
    ("Protocol Emb.", "vocab $N_P{=}5$,\ndim 128"),
]
for i, (xi, (lbl, sub)) in enumerate(zip(positions_r1, labels_r1)):
    box(ax, xi, Y1, BW, BH, lbl, sub, color=COL_BOX)

# Arrows row 1
for i in range(3):
    arrow(ax, positions_r1[i] + BW / 2, Y1,
          positions_r1[i + 1] - BW / 2, Y1)

# Label on the first arrow

# Protocol token feeds into step projection (diagonal arrow)
# From Protocol Embedding box bottom → into Step Projection
arrow(ax, positions_r1[4] - BW / 2, Y1,
      positions_r1[3] + BW / 2, Y1)

# ---- ROW 2: Sequence encoder --------------------------------------------
Y2 = 0.60
positions_r2 = [0.22, 0.40, 0.58, 0.76]
labels_r2 = [
    ("BiGRU", "64 units/direction\nT×128 output"),
    ("Transformer", "2 layers, $H{=}4$ heads\n$d_{\\mathrm{ff}}{=}256$, pre-LN"),
    ("Final Pool", "AttentionPool+last step\nLinear(256,128)+GELU"),
    ("$\\mathbf{h}$", "$\\in \\mathbb{R}^{128}$"),
]
bw_r2 = [BW, BW, BW, 0.08]
bc_r2 = [COL_BOX, COL_BOX, COL_BOX, COL_BOX2]
for xi, (lbl, sub), bw, col in zip(positions_r2, labels_r2, bw_r2, bc_r2):
    box(ax, xi, Y2, bw, BH, lbl, sub if lbl != "$\\mathbf{h}$" else None, color=col)

# Arrow from row1 step projection down to BiGRU
arrow(ax, positions_r1[3], Y1 - BH / 2,
      positions_r2[0] + 0.03, Y2 + BH / 2)
label_between(ax, 0.50, 0.705, "T step embeddings")

# Arrows within row 2
for i in range(len(positions_r2) - 1):
    arrow(ax, positions_r2[i] + bw_r2[i] / 2, Y2,
          positions_r2[i + 1] - bw_r2[i + 1] / 2, Y2)

# Protocol embedding for router (separate path, feeds into router below)
# Just a text label to keep diagram clean
ax.text(0.92, Y2 + 0.02, "Protocol emb.\n$c_T{\\to}\\mathbb{R}^{32}$",
        ha="center", va="center", fontsize=FONT_SIZE - 1,
        style="italic", color="#444444", fontfamily="DejaVu Sans")

# ---- ROW 3: Router and experts ------------------------------------------
Y3 = 0.39       # router and z row
Y_EXP = 0.13    # experts row (parallel, side by side)
router_x = 0.20
box(ax, router_x, Y3, BW, BH, "Router", "Linear(160,4)\nsoftmax", color=COL_BOX2)

arrow(ax, positions_r2[3], Y2 - BH / 2,
      router_x + BW / 2, Y3 + BH / 3)
arrow(ax, 0.92, Y2 - 0.07, router_x + BW / 2, Y3 + BH / 6)

z_x = 0.84
exp_xs = [0.30, 0.42, 0.54, 0.66]
EXW = 0.10
for j, ex in enumerate(exp_xs):
    box(ax, ex, Y_EXP, EXW, BH,
        f"$\\mathrm{{Ex}}_{j+1}$", "LN,Lin(256),\nGELU,Lin(128)",
        color=COL_BOX)
    arrow(ax, router_x, Y3 - BH / 2, ex - 0.02, Y_EXP + BH / 2)

box(ax, z_x, Y3, 0.08, BH, "$\\mathbf{z}$",
    "$\\mathbf{h}+g_{k^*}\\cdot \\mathrm{Ex}_{k^*}(\\mathbf{h})$",
    color=COL_BOX2)
for ex in exp_xs:
    arrow(ax, ex + 0.02, Y_EXP + BH / 2, z_x - 0.04, Y3 - BH * 0.2)

# Top-1 selection label
ax.text(0.06, Y3 - 0.17, "top-1 gate:\n$k^*{=}\\arg\\max g_k$\n(one expert active)",
        ha="center", va="center", fontsize=FONT_SIZE - 1,
        color=COL_ARROW, fontfamily="DejaVu Sans")

# ---- ROW 4: Output heads ------------------------------------------------
Y4 = -0.10
head_xs = [0.68, 0.80, 0.92]
head_labels = ["Binary\n$p_2$\n(2 classes)", "Category\n$p_6$\n(6 classes)",
               "Attack type\n$p_{19}$\n(19 classes)"]
for hx, hl in zip(head_xs, head_labels):
    box(ax, hx, Y4, 0.105, BH, hl, color=COL_HEAD)
    arrow(ax, z_x, Y3 - BH / 2, hx, Y4 + BH / 2)

# ---- Title label --------------------------------------------------------
ax.text(0.5, 0.97, "TRACE-MoE Data-Flow Architecture",
        ha="center", va="top", fontsize=FONT_SIZE + 1,
        fontweight="bold", fontfamily="DejaVu Sans")

# ---- Section labels on the left -----------------------------------------
for y, lbl in [(Y1, "Input\nencoding"), (Y2, "Sequence\nencoder"),
               (Y_EXP, "MoE layer"), (Y4, "Output\nheads")]:
    ax.text(0.00, y, lbl, ha="left", va="center", fontsize=FONT_SIZE - 1,
            color="#666666", fontfamily="DejaVu Sans",
            rotation=0)

plt.tight_layout(pad=0.1)
fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", format="pdf")
fig.savefig(OUT_TIF, dpi=DPI, bbox_inches="tight", format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_TIF}")
