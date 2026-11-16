"""
gen_fig5_cost_robustness.py
===========================
Generate Fig 5 for Paper 2: compute cost vs adversarial robustness.

Data source (manuscript tables only — no new run)
-------------------------------------------------
  Table 1 (tab:architectures):  MFLOPs/window
    TRACE-MoE    : 11.54
    Transformer  :  8.84
    BiLSTM-DNN   :  3.07
    TF-SAGE-lite :  2.51

  Table 4 (tab:h2h3):  AUC-robustness mean ± std over 5 seeds
    Clean-trained:
      TRACE-MoE    : 0.636 ± 0.003
      TF-SAGE-lite : 0.648 ± 0.022
      Transformer  : 0.578 ± 0.036
      BiLSTM-DNN   : 0.486 ± 0.032
    Adv-FT:
      TRACE-MoE    : 0.694 ± 0.014
      TF-SAGE-lite : 0.685 ± 0.017
      Transformer  : 0.676 ± 0.014
      BiLSTM-DNN   : 0.608 ± 0.014

Output files
------------
  documents_v2/paper2/figures/Fig5_cost_robustness.pdf
  documents_v2/paper2/figures/Fig5_cost_robustness.tif

Design (Springer Nature / Cluster Computing)
--------------------------------------------
  Width : 88 mm (single-column)
  DPI   : 600
  Rule  : NO result numbers in figure body; NO verdict outcomes.
          Hollow marker = clean model; filled marker = adv-FT model.
          Arrow from hollow to filled per model.
          ±1 SD vertical error bars on both points.
"""

import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_PDF = FIG_DIR / "Fig5_cost_robustness.pdf"
OUT_TIF = FIG_DIR / "Fig5_cost_robustness.tif"

# ---------------------------------------------------------------------------
# Data — EXACT values from manuscript Tables 1 and 4 only
# ---------------------------------------------------------------------------
MODELS = [
    # (display_name, MFLOPs, auc_clean, sd_clean, auc_adv, sd_adv)
    ("TRACE-MoE",    11.54, 0.636, 0.003, 0.694, 0.014),
    ("Transformer",   8.84, 0.578, 0.036, 0.676, 0.014),
    ("BiLSTM-DNN",    3.07, 0.486, 0.032, 0.608, 0.014),
    ("TF-SAGE-lite",  2.51, 0.648, 0.022, 0.685, 0.017),
]

# Colour per model (same palette as Fig 3)
COLOURS = {
    "TRACE-MoE":    "#1A6FBF",
    "Transformer":  "#E37222",
    "BiLSTM-DNN":   "#3DAA55",
    "TF-SAGE-lite": "#9B1E8E",
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_WIDTH_INCH  = 88 / 25.4   # single-column
FIG_HEIGHT_INCH = 80 / 25.4
DPI = 600
FONT_SIZE = 7

fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=DPI)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for name, flops, auc_c, sd_c, auc_a, sd_a in MODELS:
    col = COLOURS[name]

    # Arrow from clean (hollow) to adv-FT (filled)
    ax.annotate(
        "", xy=(flops, auc_a), xytext=(flops, auc_c),
        arrowprops=dict(
            arrowstyle="-|>",
            color=col,
            lw=0.9,
            mutation_scale=6,
            shrinkA=5,
            shrinkB=5,
        ),
        zorder=2,
    )

    # Clean point (hollow)
    ax.errorbar(flops, auc_c, yerr=sd_c,
                fmt="o", markersize=6,
                markerfacecolor="white",
                markeredgecolor=col,
                markeredgewidth=1.2,
                ecolor=col, elinewidth=0.9, capsize=3,
                zorder=3)

    # Adv-FT point (filled)
    ax.errorbar(flops, auc_a, yerr=sd_a,
                fmt="o", markersize=6,
                markerfacecolor=col,
                markeredgecolor=col,
                markeredgewidth=1.2,
                ecolor=col, elinewidth=0.9, capsize=3,
                label=name,
                zorder=4)

# Axis styling
ax.set_xlabel("Compute cost (MFLOPs / window)",
              fontsize=FONT_SIZE, fontfamily="DejaVu Sans")
ax.set_ylabel("AUC-robustness",
              fontsize=FONT_SIZE, fontfamily="DejaVu Sans")
ax.set_xlim(1.5, 13.5)
ax.set_ylim(0.44, 0.72)
ax.tick_params(labelsize=FONT_SIZE - 1)
ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc")
ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#eeeeee")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend: label each model using the filled-point entry
ax.legend(fontsize=FONT_SIZE - 1, framealpha=0.9, edgecolor="#aaaaaa",
          loc="lower right", handlelength=1.0)

# Hollow / filled legend annotation
ax.text(0.03, 0.97,
        "○ Clean model   ● After adv. fine-tuning\n"
        "Arrows show improvement from adv-FT",
        transform=ax.transAxes,
        fontsize=FONT_SIZE - 1.5, va="top", ha="left",
        color="#444444", fontfamily="DejaVu Sans",
        linespacing=1.4)

plt.tight_layout(pad=0.4)
fig.savefig(str(OUT_PDF), dpi=DPI, bbox_inches="tight", format="pdf")
fig.savefig(str(OUT_TIF), dpi=DPI, bbox_inches="tight", format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_TIF}")
