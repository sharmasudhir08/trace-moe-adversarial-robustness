"""
gen_fig2_workflow.py
====================
Fig 2 for Paper 2: adversarial robustness evaluation workflow (re-laid out for the printed width).

Outputs: figures/Fig2_workflow.pdf (vector) and figures/Fig2_workflow.tif (600 dpi).
Drawn at the printed width (5.49 in = text width, included without scaling); all text is at least 8 pt at that size.
Previous version: gen_fig2_workflow.py.backup_20261118 (7 pt and 6 pt text at 7.0 in width).
No result numbers or verdicts appear in the figure (the 4,905 evaluation windows are a design quantity).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIG_DIR, "Fig2_workflow.pdf")
OUT_TIF = os.path.join(FIG_DIR, "Fig2_workflow.tif")

W, H, DPI = 5.49, 6.35, 600
COL_DATA, COL_TRAIN, COL_EVAL, COL_HYP, COL_PCRP = "#D0E4F7", "#E8E8E8", "#E8F5E9", "#FFF3E0", "#F3E5F5"
COL_ARROW, COL_EDGE = "#2255AA", "#555555"
FS, FS_SUB = 8.5, 8.2
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(W, 5.75), dpi=DPI)
ax.set_position([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, sub=None, color=COL_TRAIN):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                linewidth=0.6, edgecolor=COL_EDGE, facecolor=color, zorder=2))
    if sub:
        extra = 0.08 * max(0, sub.count(chr(10)) - 1)
        ax.text(x, y - 0.14 - extra, title, ha="center", va="center", fontsize=FS, fontweight="bold", zorder=3)
        ax.text(x, y + 0.12 + extra * 0.5, sub, ha="center", va="center", fontsize=FS_SUB, style="italic", linespacing=1.15, zorder=3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=FS, fontweight="bold", zorder=3)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=4)


def note(x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FS_SUB, color="#444444", linespacing=1.1)


# in-figure title removed (the caption carries it)

# data and split
box(W / 2, 0.68, 5.0, 0.62, "CICIoMT2024 (Wi-Fi + MQTT)", "45 features, 19 classes, seeds 42 to 46, 40 IoMT devices", COL_DATA)
arrow(W / 2, 0.99, W / 2, 1.2)
note(W / 2 + 0.08, 1.095, "file-level split")
box(W / 2, 1.55, 5.0, 0.7, "Train / validation / test split",
    "80% of capture files (85% of rows train, 15% validation),\n20% of capture files test; deduplicated", COL_DATA)

# three branches
XT, XP, XE = 1.0, 3.0, 4.65
y3 = 2.62
for xx, lab, ha in [(XT, "train windows", "left"), (XP, "train windows", "left"), (XE, "test windows", "right")]:
    arrow(xx, 1.9, xx, y3 - 0.33)
note(XT + 0.06, 2.1, "train windows")
note(XP + 0.06, 2.1, "train windows")
note(XE + 0.06, 2.1, "test\nwindows")
box(XT, y3, 1.92, 0.66, "Train four architectures", "TRACE-MoE, Transformer,\nBiLSTM-DNN, TF-SAGE-lite", COL_TRAIN)
box(XP, y3, 1.55, 0.66, "PCRP fitting", "Gaussian profiles,\n6 traffic categories", COL_PCRP)
box(XE, y3, 1.5, 0.66, "Test set (held out)", "4,905 windows,\nclass-balanced", COL_DATA)

# adversarial fine-tuning
y4 = 3.8
box(1.95, y4, 2.4, 0.92, "Adversarial fine-tuning", "3 epochs, 5-step PGD,\n$\\varepsilon{\\sim}\\mathrm{Uniform}[0,0.25]$,\n18 mutable features", COL_TRAIN)
arrow(1.5, y3 + 0.33, 1.5, y4 - 0.46)

# evaluation
y5 = 4.95
box(W / 2, y5, 5.0, 0.78, "Robustness evaluation",
    "$\\varepsilon \\in \\{0, 0.05, 0.10, 0.25, 0.50\\}$, macro-F1 curves, PCRP discrepancy;\nH6: 5-step PGD on a 1% subsample", COL_EVAL)
arrow(0.6, y3 + 0.33, 0.6, y5 - 0.39)
note(0.66, 4.45, "clean model")
arrow(1.95, y4 + 0.46, 1.95, y5 - 0.39)
note(2.01, 4.42, "fine-tuned model")
arrow(XP + 0.3, y3 + 0.33, XP + 0.3, y5 - 0.39)
arrow(XE, y3 + 0.33, XE, y5 - 0.39)

# hypotheses
y6 = 5.88
box(1.9, y6, 3.2, 0.5, "Hypothesis tests  H1 to H6", None, COL_HYP)
arrow(1.9, y5 + 0.39, 1.9, y6 - 0.25)

legend_items = [mpatches.Patch(facecolor=c, edgecolor=COL_EDGE, label=l) for c, l in
                [(COL_DATA, "Data"), (COL_TRAIN, "Model training"), (COL_PCRP, "PCRP fingerprint"),
                 (COL_EVAL, "Evaluation"), (COL_HYP, "Hypothesis tests")]]
ax.legend(handles=legend_items, loc="center right", bbox_to_anchor=(1.0, 0.07), fontsize=FS_SUB, framealpha=0.9,
          edgecolor=COL_EDGE, ncol=1, handlelength=1.2, labelspacing=0.25, borderpad=0.4)

fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.03, format="pdf")
fig.savefig(OUT_TIF, dpi=DPI, bbox_inches="tight", pad_inches=0.03, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("Saved:", OUT_PDF)
print("Saved:", OUT_TIF)
