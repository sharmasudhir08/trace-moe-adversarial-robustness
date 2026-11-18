"""
gen_fig1_dataflow.py
====================
Fig 1 for Paper 2: TRACE-MoE data-flow architecture diagram (re-laid out for the printed width).

Outputs: figures/Fig1_dataflow.pdf (vector) and figures/Fig1_dataflow.tif (600 dpi).
The figure is drawn at its printed width (5.49 in = text width, included without scaling), and every text
element is at least 8 pt at that size. The previous version (backup: gen_fig1_dataflow.py.backup_20261118)
used 7 pt and 6 pt text at 6.9 in width, i.e. about 5.5 pt and 4.7 pt when printed.

Architecture details (src/trace_iomt/models/networks.py and configs/): T=16 flows, F=45 features, N_P=5 protocol
tokens; FeatureTokenizer d_t=16; AttentionPool over F; step embedding concat -> Linear(61,128)+LN+GELU;
protocol embedding dim 128 (32 for the router); BiGRU(64 units/direction) -> Transformer (2 layers, H=4, ff=256,
pre-LN) -> final pool AttentionPool(T)+last step -> Linear(256,128)+GELU -> h in R^128; router Linear(160,4)+softmax;
top-1: k* = argmax g, z = h + g_k* Ex_k*(h); expert LN, Lin(256), GELU, Lin(128); heads Linear(128,2/6/19).
No results, verdicts, file names or hypothesis outcomes appear in the figure.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIG_DIR, "Fig1_dataflow.pdf")
OUT_TIF = os.path.join(FIG_DIR, "Fig1_dataflow.tif")

W, H, DPI = 5.49, 7.45, 600
COL_BOX, COL_BOX2, COL_ARROW, COL_HEAD, COL_EDGE = "#E8E8E8", "#D0E4F7", "#2255AA", "#F5E6CC", "#555555"
FS = 8.5       # box titles
FS_SUB = 8.2   # box sub-labels and annotations (never below 8 pt)
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(W, 6.6), dpi=DPI)
ax.set_position([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)       # y grows downward, units are inches
ax.axis("off")
fig.patch.set_facecolor("white")
BH = 0.62


def box(x, y, w, title, sub=None, color=COL_BOX, h=BH):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                linewidth=0.6, edgecolor=COL_EDGE, facecolor=color, zorder=2))
    if sub:
        ax.text(x, y - 0.13, title, ha="center", va="center", fontsize=FS, fontweight="bold", zorder=3)
        ax.text(x, y + 0.13, sub, ha="center", va="center", fontsize=FS_SUB, style="italic", linespacing=1.15, zorder=3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=FS, fontweight="bold", zorder=3)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=4)


def heading(y, text):
    ax.text(0.02, y, text, ha="left", va="center", fontsize=FS_SUB, color="#666666", fontweight="bold")


BW = 1.62
X = [0.83, 2.745, 4.66]          # three-column centres

# in-figure title removed (the caption carries it)

# ---- Input encoding -----------------------------------------------------
heading(0.45, "Input encoding")
y1 = 0.98
box(X[0], y1, BW, "Input window", "$T{=}16$ flows,\n$F{=}45$ features")
box(X[1], y1, BW, "FeatureTokenizer", "per-feature affine,\n$d_t{=}16$")
box(X[2], y1, BW, "AttentionPool", "over $F{=}45$ tokens")
arrow(X[0] + BW / 2, y1, X[1] - BW / 2, y1)
arrow(X[1] + BW / 2, y1, X[2] - BW / 2, y1)
y2 = 1.95
box(X[0], y2, BW, "Protocol emb.", "vocab $N_P{=}5$,\ndim 128")
box(X[1], y2, BW, "Step projection", "concat, Lin(61,128),\nLN, GELU")
arrow(X[2] - 0.3, y1 + BH / 2, X[1] + 0.45, y2 - BH / 2)
arrow(X[0] + BW / 2, y2, X[1] - BW / 2, y2)

# ---- Sequence encoder ---------------------------------------------------
heading(2.5, "Sequence encoder")
y3 = 3.03
box(X[0], y3, BW, "BiGRU", "64 units/direction,\n$T{\\times}128$ output")
box(X[1], y3, BW, "Transformer", "2 layers, $H{=}4$ heads,\n$d_{\\mathrm{ff}}{=}256$, pre-LN")
box(X[2], y3, BW, "Final pool", "AttentionPool, last step,\nLin(256,128), GELU")
arrow(X[1] - 0.45, y2 + BH / 2, X[0] + 0.3, y3 - BH / 2)
ax.text(X[0] + 1.18, (y2 + y3) / 2 + 0.02, "$T$ step embeddings", ha="left", va="center", fontsize=FS_SUB, color="#444444")
arrow(X[0] + BW / 2, y3, X[1] - BW / 2, y3)
arrow(X[1] + BW / 2, y3, X[2] - BW / 2, y3)

# ---- MoE layer ----------------------------------------------------------
heading(3.62, "MoE layer (top-1 gate: $k^{*}{=}\\arg\\max_k g_k$, one expert active)")
y4 = 4.15
box(X[2], y4, 1.1, "$\\mathbf{h}$", "$\\in \\mathbb{R}^{128}$", color=COL_BOX2)
box(X[1], y4, BW, "Router", "Lin(160,4), softmax", color=COL_BOX2)
box(X[0], y4, BW, "Protocol emb.", "$c_T{\\to}\\mathbb{R}^{32}$")
arrow(X[2], y3 + BH / 2, X[2], y4 - BH / 2)
arrow(X[2] - 0.55, y4, X[1] + BW / 2, y4)
arrow(X[0] + BW / 2, y4, X[1] - BW / 2, y4)

y5 = 5.05
EXW = 1.2
EXC = [0.62, 2.0, 3.38, 4.76]
for j, ex in enumerate(EXC):
    box(ex, y5, EXW, f"$\\mathrm{{Ex}}_{j + 1}$", "LN, Lin(256),\nGELU, Lin(128)")
    arrow(X[1], y4 + BH / 2, ex, y5 - BH / 2)
y6 = 5.9
ZW = 2.6
box(X[1], y6, ZW, "$\\mathbf{z}$", "$\\mathbf{h}+g_{k^{*}}\\,\\mathrm{Ex}_{k^{*}}(\\mathbf{h})$", color=COL_BOX2, h=0.56)
for ex in EXC:
    arrow(ex, y5 + BH / 2, X[1] + (ex - X[1]) * 0.3, y6 - 0.28)

# ---- Output heads -------------------------------------------------------
heading(6.4, "Output heads")
y7 = 6.95
for hx, t, sub in zip(X, ["Binary", "Category", "Attack type"], ["$p_{2}$, 2 classes", "$p_{6}$, 6 classes", "$p_{19}$, 19 classes"]):
    box(hx, y7, BW, t, sub, color=COL_HEAD)
    arrow(X[1] + (hx - X[1]) * 0.3, y6 + 0.28, hx, y7 - BH / 2)

fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.03, format="pdf")
fig.savefig(OUT_TIF, dpi=DPI, bbox_inches="tight", pad_inches=0.03, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("Saved:", OUT_PDF)
print("Saved:", OUT_TIF)
