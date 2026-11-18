"""Fig. 4: mean PCRP discrepancy (squared Mahalanobis distance) versus PGD budget.

Reads results_v3_paper2/twin_h6/<model>_official_T16_seed<s>_twin_h6_track.json (mean_D_attack per eps),
averages over seeds 42 to 46, shades +/-1 SD across seeds. Printed width equals the LaTeX width (no scaling).
Outputs figures/Fig4_pcrp.pdf (vector) and figures/Fig4_pcrp.tif (600 dpi).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "results_v3_paper2" / "twin_h6"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
MODELS = [("trace_moe", "TRACE-MoE", "#1b7837", "o"), ("transformer", "Transformer", "#7b2d8e", "s"),
          ("bilstm_dnn", "BiLSTM-DNN", "#d9534f", "^"), ("tfsage_lite", "TF-SAGE-lite", "#3b8fc4", "D")]
SEEDS = range(42, 47)
FONT = 9
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT})

fig, ax = plt.subplots(figsize=(4.94, 3.5), dpi=600)
eps = None
for key, name, col, mk in MODELS:
    vals = []
    for s in SEEDS:
        j = json.load(open(SRC / f"{key}_official_T16_seed{s}_twin_h6_track.json"))
        eps = [r["eps"] for r in j["per_eps"]]
        vals.append([r["mean_D_attack"] for r in j["per_eps"]])
    v = np.array(vals)
    m, sd = v.mean(0), v.std(0, ddof=1)
    ax.plot(eps, m, color=col, marker=mk, markersize=4, linewidth=1.2, label=name)
    ax.fill_between(eps, m - sd, m + sd, color=col, alpha=0.15, linewidth=0)
    print(name, np.round(m, 2))
ax.set_xticks(eps)
ax.set_xticklabels(["0", "0.05", "0.10", "0.25", "0.50"], fontsize=FONT - 1, rotation=45, ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", labelsize=FONT - 1)
ax.set_xlabel(r"PGD budget $\varepsilon$ ($\ell_\infty$)", fontsize=FONT)
ax.set_ylabel("Squared Mahalanobis distance", fontsize=FONT)
ax.grid(linestyle=":", linewidth=0.4, color="#cccccc")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=FONT - 1, frameon=True, framealpha=0.9, edgecolor="#aaaaaa", loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "Fig4_pcrp.pdf", dpi=600, bbox_inches="tight", format="pdf")
fig.savefig(FIG / "Fig4_pcrp.tif", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("saved")
