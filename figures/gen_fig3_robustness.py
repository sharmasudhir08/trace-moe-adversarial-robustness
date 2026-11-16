"""
gen_fig3_robustness.py
======================
Generate Fig 3 for Paper 2: adversarial robustness curves (clean model and
adv-FT model), 19-class macro-F1 vs PGD epsilon.

Output files
------------
  documents_v2/paper2/figures/Fig3_robustness.pdf  (vector, for LaTeX)
  documents_v2/paper2/figures/Fig3_robustness.tif  (600 DPI, for Springer)

Design constraints (Springer Nature / Cluster Computing)
---------------------------------------------------------
  Width : 174 mm (double-column)
  Height: 90 mm
  DPI   : 600
  Rule  : NO result numbers in legend; NO verdict outcomes.
          Legend shows model display names only.
          Metric and label level (19-class macro-F1) on y-axis label.

Data source
-----------
  results_v3_paper2/robustness/{model}_official_T16_seed{k}[__v2]/metrics.json
  Keys: curve_clean_model, curve_adv_finetuned
  Sub-keys: "0.0", "0.05", "0.1", "0.25", "0.5"
            -> label19.macro_f1

AUC formula (from scripts/analyze_h5_ablation.py)
--------------------------------------------------
  EPS = [0.0, 0.05, 0.1, 0.25, 0.5]
  auc = np.trapezoid(f1s, EPS) / (EPS[-1] - EPS[0])

Table 4 reference values (adv-FT panel, mean over seeds 42-46)
---------------------------------------------------------------
  TRACE-MoE      : 0.6943
  TF-SAGE-lite   : 0.6852
  Transformer    : 0.6756
  BiLSTM-DNN     : 0.6081
"""

import json
import os
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(r"D:\Sharda University\CICIoMT2024\trace-iomt")
RB   = ROOT / "results_v3_paper2" / "robustness"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_PDF = FIG_DIR / "Fig3_robustness.pdf"
OUT_TIF = FIG_DIR / "Fig3_robustness.tif"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS     = [0.0, 0.05, 0.1, 0.25, 0.5]
EPS_STR = ["0.0", "0.05", "0.1", "0.25", "0.5"]
SEEDS   = [42, 43, 44, 45, 46]

# (internal_name, display_name, search_prefix)
MODELS = [
    ("trace_moe",   "TRACE-MoE",    "trace_moe_official_T16"),
    ("transformer", "Transformer",  "transformer_official_T16"),
    ("bilstm_dnn",  "BiLSTM-DNN",   "bilstm_dnn_official_T16"),
    ("tfsage_lite", "TF-SAGE-lite", "tfsage_lite_official_T16"),
]

# Table 4 reference AUC-robustness (adv-FT, mean over seeds 42-46)
TABLE4_AUC = {
    "trace_moe":   0.6943,
    "transformer": 0.6756,
    "bilstm_dnn":  0.6081,
    "tfsage_lite": 0.6852,
}

# Colour palette (4-colour, accessible, greyscale-distinguishable)
COLOURS = {
    "trace_moe":   "#1A6FBF",   # strong blue
    "transformer": "#E37222",   # orange
    "bilstm_dnn":  "#3DAA55",   # green
    "tfsage_lite": "#9B1E8E",   # purple
}
LINESTYLES = {
    "trace_moe":   "-",
    "transformer": "--",
    "bilstm_dnn":  "-.",
    "tfsage_lite": ":",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_run_dir(prefix: str, seed: int) -> pathlib.Path | None:
    """Return the run directory matching <prefix>_seed<seed>[optional suffix]."""
    seed_str = f"_seed{seed}"
    for d in RB.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(prefix + seed_str):
            return d
    return None


def load_curves(mname: str, prefix: str):
    """Load per-epsilon macro-F1 for clean and adv-FT models across all seeds.

    Returns
    -------
    clean : list of list[float]  (n_seeds × 5)
    adv   : list of list[float]  (n_seeds × 5)
    """
    clean_all, adv_all = [], []
    missing = []
    for seed in SEEDS:
        d = find_run_dir(prefix, seed)
        if d is None:
            missing.append(seed)
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            missing.append(seed)
            continue
        with open(mj, encoding="utf-8") as fh:
            m = json.load(fh)
        clean_all.append([m["curve_clean_model"][e]["label19"]["macro_f1"]
                          for e in EPS_STR])
        adv_all.append([m["curve_adv_finetuned"][e]["label19"]["macro_f1"]
                        for e in EPS_STR])
    if missing:
        print(f"  WARNING: {mname} missing seeds {missing}", file=sys.stderr)
    return clean_all, adv_all


def auc_robustness(f1s: list[float]) -> float:
    """Trapezoidal AUC normalised by epsilon range (matches analyze_h5_ablation.py)."""
    return float(np.trapezoid(f1s, EPS)) / (EPS[-1] - EPS[0])


# ---------------------------------------------------------------------------
# Load data and compute statistics
# ---------------------------------------------------------------------------
print("Loading per-epsilon curves ...")
data = {}
for mname, dname, prefix in MODELS:
    clean_all, adv_all = load_curves(mname, prefix)
    n = len(clean_all)
    if n == 0:
        print(f"  ERROR: no data for {mname}", file=sys.stderr)
        sys.exit(1)

    clean_arr = np.array(clean_all)   # (n, 5)
    adv_arr   = np.array(adv_all)

    # Per-seed AUC-robustness (adv-FT panel)
    auc_per_seed = [auc_robustness(adv_arr[i]) for i in range(n)]
    mean_auc  = float(np.mean(auc_per_seed))
    ref_auc   = TABLE4_AUC[mname]
    diff      = abs(mean_auc - ref_auc)

    print(f"  {dname:14s}  n={n}  "
          f"AUC-adv computed={mean_auc:.4f}  "
          f"Table4={ref_auc:.4f}  |diff|={diff:.4f}"
          + ("  <<< MISMATCH" if diff > 0.0010 else "  OK"))

    data[mname] = {
        "display": dname,
        "clean_mean": clean_arr.mean(axis=0),
        "clean_sd":   clean_arr.std(axis=0, ddof=1) if n > 1 else np.zeros(5),
        "adv_mean":   adv_arr.mean(axis=0),
        "adv_sd":     adv_arr.std(axis=0, ddof=1) if n > 1 else np.zeros(5),
        "mean_auc":   mean_auc,
        "ref_auc":    ref_auc,
        "n_seeds":    n,
    }

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
FIG_WIDTH_INCH  = 174 / 25.4
FIG_HEIGHT_INCH =  90 / 25.4
DPI = 600
FONT_SIZE = 7

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=DPI,
                         sharey=True)
fig.patch.set_facecolor("white")

PANEL_TITLES = ["(a) Clean model", "(b) After adversarial fine-tuning"]
CURVE_KEYS   = [("clean_mean", "clean_sd"), ("adv_mean", "adv_sd")]

for ax, title, (mk, sk) in zip(axes, PANEL_TITLES, CURVE_KEYS):
    ax.set_facecolor("white")
    for mname, dname, _ in MODELS:
        d    = data[mname]
        mean = d[mk]
        sd   = d[sk]
        col  = COLOURS[mname]
        ls   = LINESTYLES[mname]
        ax.plot(EPS, mean, color=col, linestyle=ls, linewidth=1.2,
                label=d["display"])
        ax.fill_between(EPS, mean - sd, mean + sd,
                        color=col, alpha=0.12, linewidth=0)

    ax.set_title(title, fontsize=FONT_SIZE, fontweight="bold",
                 fontfamily="DejaVu Sans", pad=3)
    ax.set_xlabel("PGD perturbation budget $\\varepsilon$",
                  fontsize=FONT_SIZE, fontfamily="DejaVu Sans")
    ax.set_xticks(EPS)
    ax.set_xticklabels([str(e) for e in EPS], fontsize=FONT_SIZE - 1)
    ax.yaxis.set_tick_params(labelsize=FONT_SIZE - 1)
    ax.set_ylim(0.25, 0.85)
    ax.set_xlim(-0.01, 0.52)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc")
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("19-class macro-F1", fontsize=FONT_SIZE,
                   fontfamily="DejaVu Sans")

# Single shared legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.04),
           ncol=4,
           fontsize=FONT_SIZE,
           framealpha=0.9,
           edgecolor="#aaaaaa",
           handlelength=2.0)

# suptitle removed: title information is in the LaTeX caption

plt.tight_layout(pad=0.3, rect=[0, 0.08, 1, 1])

fig.savefig(str(OUT_PDF), dpi=DPI, bbox_inches="tight", format="pdf")
fig.savefig(str(OUT_TIF), dpi=DPI, bbox_inches="tight", format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_TIF}")
