"""Paper 2, S1: PCRP detection at fixed false-positive rates, from saved artifacts only (no attack, no training).

Reads results_v3_paper2/e5_pcrp_variants/pcrp_variants.json, which already stores roc_auc, tpr_at_1pct_fpr and
tpr_at_5pct_fpr per model, seed, epsilon and variant (predicted, y_true = oracle, min_c = nearest centroid).
Writes to results_v3_paper2/saved_artifact_analyses_20261118/:
  s1_pcrp_tpr_summary.csv   mean and sample SD over the five seeds, every model, eps and variant
  s1_pcrp_tpr_table.tex     draft LaTeX table (TPR at 1% FPR, eps 0.25 and 0.50) for the author to review
  s1_check_against_table15.txt   comparison of the mean ROC-AUC with the values printed in the manuscript table
Nothing in the manuscript is changed by this script.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_v3_paper2"
OUT = R / "saved_artifact_analyses_20261118"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["trace_moe", "transformer", "bilstm_dnn", "tfsage_lite"]
NAMES = {"trace_moe": "TRACE-MoE", "transformer": "Transformer", "bilstm_dnn": "BiLSTM-DNN", "tfsage_lite": "TF-SAGE-lite"}
VARS = ["predicted", "y_true", "min_c"]
EPS = ["0.05", "0.1", "0.25", "0.5"]

pv = json.load(open(R / "e5_pcrp_variants" / "pcrp_variants.json"))
rows = []
for m in MODELS:
    seeds = sorted(pv[m].keys())
    for e in EPS:
        for v in VARS:
            vals = {k: np.array([pv[m][s][e][v][k] for s in seeds], float) for k in ("roc_auc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr")}
            rows.append(dict(model=m, eps=float(e), variant=v, n_seeds=len(seeds),
                             **{f"{k}_mean": a.mean() for k, a in vals.items()},
                             **{f"{k}_sd": a.std(ddof=1) for k, a in vals.items()}))
df = pd.DataFrame(rows)
df.to_csv(OUT / "s1_pcrp_tpr_summary.csv", index=False)

# draft LaTeX table: TPR at 1% FPR, eps 0.25 and 0.50
def cell(m, e, v):
    r = df[(df.model == m) & (df.eps == float(e)) & (df.variant == v)].iloc[0]
    return f"{r.tpr_at_1pct_fpr_mean:.3f}"
tex = ["\\begin{table}[tbp]",
       "\\caption{Mean true-positive rate of the three PCRP detector variants at a 1\\% false-positive rate on clean windows, over five seeds (5-step PGD, 4,905 evaluation windows).}",
       "\\label{tab:pcrp_tpr}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccc@{}}", "\\toprule",
       " & \\multicolumn{3}{c}{$\\varepsilon = 0.25$} & \\multicolumn{3}{c}{$\\varepsilon = 0.50$} \\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Model & predicted & oracle & nearest & predicted & oracle & nearest \\\\", "\\midrule"]
for m in MODELS:
    tex.append(f"{NAMES[m]} & " + " & ".join(cell(m, e, v) for e in ("0.25", "0.5") for v in VARS) + " \\\\")
tex += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
(OUT / "s1_pcrp_tpr_table.tex").write_text("\n".join(tex), encoding="utf-8")

# consistency with the ROC-AUC values printed in the manuscript (Table 15)
paper = {"trace_moe": [0.786, 0.783, 0.981, 0.854, 0.853, 0.992], "transformer": [0.788, 0.786, 0.981, 0.857, 0.854, 0.992],
         "bilstm_dnn": [0.783, 0.786, 0.986, 0.861, 0.864, 0.994], "tfsage_lite": [0.778, 0.776, 0.980, 0.847, 0.845, 0.992]}
lines = []
for m in MODELS:
    got = [round(float(df[(df.model == m) & (df.eps == float(e)) & (df.variant == v)].roc_auc_mean.iloc[0]), 3) for e in ("0.25", "0.5") for v in VARS]
    lines.append(f"{m}: file {got} | manuscript {paper[m]} | {'MATCH' if got == paper[m] else 'DIFFERENT'}")
(OUT / "s1_check_against_table15.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(df[(df.eps.isin([0.25, 0.5]))][["model", "eps", "variant", "tpr_at_1pct_fpr_mean", "tpr_at_5pct_fpr_mean"]].round(3).to_string(index=False))
print("wrote", OUT)
