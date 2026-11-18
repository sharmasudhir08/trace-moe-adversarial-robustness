"""Paper 2, S2: class-wise (per attack type) robustness, from saved artifacts only (no attack, no training).

Reads the per-class F1 stored in results_v3_paper2/robustness/<model>_official_T16_seed<s>/metrics.json
(curve_clean_model and curve_adv_finetuned, label19, every epsilon) and averages over the five seeds.
Writes to results_v3_paper2/saved_artifact_analyses_20261118/:
  s2_classwise_long.csv        model, curve, eps, class, mean_f1, sd_f1, mean_support, n_seeds
  s2_classwise_eps050_ft.csv   wide table: class by model, mean F1 at eps 0.50 after adversarial fine-tuning
  s2_classwise_summary.md      TRACE-MoE classes with the lowest F1 at eps 0.25 and 0.50, and support counts
Classes with no windows in the 4,905-window evaluation set are skipped. Up to 263 windows per class, so rare-class
values are noisy and should be read with the support column.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_v3_paper2"
OUT = R / "saved_artifact_analyses_20261118"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["trace_moe", "transformer", "bilstm_dnn", "tfsage_lite"]
EPS = ["0.0", "0.05", "0.1", "0.25", "0.5"]
CURVES = {"clean": "curve_clean_model", "ft": "curve_adv_finetuned"}

def run_dir(m, s):
    n = f"{m}_official_T16_seed{s}"
    d = R / "robustness" / n
    v2 = R / "robustness" / (n + "__v2")
    return v2 if (s == 42 and v2.exists()) else d

recs = []
for m in MODELS:
    for s in range(42, 47):
        j = json.load(open(run_dir(m, s) / "metrics.json"))
        for cname, key in CURVES.items():
            for e in EPS:
                pc = j[key][e]["label19"]["per_class"]
                for cls, d in pc.items():
                    if d.get("support", 0) > 0:
                        recs.append(dict(model=m, curve=cname, eps=float(e), cls=cls, seed=s, f1=d["f1"], support=d["support"]))
df = pd.DataFrame(recs)
g = df.groupby(["model", "curve", "eps", "cls"]).agg(mean_f1=("f1", "mean"), sd_f1=("f1", lambda x: x.std(ddof=1)),
                                                     mean_support=("support", "mean"), n_seeds=("seed", "nunique")).reset_index()
g.to_csv(OUT / "s2_classwise_long.csv", index=False)
w = g[(g.curve == "ft") & (g.eps == 0.5)].pivot(index="cls", columns="model", values="mean_f1")[MODELS].round(3)
w.to_csv(OUT / "s2_classwise_eps050_ft.csv")
lines = ["# S2: class-wise robustness after adversarial fine-tuning (mean F1 over five seeds)\n"]
for e in (0.25, 0.5):
    t = g[(g.model == "trace_moe") & (g.curve == "ft") & (g.eps == e)].sort_values("mean_f1").head(6)
    lines.append(f"\n## TRACE-MoE, eps {e}: six lowest classes\n")
    lines.append(t[["cls", "mean_f1", "sd_f1", "mean_support"]].round(3).to_markdown(index=False))
lines.append("\n## Class-wise F1 at eps 0.50 after fine-tuning, all models\n")
lines.append(w.to_markdown())
(OUT / "s2_classwise_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("wrote", OUT)
