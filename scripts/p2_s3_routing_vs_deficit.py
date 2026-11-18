"""Paper 2, S3 (optional, exploratory): link between router instability and the eps 0.25 deficit, from saved artifacts only.

Instability: results_v3_paper2/e4_routing_stability/routing_instability.json (clean-trained TRACE-MoE, eps 0.25, per seed).
Deficit: after adversarial fine-tuning, F1 at eps 0.25 of the best other architecture minus TRACE-MoE, per seed
(positive = TRACE-MoE is behind). Both are read from stored files; n = 5 seeds, so this is descriptive only.
Writes results_v3_paper2/saved_artifact_analyses_20261118/s3_routing_vs_deficit.{csv,md}.
Note: instability is measured on clean-trained models, the deficit on fine-tuned models, so this is an indirect link at best.
"""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_v3_paper2"
OUT = R / "saved_artifact_analyses_20261118"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["trace_moe", "transformer", "bilstm_dnn", "tfsage_lite"]
SEEDS = range(42, 47)

def f1_ft_025(m, s):
    n = f"{m}_official_T16_seed{s}"
    v2 = R / "robustness" / (n + "__v2")
    d = v2 if (s == 42 and v2.exists()) else R / "robustness" / n
    return json.load(open(d / "metrics.json"))["curve_adv_finetuned"]["0.25"]["label19"]["macro_f1"]

ri = json.load(open(R / "e4_routing_stability" / "routing_instability.json"))
rows = []
for s in SEEDS:
    f = {m: f1_ft_025(m, s) for m in MODELS}
    best_other = max(f[m] for m in MODELS if m != "trace_moe")
    rank = 1 + sum(f[m] > f["trace_moe"] for m in MODELS if m != "trace_moe")
    rows.append(dict(seed=s, instability_eps025=ri[str(s)]["0.25"]["instability_rate"], f1_trace_moe=f["trace_moe"],
                     f1_best_other=best_other, deficit=best_other - f["trace_moe"], rank=rank))
df = pd.DataFrame(rows)
df.to_csv(OUT / "s3_routing_vs_deficit.csv", index=False)
rho, _ = spearmanr(df.instability_eps025, df.deficit)
r_p, _ = pearsonr(df.instability_eps025, df.deficit)
# exact permutation p-value for the Spearman coefficient (n = 5: 120 permutations), two-sided
x = df.instability_eps025.rank().to_numpy(); y = df.deficit.rank().to_numpy()
perm = [np.corrcoef(x, np.array(p))[0, 1] for p in itertools.permutations(y)]
p_exact = float(np.mean(np.abs(perm) >= abs(np.corrcoef(x, y)[0, 1]) - 1e-12))
md = ["# S3 (exploratory): router instability versus the eps 0.25 deficit, five seeds\n", df.round(4).to_markdown(index=False),
      f"\nSpearman rho = {rho:.3f} (exact two-sided permutation p = {p_exact:.3f}); Pearson r = {r_p:.3f}. n = 5, descriptive only.\n",
      "Instability is from clean-trained models, the deficit from fine-tuned models."]
(OUT / "s3_routing_vs_deficit.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("wrote", OUT)
