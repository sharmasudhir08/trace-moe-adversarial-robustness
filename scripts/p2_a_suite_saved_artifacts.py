"""Paper 2, analyses A1 to A7 from saved artifacts only (no training, no attack, no inference).

Inputs (all existing): results_v3_paper2/robustness/<model>_official_T16_seed<s>[__v2]/metrics.json,
saved_artifact_analyses_20261118/s2_classwise_long.csv, e4_routing_stability/routing_instability.json,
twin_h6/<model>_official_T16_seed<s>_twin_h6_track.json, e6_transfer_multi_eps/transfer_multi_eps.json.
Output: results_v3_paper2/saved_artifact_analyses_20261118/a_suite/ (a1..a7 csv and a_suite_summary.md).
Items that need per-window or per-class data that were not saved are reported as NOT COMPUTABLE, with the reason.
"""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, binomtest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_v3_paper2"
S = R / "saved_artifact_analyses_20261118"
OUT = S / "a_suite"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["trace_moe", "transformer", "bilstm_dnn", "tfsage_lite"]
SEEDS = range(42, 47)
EPS = ["0.0", "0.05", "0.1", "0.25", "0.5"]
md = []


def run_dir(m, s):
    n = f"{m}_official_T16_seed{s}"
    v2 = R / "robustness" / (n + "__v2")
    return v2 if (s == 42 and v2.exists()) else R / "robustness" / n


J = {(m, s): json.load(open(run_dir(m, s) / "metrics.json")) for m in MODELS for s in SEEDS}

# ---------------------------------------------------------------- A1 conditional ASR (bounds only)
md.append("# A1: conditional ASR (bounds; exact values need per-window clean and attacked predictions, which were not saved)\n")
md.append("ASR = 1 - attack_detection_rate on the attack windows. f = ASR at eps 0 (clean miss floor). Conditional ASR = P(evaded | clean-detected) lies in "
          "[(ASR-f)/(1-f), min(1, ASR/(1-f))]; the lower end is exact if no clean-missed window becomes detected under attack.\n")
rows = []
for cn, key in (("clean-trained", "curve_clean_model"), ("fine-tuned", "curve_adv_finetuned")):
    for m in MODELS:
        for e in EPS[1:]:
            asr = np.array([1 - J[(m, s)][key][e]["attack_detection_rate"] for s in SEEDS])
            f = np.array([1 - J[(m, s)][key]["0.0"]["attack_detection_rate"] for s in SEEDS])
            lo = (asr - f) / (1 - f)
            hi = np.minimum(1, asr / (1 - f))
            rows.append(dict(curve=cn, model=m, eps=float(e), asr_uncond=asr.mean(), clean_miss_floor=f.mean(),
                             cond_asr_lower=lo.mean(), cond_asr_upper=hi.mean(), floor_share_of_asr=(f / asr).mean()))
a1 = pd.DataFrame(rows)
a1.to_csv(OUT / "a1_conditional_asr_bounds.csv", index=False)
md.append(a1.round(3).to_markdown(index=False))

# ---------------------------------------------------------------- A2 per-class hardness
L = pd.read_csv(S / "s2_classwise_long.csv")
md.append("\n\n# A2: per-class degradation DeltaF1 = F1(eps 0) - F1(eps 0.50), mean over five seeds\n")
a2 = []
for cur in ("clean", "ft"):
    w = {}
    for m in MODELS:
        d = L[(L.model == m) & (L.curve == cur)].pivot(index="cls", columns="eps", values="mean_f1")
        w[m] = d[0.0] - d[0.5]
    D = pd.DataFrame(w)
    rk = D.rank(ascending=False)  # 1 = most degraded
    D["mean_delta"] = D[MODELS].mean(1)
    D["rank_spread"] = rk.max(1) - rk.min(1)
    D["rank_trace_moe"] = rk["trace_moe"]
    D["rank_others_mean"] = rk[[m for m in MODELS if m != "trace_moe"]].mean(1)
    D = D.sort_values("mean_delta", ascending=False)
    D.insert(0, "curve", cur)
    a2.append(D.reset_index())
    md.append(f"\n## {cur}-trained curve\n" if cur == "clean" else "\n## adversarially fine-tuned curve\n")
    md.append(D.drop(columns="curve").round(3).to_markdown())
    md.append(f"\nMost degraded (pooled mean): {', '.join(D.index[:3])}. Least degraded: {', '.join(D.index[-3:])}.")
    dis = D[(D.rank_trace_moe - D.rank_others_mean).abs() >= 5]
    md.append(f"TRACE-MoE rank differs from the mean rank of the other three by >= 5 places: {', '.join(dis.index) or 'none'}.")
pd.concat(a2).to_csv(OUT / "a2_class_degradation.csv", index=False)

# ---------------------------------------------------------------- A3 routing vs per-class drop
md.append("\n\n# A3: routing instability versus per-class F1 drop\n")
md.append("NOT COMPUTABLE per class: routing_instability.json stores instability_rate, jaccard and the expert distribution per seed and eps only, "
          "not per attack class (and only for clean-trained TRACE-MoE). The per-seed version (n = 5) was already reported as S3 "
          "(Spearman 0.80, exact p = 0.13). A per-class analysis needs the per-window router choices, which were not saved.\n")

# ---------------------------------------------------------------- A4 seed variance per class
md.append("\n# A4: across-seed SD of class F1 at eps 0.50\n")
a4 = []
for cur in ("clean", "ft"):
    V = L[(L.curve == cur) & (L.eps == 0.5)].pivot(index="cls", columns="model", values="sd_f1")[MODELS]
    V["mean_sd"] = V.mean(1)
    V = V.sort_values("mean_sd", ascending=False)
    top5, bot5 = list(V.index[:5]), list(V.index[-5:])
    per_model_top = {m: set(V[m].sort_values(ascending=False).index[:5]) for m in MODELS}
    common = set.intersection(*per_model_top.values())
    in_ge2 = [c for c in V.index if sum(c in per_model_top[m] for m in MODELS) >= 2]
    tcpip = [c for c in V.index if c.startswith("TCP_IP")]
    md.append(f"\n## {cur}-trained\n")
    md.append(V.round(3).to_markdown())
    md.append(f"\nMost variable (mean SD): {', '.join(top5)}. Least variable: {', '.join(bot5)}.")
    md.append(f"In the five most variable classes of every architecture: {sorted(common) or 'none'}; of at least two architectures: {in_ge2}.")
    md.append(f"TCP/IP classes among the top-5 pooled: {[c for c in top5 if c in tcpip]}.")
    V.insert(0, "curve", cur)
    a4.append(V.reset_index())
pd.concat(a4).to_csv(OUT / "a4_class_seed_sd.csv", index=False)

# ---------------------------------------------------------------- A5 PCRP score distributions
md.append("\n\n# A5: PCRP discrepancy distribution summary (attack windows, 5-step PGD, 1% subsample; saved: mean and SD per seed and eps)\n")
rows = []
for m in MODELS:
    for s in SEEDS:
        j = json.load(open(R / "twin_h6" / f"{m}_official_T16_seed{s}_twin_h6_track.json"))
        for r in j["per_eps"]:
            rows.append(dict(model=m, seed=s, eps=r["eps"], mean_D=r["mean_D_attack"], sd_D=r["std_D_attack"]))
P = pd.DataFrame(rows)
a5 = P.groupby(["model", "eps"]).agg(mean_D=("mean_D", "mean"), sd_D_pooled=("sd_D", "mean"), sd_of_seed_means=("mean_D", lambda x: x.std(ddof=1))).reset_index()
base = a5[a5.eps == 0.0].set_index("model")
a5["shift_in_mean"] = a5.apply(lambda r: r.mean_D - base.loc[r.model, "mean_D"], axis=1)
a5["shift_over_window_sd"] = a5.shift_in_mean / a5.sd_D_pooled
a5.to_csv(OUT / "a5_pcrp_distribution.csv", index=False)
md.append(a5[a5.eps.isin([0.0, 0.25, 0.5])].round(3).to_markdown(index=False))
md.append("\nThe saved files hold only mean and SD of D over windows, not the windows themselves, so overlap is characterised here by the shift in the mean "
          "relative to the window-level SD (last column); histograms need the per-window scores (the S6 export, once run).")

# ---------------------------------------------------------------- A6 transfer asymmetry
T = json.load(open(R / "e6_transfer_multi_eps" / "transfer_multi_eps.json"))
md.append("\n\n# A6: transfer asymmetry (ASR of surrogate A on target B minus ASR of B on A, per seed; one-sided exact Wilcoxon and sign test, n = 5 seeds)\n")
rows = []
for e in ("0.05", "0.1", "0.25", "0.5"):
    asr = {}
    for s in SEEDS:
        for r in T[str(s)][e]:
            if r["surrogate"] in MODELS and r["target"] in MODELS:
                asr[(s, r["surrogate"], r["target"])] = r["asr"]
    for A, B in itertools.combinations(MODELS, 2):
        d = np.array([asr[(s, A, B)] - asr[(s, B, A)] for s in SEEDS])
        hi, lo = (A, B) if d.mean() >= 0 else (B, A)
        dd = np.abs(d) * 1.0 if d.mean() >= 0 else -d
        try:
            pw = wilcoxon(dd, alternative="greater", method="exact").pvalue
        except ValueError:
            pw = float("nan")
        pos = int((dd > 0).sum())
        rows.append(dict(eps=float(e), higher_direction=f"{hi}->{lo}", mean_diff=abs(d.mean()), seeds_in_that_direction=pos,
                         wilcoxon_p_one_sided=pw, sign_p_one_sided=binomtest(pos, 5, 0.5, alternative="greater").pvalue))
a6 = pd.DataFrame(rows)
a6.to_csv(OUT / "a6_transfer_asymmetry.csv", index=False)
for e in (0.25, 0.5):
    t = a6[a6.eps == e].sort_values("mean_diff", ascending=False)
    md.append(f"\n## eps {e}\n")
    md.append(t.round(4).to_markdown(index=False))
md.append("\nWith five seeds the smallest attainable one-sided exact p is 0.031, and 6 pairs x 4 budgets are examined, so these are descriptive, not corrected.")

# ---------------------------------------------------------------- A7 clean F1 versus attacked F1 per class
md.append("\n\n# A7: clean F1 versus attacked F1 (eps 0.50) per class, 19 classes x 4 architectures\n")
a7 = []
for cur in ("clean", "ft"):
    rows = []
    for m in MODELS:
        d = L[(L.model == m) & (L.curve == cur)].pivot(index="cls", columns="eps", values="mean_f1")
        for c in d.index:
            rows.append(dict(curve=cur, model=m, cls=c, f1_clean=d.loc[c, 0.0], f1_attack=d.loc[c, 0.5]))
    X = pd.DataFrame(rows)
    rho, p = spearmanr(X.f1_clean, X.f1_attack)
    per = {m: spearmanr(X[X.model == m].f1_clean, X[X.model == m].f1_attack) for m in MODELS}
    md.append(f"\n## {cur}-trained curve: Spearman rho (clean F1 vs attacked F1) pooled over 76 class-architecture points = {rho:.3f} (p = {p:.2g}); "
              + "; ".join(f"{m}: {v[0]:.2f} (p={v[1]:.2g})" for m, v in per.items()))
    rob = X[(X.f1_clean >= 0.9) & (X.f1_attack >= 0.8)]
    brit = X[(X.f1_clean >= 0.9) & (X.f1_attack < 0.3)]
    weak = X[(X.f1_clean < 0.3) & (X.f1_attack < X.f1_clean)]
    md.append(f"\nRobustly correct (clean F1 >= 0.90 and attacked F1 >= 0.80): {len(rob)} points, e.g. {sorted(set(rob.cls))[:6]}.")
    md.append(f"Accurate but brittle (clean >= 0.90, attacked < 0.30): {len(brit)} points, e.g. {sorted(set(brit.cls))[:6]}.")
    md.append(f"Already weak and further hurt (clean < 0.30 and attacked below clean): {len(weak)} points, e.g. {sorted(set(weak.cls))[:6]}.")
    a7.append(X)
pd.concat(a7).to_csv(OUT / "a7_clean_vs_attack_f1.csv", index=False)
md.append("\nThresholds (0.90, 0.80, 0.30) are descriptive cut-offs chosen here, not pre-registered. ASR per class is available per class in "
          "metrics.json (asr_per_class) but is a binary-evasion rate, not the 19-class F1 used here.")

(OUT / "a_suite_summary.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("wrote", OUT)
