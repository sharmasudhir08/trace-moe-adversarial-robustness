"""Paper 2, S6 analysis from the saved per-window arrays (results_v3_paper2/e5_pcrp_perwindow/<model>/seed<s>.npz). No model is run.

  S6a  exact conditional ASR: evasion = the 6-category head predicts Benign (category 0) after PGD; 'clean-detected' = the clean pass does not predict
       Benign. (The binary head is not stored; this proxy is compared with the stored 1 - attack_detection_rate.)
  S6b  score distributions: clean versus attacked D at eps 0.25 and 0.50 (mean, SD, quantiles, overlap coefficient, AUC), predicted-class and nearest-centroid.
  S6c  which attack classes evade the PCRP: per true class TPR at the 1% FPR threshold (99th percentile of the clean attack-window scores, as in e5), per seed then averaged.
Writes results_v3_paper2/saved_artifact_analyses_20261118/s6_*.csv and s6_analysis.md.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_v3_paper2"
OUT = R / "saved_artifact_analyses_20261118"
MODELS = ["trace_moe", "transformer", "bilstm_dnn", "tfsage_lite"]
SEEDS = range(42, 47)
EPS = ["0.05", "0.1", "0.25", "0.5"]
Z = {m: {s: np.load(R / "e5_pcrp_perwindow" / m / f"seed{s}.npz") for s in SEEDS} for m in MODELS}
CLS = json.load(open(R / "robustness" / "trace_moe_official_T16_seed43" / "metrics.json"))["curve_clean_model"]["0.25"]["label19"]["per_class"]
md = []


def run_dir(m, s):
    n = f"{m}_official_T16_seed{s}"
    v2 = R / "robustness" / (n + "__v2")
    return v2 if (s == 42 and v2.exists()) else R / "robustness" / n


# ---------- S6a
rows = []
for m in MODELS:
    for e in EPS:
        cu, co, cm, px, st = [], [], [], [], []
        for s in SEEDS:
            z = Z[m][s]
            cdet = z["clean_pred6"] != 0
            ev = z[f"eps{e}_pred6"] == 0
            cu.append(ev.mean())
            co.append(ev[cdet].mean())
            cm.append(ev[~cdet].mean() if (~cdet).any() else np.nan)
            j = json.load(open(run_dir(m, s) / "metrics.json"))
            px.append(1 - j["curve_clean_model"][e]["attack_detection_rate"])
            st.append((~cdet).mean())
        rows.append(dict(model=m, eps=float(e), asr_proxy=np.mean(cu), asr_stored=np.mean(px), clean_missed_share=np.mean(st),
                         cond_asr_given_clean_detected=np.mean(co), asr_given_clean_missed=np.nanmean(cm)))
a1 = pd.DataFrame(rows)
a1.to_csv(OUT / "s6a_conditional_asr_exact.csv", index=False)
md.append("# S6a: exact conditional ASR (clean-trained models; evasion = 6-category head predicts Benign)\n")
md.append(a1.round(3).to_markdown(index=False))
md.append("\nasr_proxy is computed from the saved per-window predictions of the 6-category head; asr_stored is 1 - attack_detection_rate from metrics.json (binary head). "
          "A difference between them is the proxy error.\n")

# ---------- S6b
rows = []
for m in MODELS:
    for e in ["0.25", "0.5"]:
        for var in ("pred", "min"):
            cl, ad, au, ov, thr_tpr = [], [], [], [], []
            for s in SEEDS:
                z = Z[m][s]
                c, a = z[f"clean_d_{var}"], z[f"eps{e}_d_{var}"]
                cl.append((c.mean(), c.std(), np.quantile(c, 0.99)))
                ad.append((a.mean(), a.std(), np.quantile(a, 0.5)))
                au.append(roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(a))], np.r_[c, a]))
                hi = np.quantile(np.r_[c, a], 0.995)
                bins = np.linspace(0, hi, 101)
                hc, _ = np.histogram(np.clip(c, 0, hi), bins, density=False)
                ha, _ = np.histogram(np.clip(a, 0, hi), bins, density=False)
                ov.append(np.minimum(hc / hc.sum(), ha / ha.sum()).sum())
                thr_tpr.append((a > np.quantile(c, 0.99)).mean())
            rows.append(dict(model=m, eps=float(e), score=var, clean_mean=np.mean([x[0] for x in cl]), clean_sd=np.mean([x[1] for x in cl]),
                             clean_q99=np.mean([x[2] for x in cl]), attacked_mean=np.mean([x[0] for x in ad]), attacked_sd=np.mean([x[1] for x in ad]),
                             attacked_median=np.mean([x[2] for x in ad]), overlap_coefficient=np.mean(ov), roc_auc=np.mean(au), tpr_at_1pct_fpr=np.mean(thr_tpr)))
b = pd.DataFrame(rows)
b.to_csv(OUT / "s6b_score_distributions.csv", index=False)
md.append("\n\n# S6b: PCRP score distributions, clean attack windows versus attacked windows (mean over five seeds)\n")
md.append("pred = predicted-class variant, min = nearest-centroid variant. overlap_coefficient = sum of the minima of the two normalised histograms (1 = identical, 0 = disjoint).\n")
md.append(b.round(3).to_markdown(index=False))

# ---------- S6c
rows = []
for m in MODELS:
    for e in ["0.25", "0.5"]:
        for var in ("pred", "min"):
            per = {}
            for s in SEEDS:
                z = Z[m][s]
                thr = np.quantile(z[f"clean_d_{var}"], 0.99)
                t19 = z[f"eps{e}_true19"]
                hit = z[f"eps{e}_d_{var}"] > thr
                for c in np.unique(t19):
                    per.setdefault(int(c), []).append(hit[t19 == c].mean())
            for c, v in per.items():
                rows.append(dict(model=m, eps=float(e), score=var, cls19=c, tpr_at_1pct_fpr=np.mean(v), sd_over_seeds=np.std(v, ddof=1)))
c = pd.DataFrame(rows)
# class names from the stored per-class dict order (label19 index order)
names = list(CLS.keys())
c["cls"] = c.cls19.map(lambda i: names[i] if i < len(names) else str(i))
c.to_csv(OUT / "s6c_pcrp_tpr_by_class.csv", index=False)
md.append("\n\n# S6c: true-positive rate at 1% FPR by attack class (PCRP, mean over seeds)\n")
md.append("Class names are taken from the label-19 order of the stored per-class metrics; checked below against the number of classes.\n")
for var, lab in (("min", "nearest-centroid"), ("pred", "predicted-class")):
    for e in (0.25, 0.5):
        p = c[(c.score == var) & (c.eps == e)].pivot(index="cls", columns="model", values="tpr_at_1pct_fpr")[MODELS]
        p["mean"] = p.mean(1)
        p = p.sort_values("mean")
        md.append(f"\n## {lab}, eps {e}: classes sorted by mean TPR over architectures (lowest first)\n")
        md.append(p.round(3).to_markdown())
(OUT / "s6_analysis.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("n class names:", len(names), "n classes in arrays:", c.cls19.nunique())
