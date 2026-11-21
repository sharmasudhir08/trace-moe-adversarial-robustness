"""Paper 4 saved-score analyses T1 (decomposition table), T2 (operating points, temperature) and T11 (Charter ESA2 Spearman).
Post-hoc exploratory (T1, T2) and pre-registered ESA2 rule (T11). CPU only, no training. Reads score_export/*.npz and saved_output_analyses_20261118/per_run.csv.
Writes only into results_v3_paper3/p4_review_analyses_<date>/{T1,T2,T11}. Fixed before running: temperature grid {0.5,1,2,5}, bootstrap 10,000 (rng 42),
OSCR with argmax over known classes only. Score orientation: higher = more unknown (-energy, -MSP, -MaxLogit). Usage: python scripts/p4_rev_saved_scores.py --date 20261121
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
SE = R3 / "score_export"
PR = pd.read_csv(R3 / "saved_output_analyses_20261118" / "per_run.csv")
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
OUT = R3 / f"p4_review_analyses_{a.date}"
for t in ("T1", "T2", "T11"):
    (OUT / t).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
B = 10_000
TEMPS = [0.5, 1, 2, 5]


def ci(v):
    v = np.asarray(v, float)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fpr_at_tpr(y, s, t=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[min(np.searchsorted(tpr, t, side="left"), len(fpr) - 1)])


def tpr_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[np.searchsorted(fpr, f, side="right") - 1])


def oscr(unk, s, correct):
    o = np.argsort(s, kind="stable")                       # accept low scores first
    u, k, c = unk[o], 1 - unk[o], correct[o] * (1 - unk[o])
    fpr = np.concatenate([[0], np.cumsum(u) / max(u.sum(), 1)])
    ccr = np.concatenate([[0], np.cumsum(c) / max(k.sum(), 1)])
    return float(np.trapezoid(ccr, fpr))


rows2, rows11, tmp = [], [], []
t0 = time.time()
for f in sorted(SE.glob("*.npz")):
    run = f.stem
    if run.endswith("_1epoch_backup") or "__v" in run:
        continue
    model = run.split("_loao_")[0]
    rest = run.split("_loao_T16_")[1]
    held, seed = rest.rsplit("_seed", 1)
    seed = int(seed)
    z = np.load(f)
    y19, hid = z["y19"].astype(int), int(z["hid"])
    unk = (y19 == hid).astype(int)
    lg = z["logits19"].astype(np.float32)
    row = dict(model=model, held_out_class=held, seed=seed, n_unknown=int(unk.sum()), n_known=int((1 - unk).sum()))
    sc = {"energy": -z["energy"].astype(np.float64), "msp": -z["msp"].astype(np.float64), "maxlogit": -z["maxlogit"].astype(np.float64)}
    lk = lg.copy(); lk[:, hid] = -np.inf
    correct = (lk.argmax(1) == y19).astype(int)
    for k, s in sc.items():
        row[f"{k}_auroc"] = float(roc_auc_score(unk, s))
        row[f"{k}_fpr95"] = fpr_at_tpr(unk, s)
        row[f"{k}_tpr_at_fpr1pct"] = tpr_at_fpr(unk, s)
        row[f"{k}_aupr"] = float(average_precision_score(unk, s))
        row[f"{k}_oscr"] = oscr(unk, s, correct)
    # temperature (from the saved float16 logits)
    lgt = lg.astype(np.float64)
    for T in TEMPS:
        l = lgt / T
        mx = l.max(1, keepdims=True)
        lse = (mx[:, 0] + np.log(np.exp(l - mx).sum(1)))
        msp_T = np.exp(mx[:, 0] - lse)
        row[f"energy_T{T}_auroc"] = float(roc_auc_score(unk, -(T * lse)))
        row[f"msp_T{T}_auroc"] = float(roc_auc_score(unk, -msp_T))
    rows2.append(row)
    # T11: Spearman between energy and (1 - msp) on held-out windows; both oriented so higher = more unknown (-energy vs 1-msp)
    m = unk.astype(bool)
    rho = float(spearmanr(-z["energy"][m].astype(np.float64), 1 - z["msp"][m].astype(np.float64)).correlation)
    rows11.append(dict(model=model, held_out_class=held, seed=seed, n_heldout=int(m.sum()), rho=rho))
    print(f"{run}: {time.time() - t0:.0f}s", flush=True)
d2 = pd.DataFrame(rows2).sort_values(["model", "held_out_class", "seed"])
d2.to_csv(OUT / "T2" / "per_run_operating_points.csv", index=False)

# ---- T2 checks against per_run.csv
m = d2.merge(PR, on=["model", "held_out_class", "seed"])
chk = {"energy_auroc_T1": np.abs(m.energy_auroc - m.A7_energy_auroc).max(), "msp_auroc_T1": np.abs(m.msp_auroc - m.A7_msp_auroc).max(),
       "maxlogit_auroc": np.abs(m.maxlogit_auroc - m.A7_maxlogit_auroc).max()}
for T in (0.5, 2, 5):
    chk[f"energy_T{T}"] = np.abs(m[f"energy_T{T}_auroc"] - m[f"A18_energy_T{T}_auroc"]).max()
# ---- T2 summary
S = ["energy", "msp", "maxlogit"]
srows, trows = [], []
for (mo, c), g in d2.groupby(["model", "held_out_class"]):
    for s in S:
        for met in ("auroc", "fpr95", "tpr_at_fpr1pct", "aupr", "oscr"):
            mean, lo, hi = ci(g[f"{s}_{met}"]); srows.append(dict(model=mo, held_out_class=c, score=s, metric=met, n_seeds=len(g), mean=mean, ci_lo=lo, ci_hi=hi))
    for met in ("auroc", "fpr95"):
        mean, lo, hi = ci(g[f"energy_{met}"].values - g[f"msp_{met}"].values)
        srows.append(dict(model=mo, held_out_class=c, score="energy_minus_msp_paired", metric=met, n_seeds=len(g), mean=mean, ci_lo=lo, ci_hi=hi))
    for T in TEMPS:
        for s in ("energy", "msp"):
            mean, lo, hi = ci(g[f"{s}_T{T}_auroc"]); trows.append(dict(model=mo, held_out_class=c, score=s, T=T, mean_auroc=mean, ci_lo=lo, ci_hi=hi))
pd.DataFrame(srows).to_csv(OUT / "T2" / "summary_operating_points.csv", index=False)
pd.DataFrame(trows).to_csv(OUT / "T2" / "temperature_energy_msp.csv", index=False)

# ---- T11 aggregate and three-tier rule
d11 = pd.DataFrame(rows11).sort_values(["model", "held_out_class", "seed"])
d11.to_csv(OUT / "T11" / "esa2_spearman_per_run.csv", index=False)
agg = []
for (mo, c), g in d11.groupby(["model", "held_out_class"]):
    mean, lo, hi = ci(g.rho)
    v = "RANKING IMPROVES" if (mean < 0.9 and hi < 0.9) else ("RANKING DOES NOT IMPROVE" if (mean >= 0.9 and lo >= 0.9) else "INCONCLUSIVE")
    agg.append(dict(model=mo, held_out_class=c, n_seeds=len(g), mean_rho=mean, ci_lo=lo, ci_hi=hi, esa2_tier=v))
pd.DataFrame(agg).to_csv(OUT / "T11" / "esa2_spearman.csv", index=False)

# ---- T1 (a): decomposition table from per_run.csv
t1 = PR[["model", "held_out_class", "seed", "A2_energy_only", "A2_binary_only", "A2_both", "A2_neither", "A2_caught_any", "A2_energy_flag_rate", "A2_binary_rate"]].sort_values(["model", "held_out_class", "seed"])
t1.to_csv(OUT / "T1" / "decomposition_per_seed.csv", index=False)
sum_err = float(np.abs(t1.A2_energy_only + t1.A2_binary_only + t1.A2_both + t1.A2_neither - 1).max())
caught_err = float(np.abs(t1.A2_energy_only + t1.A2_binary_only + t1.A2_both - t1.A2_caught_any).max())
h1 = pd.read_csv(R3 / "h8_verdict" / "H1_safety_view.csv").set_index("class")
tm = t1[t1.model == "trace_moe"]
h1_err = max(abs(float(tm[(tm.held_out_class == c) & (tm.seed == s)].A2_caught_any.iloc[0]) - float(h1.loc[c, f"seed{s}"])) for c in h1.index for s in range(42, 47))
tex = t1.round(4).to_latex(index=False, escape=True, longtable=False, caption="Safety-view decomposition per model, class and seed (post-hoc exploratory).", label="tab:supp_decomp")
(OUT / "T1" / "decomposition_per_seed.tex").write_text(tex, encoding="utf-8")
# T1 (b): ARP TRACE-MoE seed diagnosis (descriptive)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 5, figsize=(17, 6))
diag = []
for i, seed in enumerate(range(42, 47)):
    z = np.load(SE / f"trace_moe_loao_T16_ARP_Spoofing_seed{seed}.npz")
    unk = z["y19"].astype(int) == int(z["hid"])
    e = z["energy"]; thr = float(z["thr"])
    ax[0, i].hist(e[~unk], bins=80, alpha=.5, density=True, label="known"); ax[0, i].hist(e[unk], bins=80, alpha=.6, density=True, label="held-out")
    ax[0, i].axvline(thr, color="k", ls="--"); ax[0, i].set_title(f"seed {seed}: energy (higher = known)", fontsize=9)
    ax[1, i].hist(z["p_attack"][unk], bins=50); ax[1, i].axvline(.5, color="k", ls="--"); ax[1, i].set_title("held-out p_attack", fontsize=9)
    diag.append(dict(seed=seed, thr=thr, val_energy_p5=float(np.quantile(z["val_energy"], 0.05)), median_heldout_energy=float(np.median(e[unk])),
                     median_known_energy=float(np.median(e[~unk])), n_heldout=int(unk.sum())))
ax[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "T1" / "arp_seed_diagnosis.png", dpi=200); fig.savefig(OUT / "T1" / "arp_seed_diagnosis.pdf"); plt.close(fig)
pd.DataFrame(diag).to_csv(OUT / "T1" / "arp_seed_diagnosis.csv", index=False)

sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
env = f"Script `scripts/p4_rev_saved_scores.py` SHA-256 {sha}; Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}; rng 42; runs read: {len(d2)} (expected 75); {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
(OUT / "T2" / "README.md").write_text("# T2 operating points (post-hoc exploratory)\n\n" + env + "\nReproduction of saved values (max abs difference vs per_run.csv):\n\n" + "\n".join(f"- {k}: {v:.3e}" for k, v in chk.items()) + "\n\nNo temperature is selected or recommended. Higher score = more unknown; OSCR with argmax over known classes.\n", encoding="utf-8")
(OUT / "T11" / "ESA2.md").write_text("# T11 Charter ESA2 (Spearman, energy vs 1 - MSP on held-out windows)\n\n" + env + "\nSign convention chosen before computing: both scores oriented higher = more unknown (-energy vs 1 - MSP). Rule: Charter v1.4 three-tier, 0.9 threshold, bootstrap 10,000 over seeds.\n\n" + pd.DataFrame(agg).round(4).to_markdown(index=False) + "\n\nReported directionally; no overall ESA2 claim at this scope.\n", encoding="utf-8")
(OUT / "T1" / "README.md").write_text("# T1 decomposition and ARP diagnosis (post-hoc exploratory, descriptive)\n\n" + env + f"\nChecks: max |sum of four cells - 1| = {sum_err:.2e}; max |caught_any - (energy_only+binary_only+both)| = {caught_err:.2e}; max |TRACE-MoE caught_any - H1 Table per seed| = {h1_err:.2e}.\n\n" + pd.DataFrame(diag).round(4).to_markdown(index=False) + "\n", encoding="utf-8")
print(json.dumps({"T2_checks": {k: float(v) for k, v in chk.items()}, "T1": [sum_err, caught_err, h1_err]}, indent=1))
