"""Paper 4 part-2 analyses U1, U2, U6, U9, U10, U11, U12 (post hoc, exploratory; CPU only; no training; reads score_export and zero_day predictions; new files only).
Specification: documents_v2/paper4/future-work_part2.md. Fixed before running: bootstrap 10,000 over the five seeds (rng 42), percentile 95% interval; U12 quantile grid {1,2,5,10};
U10 GEN gamma 0.1 with M = 19; U2 KL epsilon 1e-6. Score orientation: higher = more unknown. Held-out class logit/column is kept in softmax-type scores (as exported).
Usage: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U6 U12 U9 U10 U1 U11 U2"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
SE, ZD = R3 / "score_export", R3 / "zero_day"
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--tasks", nargs="+", required=True); a = ap.parse_args()
OUT = R3 / f"p4_review_analyses_{a.date}" / "part2"
PR = pd.read_csv(R3 / "saved_output_analyses_20261118" / "per_run.csv").set_index(["model", "held_out_class", "seed"])
ESA1 = pd.read_csv(R3 / "h8_verdict" / "ESA1_calibration.csv").set_index(["model", "class"])
H4 = pd.read_csv(R3 / "h8_verdict" / "H4_closed_set_degradation.csv").set_index(["model", "class"])
P1PC = pd.read_csv(ROOT / "results_v2" / "tables" / "per_class_f1_official.csv").set_index("class")
RNG = np.random.default_rng(42); B = 10_000
CLS = json.load(open(ZD / "trace_moe_loao_T16_ARP_Spoofing_seed42" / "metrics.json"))["test"]["label19"]["per_class"]; CLS = list(CLS)
HI = json.load(open(ZD / "trace_moe_loao_T16_ARP_Spoofing_seed42" / "metrics.json"))["hierarchy"]
P19, P6 = np.array(HI["parent19"]), np.array(HI["parent6"])
CATS = ["Benign", "Spoofing", "Recon", "MQTT", "DoS", "DDoS"]
RUNS = []
for f in sorted(SE.glob("*.npz")):
    n = f.stem
    if n.endswith("_1epoch_backup") or "__v" in n: continue
    model, rest = n.split("_loao_T16_"); held, seed = rest.rsplit("_seed", 1)
    RUNS.append((n, model, held, int(seed)))
assert len(RUNS) == 75, len(RUNS)
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = sorted(set(r[2] for r in RUNS))


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan,) * 3
    bs = v[RNG.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def write(task, name, df):
    d = OUT / task; d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if p.exists(): raise FileExistsError(p)
    df.to_csv(p, index=False)


def readme(task, text, t0, nruns=75):
    d = OUT / task; d.mkdir(parents=True, exist_ok=True)
    p = d / "README.md"
    if p.exists(): raise FileExistsError(p)
    p.write_text(f"# {task} (post hoc, exploratory, CPU, no training)\n\nCommand: python scripts/p4_rev2_batch1.py --date {a.date} --tasks {task}\nScript SHA-256: {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}\n"
                 f"Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: {nruns}. rng seed 42, bootstrap {B} over the five seeds.\n"
                 f"Started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}, ended {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n{text}\n", encoding="utf-8")


def seedci(df, keys, cols):
    out = []
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for c in cols:
            m, lo, hi = ci(g.sort_values("seed")[c].to_numpy()); row[c] = m; row[c + "_lo"] = lo; row[c + "_hi"] = hi
        row["n_seeds"] = g.seed.nunique(); out.append(row)
    return pd.DataFrame(out)


def fpr95(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[min(np.searchsorted(tpr, 0.95, side="left"), len(fpr) - 1)])


def load(n):
    z = np.load(SE / f"{n}.npz"); y = z["y19"].astype(int); hid = int(z["hid"]); return z, y, hid, (y == hid)


# ---------------- U6 and U12 (energy threshold)
def u6_u12():
    t0 = time.time(); r6, r12, chk6, chk12 = [], [], [], []
    for n, m, h, s in RUNS:
        z, y, hid, unk = load(n); e = z["energy"].astype(np.float64); thr = float(z["thr"]); known = ~unk
        fl = e < thr
        chk6.append(abs(fl[known].mean() - ESA1.loc[(m, h), f"far_s{s}"]))
        for c in range(19):
            if c == hid: continue
            mk = y == c
            if mk.sum() == 0: continue
            r6.append(dict(model=m, held_out_class=h, seed=s, known_class=CLS[c], n=int(mk.sum()), flag_rate=float(fl[mk].mean()), contribution_to_far=float(fl[mk].sum() / known.sum())))
        ve = z["val_energy"].astype(np.float64)
        for q in (1, 2, 5, 10):
            tq = float(np.percentile(ve, q))
            if q == 5: chk12.append(abs(tq - thr))
            f = e < tq
            r12.append(dict(model=m, held_out_class=h, seed=s, quantile=q, threshold=tq, caught_energy=float(f[unk].mean()), far_known=float(f[known].mean())))
    d6 = pd.DataFrame(r6); write("U6", "far_by_known_class_per_run.csv", d6)
    s6 = seedci(d6, ["model", "held_out_class", "known_class"], ["flag_rate", "contribution_to_far"]); s6["exceeds_5pct"] = s6.flag_rate > 0.05
    write("U6", "far_by_known_class.csv", s6)
    top = s6.sort_values("contribution_to_far", ascending=False).groupby(["model", "held_out_class"]).head(3)
    readme("U6", f"Flag = energy < exported thr (5th percentile of known validation energy). Check: overall known FAR equals ESA1 far_s<seed> within {max(chk6):.2e} (criterion 1e-6; {'PASS' if max(chk6) < 1e-6 else 'FAIL'}).\n\nTop three known classes by contribution per model and held-out class:\n\n" + top.round(4).to_markdown(index=False), t0)
    d12 = pd.DataFrame(r12); write("U12", "threshold_quantiles_per_run.csv", d12)
    s12 = seedci(d12, ["model", "held_out_class", "quantile"], ["caught_energy", "far_known"]); write("U12", "threshold_quantiles.csv", s12)
    pooled = seedci(d12.groupby(["model", "seed", "quantile"], as_index=False)[["caught_energy", "far_known"]].mean(), ["model", "quantile"], ["caught_energy", "far_known"])
    readme("U12", f"Threshold = q-th percentile of known validation energy (q in 1, 2, 5, 10); energy flag only. Check: q = 5 threshold equals exported thr, max difference {max(chk12):.2e} (criterion 1e-6; {'PASS' if max(chk12) < 1e-6 else 'FAIL'}).\n\nMean over the five classes and five seeds:\n\n" + pooled.round(4).to_markdown(index=False), t0)
    print("U6/U12 done", max(chk6), max(chk12), flush=True)


# ---------------- U9
def aurc(score, wrong):
    o = np.argsort(score, kind="stable"); e = wrong[o].astype(np.float64)
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def u9():
    t0 = time.time(); rows = []
    for n, m, h, s in RUNS:
        z, y, hid, unk = load(n); k = ~unk
        lk = z["logits19"][k].astype(np.float32); lk[:, hid] = -np.inf
        wrong = (lk.argmax(1) != y[k]).astype(int)
        if wrong.sum() == 0 or wrong.sum() == len(wrong): continue
        se, sm = -z["energy"][k].astype(np.float64), -z["msp"][k].astype(np.float64)
        rows.append(dict(model=m, held_out_class=h, seed=s, n_known=int(k.sum()), error_rate=float(wrong.mean()), auroc_energy=roc_auc_score(wrong, se), auroc_msp=roc_auc_score(wrong, sm),
                         aurc_energy=aurc(se, wrong), aurc_msp=aurc(sm, wrong)))
        print("U9", n, flush=True)
    d = pd.DataFrame(rows); d["auroc_diff_energy_minus_msp"] = d.auroc_energy - d.auroc_msp; d["aurc_diff_energy_minus_msp"] = d.aurc_energy - d.aurc_msp
    write("U9", "misclassification_detection_per_run.csv", d)
    s = seedci(d, ["model", "held_out_class"], ["error_rate", "auroc_energy", "auroc_msp", "auroc_diff_energy_minus_msp", "aurc_energy", "aurc_msp", "aurc_diff_energy_minus_msp"]); write("U9", "misclassification_detection.csv", s)
    p = seedci(d.groupby(["model", "seed"], as_index=False).mean(numeric_only=True), ["model"], ["auroc_energy", "auroc_msp", "auroc_diff_energy_minus_msp", "aurc_diff_energy_minus_msp"])
    readme("U9", "Known windows only (held-out class excluded); a prediction is wrong if the argmax over the 18 known-class logits differs from y19 (held-out logit masked, as in T2). Score = -energy or -MSP (higher = more likely wrong); AURC accepts the lowest scores first. Float16 stored logits are used only for the argmax.\n\nPooled over classes (mean over the five classes per seed, then seed interval):\n\n" + p.round(4).to_markdown(index=False), t0, len(d))


# ---------------- U10
def u10():
    t0 = time.time(); rows = []; maxd = 0.0
    for n, m, h, s in RUNS:
        z, y, hid, unk = load(n); u = unk.astype(int)
        L = z["logits19"].astype(np.float32)
        lse = np.logaddexp.reduce(L, axis=1)
        maxd = max(maxd, float(np.abs(lse - z["energy"]).max()))
        P = np.exp(L - lse[:, None]); g = 0.1
        gen = (P ** g * (1 - P) ** g).sum(1)
        sc = {"energy_exported": -z["energy"].astype(np.float64), "gen_gamma0.1": gen.astype(np.float64), "logit_norm": -np.linalg.norm(L, axis=1).astype(np.float64), "maxlogit_minus_mean": -(L.max(1) - L.mean(1)).astype(np.float64)}
        row = dict(model=m, held_out_class=h, seed=s)
        for k, v in sc.items():
            row[f"{k}_auroc"] = float(roc_auc_score(u, v)); row[f"{k}_fpr95"] = fpr95(u, v)
        for k in ("gen_gamma0.1", "logit_norm", "maxlogit_minus_mean"):
            row[f"{k}_auroc_minus_energy"] = row[f"{k}_auroc"] - row["energy_exported_auroc"]
        rows.append(row); print("U10", n, flush=True)
    d = pd.DataFrame(rows); write("U10", "logit_scores_per_run.csv", d)
    cols = [c for c in d.columns if c.endswith("_auroc") or c.endswith("_fpr95") or c.endswith("_minus_energy")]
    write("U10", "logit_scores.csv", seedci(d, ["model", "held_out_class"], cols))
    pooled = seedci(d.groupby(["model", "seed"], as_index=False).mean(numeric_only=True), ["model"], [c for c in cols if c.endswith("_minus_energy")])
    readme("U10", f"Scores from float16 logits19 converted to float32 (softmax over all 19 logits for GEN, M = 19, gamma 0.1). Energy reference = exported float32 energy. Check: -logsumexp of the stored float16 logits versus exported energy: max |difference| = {maxd:.3e} (criterion 1e-3; {'PASS' if maxd < 1e-3 else 'ABOVE CRITERION: float16 storage; AUROC differences are indicative'}).\n\nPaired AUROC difference to energy, mean over the five classes per seed, then seed interval:\n\n" + pooled.round(4).to_markdown(index=False), t0)


# ---------------- U1 and U11 and U2 (predictions.npz)
def u1():
    t0 = time.time(); r19, r6, ck = [], [], []
    for n, m, h, s in RUNS:
        z = np.load(ZD / n / "predictions.npz"); yt = z["y_true_label19"].astype(int); hid = CLS.index(h); unk = yt == hid
        yp = z["y_pred_label19"].astype(int)[unk]; sh = np.bincount(yp, minlength=19) / len(yp)
        sh6 = np.bincount(P19[yp], minlength=6) / len(yp)
        att = (z["y_pred_label2"].astype(int)[unk] == 1).mean()
        ck.append((abs(sh.sum() - 1), abs(att - PR.loc[(m, h, s), "A2_binary_rate"])))
        for c in range(19): r19.append(dict(model=m, held_out_class=h, seed=s, predicted_class=CLS[c], share=sh[c], is_sibling=(P19[c] == P19[hid] and c != hid), is_benign=(c == 1)))
        for c in range(6): r6.append(dict(model=m, held_out_class=h, seed=s, predicted_category=CATS[c], share=sh6[c], is_sibling_category=(c == P19[hid])))
        print("U1", n, flush=True)
    d19, d6 = pd.DataFrame(r19), pd.DataFrame(r6)
    s19 = seedci(d19, ["model", "held_out_class", "predicted_class", "is_sibling", "is_benign"], ["share"]); write("U1", "absorption_19.csv", s19)
    s6 = seedci(d6, ["model", "held_out_class", "predicted_category", "is_sibling_category"], ["share"]); write("U1", "absorption_6.csv", s6)
    top = s19.sort_values("share", ascending=False).groupby(["model", "held_out_class"]).head(3)
    sib = d6[d6.is_sibling_category].groupby(["model", "held_out_class"]).share.mean().round(3).reset_index()
    readme("U1", f"Held-out windows = test windows whose true 19-class label is the held-out class; shares of predicted 19-class labels and 6-class categories (mean over seeds with seed interval). Checks: shares sum to 1 (max deviation {max(c[0] for c in ck):.1e}); binary-head attack share equals A2_binary_rate (max difference {max(c[1] for c in ck):.1e}).\n\nTop three absorbing classes:\n\n" + top.round(4).to_markdown(index=False)
           + "\n\nShare absorbed by the held-out class's own 6-category (siblings plus other classes of that category):\n\n" + sib.to_markdown(index=False), t0)


def u11():
    t0 = time.time(); rows = []; chk = []
    for n, m, h, s in RUNS:
        z = np.load(ZD / n / "predictions.npz"); yt = z["y_true_label19"].astype(int); yp = z["y_pred_label19"].astype(int); hid = CLS.index(h)
        k = yt != hid; known = [c for c in range(19) if c != hid]
        f1 = f1_score(yt[k], yp[k], labels=known, average=None, zero_division=0)
        chk.append(abs(f1.mean() - H4.loc[(m, h), f"p3c_s{s}"]))
        for c, v in zip(known, f1):
            base = float(P1PC.loc[CLS[c], m]); rows.append(dict(model=m, held_out_class=h, seed=s, known_class=CLS[c], f1_loao=float(v), f1_reference_baseline=base, degradation=base - float(v), is_sibling=bool(P19[c] == P19[hid])))
        print("U11", n, flush=True)
    d = pd.DataFrame(rows); write("U11", "h4_per_class_per_run.csv", d)
    s = seedci(d, ["model", "held_out_class", "known_class", "is_sibling"], ["degradation", "f1_loao"]); write("U11", "h4_per_class.csv", s)
    top = s.sort_values("degradation", ascending=False).groupby(["model", "held_out_class"]).head(3)
    sib = d.groupby(["model", "held_out_class", "is_sibling"]).degradation.mean().round(4).reset_index()
    readme("U11", f"Degradation = full-model per-class F1 (mean over seeds, results_v2/tables/per_class_f1_official.csv) minus per-class F1 of the LOAO run on the known test windows (held-out windows excluded), positive = worse. Check: mean over the 18 known classes of the LOAO F1 equals the H4 p3c value, max difference {max(chk):.2e} (criterion 1e-6; {'PASS' if max(chk) < 1e-6 else 'FAIL'}). The full-model baseline is the seed-mean per-class F1 (per-seed per-class values are not stored in that table), so per-seed degradations here are not identical to the H4 per-seed degradations.\n\nThree largest mean degradations per model and held-out class:\n\n" + top.round(4).to_markdown(index=False) + "\n\nMean degradation siblings vs others:\n\n" + sib.to_markdown(index=False), t0)


def u2():
    t0 = time.time(); rows = []; chk = []; eps = 1e-6
    for n, m, h, s in RUNS:
        z = np.load(ZD / n / "predictions.npz"); zs, y, hid, unk = load(n); u = unk.astype(int); thr = float(zs["thr"]); e = zs["energy"].astype(np.float64)
        p19, p6, p2 = [z[f"proba_label{k}"].astype(np.float32) for k in (19, 6, 2)]
        a6 = np.zeros_like(p6); [np.add.at(a6, (slice(None), P19[c]), 0) for c in []]
        M19 = np.zeros((19, 6), np.float32); M19[np.arange(19), P19] = 1; a6 = p19 @ M19
        M6 = np.zeros((6, 2), np.float32); M6[np.arange(6), P6] = 1; a2 = p6 @ M6
        kl = lambda P, Q: (P * (np.log(P + eps) - np.log(Q + eps))).sum(1)
        A = kl(a6, p6) + kl(a2, p2)
        Bi = ((P19[p19.argmax(1)] != p6.argmax(1)) | (P6[p6.argmax(1)] != p2.argmax(1))).astype(int)
        ea = float(roc_auc_score(u, -e)); chk.append(abs(ea - PR.loc[(m, h, s), "A7_energy_auroc"]))
        fl = (e < thr) | (Bi == 1); fe = e < thr
        rows.append(dict(model=m, held_out_class=h, seed=s, energy_auroc=ea, scoreA_auroc=float(roc_auc_score(u, A)), scoreA_fpr95=fpr95(u, A), scoreB_auroc=float(roc_auc_score(u, Bi)),
                         share_inconsistent_heldout=float(Bi[unk].mean()), share_inconsistent_known=float(Bi[~unk].mean()), caught_energy=float(fe[unk].mean()), far_energy=float(fe[~unk].mean()),
                         caught_energy_or_B=float(fl[unk].mean()), far_energy_or_B=float(fl[~unk].mean())))
        print("U2", n, flush=True)
    d = pd.DataFrame(rows); d["scoreA_minus_energy_auroc"] = d.scoreA_auroc - d.energy_auroc; d["scoreB_minus_energy_auroc"] = d.scoreB_auroc - d.energy_auroc
    write("U2", "head_disagreement_per_run.csv", d)
    cols = [c for c in d.columns if c not in ("model", "held_out_class", "seed")]
    write("U2", "head_disagreement_auroc.csv", seedci(d, ["model", "held_out_class"], cols))
    pooled = seedci(d.groupby(["model", "seed"], as_index=False).mean(numeric_only=True), ["model"], ["scoreA_minus_energy_auroc", "scoreB_minus_energy_auroc", "caught_energy", "caught_energy_or_B", "far_energy", "far_energy_or_B"])
    readme("U2", f"Score A = KL(p19 aggregated to 6 categories || p6) + KL(p6 aggregated to 2 || p2), epsilon 1e-6; score B = 1 if the argmax of the 19-class head (mapped to its category) differs from the argmax of the 6-class head, or the 6-class argmax (mapped to binary) differs from the binary head. OR rule: energy flag (exported thr) OR B = 1. Probabilities stored as float16. Check: energy AUROC equals per_run A7_energy_auroc, max difference {max(chk):.2e} (criterion 1e-6; {'PASS' if max(chk) < 1e-6 else 'FAIL'}).\n\nMean over the five classes per seed, then seed interval:\n\n" + pooled.round(4).to_markdown(index=False), t0)


T = {"U6": u6_u12, "U12": u6_u12, "U9": u9, "U10": u10, "U1": u1, "U11": u11, "U2": u2}
done = set()
for t in a.tasks:
    f = T[t]
    if f in done: continue
    f(); done.add(f)
print("finished", a.tasks)
