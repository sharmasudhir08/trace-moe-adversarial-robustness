"""Paper 4 T5 (calibration and risk-coverage) and T9 (safety-view false alarms, event-level alerting). Post-hoc exploratory, descriptive. CPU only.
Inputs: score_export/*.npz, zero_day/<run>/predictions.npz (y_true_label2, y_true_label19), saved_output_analyses_20261118/per_run.csv and, for the event-level
part of T9, pcrp_cache_<date>/ (file id of each test window, made by scripts/p4_rev_pcrp_t4_u3.py). Settings fixed now:
T5: 15 equal-width confidence bins; error = held-out window or misclassified known window (argmax of the 19-class logits); score oriented higher = more unknown
    (-energy, -MSP), the lowest scores are accepted first; AURC = mean risk over all coverage points; risk-coverage grid 0.05..1 stored.
T9: alert = (energy < thr) OR (p_attack > 0.5); event rule grid n in {4,16,64}, m in {ceil(n/2), n}; the count is over the last n windows of the same file
    (fewer at the start of a file); an event is a maximal run of consecutive firing windows within a file. No rule is recommended.
Usage: python scripts/p4_rev_t5_t9.py --date 20261121
"""
import argparse, hashlib, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
BASE = R3 / f"p4_review_analyses_{a.date}"
for t in ("T5", "T9"):
    (BASE / t).mkdir(parents=True, exist_ok=True)
(BASE / "T5" / "figures").mkdir(exist_ok=True)
CACHE = R3 / f"pcrp_cache_{a.date}"
PR = pd.read_csv(R3 / "saved_output_analyses_20261118" / "per_run.csv").set_index(["model", "held_out_class", "seed"])
rng = np.random.default_rng(42)
NB = 15
GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def bins(conf, correct):
    b = np.minimum((conf * NB).astype(int), NB - 1)
    return (np.bincount(b, minlength=NB).astype(float), np.bincount(b, weights=correct, minlength=NB), np.bincount(b, weights=conf, minlength=NB))


def ece(n, c, s):
    return float(np.sum(np.abs(c - s)) / n.sum())


def ci(v):
    v = np.asarray(v, float)
    bs = v[rng.integers(0, len(v), (10_000, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


ece_rows, aurc_rows, rc_rows, fa_rows, ev_rows = [], [], [], [], []
pool = {m: {"h19": np.zeros((3, NB)), "h2": np.zeros((3, NB))} for m in MODELS}
chk_rc, chk_a1, chk_caught, chk_order = [], [], [], []
t0 = time.time()
for m in MODELS:
    for held in CLASSES:
        safe = held.replace("/", "_")
        for seed in SEEDS:
            run = f"{m}_loao_T16_{safe}_seed{seed}"
            z = np.load(R3 / "score_export" / f"{run}.npz")
            pz = np.load(R3 / "zero_day" / run / "predictions.npz")
            y19, hid = z["y19"].astype(int), int(z["hid"])
            assert (pz["y_true_label19"].astype(int) == y19).all(), "label order mismatch between predictions.npz and score_export"
            y2 = pz["y_true_label2"].astype(int)
            unk = y19 == hid
            kn = ~unk
            lg = z["logits19"].astype(np.float32)
            lg_ = lg - lg.max(1, keepdims=True)
            pr = np.exp(lg_); pr /= pr.sum(1, keepdims=True)
            pred = lg.argmax(1)
            # ---- T5 calibration on known windows
            conf = pr.max(1)[kn]; corr = (pred[kn] == y19[kn]).astype(float)
            n, c, s = bins(conf, corr)
            pool[m]["h19"] += np.stack([n, c, s])
            pa = z["p_attack"].astype(float)[kn]; p2 = np.where(pa >= 0.5, pa, 1 - pa); c2 = ((pa >= 0.5).astype(int) == y2[kn]).astype(float)
            n2, cc2, s2 = bins(p2, c2)
            pool[m]["h2"] += np.stack([n2, cc2, s2])
            ece_rows.append(dict(model=m, held_out_class=held, seed=seed, n_known=int(kn.sum()), ece_label19=ece(n, c, s), acc_label19=float(corr.mean()),
                                 mean_conf_label19=float(conf.mean()), ece_binary=ece(n2, cc2, s2), acc_binary=float(c2.mean())))
            # ---- T5 risk-coverage
            err = (unk | (pred != y19)).astype(float)
            for sc, sv in (("energy", -z["energy"].astype(np.float64)), ("msp", -z["msp"].astype(np.float64))):
                o = np.argsort(sv, kind="stable")
                risk = np.cumsum(err[o]) / np.arange(1, len(o) + 1)
                aurc_rows.append(dict(model=m, held_out_class=held, seed=seed, score=sc, aurc=float(risk.mean()), risk_at_full_coverage=float(risk[-1])))
                idx = np.minimum((GRID * len(o)).astype(int) - 1, len(o) - 1)
                for g, i in zip(GRID, idx):
                    rc_rows.append(dict(model=m, held_out_class=held, seed=seed, score=sc, coverage=g, risk=float(risk[i])))
                chk_rc.append(abs(float(risk[-1]) - float(err.sum() / len(err))))
            # ---- T9 (a) alert rates
            fE = z["energy"] < float(z["thr"]); fB = z["p_attack"] > 0.5; alert = fE | fB
            benign = kn & (y2 == 0); attack = kn & (y2 == 1)
            fa_rows.append(dict(model=m, held_out_class=held, seed=seed, n_benign=int(benign.sum()), n_known_attack=int(attack.sum()), n_heldout=int(unk.sum()),
                                alert_rate_benign=float(alert[benign].mean()), alert_rate_known_attack=float(alert[attack].mean()), alert_rate_heldout=float(alert[unk].mean()),
                                energy_only_rate_benign=float(fE[benign].mean()), binary_only_rate_benign=float(fB[benign].mean()),
                                energy_flag_rate_known_attack=float(fE[attack].mean()), binary_rate_known_attack=float(fB[attack].mean())))
            chk_a1.append(abs(float(fB[benign].mean()) - float(PR.loc[(m, held, seed), "A1_binary_fpr_on_benign"])))
            chk_caught.append(abs(float(alert[unk].mean()) - float(PR.loc[(m, held, seed), "A2_caught_any"])))
            # ---- T9 (b) event level
            cf = CACHE / f"{safe}_seed{seed}.npz"
            if cf.exists():
                fid = np.load(cf)["file_id"].astype(int)
                assert len(fid) == len(y19)
                chg = np.flatnonzero(np.diff(fid) != 0) + 1
                fstart = np.zeros(len(fid), int)
                bounds = np.r_[0, chg]
                fstart = bounds[np.searchsorted(bounds, np.arange(len(fid)), side="right") - 1]
                chk_order.append(len(np.unique(fid)) - (len(chg) + 1))        # 0 when every file is one contiguous block
                S = np.r_[0, np.cumsum(alert.astype(np.int64))]
                idxs = np.arange(len(fid))
                for nn in (4, 16, 64):
                    for mm in sorted({math.ceil(nn / 2), nn}):
                        lo = np.maximum(idxs - nn + 1, fstart)
                        fire = (S[idxs + 1] - S[lo]) >= mm
                        prev = np.r_[False, fire[:-1]] & np.r_[False, fid[1:] == fid[:-1]]
                        ev_start = fire & ~prev
                        # per-file summary
                        nf = fid.max() + 1
                        n_win = np.bincount(fid, minlength=nf); n_unk = np.bincount(fid[unk], minlength=nf); n_ben = np.bincount(fid[benign], minlength=nf)
                        ev = np.bincount(fid[ev_start], minlength=nf)
                        anyfire = np.bincount(fid[fire], minlength=nf) > 0
                        ho_files = n_unk > 0.5 * n_win; ben_files = n_ben > 0.5 * n_win
                        ev_rows.append(dict(model=m, held_out_class=held, seed=seed, n=nn, m=mm,
                                            window_fire_rate_benign=float(fire[benign].mean()), window_fire_rate_known_attack=float(fire[attack].mean()),
                                            window_fire_rate_heldout=float(fire[unk].mean()),
                                            events_per_10k_windows_benign_files=float(ev[ben_files].sum() / max(n_win[ben_files].sum(), 1) * 1e4) if ben_files.any() else np.nan,
                                            n_benign_files=int(ben_files.sum()), n_heldout_files=int(ho_files.sum()),
                                            heldout_file_detection=float(anyfire[ho_files].mean()) if ho_files.any() else np.nan))
        print(m, held, f"{time.time() - t0:.0f}s", flush=True)

E5 = pd.DataFrame(ece_rows); E5.to_csv(BASE / "T5" / "ece_per_run.csv", index=False)
A5 = pd.DataFrame(aurc_rows); A5.to_csv(BASE / "T5" / "aurc_per_run.csv", index=False)
RC = pd.DataFrame(rc_rows); RC.groupby(["model", "held_out_class", "score", "coverage"]).risk.mean().reset_index().to_csv(BASE / "T5" / "risk_coverage_mean_over_seeds.csv", index=False)
summ = []
for (mo, c, sc), g in A5.groupby(["model", "held_out_class", "score"]):
    mean, lo, hi = ci(g.aurc); summ.append(dict(model=mo, held_out_class=c, score=sc, metric="aurc", mean=mean, ci_lo=lo, ci_hi=hi))
for (mo, c), g in E5.groupby(["model", "held_out_class"]):
    for met in ("ece_label19", "ece_binary"):
        mean, lo, hi = ci(g[met]); summ.append(dict(model=mo, held_out_class=c, score="-", metric=met, mean=mean, ci_lo=lo, ci_hi=hi))
pd.DataFrame(summ).to_csv(BASE / "T5" / "summary_ece_aurc.csv", index=False)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for j, m in enumerate(MODELS):
    for i, h in enumerate(("h19", "h2")):
        n, c, s = pool[m][h]
        k = n > 0
        ax[i, j].plot([0, 1], [0, 1], "k--", lw=1); ax[i, j].plot(s[k] / n[k], c[k] / n[k], "o-", ms=3)
        ax[i, j].set_title(f"{m}, {'19-class' if h == 'h19' else 'binary'} head, known windows pooled", fontsize=9)
        ax[i, j].set_xlabel("mean confidence"); ax[i, j].set_ylabel("accuracy")
fig.tight_layout(); fig.savefig(BASE / "T5" / "figures" / "reliability_pooled.png", dpi=200); fig.savefig(BASE / "T5" / "figures" / "reliability_pooled.pdf"); plt.close(fig)
F9 = pd.DataFrame(fa_rows); F9.to_csv(BASE / "T9" / "safety_view_false_alarms.csv", index=False)
V9 = pd.DataFrame(ev_rows); V9.to_csv(BASE / "T9" / "event_level.csv", index=False)
env = f"Script `scripts/p4_rev_t5_t9.py` SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; Python {sys.version.split()[0]}, numpy {np.__version__}; rng 42; runs: {len(E5)} (expected 75); {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
(BASE / "T5" / "README.md").write_text("# T5 calibration and risk-coverage (post-hoc exploratory)\n\n" + env + f"\nCheck: max |risk at coverage 1 - independent (held-out + misclassified known)/N| = {max(chk_rc):.2e}.\n", encoding="utf-8")
(BASE / "T9" / "README.md").write_text("# T9 safety-view false alarms and event-level alerting (post-hoc exploratory, descriptive)\n\n" + env +
    f"\nChecks: max |held-out alert rate - A2_caught_any (Table II per seed)| = {max(chk_caught):.2e}; max |binary-only benign rate - A1_binary_fpr_on_benign| = {max(chk_a1):.2e}; "
    f"event-level runs: {len(V9) // 6 if len(V9) else 0}; non-contiguous file blocks found (must be 0): {sum(chk_order) if chk_order else 'cache not found'}.\n", encoding="utf-8")
print("done", max(chk_rc), max(chk_a1), max(chk_caught), sum(chk_order) if chk_order else "no cache")
