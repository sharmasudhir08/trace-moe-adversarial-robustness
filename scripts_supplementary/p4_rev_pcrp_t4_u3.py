"""Paper 4 T4 (PCRP estimator and aggregation variants) and U3 (fixed-rule fusion of energy and PCRP). Post-hoc exploratory; no verdict.
CPU only, no model training. Reuses the H5 scoring logic of scripts/24_twin_score_paper4.py (not modified): same windows (build_windowed, seed-specific),
same twin profile, D_min = min over label6 categories of sqrt(Mahalanobis^2 / d) on the LAST row of each window.

T4 variants (fixed now): covariance estimator in {reference (saved twin profile), Ledoit-Wolf, MinCovDet (support_fraction=None, random_state 42,
class-stratified subsample of at most 50,000 rows per category)} x aggregation over the last k in {1, 4, 16} rows of the window with mean and min.
Variant estimators are refit on the rows spanned by the known-class TRAINING windows of the LOAO run (scaled as in wd.X); nothing is fitted on test windows.
U3 (fixed now): rank-average of energy and PCRP (equal weights); OR rule with energy threshold = the run's saved `thr` and PCRP threshold = 95th percentile of
PCRP D_min on known validation windows. Reported per model, class and seed.
Pass criterion: reference variant AUROC equals twin_h5_score.json within 1e-6, else the script stops (reference mismatch).
Also writes a cache (pcrp_cache/<class>_seed<s>.npz: D_min of test windows, file id of each test window, held-out last-row features) used by T6 and T10.
Usage: python scripts/p4_rev_pcrp_t4_u3.py --date 20261121 [--classes ...] [--seeds ...]   (needs >= 6 GB free RAM)
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.experiment import load_prepared  # noqa: E402
from trace_iomt.data.windows import build_windowed  # noqa: E402

R3 = ROOT / "results_v3_paper3"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
KS = [1, 4, 16]
MCD_MAX = 50_000


def dmin_rows(X, mus, precs):
    d = X.shape[1]
    best = np.full(len(X), np.inf)
    for mu, P in zip(mus, precs):
        diff = X - mu
        md2 = np.maximum(np.einsum("nd,nd->n", diff @ P, diff), 0.0)
        best = np.minimum(best, np.sqrt(md2 / d))
    return best


def fpr95(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[min(np.searchsorted(tpr, 0.95, side="left"), len(fpr) - 1)])


def span_mask(n, starts, T):
    diff = np.zeros(n + 1, np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, starts + T, -1)
    return np.cumsum(diff[:-1]) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    a = ap.parse_args()
    out = R3 / f"p4_review_analyses_{a.date}"
    (out / "T4").mkdir(parents=True, exist_ok=True); (out / "U3").mkdir(parents=True, exist_ok=True)
    cache = R3 / f"pcrp_cache_{a.date}"; cache.mkdir(exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    T = int(cfg["windows"]["length"])
    df = load_prepared(cfg)
    srcfile = df["source_file"]
    log = open(out / "T4" / "run_log.txt", "a", encoding="utf-8")
    def say(s):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + s
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say(f"start; script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    t4_path, u3_path, st_path = out / "T4" / "pcrp_variants_per_seed.csv", out / "U3" / "fusion_energy_pcrp.csv", out / "T4" / "estimator_status.csv"
    done = set()
    if t4_path.exists():
        with open(t4_path, newline="", encoding="utf-8") as f:
            done = {(r["held_out_class"], int(r["seed"])) for r in csv.DictReader(f)}
    f4 = open(t4_path, "a", newline="", encoding="utf-8"); f3 = open(u3_path, "a", newline="", encoding="utf-8"); fs = open(st_path, "a", newline="", encoding="utf-8")
    w4 = csv.DictWriter(f4, ["held_out_class", "seed", "estimator", "aggregation", "k", "auroc", "fpr95", "n_unknown", "n_known"])
    w3 = csv.DictWriter(f3, ["model", "held_out_class", "seed", "n_unknown", "n_known", "energy_auroc", "pcrp_auroc", "rankavg_auroc", "pcrp_thr_q95_known_val",
                             "energy_only_caught", "energy_only_known_far", "pcrp_only_caught", "pcrp_only_known_far", "or_caught", "or_known_far"])
    ws = csv.DictWriter(fs, ["held_out_class", "seed", "estimator", "status"])
    for fh, ww in ((f4, w4), (f3, w3), (fs, ws)):
        if fh.tell() == 0:
            ww.writeheader()
    for held in a.classes:
        safe = held.replace("/", "_")
        prof = np.load(R3 / "twin" / f"twin_profile_loao_{safe}.npz", allow_pickle=False)
        mus_ref, precs_ref = prof["mus"], prof["inv_covs"]
        cats = [str(x) for x in prof["category_names"]]
        feat_names = [str(x) for x in prof["feature_names"]]
        for seed in a.seeds:
            if (cache / f"{safe}_seed{seed}.npz").exists():
                continue
            t0 = time.time()
            c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held}}
            wd = build_windowed(df, c2, regime="loao", T=T, seed=seed)
            assert wd.features == feat_names, "feature-order mismatch"
            hid = wd.hierarchy.index["label19"][held]
            st = wd.starts["test"]
            last = st + T - 1
            y19 = wd.labels("test", "label19")
            unk = (y19 == hid).astype(int)
            N = len(wd.X)
            # file id of each test window (source file of the last row)
            files = srcfile.loc[wd.row_index[last]].to_numpy()
            fnames, fid = np.unique(files, return_inverse=True)
            # row-level D_min for all rows needed by the largest k
            need = np.zeros(N, bool)
            for j in range(max(KS)):
                need[last - j] = True
            pos = np.cumsum(need) - 1
            Xn = wd.X[need].astype(np.float64)
            # training rows and label6 per row for refits
            trmask = span_mask(N, wd.starts["train"], T)
            y6 = wd.y["label6"]
            cat_idx = [wd.hierarchy.index["label6"][c] for c in cats]
            rng = np.random.default_rng(42)
            est = {"reference": (mus_ref, precs_ref)}
            for name in ("ledoit_wolf", "mincovdet"):
                mus_v, precs_v, ok = [], [], True
                for ci in cat_idx:
                    xc = wd.X[trmask & (y6 == ci)].astype(np.float64)
                    try:
                        if len(xc) < 5:
                            raise ValueError(f"only {len(xc)} rows")
                        if name == "ledoit_wolf":
                            m_ = LedoitWolf().fit(xc)
                        else:
                            if len(xc) > MCD_MAX:
                                xc = xc[np.sort(rng.choice(len(xc), MCD_MAX, replace=False))]
                            m_ = MinCovDet(support_fraction=None, random_state=42).fit(xc)
                        mus_v.append(m_.location_); precs_v.append(m_.precision_)
                    except Exception as exc:                         # report; do not substitute another estimator
                        ws.writerow(dict(held_out_class=held, seed=seed, estimator=name, status=f"FAILED category {ci}: {exc}")); ok = False
                        break
                if ok:
                    est[name] = (np.stack(mus_v), np.stack(precs_v))
                    ws.writerow(dict(held_out_class=held, seed=seed, estimator=name, status="ok"))
            fs.flush()
            ref_D = None
            for name, (mu, P) in est.items():
                Drow = dmin_rows(Xn, mu, P)
                for k in KS:
                    cols = np.stack([Drow[pos[last - j]] for j in range(k)], 1)
                    for agg in (["last"] if k == 1 else ["mean", "min"]):
                        s = cols[:, 0] if k == 1 else (cols.mean(1) if agg == "mean" else cols.min(1))
                        w4.writerow(dict(held_out_class=held, seed=seed, estimator=name, aggregation=agg, k=k, auroc=float(roc_auc_score(unk, s)),
                                         fpr95=fpr95(unk, s), n_unknown=int(unk.sum()), n_known=int((1 - unk).sum())))
                        if name == "reference" and k == 1:
                            ref_D = s.copy()
                del Drow
            f4.flush()
            # reference reproduction (stop if mismatch)
            js = json.loads((R3 / "zero_day" / f"trace_moe_loao_T16_{safe}_seed{seed}" / "twin_h5_score.json").read_text())
            diff_ref = abs(float(roc_auc_score(unk, ref_D)) - js["auroc"])
            say(f"{held} seed {seed}: reference AUROC {roc_auc_score(unk, ref_D):.6f} vs saved {js['auroc']:.6f} (diff {diff_ref:.2e})")
            if diff_ref > 1e-6:
                raise SystemExit(f"STOP: reference mismatch {diff_ref:.3e} for {held} seed {seed}; do not continue without a decision")
            # validation known windows for the PCRP threshold (U3)
            vst = wd.starts["val"]
            assert len(vst) > 0, "no validation windows: stop and ask"
            yv = wd.labels("val", "label19")
            Dv = dmin_rows(wd.X[vst + T - 1][yv != hid].astype(np.float64), mus_ref, precs_ref)
            q95 = float(np.quantile(Dv, 0.95))
            for m in MODELS:
                se = np.load(R3 / "score_export" / f"{m}_loao_T16_{safe}_seed{seed}.npz")
                assert len(se["y19"]) == len(unk) and (se["y19"] == y19).all(), "score_export order mismatch"
                en = -se["energy"].astype(np.float64)
                flagE = se["energy"] < float(se["thr"]); flagP = ref_D > q95; flagO = flagE | flagP
                ra = (rankdata(en) + rankdata(ref_D)) / (2.0 * len(en))
                kn = unk == 0
                w3.writerow(dict(model=m, held_out_class=held, seed=seed, n_unknown=int(unk.sum()), n_known=int(kn.sum()), energy_auroc=float(roc_auc_score(unk, en)),
                                 pcrp_auroc=float(roc_auc_score(unk, ref_D)), rankavg_auroc=float(roc_auc_score(unk, ra)), pcrp_thr_q95_known_val=q95,
                                 energy_only_caught=float(flagE[unk == 1].mean()), energy_only_known_far=float(flagE[kn].mean()),
                                 pcrp_only_caught=float(flagP[unk == 1].mean()), pcrp_only_known_far=float(flagP[kn].mean()),
                                 or_caught=float(flagO[unk == 1].mean()), or_known_far=float(flagO[kn].mean())))
            f3.flush()
            np.savez_compressed(cache / f"{safe}_seed{seed}.npz", dmin=ref_D.astype(np.float32), file_id=fid.astype(np.int32), file_names=fnames.astype(str),
                                y19=y19.astype(np.int8), hid=hid, unk_X_last=wd.X[last][unk == 1].astype(np.float32))
            say(f"{held} seed {seed} done in {time.time() - t0:.0f}s; estimators ok: {list(est)}")
            del wd, Xn, need, pos, trmask
            gc.collect()
    say("T4 and U3 finished")


if __name__ == "__main__":
    main()
