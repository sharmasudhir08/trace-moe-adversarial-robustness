"""Paper 4 T6: window/file-level (cluster) bootstrap for H1 (caught-any, TRACE-MoE), H2 (energy AUROC minus MSP AUROC, per model) and H5 (PCRP AUROC).
Post-hoc exploratory; no verdict is recomputed or changed. CPU only. Needs pcrp_cache_<date>/ (made by scripts/p4_rev_pcrp_t4_u3.py) and score_export/.

Two-level bootstrap, 2,000 resamples (fixed now, rng 42): each resample (i) draws 5 seeds with replacement (paired across metrics) and (ii) draws test
source_file clusters with replacement, separately among held-out files and among known files (one draw per resample, applied by file name to every seed).
AUROC under file weights is computed from per-file score histograms (4,096 quantile bins per run; ties inside a bin count one half); the binned point
estimate is compared with the exact AUROC and the maximum difference is reported. Seed-only intervals (file set fixed) are recomputed in the same script.
A held-out class with fewer than 2 test files cannot be cluster-bootstrapped: reported, and only the seed-only interval is given.
Usage: python scripts/p4_rev_t6_cluster_bootstrap.py --date 20261121
"""
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
NB, NBIN = 2000, 4096
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
OUT = R3 / f"p4_review_analyses_{a.date}" / "T6"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = R3 / f"pcrp_cache_{a.date}"
rng = np.random.default_rng(42)


def file_hist(score, unk, fid, nfile):
    """per-file histograms of the score for unknown and known windows."""
    edges = np.unique(np.quantile(score, np.linspace(0, 1, NBIN + 1)))
    b = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 1)
    nb = len(edges)
    u = unk == 1
    Hu = np.bincount(fid[u] * nb + b[u], minlength=nfile * nb).reshape(nfile, nb).astype(np.float64)
    Hk = np.bincount(fid[~u] * nb + b[~u], minlength=nfile * nb).reshape(nfile, nb).astype(np.float64)
    return Hu, Hk


def auc_from_hist(hu, hk):
    cu, ck = hu.sum(), hk.sum()
    cum_k = np.cumsum(hk) - hk
    return float((hu * (cum_k + 0.5 * hk)).sum() / (cu * ck))


def main():
    t0 = time.time()
    # per class: file names (union), per seed: per-file stats per metric
    res_point, rows = {}, []
    H = {}   # (class, metric, seed) -> (Hu, Hk, names) or ('caught', n_caught_per_file, n_unk_per_file)
    status = []
    maxdiff = 0.0
    for held in CLASSES:
        safe = held.replace("/", "_")
        H.clear()
        allnames = set()
        data = {}
        for s in SEEDS:
            c = np.load(CACHE / f"{safe}_seed{s}.npz")
            data[s] = c
            allnames |= set(c["file_names"].tolist())
        names = sorted(allnames); nidx = {n: i for i, n in enumerate(names)}
        for s in SEEDS:
            c = data[s]
            fid = np.array([nidx[n] for n in c["file_names"].tolist()])[c["file_id"]]
            unk = (c["y19"].astype(int) == int(c["hid"])).astype(int)
            nf = len(names)
            H[(held, "pcrp", s)] = (*file_hist(c["dmin"].astype(np.float64), unk, fid, nf), None)
            exact = float(roc_auc_score(unk, c["dmin"]))
            maxdiff = max(maxdiff, abs(auc_from_hist(H[(held, "pcrp", s)][0].sum(0), H[(held, "pcrp", s)][1].sum(0)) - exact))
            res_point[(held, "pcrp", s)] = exact
            for m in MODELS:
                se = np.load(R3 / "score_export" / f"{m}_loao_T16_{safe}_seed{s}.npz")
                assert len(se["y19"]) == len(unk) and (se["y19"].astype(int) == c["y19"].astype(int)).all(), "order mismatch"
                en, ms = -se["energy"].astype(np.float64), -se["msp"].astype(np.float64)
                H[(held, f"energy_{m}", s)] = (*file_hist(en, unk, fid, nf), None)
                H[(held, f"msp_{m}", s)] = (*file_hist(ms, unk, fid, nf), None)
                ea, ma = float(roc_auc_score(unk, en)), float(roc_auc_score(unk, ms))
                maxdiff = max(maxdiff, abs(auc_from_hist(H[(held, f"energy_{m}", s)][0].sum(0), H[(held, f"energy_{m}", s)][1].sum(0)) - ea), abs(auc_from_hist(H[(held, f"msp_{m}", s)][0].sum(0), H[(held, f"msp_{m}", s)][1].sum(0)) - ma))
                res_point[(held, f"energy_{m}", s)], res_point[(held, f"msp_{m}", s)] = ea, ma
                if m == "trace_moe":
                    caught = ((se["energy"] < float(se["thr"])) | (se["p_attack"] > 0.5)) & (unk == 1)
                    nc = np.bincount(fid[caught], minlength=nf).astype(float)
                    nu = np.bincount(fid[unk == 1], minlength=nf).astype(float)
                    H[(held, "caught", s)] = (nc, nu, None)
                    res_point[(held, "caught", s)] = float(caught.sum() / unk.sum())
        # held-out and known file lists (by name)
        hu_files = [i for i in range(len(names)) if sum(H[(held, "pcrp", s)][0][i].sum() for s in SEEDS) > 0]
        kn_files = [i for i in range(len(names)) if sum(H[(held, "pcrp", s)][1][i].sum() for s in SEEDS) > 0]
        status.append(dict(held_out_class=held, n_heldout_files=len(hu_files), n_known_files=len(kn_files)))
        print(held, "held-out files", len(hu_files), "known files", len(kn_files), f"{time.time() - t0:.0f}s", flush=True)
        # bootstrap
        seed_draw = rng.integers(0, len(SEEDS), (NB, len(SEEDS)))
        can_cluster = len(hu_files) >= 2 and len(kn_files) >= 2
        wu = np.zeros((NB, len(names))); wk = np.zeros((NB, len(names)))
        if can_cluster:
            for b in range(NB):
                wu[b] = np.bincount(rng.choice(hu_files, len(hu_files)), minlength=len(names))
                wk[b] = np.bincount(rng.choice(kn_files, len(kn_files)), minlength=len(names))
        ones_u, ones_k = np.zeros(len(names)), np.zeros(len(names)); ones_u[hu_files] = 1; ones_k[kn_files] = 1

        def metric_seed(kind, s, wu_, wk_):
            if kind == "caught":
                nc, nu, _ = H[(held, "caught", s)]
                return float((wu_ * nc).sum() / (wu_ * nu).sum())
            hu, hk, _ = H[(held, kind, s)]
            return auc_from_hist(wu_ @ hu, wk_ @ hk)

        kinds = ["caught", "pcrp"] + [f"energy_{m}" for m in MODELS] + [f"msp_{m}" for m in MODELS]
        # per (b, seed) values, seed-only (weights = 1) and cluster
        seed_only = {k: np.array([metric_seed(k, s, ones_u, ones_k) for s in SEEDS]) for k in kinds}
        clus = {k: np.zeros((NB, len(SEEDS))) for k in kinds} if can_cluster else None
        if can_cluster:
            for b in range(NB):
                for k in kinds:
                    for j, s in enumerate(SEEDS):
                        clus[k][b, j] = metric_seed(k, s, wu[b], wk[b])
        # window-level bootstrap (used when a class has fewer than 2 held-out files): windows resampled with replacement, via multinomial draws of the score histograms
        wrng = np.random.default_rng(43)
        win = {k: np.zeros((NB, len(SEEDS))) for k in kinds}
        for j, s_ in enumerate(SEEDS):
            for k in kinds:
                if k == "caught":
                    nc, nu_, _ = H[(held, "caught", s_)]
                    p_, n_ = nc.sum() / nu_.sum(), int(nu_.sum())
                    win[k][:, j] = wrng.binomial(n_, p_, NB) / n_
                else:
                    hu, hk, _ = H[(held, k, s_)]
                    hus, hks = hu.sum(0), hk.sum(0)
                    nu_, nk_ = int(hus.sum()), int(hks.sum())
                    U = wrng.multinomial(nu_, hus / nu_, size=NB).astype(float); K_ = wrng.multinomial(nk_, hks / nk_, size=NB).astype(float)
                    cum_k = np.cumsum(K_, axis=1) - K_
                    win[k][:, j] = (U * (cum_k + 0.5 * K_)).sum(1) / (nu_ * nk_)
        targets = [("H1_caught_any_trace_moe", ["caught"], None), ("H5_pcrp_auroc", ["pcrp"], None)] + \
                  [(f"H2_energy_minus_msp_{m}", [f"energy_{m}"], f"msp_{m}") for m in MODELS]
        for name, (k1,), k2 in [(t[0], t[1], t[2]) for t in targets]:
            f = (lambda D: D[k1] - D[k2]) if k2 else (lambda D: D[k1])
            pt = f(seed_only).mean()
            so = np.array([f(seed_only)[draw].mean() for draw in seed_draw])
            row = dict(held_out_class=held, quantity=name, point_mean_over_seeds=float(pt), seed_only_lo=float(np.percentile(so, 2.5)), seed_only_hi=float(np.percentile(so, 97.5)),
                       n_heldout_files=len(hu_files), n_known_files=len(kn_files))
            wl = np.array([f({k: win[k][b] for k in (k1, k2) if k})[seed_draw[b]].mean() for b in range(NB)])
            row.update(window_lo=float(np.percentile(wl, 2.5)), window_hi=float(np.percentile(wl, 97.5)))
            if can_cluster:
                cl = np.array([f({k: clus[k][b] for k in (k1, k2) if k})[seed_draw[b]].mean() for b in range(NB)])
                row.update(cluster_lo=float(np.percentile(cl, 2.5)), cluster_hi=float(np.percentile(cl, 97.5)))
            else:
                row.update(cluster_lo=np.nan, cluster_hi=np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cluster_bootstrap.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "file_counts.csv", index=False)
    # point-estimate checks against the paper's saved tables
    h1 = pd.read_csv(R3 / "h8_verdict" / "H1_safety_view.csv").set_index("class")["mean"]
    h5 = pd.read_csv(R3 / "h8_verdict" / "twin_h5_verdict.csv")
    h5 = h5[h5.model == "trace_moe"].set_index("held_out_class")["mean_auroc"]
    d1 = max(abs(float(out[(out.held_out_class == c) & (out.quantity == "H1_caught_any_trace_moe")].point_mean_over_seeds.iloc[0]) - h1[c]) for c in CLASSES)
    d5 = max(abs(float(out[(out.held_out_class == c) & (out.quantity == "H5_pcrp_auroc")].point_mean_over_seeds.iloc[0]) - h5[c]) for c in CLASSES)
    (OUT / "README.md").write_text(
        "# T6 cluster bootstrap (post-hoc exploratory)\n\n"
        f"Script `scripts/p4_rev_t6_cluster_bootstrap.py` SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; Python {sys.version.split()[0]}, numpy {np.__version__}; rng 42; {NB} resamples; {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
        f"Checks: max |binned AUROC - exact AUROC| over all runs = {maxdiff:.2e}; max |point H1 mean - H1_safety_view.csv| = {d1:.2e}; max |point H5 mean - twin_h5_verdict.csv| = {d5:.2e} (H5 table is rounded to 6 decimals).\n"
        "Seed-only intervals use 2,000 resamples here and may differ slightly from the paper's 10,000-resample intervals (rng stream differs).\n\n"
        + out.round(4).to_markdown(index=False) + "\n\nFile counts per class: see `file_counts.csv`. Classes with fewer than 2 held-out files have no cluster interval.\n", encoding="utf-8")
    print("done", f"max binned diff {maxdiff:.2e}", f"H1 diff {d1:.2e}", f"H5 diff {d5:.2e}")


if __name__ == "__main__":
    main()
