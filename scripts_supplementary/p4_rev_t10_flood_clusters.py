"""Paper 4 T10: flood-class heterogeneity (post-hoc exploratory; null results are acceptable). CPU only.
Classes: TCP_IP-DoS-UDP and MQTT-DDoS-Publish_Flood. Input: pcrp_cache_<date>/ (held-out last-row scaled features, source-file ids; made by
scripts/p4_rev_pcrp_t4_u3.py) and score_export/ (energy). Fixed now: k-means with k in {2,3,4,5}, rng 42, n_init 10, fitted on at most 50,000 held-out
windows of the run; silhouette on a random sample of 10,000 of those windows (random_state 42); cross-tabulation of clusters with source_file;
per (model, seed, cluster) energy AUROC = held-out windows of the cluster against ALL known test windows of the run (score -energy, higher = more unknown).
Pass criterion: the AUROC over all held-out windows of a run (no clustering) equals A7_energy_auroc of per_run.csv within 1e-9.
Plot: PCA 2-D of the held-out windows coloured by k-means cluster (k = 3) for inspection only.
Usage: python scripts/p4_rev_t10_flood_clusters.py --date 20261121
"""
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
FLOOD = ["TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
KS = [2, 3, 4, 5]
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
OUT = R3 / f"p4_review_analyses_{a.date}" / "T10"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
CACHE = R3 / f"pcrp_cache_{a.date}"
PR = pd.read_csv(R3 / "saved_output_analyses_20261118" / "per_run.csv").set_index(["model", "held_out_class", "seed"])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

crow, arow, sil, chk = [], [], [], []
fig, axs = plt.subplots(len(FLOOD), len(SEEDS), figsize=(18, 7))
for ci, held in enumerate(FLOOD):
    safe = held.replace("/", "_")
    for si, seed in enumerate(SEEDS):
        c = np.load(CACHE / f"{safe}_seed{seed}.npz")
        y19, hid = c["y19"].astype(int), int(c["hid"])
        unk = y19 == hid
        X = c["unk_X_last"].astype(np.float64)
        assert len(X) == unk.sum()
        fnames = c["file_names"].astype(str)
        ffile = fnames[c["file_id"][unk]]
        rng = np.random.default_rng(42)
        fit_idx = np.sort(rng.choice(len(X), min(50_000, len(X)), replace=False))
        sil_idx = rng.choice(fit_idx, min(10_000, len(fit_idx)), replace=False)
        en = {m: -np.load(R3 / "score_export" / f"{m}_loao_T16_{safe}_seed{seed}.npz")["energy"].astype(np.float64) for m in MODELS}
        for m in MODELS:
            assert len(en[m]) == len(unk)
            exact = float(roc_auc_score(unk.astype(int), en[m]))
            chk.append(abs(exact - float(PR.loc[(m, held, seed), "A7_energy_auroc"])))
        for k in KS:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X[fit_idx])
            lab = km.predict(X)
            sil.append(dict(held_out_class=held, seed=seed, k=k, silhouette=float(silhouette_score(X[sil_idx], km.predict(X[sil_idx]))), n_heldout=len(X),
                            cluster_sizes=";".join(str(int((lab == j).sum())) for j in range(k))))
            ct = pd.crosstab(lab, ffile)
            for j in ct.index:
                for fn, n in ct.loc[j].items():
                    if n:
                        crow.append(dict(held_out_class=held, seed=seed, k=k, cluster=int(j), source_file=fn, n_windows=int(n)))
            for m in MODELS:
                known = en[m][~unk]
                for j in range(k):
                    mj = lab == j
                    if mj.sum() == 0:
                        continue
                    arow.append(dict(held_out_class=held, seed=seed, model=m, k=k, cluster=j, n_cluster=int(mj.sum()),
                                     energy_auroc=float(roc_auc_score(np.r_[np.ones(mj.sum()), np.zeros(len(known))], np.r_[en[m][unk][mj], known]))))
        km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X[fit_idx]); lab3 = km3.predict(X)
        p = PCA(n_components=2, random_state=42).fit(X[fit_idx]).transform(X[sil_idx])
        axs[ci, si].scatter(p[:, 0], p[:, 1], c=km3.predict(X[sil_idx]), s=2, cmap="tab10"); axs[ci, si].set_title(f"{held} seed {seed}", fontsize=8)
        print(held, seed, "done", flush=True)
fig.tight_layout(); fig.savefig(OUT / "figures" / "pca_clusters_k3.png", dpi=200); fig.savefig(OUT / "figures" / "pca_clusters_k3.pdf"); plt.close(fig)
pd.DataFrame(crow).to_csv(OUT / "clusters.csv", index=False)
pd.DataFrame(arow).to_csv(OUT / "auroc_by_cluster.csv", index=False)
sd = pd.DataFrame(sil); sd.to_csv(OUT / "silhouette.csv", index=False)
(OUT / "README.md").write_text(
    "# T10 flood-class heterogeneity (post-hoc exploratory)\n\n"
    f"Script `scripts/p4_rev_t10_flood_clusters.py` SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; Python {sys.version.split()[0]}, numpy {np.__version__}; rng 42; {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
    f"Pass check: max |all-window AUROC - A7_energy_auroc| over {len(chk)} runs = {max(chk):.2e} (criterion 1e-9).\n\n"
    "Mean silhouette by class and k:\n\n" + sd.groupby(["held_out_class", "k"]).silhouette.mean().round(4).reset_index().to_markdown(index=False) + "\n\n"
    "Files: `clusters.csv` (cluster by source file), `auroc_by_cluster.csv`, `silhouette.csv`, `figures/`. Whatever is found is reported; a null result is acceptable.\n", encoding="utf-8")
print("done; max check diff", max(chk))
