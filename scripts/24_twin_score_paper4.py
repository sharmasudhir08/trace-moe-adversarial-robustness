"""Step 24 — Score Paper 4 H5 twin-alone AUROC per (model x held_out_class x seed).

For each LOAO run under results_v3_paper3/zero_day/<model>_loao_T16_<class>_seed<s>/:
  1. Load predictions.npz (proba_label19, proba_label6, y_true_label19).
  2. Rebuild windowed test features with build_windowed(regime='loao',
     loao_class=<class>, seed=<s>) — same pipeline Paper 4 inference used.
  3. For each test window, take the LAST-ROW feature vector.
  4. Load that class's LOAO-specific twin profile from
     results_v3_paper3/twin/twin_profile_loao_<class>.npz.
  5. Compute per-window min-cluster Mahalanobis discrepancy D_min(x) =
     min over categories { Mahalanobis(x, mu_c, inv_cov_c) }. This is the
     "nearest-known-cluster" distance — a natural standalone signal for
     unknown-attack detection: known-class flows should sit near their cluster
     (low D_min), unknown held-out-class flows should sit far from every
     known cluster (high D_min).
  6. Label held-out-class test rows as positive (unknown), other rows as negative.
  7. Compute AUROC and FPR@95TPR of D_min as a score for unknown-vs-known separation.
  8. Write results_v3_paper3/zero_day/<run>/twin_h5_score.json.

Pre-registered by Paper 4 Charter v1.2 (DRAFT; not yet locked).
CPU only. No modification to existing metrics.json or ood block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.experiment import load_prepared  # noqa: E402
from trace_iomt.data.windows import build_windowed  # noqa: E402

MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
# Default seed list; will fall back to whatever runs exist on disk
SEEDS = [42, 43, 44, 45, 46]
CLASSES = [
    "ARP_Spoofing",
    "Recon-VulScan",
    "MQTT-Malformed_Data",
    "TCP_IP-DoS-UDP",
    "MQTT-DDoS-Publish_Flood",
]


def mahalanobis_batch(X, mu, inv_cov):
    diff = X - mu[None, :]
    tmp = diff @ inv_cov
    md2 = np.einsum("nd,nd->n", tmp, diff)
    md2 = np.maximum(md2, 0.0)
    d = X.shape[1]
    return np.sqrt(md2 / d)


def ood_metrics_from_score(score, is_unknown):
    """AUROC of score separating unknown (1) from known (0), plus FPR at TPR=0.95."""
    y = is_unknown.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": float("nan"), "fpr_at_95tpr": float("nan"),
                "n_known": int((1 - y).sum()), "n_unknown": int(y.sum())}
    fpr, tpr, _ = roc_curve(y, score)
    auroc = float(roc_auc_score(y, score))
    idx = np.searchsorted(tpr, 0.95, side="left")
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])
    return {"auroc": auroc, "fpr_at_95tpr": fpr95,
            "n_known": int((1 - y).sum()), "n_unknown": int(y.sum())}


def main():
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    res_dir = Path(cfg["paths"]["results_dir"])
    twin_dir = res_dir / "twin"
    assert twin_dir.exists(), f"Missing twin dir: {twin_dir}"

    df = load_prepared(cfg)
    print(f"[data] df shape = {df.shape}")

    T = int(cfg["windows"]["length"])

    n_done = n_skipped = n_errors = 0
    for held in CLASSES:
        safe = held.replace("/", "_")
        prof_file = twin_dir / f"twin_profile_loao_{safe}.npz"
        if not prof_file.exists():
            print(f"[skip] no profile for {held!r}")
            continue
        prof = np.load(prof_file, allow_pickle=False)
        mus = prof["mus"]
        inv_covs = prof["inv_covs"]
        category_names = [str(x) for x in prof["category_names"]]
        feature_names = [str(x) for x in prof["feature_names"]]
        print(f"\n[{held!r}] twin has {len(category_names)} categories: {category_names}")

        for seed in SEEDS:
            # Build windowed LOAO data for this held-out class with this seed
            c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held}}
            try:
                wd = build_windowed(df, c2, regime="loao", T=T, seed=seed)
            except Exception as exc:
                n_errors += 1
                print(f"  [err seed {seed}] build_windowed failed: {exc}")
                continue
            if wd.features != feature_names:
                n_errors += 1
                print(f"  [err seed {seed}] feature-order mismatch")
                continue
            starts_test = wd.starts["test"]
            if len(starts_test) == 0:
                n_skipped += 1
                continue
            last_idx = starts_test + T - 1
            X_last = wd.X[last_idx].astype(np.float64)
            y19 = wd.labels("test", "label19")
            # held-out class index in hierarchy label19 order
            hid = wd.hierarchy.index["label19"].get(held)
            if hid is None:
                n_errors += 1
                print(f"  [err seed {seed}] {held!r} not in hierarchy")
                continue
            is_unknown = (y19 == hid)

            # Per-category Mahalanobis, then min over categories (D_min = nearest known cluster distance)
            D_per_cat = np.zeros((len(category_names), len(X_last)), dtype=np.float64)
            for ci in range(len(category_names)):
                D_per_cat[ci] = mahalanobis_batch(X_last, mus[ci], inv_covs[ci])
            D_min = D_per_cat.min(axis=0)
            D_argmin_cat = D_per_cat.argmin(axis=0)

            for model in MODELS:
                run_dir = res_dir / "zero_day" / f"{model}_loao_T16_{safe}_seed{seed}"
                preds_file = run_dir / "predictions.npz"
                if not preds_file.exists():
                    n_skipped += 1
                    continue
                # ood_metrics from D_min
                m = ood_metrics_from_score(D_min, is_unknown)
                m.update({
                    "held_out_class": held,
                    "seed": int(seed),
                    "model": model,
                    "mean_D_min_unknown": float(D_min[is_unknown].mean()) if is_unknown.any() else float("nan"),
                    "mean_D_min_known": float(D_min[~is_unknown].mean()) if (~is_unknown).any() else float("nan"),
                    "std_D_min": float(D_min.std()),
                    "twin_profile": str(prof_file.relative_to(ROOT)),
                    "n_test_windows": int(len(X_last)),
                    "charter_version": "Paper 4 Charter v1.2 (DRAFT), H5 twin-alone AUROC",
                })
                (run_dir / "twin_h5_score.json").write_text(json.dumps(m, indent=2))
                n_done += 1
            print(f"  [seed {seed}] N_test = {len(X_last):,}  D_min(known)={D_min[~is_unknown].mean():.3f}  "
                  f"D_min(unknown)={D_min[is_unknown].mean():.3f}  AUROC={m['auroc']:.4f}  FPR@95={m['fpr_at_95tpr']:.4f}")

    print(f"\n[done] runs scored: {n_done}  skipped (no predictions): {n_skipped}  errors: {n_errors}")


if __name__ == "__main__":
    main()
