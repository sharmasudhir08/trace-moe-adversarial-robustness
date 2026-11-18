"""E5 — PCRP variants: y_true, min_c, ROC-AUC as adversarial detector.

Computes per-window Mahalanobis distances with three class choices:
  - predicted: D(x, c_hat)  — current H6 method
  - y_true:    D(x, y_true) — ground truth category
  - min_c:     min_c D(x, c) — closest centroid regardless of label

Then evaluates each as a binary detector (clean=0 vs adversarial=1) using ROC-AUC
and TPR@1%FPR and TPR@5%FPR.

Runs on all seeds and epsilon values in {0, 0.05, 0.1, 0.25, 0.5}.

Usage:
    python scripts/e5_pcrp_variants.py --config configs/paper2_robustness.yaml --regime official
"""
import json, sys, pathlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from _common import setup
from trace_iomt.attacks import pgd, constraint_tensors, attack_eval_starts
from trace_iomt.experiment import load_or_train_deep, load_prepared
from trace_iomt.train import iterate
from trace_iomt.utils import resolve_device

SEEDS    = [42, 43, 44, 45, 46]
EPS_LIST = [0.0, 0.05, 0.1, 0.25, 0.5]
MODELS   = ["trace_moe", "bilstm_dnn", "transformer", "tfsage_lite"]
TWIN_PATH = pathlib.Path("results_v2/twin/twin_profile_main.npz")


def add_args(p):
    p.add_argument("--regime", default="official")


def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def load_twin():
    tp = np.load(str(TWIN_PATH))
    return {
        "mus":          tp["mus"].astype(np.float64),          # (6, 45)
        "inv_covs":     tp["inv_covs"].astype(np.float64),     # (6, 45, 45)
        "category_names": tp["category_names"].tolist(),
        "scaler_center": tp["scaler_center"].astype(np.float64),
        "scaler_scale":  tp["scaler_scale"].astype(np.float64),
        "scaler_clip":   float(tp["scaler_clip"]),
    }


def maha_dist(x_row, mu, inv_cov):
    diff = x_row - mu
    return float(np.sqrt(diff @ inv_cov @ diff))


def compute_distances_batch(X_last, true_cat6, pred_cat6, twin):
    """
    X_last:    (N, 45) last-flow features (raw scaled values)
    true_cat6: (N,) int — true 6-category labels
    pred_cat6: (N,) int — predicted 6-category labels
    Returns arrays of shape (N,) for predicted, y_true, min_c variants.
    """
    mus, inv_covs = twin["mus"], twin["inv_covs"]
    N = len(X_last)
    d_pred   = np.empty(N)
    d_true   = np.empty(N)
    d_min    = np.empty(N)
    for i in range(N):
        x = X_last[i]
        pc, tc = int(pred_cat6[i]), int(true_cat6[i])
        d_pred[i] = maha_dist(x, mus[pc], inv_covs[pc])
        d_true[i] = maha_dist(x, mus[tc], inv_covs[tc])
        d_min[i]  = min(maha_dist(x, mus[c], inv_covs[c]) for c in range(len(mus)))
    return d_pred, d_true, d_min


def tpr_at_fpr(scores, labels, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(idx, len(tpr)-1)])


def eval_detector(d_clean, d_adv):
    """Return dict of detection metrics using Mahalanobis distance as detector score
    (higher D = more anomalous = adversarial)."""
    scores = np.concatenate([d_clean, d_adv])
    labels = np.concatenate([np.zeros(len(d_clean)), np.ones(len(d_adv))])
    if len(np.unique(labels)) < 2:
        return {"roc_auc": float("nan"), "tpr_at_1pct_fpr": float("nan"), "tpr_at_5pct_fpr": float("nan")}
    return {
        "roc_auc":           float(roc_auc_score(labels, scores)),
        "tpr_at_1pct_fpr":   tpr_at_fpr(scores, labels, 0.01),
        "tpr_at_5pct_fpr":   tpr_at_fpr(scores, labels, 0.05),
        "mean_d_clean":      float(d_clean.mean()),
        "mean_d_adv":        float(d_adv.mean()),
    }


def run_model_seed(model, wd, cfg, device, starts, twin):
    rc = cfg["robustness"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = {i for i, c in enumerate(wd.hierarchy.classes["label19"])
                if wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0}
    model.eval()

    # Collect clean window data
    clean_last, clean_pred6, clean_true6, clean_is_att = [], [], [], []
    for x, p, y in iterate(wd, starts, 512, device):
        is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
        with torch.no_grad():
            logits, _ = model(x, p)
        pred6 = logits["label6"].argmax(-1).cpu().numpy()
        true6 = y["label6"].cpu().numpy()
        last  = x[:, -1].cpu().numpy()   # last flow raw features
        clean_last.append(last)
        clean_pred6.append(pred6)
        clean_true6.append(true6)
        clean_is_att.append(is_attack.cpu().numpy())

    clean_last    = np.concatenate(clean_last)
    clean_pred6   = np.concatenate(clean_pred6)
    clean_true6   = np.concatenate(clean_true6)
    clean_is_att  = np.concatenate(clean_is_att).astype(bool)

    # Clean distances (attack windows only as reference for the detector)
    d_pred_clean, d_true_clean, d_min_clean = compute_distances_batch(
        clean_last[clean_is_att], clean_true6[clean_is_att], clean_pred6[clean_is_att], twin)

    results = {}
    for eps in EPS_LIST:
        if eps == 0.0:
            results["0.0"] = {
                "predicted": {"mean_d": float(d_pred_clean.mean()), "std_d": float(d_pred_clean.std())},
                "y_true":    {"mean_d": float(d_true_clean.mean()), "std_d": float(d_true_clean.std())},
                "min_c":     {"mean_d": float(d_min_clean.mean()),  "std_d": float(d_min_clean.std())},
            }
            continue

        # Adversarial pass
        adv_last, adv_pred6, adv_true6 = [], [], []
        for x, p, y in iterate(wd, starts, 512, device):
            is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
            if not is_attack.any():
                continue
            adv = pgd(model, x[is_attack], p[is_attack], y["label19"][is_attack],
                      eps, rc["pgd_steps"], rc["pgd_step_frac"], mask, lower, upper,
                      rand_init=True, momentum=rc.get("momentum", 0.0))
            with torch.no_grad():
                logits, _ = model(adv, p[is_attack])
            pred6 = logits["label6"].argmax(-1).cpu().numpy()
            true6 = y["label6"][is_attack].cpu().numpy()
            last  = adv[:, -1].cpu().numpy()
            adv_last.append(last); adv_pred6.append(pred6); adv_true6.append(true6)

        if not adv_last:
            continue
        adv_last  = np.concatenate(adv_last)
        adv_pred6 = np.concatenate(adv_pred6)
        adv_true6 = np.concatenate(adv_true6)

        d_pred_adv, d_true_adv, d_min_adv = compute_distances_batch(adv_last, adv_true6, adv_pred6, twin)

        results[str(eps)] = {
            "predicted": {**eval_detector(d_pred_clean, d_pred_adv)},
            "y_true":    {**eval_detector(d_true_clean, d_true_adv)},
            "min_c":     {**eval_detector(d_min_clean, d_min_adv)},
        }
        for variant in ("predicted", "y_true", "min_c"):
            r = results[str(eps)][variant]
            print(f"  eps={eps}  {variant:10s}  AUC={r.get('roc_auc', float('nan')):.4f}"
                  f"  mean_D_clean={r['mean_d_clean']:.2f}  mean_D_adv={r['mean_d_adv']:.2f}", flush=True)
    return results


def main():
    args, cfg, log = setup("E5 PCRP variants", "e5_pcrp_variants", add_args)
    df = load_prepared(cfg)
    device = resolve_device(cfg["train"]["device"])
    twin = load_twin()
    all_results = {}

    for model_name in MODELS:
        all_results[model_name] = {}
        wd = None
        for seed in SEEDS:
            print(f"\n=== {model_name}  seed {seed} ===", flush=True)
            set_seed(seed)
            try:
                model, wd = load_or_train_deep(df, cfg, model_name, args.regime, seed, "main", log,
                                               strict=True, wd=wd)
            except Exception as e:
                print(f"  SKIP: {e}", flush=True)
                continue
            model.eval()
            starts = attack_eval_starts(wd, cfg["robustness"]["max_eval_windows"], seed, 0)
            all_results[model_name][str(seed)] = run_model_seed(model, wd, cfg, device, starts, twin)

    out_path = pathlib.Path(cfg["paths"]["results_dir"]) / "e5_pcrp_variants" / "pcrp_variants.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)

    # Summary: ROC-AUC at eps=0.25 for predicted vs y_true vs min_c
    print("\n=== ROC-AUC summary (mean over seeds, eps=0.25) ===")
    print(f"  {'model':20s}  predicted   y_true     min_c")
    for m in MODELS:
        vals = {v: [] for v in ("predicted", "y_true", "min_c")}
        for s in SEEDS:
            r = all_results[m].get(str(s), {}).get("0.25", {})
            for v in vals:
                vals[v].append(r.get(v, {}).get("roc_auc", float("nan")))
        means = {v: np.nanmean(vals[v]) for v in vals}
        print(f"  {m:20s}  {means['predicted']:.4f}      {means['y_true']:.4f}     {means['min_c']:.4f}")


if __name__ == "__main__":
    main()
