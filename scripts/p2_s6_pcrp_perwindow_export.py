"""Paper 2, S6: save per-window PCRP scores so that detector analyses can be done from saved artifacts.

Same procedure as scripts/e5_pcrp_variants.py (same saved clean checkpoints loaded with strict=True, same evaluation
windows, same seeds, same 20-step PGD settings from the config, same Mahalanobis code imported from e5), but the
per-window quantities are written to disk instead of only the summary metrics. NOTHING IS TRAINED: clean checkpoints
are loaded, PGD is run on attack windows, scores are computed. The fine-tuned models are NOT part of this script.

Per model and seed it writes results_v3_paper2/e5_pcrp_perwindow/<model>/seed<s>.npz with
  clean_*  (attack windows, eps 0): true19, true6, pred19, pred6, d_pred, d_true, d_min
  eps<e>_* (the same windows after PGD at each eps in 0.05, 0.1, 0.25, 0.5): same keys
and results_v3_paper2/e5_pcrp_perwindow/<model>/seed<s>_check.json, which recomputes ROC-AUC and TPR at 1% and 5% FPR
from the saved arrays and compares them with the stored results_v3_paper2/e5_pcrp_variants/pcrp_variants.json.
Existing seed files are skipped unless --overwrite is given (nothing is overwritten silently).

Usage (from the repository root, one process per model so they can run in parallel):
  python scripts/p2_s6_pcrp_perwindow_export.py --config configs/paper2_robustness.yaml --regime official --models trace_moe
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))
from _common import setup  # noqa: E402
from e5_pcrp_variants import EPS_LIST, SEEDS, compute_distances_batch, eval_detector, load_twin, set_seed  # noqa: E402
from trace_iomt.attacks import attack_eval_starts, constraint_tensors, pgd  # noqa: E402
from trace_iomt.experiment import load_or_train_deep, load_prepared  # noqa: E402
from trace_iomt.train import iterate  # noqa: E402
from trace_iomt.utils import resolve_device  # noqa: E402

ALL_MODELS = ["trace_moe", "bilstm_dnn", "transformer", "tfsage_lite"]


def add(p):
    p.add_argument("--regime", default="official")
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--overwrite", action="store_true")


def export_one(model, wd, cfg, device, starts, twin):
    rc = cfg["robustness"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = {i for i, c in enumerate(wd.hierarchy.classes["label19"])
                if wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0}
    model.eval()
    out = {}
    # clean pass (identical order and filtering to e5)
    c_last, c_p6, c_p19, c_t6, c_t19, c_att = [], [], [], [], [], []
    for x, p, y in iterate(wd, starts, 512, device):
        is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
        with torch.no_grad():
            logits, _ = model(x, p)
        c_p6.append(logits["label6"].argmax(-1).cpu().numpy())
        c_p19.append(logits["label19"].argmax(-1).cpu().numpy())
        c_t6.append(y["label6"].cpu().numpy())
        c_t19.append(y["label19"].cpu().numpy())
        c_last.append(x[:, -1].cpu().numpy())
        c_att.append(is_attack.cpu().numpy())
    c_last, c_p6, c_p19 = np.concatenate(c_last), np.concatenate(c_p6), np.concatenate(c_p19)
    c_t6, c_t19, c_att = np.concatenate(c_t6), np.concatenate(c_t19), np.concatenate(c_att).astype(bool)
    dp, dt, dm = compute_distances_batch(c_last[c_att], c_t6[c_att], c_p6[c_att], twin)
    out.update({"clean_true19": c_t19[c_att], "clean_true6": c_t6[c_att], "clean_pred19": c_p19[c_att],
                "clean_pred6": c_p6[c_att], "clean_d_pred": dp, "clean_d_true": dt, "clean_d_min": dm})
    clean_d = (dp, dt, dm)
    check = {}
    for eps in EPS_LIST:
        if eps == 0.0:
            continue
        a_last, a_p6, a_p19, a_t6, a_t19 = [], [], [], [], []
        for x, p, y in iterate(wd, starts, 512, device):
            is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
            if not is_attack.any():
                continue
            adv = pgd(model, x[is_attack], p[is_attack], y["label19"][is_attack], eps, rc["pgd_steps"],
                      rc["pgd_step_frac"], mask, lower, upper, rand_init=True, momentum=rc.get("momentum", 0.0))
            with torch.no_grad():
                logits, _ = model(adv, p[is_attack])
            a_p6.append(logits["label6"].argmax(-1).cpu().numpy())
            a_p19.append(logits["label19"].argmax(-1).cpu().numpy())
            a_t6.append(y["label6"][is_attack].cpu().numpy())
            a_t19.append(y["label19"][is_attack].cpu().numpy())
            a_last.append(adv[:, -1].cpu().numpy())
        a_last, a_p6, a_p19 = np.concatenate(a_last), np.concatenate(a_p6), np.concatenate(a_p19)
        a_t6, a_t19 = np.concatenate(a_t6), np.concatenate(a_t19)
        ap, at, am = compute_distances_batch(a_last, a_t6, a_p6, twin)
        k = f"eps{eps}"
        out.update({f"{k}_true19": a_t19, f"{k}_true6": a_t6, f"{k}_pred19": a_p19, f"{k}_pred6": a_p6,
                    f"{k}_d_pred": ap, f"{k}_d_true": at, f"{k}_d_min": am})
        check[str(eps)] = {"predicted": eval_detector(clean_d[0], ap), "y_true": eval_detector(clean_d[1], at),
                           "min_c": eval_detector(clean_d[2], am)}
        print(f"  eps={eps}  windows={len(ap)}  AUC predicted={check[str(eps)]['predicted']['roc_auc']:.4f}"
              f"  min_c={check[str(eps)]['min_c']['roc_auc']:.4f}", flush=True)
    return out, check


def main():
    args, cfg, log = setup("S6 PCRP per-window export", "p2_s6_pcrp_perwindow_export", add)
    df = load_prepared(cfg)
    device = resolve_device(cfg["train"]["device"])
    if device.type != "cuda":
        raise SystemExit("GPU required (project rule: never run on CPU); torch.cuda.is_available() is False")
    twin = load_twin()
    root = pathlib.Path(cfg["paths"]["results_dir"])
    stored_path = root / "e5_pcrp_variants" / "pcrp_variants.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8")) if stored_path.exists() else {}
    for mname in args.models:
        wd = None
        outdir = root / "e5_pcrp_perwindow" / mname
        outdir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            f = outdir / f"seed{seed}.npz"
            if f.exists() and not args.overwrite:
                print(f"SKIP {f} exists (use --overwrite)", flush=True)
                continue
            print(f"\n=== {mname} seed {seed} ===", flush=True)
            set_seed(seed)
            model, wd = load_or_train_deep(df, cfg, mname, args.regime, seed, "main", log, strict=True, wd=wd)
            model.eval()
            starts = attack_eval_starts(wd, cfg["robustness"]["max_eval_windows"], seed, 0)
            arrays, check = export_one(model, wd, cfg, device, starts, twin)
            np.savez_compressed(f, **arrays)
            cmp = {}
            for e, per_v in check.items():
                for v, r in per_v.items():
                    s = stored.get(mname, {}).get(str(seed), {}).get(e, {}).get(v, {})
                    cmp[f"{e}/{v}"] = {k: {"new": r[k], "stored": s.get(k), "abs_diff": (abs(r[k] - s[k]) if s.get(k) is not None else None)}
                                      for k in ("roc_auc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr")}
            (outdir / f"seed{seed}_check.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
            print(f"saved {f}", flush=True)


if __name__ == "__main__":
    main()
