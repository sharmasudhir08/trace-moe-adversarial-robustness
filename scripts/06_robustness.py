"""Step 6. Adversarial robustness at 19 classes (contribution C2, hypotheses H6 and H7).

For each model: constrained PGD on attack windows at every eps (robustness.epsilons), then a few
epochs of adversarial fine-tuning and the same curve again. Tree models are scored on transfer
examples crafted against each neural model. Trained models from step 4 are reused when present.

Results: results/robustness/<model>_official_T<T>_seed<k>/metrics.json with
  "curve_clean_model": {eps: metrics}, "curve_adv_finetuned": {eps: metrics}
(each with the attack success rate per attack class, used by the H6 test)
"""
import time

import numpy as np

from _common import setup
from trace_iomt.attacks import adversarial_finetune, attack_eval_starts, robustness_curve, transfer_matrix
from trace_iomt.experiment import (classical_proba, load_or_fit_classical, load_or_train_deep, load_prepared,
                                   run_dir, save_run)
from trace_iomt.train import evaluate
from trace_iomt.utils import now_iso, resolve_device, save_json, set_seed


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--regime", default="official")


def main():
    args, cfg, log = setup("Adversarial robustness", "06_robustness", add)
    rc = cfg["robustness"]
    models = args.models or rc["models"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    df = load_prepared(cfg)
    device = resolve_device(cfg["train"]["device"])
    T = cfg["windows"]["length"]
    for seed in seeds:
        # 1) load (or train) every clean model for this seed, and the tree models
        wd, clean, trees = None, {}, {}
        for mname in models:
            clean[mname], wd = load_or_train_deep(df, cfg, mname, args.regime, seed, "main", log, wd=wd,
                                                   strict=True)
        for tname in rc.get("tree_models", []):
            try:
                clf, _, _ = load_or_fit_classical(tname, wd, cfg, seed, args.regime, "main", log,
                                                   strict=True)
            except ImportError as e:
                log.warning(f"Skipping tree model {tname}: {e}"); continue
            trees[tname] = (lambda c: (lambda X: classical_proba(c, X, wd.hierarchy.n("label19"))))(clf)
        starts = attack_eval_starts(wd, rc["max_eval_windows"], seed, 0)

        # 2) black-box transfer matrix on the clean models
        if rc.get("transfer_matrix", True) and len(clean) > 1:
            set_seed(seed)
            rows = transfer_matrix(clean, trees, wd, cfg, device, starts, log)
            save_json({"seed": seed, "regime": args.regime, "rows": rows},
                      run_dir(cfg, "robustness", f"transfer_matrix_{args.regime}_T{T}_seed{seed}") / "transfer_matrix.json")

        # 3) white-box curves, adversarial fine-tuning, curves again
        for mname, model in clean.items():
            log.info(f"[{mname} seed {seed}] PGD on {len(starts):,} test windows")
            set_seed(seed)
            started_at = now_iso()
            t0 = time.time()
            curve = robustness_curve(model, wd, cfg, device, starts, log, extra=trees)
            attack_s = time.time() - t0
            c2 = {**cfg, "train": {**cfg["train"], "seed": seed}}
            model = adversarial_finetune(model, wd, c2, device, log)
            clean_after, _, _ = evaluate(model, wd, "test", device, cfg)
            curve_adv = robustness_curve(model, wd, cfg, device, starts, log)
            name = f"{mname}_{args.regime}_T{T}_seed{seed}"
            meta = dict(experiment="robustness", run=name, model=mname, family="deep", regime=args.regime,
                        seed=seed, T=T, tag="", started_at=started_at,
                        attack_seconds=round(attack_s, 1), n_attack_windows=int(len(starts)),
                        epsilons=rc["epsilons"], mutable_features=rc["mutable_features"],
                        curve_clean_model=curve, curve_adv_finetuned=curve_adv)
            save_run(run_dir(cfg, "robustness", name), cfg, meta, clean_after)


if __name__ == "__main__":
    main()
