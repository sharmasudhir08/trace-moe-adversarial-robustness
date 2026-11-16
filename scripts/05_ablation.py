"""Step 5. Ablation of TRACE-MoE: remove or replace one component at a time, and vary the window length T.

Variants (ablation.variants in the config): dense_expert (same size, no routing; hypothesis H4),
protocol_routing, no_experts, no_balance_loss, no_hierarchy, plain_ce_loss, no_protocol_token.
Results: results/ablation/<model>_official_T<T>_<variant>_seed<k>/
"""
from _common import setup
from trace_iomt.config import apply_overrides
from trace_iomt.experiment import load_prepared, run_deep


def add(p):
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--regime", default="official")
    p.add_argument("--skip-window-sweep", action="store_true")


def main():
    args, cfg, log = setup("Ablation", "05_ablation", add)
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    base = cfg["ablation"].get("model", cfg["main"]["reference_model"])
    exclude = set(cfg["ablation"].get("exclude_variants", []))
    df = load_prepared(cfg)
    for seed in seeds:
        for name, overrides in cfg["ablation"]["variants"].items():
            if name in exclude:
                log.info(f"Skipping ablation variant '{name}' (listed in ablation.exclude_variants)")
                continue
            c = apply_overrides({**cfg, "model": {**cfg["model"], "name": base}}, overrides)
            run_deep(df, c, c["model"]["name"], args.regime, seed, "ablation", log, tag=name)
        if not args.skip_window_sweep:
            for T in cfg["ablation"]["window_lengths"]:
                if T == cfg["windows"]["length"]:
                    continue        # same as the 'full' variant
                run_deep(df, cfg, base, args.regime, seed, "ablation", log, T=T, tag="full")


if __name__ == "__main__":
    main()
