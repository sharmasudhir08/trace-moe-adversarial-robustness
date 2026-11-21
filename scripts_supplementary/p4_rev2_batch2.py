"""Paper 4 part-2 analyses U4, U5, U7, U8 (post hoc, exploratory; CPU only; no training; new files only). Needs the metadata cache of scripts/p4_rev2_meta.py.
Fixed before running: bootstrap 10,000 over seeds (rng 42); U4 two-fold cross-fitting of p(expert | protocol) on KNOWN test windows (fold = (window position // 10000) % 2), Laplace smoothing 1;
U5 groups = protocol token of the last row (0 other/IP, 1 ICMP, 2 TCP, 3 UDP, 4 mixed), because the data have no per-window Wi-Fi/MQTT flag (Benign is one file with both);
U7 z-score of energy with each seed's own validation mean and SD, average over the five seeds on the windows common to all five seeds; U8 flags: energy (exported thr), binary head (p_attack > 0.5), either.
Usage: python scripts/p4_rev2_batch2.py --date 20261121 --tasks U5 U8 U4 U7"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
SE, ZD = R3 / "score_export", R3 / "zero_day"
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--tasks", nargs="+", required=True); a = ap.parse_args()
OUT = R3 / f"p4_review_analyses_{a.date}" / "part2"
META = OUT / "meta_cache"
PR = pd.read_csv(R3 / "saved_output_analyses_20261118" / "per_run.csv").set_index(["model", "held_out_class", "seed"])
ESA1 = pd.read_csv(R3 / "h8_verdict" / "ESA1_calibration.csv").set_index(["model", "class"])
RNG = np.random.default_rng(42); B = 10_000
CLASSES = ["ARP_Spoofing", "MQTT-DDoS-Publish_Flood", "MQTT-Malformed_Data", "Recon-VulScan", "TCP_IP-DoS-UDP"]
SEEDS = [42, 43, 44, 45, 46]; MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
TOK = ["other_IP", "ICMP", "TCP", "UDP", "mixed"]


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan,) * 3
    bs = v[RNG.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def seedci(df, keys, cols):
    out = []
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for c in cols:
            m, lo, hi = ci(g.sort_values("seed")[c].to_numpy()); row[c] = m; row[c + "_lo"] = lo; row[c + "_hi"] = hi
        row["n_seeds"] = g.seed.nunique(); out.append(row)
    return pd.DataFrame(out)


def write(task, name, df):
    d = OUT / task; d.mkdir(parents=True, exist_ok=True); p = d / name
    if p.exists(): raise FileExistsError(p)
    df.to_csv(p, index=False)


def readme(task, text, t0, n):
    d = OUT / task; d.mkdir(parents=True, exist_ok=True); p = d / "README.md"
    if p.exists(): raise FileExistsError(p)
    p.write_text(f"# {task} (post hoc, exploratory, CPU, no training)\n\nCommand: python scripts/p4_rev2_batch2.py --date {a.date} --tasks {task}\nScript SHA-256: {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}\n"
                 f"Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz, part2/meta_cache (rebuilt test-window metadata). Runs read: {n}. rng seed 42, bootstrap {B} over seeds.\n"
                 f"Started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}, ended {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n{text}\n", encoding="utf-8")


def meta(c, s):
    m = np.load(META / f"{c}_seed{s}.npz"); return {k: m[k] for k in m.files}


def score(model, c, s, mt):
    z = np.load(SE / f"{model}_loao_T16_{c}_seed{s}.npz")
    assert np.array_equal(z["y19"].astype(np.int8), mt["y19"]), f"y19 mismatch {model} {c} {s}"
    assert int(z["hid"]) == int(mt["hid"])
    return z


def u5():
    t0 = time.time(); rows = []; chk = []
    for c in CLASSES:
        for s in SEEDS:
            mt = meta(c, s); tok = mt["proto"].astype(int); hid = int(mt["hid"]); unk = mt["y19"].astype(int) == hid
            for m in MODELS:
                z = score(m, c, s, mt); fe = z["energy"] < float(z["thr"]); fb = z["p_attack"] > 0.5; fa = fe | fb; e = -z["energy"].astype(np.float64)
                chk.append((abs(fe[~unk].mean() - ESA1.loc[(m, c), f"far_s{s}"]), abs(fa[unk].mean() - PR.loc[(m, c, s), "A2_caught_any"])))
                for g in range(5):
                    k, u = (tok == g) & ~unk, (tok == g) & unk
                    au = np.nan
                    if k.sum() >= 50 and u.sum() >= 50: au = float(roc_auc_score(np.r_[np.zeros(k.sum()), np.ones(u.sum())], np.r_[e[k], e[u]]))
                    rows.append(dict(model=m, held_out_class=c, seed=s, protocol_token=TOK[g], n_known=int(k.sum()), n_heldout=int(u.sum()),
                                     far_energy_known=float(fe[k].mean()) if k.sum() else np.nan, far_binary_known=float(fb[k].mean()) if k.sum() else np.nan, far_either_known=float(fa[k].mean()) if k.sum() else np.nan,
                                     caught_energy_heldout=float(fe[u].mean()) if u.sum() else np.nan, caught_either_heldout=float(fa[u].mean()) if u.sum() else np.nan, energy_auroc_within_group=au))
        print("U5", c, flush=True)
    d = pd.DataFrame(rows); write("U5", "per_protocol_metrics_per_run.csv", d)
    cols = ["far_energy_known", "far_binary_known", "far_either_known", "caught_energy_heldout", "caught_either_heldout", "energy_auroc_within_group"]
    write("U5", "per_protocol_metrics.csv", seedci(d, ["model", "held_out_class", "protocol_token"], cols))
    pooled = seedci(d.dropna(subset=["far_energy_known"]).groupby(["model", "protocol_token", "seed"], as_index=False)[["far_energy_known", "far_either_known"]].mean(), ["model", "protocol_token"], ["far_energy_known", "far_either_known"])
    share = d.groupby(["protocol_token"])[["n_known", "n_heldout"]].sum().assign(share_known=lambda x: x.n_known / x.n_known.sum()).round(4).reset_index()
    readme("U5", f"Groups are the protocol token of the last row of each window (the token the router receives). The data contain no per-window Wi-Fi or MQTT flag (Benign is a single file that mixes both), so the Wi-Fi versus MQTT split of the brief is replaced by the transport token; MQTT runs over TCP. Checks: pooled known FAR equals ESA1 (max difference {max(c[0] for c in chk):.2e}) and pooled held-out caught-any equals A2_caught_any (max difference {max(c[1] for c in chk):.2e}); criterion 1e-6.\n\nKnown false-alarm rate by token, mean over classes and seeds (seed interval):\n\n" + pooled.round(4).to_markdown(index=False) + "\n\nWindow counts by token (summed over the 75 runs' windows, one model counted once per class and seed x3):\n\n" + share.to_markdown(index=False), t0, 75)


def u8():
    t0 = time.time(); rows = []
    for c in CLASSES:
        for s in SEEDS:
            mt = meta(c, s); hid = int(mt["hid"]); unk = mt["y19"].astype(int) == hid; idx = np.flatnonzero(unk)
            assert len(np.unique(mt["file_code"][idx])) == 1, "held-out class spans several files"
            order = idx[np.argsort(mt["row_in_file"][idx], kind="stable")]
            for m in MODELS:
                z = score(m, c, s, mt)
                for name, f in (("energy", z["energy"] < float(z["thr"])), ("binary_head", z["p_attack"] > 0.5), ("either", (z["energy"] < float(z["thr"])) | (z["p_attack"] > 0.5))):
                    fl = f[order]; first = int(np.argmax(fl)) if fl.any() else -1
                    rows.append(dict(model=m, held_out_class=c, seed=s, flag=name, n_heldout_windows=len(order), share_flagged=float(fl.mean()), first_flag_position=first,
                                     ever_flagged=bool(fl.any()), median_gap_between_flags=float(np.median(np.diff(np.flatnonzero(fl)))) if fl.sum() > 1 else np.nan))
        print("U8", c, flush=True)
    d = pd.DataFrame(rows); write("U8", "time_to_detect.csv", d)
    s = d.groupby(["model", "held_out_class", "flag"]).agg(runs=("seed", "size"), ever_flagged=("ever_flagged", "mean"), share_flagged=("share_flagged", "mean"),
                                                          first_pos_median=("first_flag_position", lambda x: float(np.median(x[x >= 0])) if (x >= 0).any() else np.nan), first_pos_max=("first_flag_position", "max")).round(4).reset_index()
    write("U8", "time_to_detect_summary.csv", s)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for i, f in enumerate(("energy", "binary_head", "either")):
        for m in MODELS:
            v = np.sort(d[(d.flag == f) & (d.model == m)].first_flag_position.to_numpy().astype(float)); v[v < 0] = np.inf
            ax[i].step(np.r_[1, v + 1], np.arange(0, len(v) + 1) / len(v), where="post", label=m)
        ax[i].set_xscale("log"); ax[i].set_title(f); ax[i].set_xlabel("held-out windows until first flag")
    ax[0].set_ylabel("share of runs (25 per model)"); ax[0].legend(fontsize=7); fig.tight_layout()
    p = OUT / "U8" / "ttd_cdf.pdf"
    if p.exists(): raise FileExistsError(p)
    fig.savefig(p)
    readme("U8", "Each held-out class has one test file; held-out windows are taken in file order (row_in_file). first_flag_position = number of held-out windows before the first flag (0 = the first window is flagged); -1 = never flagged. Descriptive only; no window-time scale is available (stride 1, no timestamps used). ttd_cdf.pdf: share of the 25 runs per model whose first flag falls within x windows (never-flagged runs stay below 1).\n\n" + s.to_markdown(index=False), t0, 75)


def u4():
    t0 = time.time(); rows, prot = [], []
    for c in CLASSES:
        for s in SEEDS:
            mt = meta(c, s); hid = int(mt["hid"]); unk = mt["y19"].astype(int) == hid; tok = mt["proto"].astype(int)
            z = np.load(ZD / f"trace_moe_loao_T16_{c}_seed{s}" / "predictions.npz"); route = z["route"].astype(int); assert len(route) == len(unk)
            ne = route.max() + 1 if route.max() >= 3 else 4
            known = np.flatnonzero(~unk); fold = (np.arange(len(known)) // 10000) % 2
            cnt_all = np.ones((5, ne)); np.add.at(cnt_all, (tok[known], route[known]), 1)
            rar = np.zeros(len(route))
            for f in (0, 1):
                cnt = np.ones((5, ne)); kk = known[fold != f]; np.add.at(cnt, (tok[kk], route[kk]), 1)
                p = cnt / cnt.sum(1, keepdims=True); tgt = known[fold == f]; rar[tgt] = -np.log(p[tok[tgt], route[tgt]])
            p_all = cnt_all / cnt_all.sum(1, keepdims=True); ui = np.flatnonzero(unk); rar[ui] = -np.log(p_all[tok[ui], route[ui]])
            au = float(roc_auc_score(unk.astype(int), rar))
            e_au = PR.loc[("trace_moe", c, s), "A7_energy_auroc"]
            rows.append(dict(held_out_class=c, seed=s, routing_rarity_auroc=au, energy_auroc=float(e_au), mean_rarity_heldout=float(rar[ui].mean()), mean_rarity_known=float(rar[known].mean())))
            for g in range(5):
                for nm, mk in (("heldout", unk & (tok == g)), ("known", ~unk & (tok == g))):
                    if mk.sum() == 0: continue
                    sh = np.bincount(route[mk], minlength=ne) / mk.sum()
                    for e_ in range(ne): prot.append(dict(held_out_class=c, seed=s, protocol_token=TOK[g], group=nm, n=int(mk.sum()), expert=e_, share=float(sh[e_])))
        print("U4", c, flush=True)
    d = pd.DataFrame(rows); write("U4", "routing_rarity_per_run.csv", d)
    s_ = seedci(d, ["held_out_class"], ["routing_rarity_auroc", "energy_auroc", "mean_rarity_heldout", "mean_rarity_known"]); write("U4", "routing_rarity.csv", s_)
    pp = pd.DataFrame(prot); write("U4", "routing_by_protocol.csv", pp.groupby(["held_out_class", "protocol_token", "group", "expert"], as_index=False).agg(n=("n", "mean"), share=("share", "mean")).round(5))
    rho = spearmanr(s_.routing_rarity_auroc, s_.energy_auroc)
    readme("U4", f"TRACE-MoE only (25 runs). Routing = expert chosen by the router on each test window (saved `route`); protocol = token of the last row (rebuilt). No validation routing is saved, so p(expert | protocol) is estimated on KNOWN test windows only, with two-fold cross-fitting (fold = (position // 10000) % 2, a known window is scored with the other fold's estimate; held-out windows are scored with the estimate from all known windows), Laplace smoothing 1; no held-out window enters the estimate. Rarity = -log p(route | protocol); AUROC held-out against known.\n\nPer class (mean over seeds, seed interval):\n\n" + s_.round(4).to_markdown(index=False) + f"\n\nSpearman correlation across the five classes between routing-rarity AUROC and energy AUROC (descriptive, n = 5): rho = {rho.statistic:.3f}, p = {rho.pvalue:.3f}.", t0, 25)


def u7():
    t0 = time.time(); rows, rows2 = [], []
    for m in MODELS:
        for c in CLASSES:
            mts = {s: meta(c, s) for s in SEEDS}; zs = {s: score(m, c, s, mts[s]) for s in SEEDS}
            common = mts[42]["gid"]
            for s in SEEDS[1:]: common = np.intersect1d(common, mts[s]["gid"], assume_unique=True)
            pos = {s: np.searchsorted(mts[s]["gid"], common) for s in SEEDS}
            for s in SEEDS: assert np.array_equal(mts[s]["gid"][pos[s]], common), "gid not sorted"
            y = (mts[42]["y19"][pos[42]].astype(int) == int(mts[42]["hid"])).astype(int)
            for s in SEEDS: assert np.array_equal(y, (mts[s]["y19"][pos[s]].astype(int) == int(mts[s]["hid"])).astype(int))
            zsc = {}; sing = {}; miss = {}
            for s in SEEDS:
                e = zs[s]["energy"][pos[s]].astype(np.float64); ve = zs[s]["val_energy"].astype(np.float64)
                zsc[s] = (e - ve.mean()) / ve.std(); sing[s] = float(roc_auc_score(y, -e)); miss[s] = (e >= float(zs[s]["thr"])) & (y == 1)
            ens = np.mean([zsc[s] for s in SEEDS], axis=0)
            jac = [float((miss[a_] & miss[b_]).sum() / max((miss[a_] | miss[b_]).sum(), 1)) for i, a_ in enumerate(SEEDS) for b_ in SEEDS[i + 1:]]
            M = np.array([miss[s][y == 1] for s in SEEDS]); nh = int(y.sum())
            rows.append(dict(model=m, held_out_class=c, n_common_windows=len(common), n_heldout_common=nh, auroc_ensemble=float(roc_auc_score(y, -ens)), mean_single_seed_auroc_same_windows=float(np.mean(list(sing.values()))),
                             best_single=float(max(sing.values())), worst_single=float(min(sing.values()))))
            rows2.append(dict(model=m, held_out_class=c, mean_miss_rate=float(M.mean()), mean_pairwise_jaccard=float(np.mean(jac)), share_missed_by_all_5=float(M.all(0).mean()), share_missed_by_any=float(M.any(0).mean()),
                              expected_share_all_5_if_independent=float(np.prod(M.mean(1)))))
            print("U7", m, c, flush=True)
    d, d2 = pd.DataFrame(rows), pd.DataFrame(rows2); d["ensemble_minus_mean_single"] = d.auroc_ensemble - d.mean_single_seed_auroc_same_windows
    write("U7", "seed_ensemble.csv", d.round(5)); write("U7", "miss_overlap.csv", d2.round(5))
    readme("U7", "Windows common to the five seeds of a model and class (test windows differ slightly between seeds because of duplicate removal; matched by global row id of the last row; held-out labels checked equal). Ensemble = mean of energy z-scores (each seed: its own known validation energy mean and SD); AUROC held-out against known on the common windows, compared with the mean of the five single-seed AUROCs on the same windows. A held-out window is missed by a seed if its energy is above that seed's threshold. Single combined run per model and class, so no seed interval; this is a descriptive ensemble note, not a new detector.\n\nEnsemble:\n\n" + d.round(4).to_markdown(index=False) + "\n\nMiss overlap:\n\n" + d2.round(4).to_markdown(index=False), t0, 75)


T = {"U5": u5, "U8": u8, "U4": u4, "U7": u7}
for t in a.tasks: T[t]()
print("finished", a.tasks)
