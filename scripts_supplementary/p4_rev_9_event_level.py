"""T9: combined safety-view false-alarm rate and event-level alerting. Exploratory, post hoc.
Window order and file identity: the exported test windows are sorted by (source_file, row_in_file) and every class has exactly one test file, so each class is one contiguous block of the
exported arrays (checked below: the number of constant-label runs equals the number of classes). Rules fixed now: n in {4, 16, 64}, m = ceil(n/2) and m = n; an alert at window i needs
at least m alerts among the last n windows of the same file (only windows with n predecessors are tested); an event is a maximal run of triggered windows."""
import sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *
t0 = time.time()
d = task_dir("T", 9)
pr = per_run_table().set_index(["model", "held_out_class", "seed"])
NS = [4, 16, 64]


def segments(y):
    ch = np.flatnonzero(np.diff(y) != 0) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(y)]
    return list(zip(st, en))


def triggers(a, n, m):
    cs = np.concatenate([[0], np.cumsum(a.astype(np.int64))])
    cnt = cs[n:] - cs[:-n]                      # alerts in windows [i-n+1, i], i from n-1
    trig = np.zeros(len(a), dtype=bool)
    trig[n - 1:] = cnt >= m
    return trig


def events(trig):
    return int((trig & ~np.r_[False, trig[:-1]]).sum())


rows_a, rows_e = [], []
dev1 = dev2 = 0.0
for mdl, c, s, name in canonical_runs():
    z = load_export(name)
    names, _ = classes19(name)
    y19, hid, thr = z["y19"].astype(int), int(z["hid"]), float(z["thr"])
    segs = segments(y19)
    assert len(segs) == len(np.unique(y19)), name
    ben = names.index("Benign")
    fe, bn = z["energy"] < thr, z["p_attack"] > 0.5
    al = fe | bn
    unk, kb, ka = y19 == hid, y19 == ben, (y19 != hid) & (y19 != ben)
    dev1 = max(dev1, abs(float(al[unk].mean()) - float(pr.loc[(mdl, c, s), "A2_caught_any"])))
    dev2 = max(dev2, abs(float((bn & ~fe)[kb].mean() + (bn & fe)[kb].mean()) - float(pr.loc[(mdl, c, s), "A1_binary_fpr_on_benign"])))
    rows_a.append(dict(model=mdl, held_out_class=c, seed=s, alert_rate_benign=float(al[kb].mean()), alert_rate_known_attack=float(al[ka].mean()), alert_rate_heldout=float(al[unk].mean()),
                       energy_only_benign=float((fe & ~bn)[kb].mean()), binary_only_benign=float((bn & ~fe)[kb].mean()), both_benign=float((fe & bn)[kb].mean())))
    for n in NS:
        for mm in sorted({math.ceil(n / 2), n}):
            ev_b = n_b = 0
            trig_u = n_u = 0
            held_file_any = False
            ka_trig = ka_n = 0
            for (a0, a1) in segments(y19):
                lab = y19[a0]
                tr = triggers(al[a0:a1], n, mm)
                valid = a1 - a0 - (n - 1)
                if valid <= 0:
                    continue
                if lab == ben:
                    ev_b += events(tr)
                    n_b += valid
                elif lab == hid:
                    trig_u += int(tr[n - 1:].sum())
                    n_u += valid
                    held_file_any = bool(tr.any())
                else:
                    ka_trig += int(tr[n - 1:].sum())
                    ka_n += valid
            rows_e.append(dict(model=mdl, held_out_class=c, seed=s, n=n, m=mm, benign_events_per_10k_windows=1e4 * ev_b / n_b if n_b else np.nan, heldout_window_detection=trig_u / n_u if n_u else np.nan,
                               heldout_file_detected=int(held_file_any), known_attack_window_trigger_rate=ka_trig / ka_n if ka_n else np.nan))
    print("done", name, f"{time.time()-t0:.0f}s", flush=True)
cmd = "python scripts/p4_rev_9_event_level.py"
if dev1 > 1e-9 or dev2 > 1e-9:
    fail_readme(d, "T9 event level", __file__, cmd, f"pass criteria failed: held-out caught-any vs A2 {dev1:.1e}; binary benign rate vs A1 {dev2:.1e}", t0)
    sys.exit(1)
da, de = pd.DataFrame(rows_a), pd.DataFrame(rows_e)
da.to_csv(d / "safety_view_false_alarms_per_run.csv", index=False)
seed_ci(da, [c_ for c_ in da.columns if c_ not in ("model", "held_out_class", "seed")], by=("model", "held_out_class")).to_csv(d / "safety_view_false_alarms.csv", index=False)
de.to_csv(d / "event_level_per_run.csv", index=False)
seed_ci(de, ["benign_events_per_10k_windows", "heldout_window_detection", "heldout_file_detected", "known_attack_window_trigger_rate"], by=("model", "held_out_class", "n", "m")).to_csv(d / "event_level.csv", index=False)
pool = da.groupby("model")[["alert_rate_benign", "energy_only_benign", "binary_only_benign", "alert_rate_known_attack", "alert_rate_heldout"]].mean().round(4)
ev = de.groupby(["model", "n", "m"])[["benign_events_per_10k_windows", "heldout_window_detection"]].mean().round(3)
res = [f"Pass criteria: held-out caught-any at n = 1 equals Table II / A2 (max |difference| {dev1:.1e}); binary-only plus both on benign equals A1_binary_fpr_on_benign (max |difference| {dev2:.1e}).", "", "Safety-view alert rates, mean over classes and seeds:", "```", pool.to_string(), "```", "",
       "Event-level rules (mean over classes and seeds; events per 10,000 Benign windows; window-level held-out detection):", "```", ev.to_string(), "```", "No rule is recommended from these test results."]
write_readme(d, "T9: safety-view false alarms and event-level alerting", __file__, cmd, "score_export/*.npz (energy, thr, p_attack, y19, hid)", 75, "\n".join(res), t0, "File identity = contiguous label block (one test file per class), verified by the number of constant-label runs.")
print("T9 done", f"{time.time()-t0:.0f}s", dev1, dev2)
