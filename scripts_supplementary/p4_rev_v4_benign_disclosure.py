"""v4 support file: false-alarm rate of the energy flag on known Benign windows, in the test and in the validation windows, per model (source file for the v4 disclosure). Exploratory, post hoc.
Test: score_export energy and thr, y19. Validation: val_energy from score_export (known classes only) with the class labels val_y_true_label19 of zero_day/<run>/predictions.npz; both arrays are in the order of
the run's validation windows (same length, checked), so the flag rate per class is energy < thr by validation class. Two aggregations are reported: the mean of the 25 per-run rates (the unit of the other analyses)
and the rate pooled over all Benign windows of the model. Safety-view alert rate on test Benign windows from T9 (per-run file)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *
t0 = time.time()
d = OUT / "V4"
d.mkdir(parents=True, exist_ok=True)
rows = []
for m, c, s, name in canonical_runs():
    z = load_export(name)
    p = np.load(ZD / name / "predictions.npz")
    names, _ = classes19(name)
    ben = names.index("Benign")
    thr = float(z["thr"])
    y, en = z["y19"].astype(int), z["energy"]
    tb = y == ben
    ve = z["val_energy"]
    vy = p["val_y_true_label19"].astype(int)
    assert len(ve) == len(vy), name
    vb = vy == ben
    rows.append(dict(model=m, held_out_class=c, seed=s, n_test_benign=int(tb.sum()), test_flag_rate=float((en[tb] < thr).mean()), test_flagged=int((en[tb] < thr).sum()),
                     n_val_benign=int(vb.sum()), val_flag_rate=float((ve[vb] < thr).mean()), val_flagged=int((ve[vb] < thr).sum()),
                     val_overall_flag_rate=float((ve < thr).mean()), n_test_known=int((y != int(z["hid"])).sum()), benign_share_of_known=float(tb.sum() / (y != int(z["hid"])).sum())))
df = pd.DataFrame(rows)
df.to_csv(d / "benign_flag_rate_per_run.csv", index=False)
t9 = pd.read_csv(OUT / "T9" / "safety_view_false_alarms_per_run.csv")
out = []
rng = np.random.default_rng(42)
for m, g in df.groupby("model"):
    rec = dict(model=m)
    for col in ("test_flag_rate", "val_flag_rate", "benign_share_of_known"):
        sm = g.groupby("seed")[col].mean().to_numpy()
        bs = rng.choice(sm, size=(10000, len(sm)), replace=True).mean(1)
        rec[col], rec[col + "_lo"], rec[col + "_hi"] = float(sm.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    rec["test_flag_rate_pooled_windows"] = float(g.test_flagged.sum() / g.n_test_benign.sum())
    rec["val_flag_rate_pooled_windows"] = float(g.val_flagged.sum() / g.n_val_benign.sum())
    rec["val_overall_flag_rate"] = float(g.val_overall_flag_rate.mean())
    t = t9[t9.model == m]
    sm = t.groupby("seed").alert_rate_benign.mean().to_numpy()
    bs = rng.choice(sm, size=(10000, len(sm)), replace=True).mean(1)
    rec["safety_view_alert_rate_benign_test"], rec["sv_lo"], rec["sv_hi"] = float(sm.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    out.append(rec)
sm = pd.DataFrame(out)
sm.to_csv(d / "benign_flag_rate_summary.csv", index=False)
print(sm.round(4).T.to_string())
(d / "README.md").write_text("# V4 support: Benign flag rate, test and validation (exploratory, post hoc)\n\nScript: p4_rev_v4_benign_disclosure.py. Source of the numbers quoted in v4_proposed_changes. Per-run file: benign_flag_rate_per_run.csv; summary: benign_flag_rate_summary.csv.\n\n```\n" + sm.round(4).T.to_string() + "\n```\n", encoding="utf-8")
print("done", f"{time.time()-t0:.0f}s")
