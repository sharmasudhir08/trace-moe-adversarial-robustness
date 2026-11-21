"""Paper 4 part-2 helper: rebuild the LOAO test windows (as in p4_rev_pcrp_t4_u3.py) and cache per-window metadata for the 25 (class, seed) pairs:
global row id of the last row, source file code, row_in_file, protocol token of the last row, 19-class label. CPU only, no training, new files only.
The windows do not depend on the model (same T = 16, same seed); the cached y19 is checked against every model's score_export y19 in p4_rev2_batch2.py.
Needs at least 6 GB of free RAM (refuses to start below). Usage: python scripts/p4_rev2_meta.py --date 20261121"""
import argparse, gc, hashlib, sys, time
from pathlib import Path
import psutil
_avail0 = psutil.virtual_memory().available / 2**30   # free RAM before this process loads anything (project rule: at least 6 GB)
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.experiment import load_prepared  # noqa: E402
from trace_iomt.data.windows import build_windowed  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
R3 = ROOT / "results_v3_paper3"
OUT = R3 / f"p4_review_analyses_{a.date}" / "part2" / "meta_cache"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["ARP_Spoofing", "MQTT-DDoS-Publish_Flood", "MQTT-Malformed_Data", "Recon-VulScan", "TCP_IP-DoS-UDP"]
SEEDS = [42, 43, 44, 45, 46]
avail = _avail0
if avail < 6.0:
    sys.exit(f"STOP: available RAM {avail:.2f} GB < 6 GB")
cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml")); T = int(cfg["windows"]["length"])
df = load_prepared(cfg)
srcfile = df["source_file"]; rowin = df["row_in_file"].to_numpy()
print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest(), flush=True)
for held in CLASSES:
    for seed in SEEDS:
        p = OUT / f"{held}_seed{seed}.npz"
        if p.exists(): continue
        t0 = time.time()
        c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held}}
        wd = build_windowed(df, c2, regime="loao", T=T, seed=seed)
        last = wd.starts["test"] + T - 1
        gid = wd.row_index[last]
        files = srcfile.loc[gid].to_numpy()
        fn, fc = np.unique(files, return_inverse=True)
        np.savez(p, gid=gid.astype(np.int64), file_code=fc.astype(np.int16), file_names=fn.astype(str), row_in_file=rowin[gid].astype(np.int64),
                 proto=wd.proto[last].astype(np.int8), y19=wd.labels("test", "label19").astype(np.int8), hid=int(wd.hierarchy.index["label19"][held]))
        del wd; gc.collect()
        print(held, seed, len(gid), f"{time.time() - t0:.0f}s, avail {psutil.virtual_memory().available / 2**30:.1f} GB", flush=True)
print("finished")
