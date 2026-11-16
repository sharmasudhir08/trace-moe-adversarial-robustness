"""Read the raw CSVs into one tidy table (features + label/meta columns) and cache it as Parquet."""
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import binary, ciciomt_category, ciciomt_label_from_filename

META = ["label19", "label6", "label2", "split", "source_file", "row_in_file", "dataset"]


def clean_name(c):
    return " ".join(str(c).strip().split())


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def _to_float32(df, cols):
    out = df[cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def load_ciciomt2024(csv_dir, max_rows_per_file=None, seed=42, logger=None):
    """Load every CICIoMT2024 flow-feature file under csv_dir. Labels come from the file names.
    Supports .csv (official release) and .xlsx (mirror used in some packaging)."""
    base = Path(csv_dir)
    files = sorted(list(base.rglob("*.csv")) + list(base.rglob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No .csv or .xlsx files under {csv_dir}. Set paths.ciciomt_csv_dir in the config.")
    rng = np.random.default_rng(seed)
    parts, manifest = [], []
    for f in files:
        label19, split = ciciomt_label_from_filename(f.name)
        if split is None:   # fall back to the folder name (.../train/... or .../test/...)
            parts_lc = [p.lower() for p in f.parts]
            split = "test" if any("test" in p for p in parts_lc) else "train"
        if f.suffix.lower() == ".xlsx":
            df = pd.read_excel(f, engine="openpyxl")
        else:
            df = pd.read_csv(f, low_memory=False)
        df.columns = [clean_name(c) for c in df.columns]
        n_full = len(df)
        df["row_in_file"] = np.arange(n_full, dtype=np.int64)   # file order = the only time proxy
        if max_rows_per_file and n_full > max_rows_per_file:
            # keep a contiguous block so that windows of consecutive flows stay meaningful
            start = int(rng.integers(0, n_full - max_rows_per_file + 1))
            df = df.iloc[start:start + max_rows_per_file]
        feats = [c for c in df.columns if c != "row_in_file"]
        out = _to_float32(df, feats)
        cat = ciciomt_category(label19)
        out["label19"], out["label6"], out["label2"] = label19, cat, binary(cat)
        out["split"], out["source_file"] = split, f.name
        out["row_in_file"] = df["row_in_file"].values
        out["dataset"] = "ciciomt2024"
        parts.append(out)
        manifest.append(dict(file=f.name, split=split, label19=label19, label6=cat,
                             rows_in_file=n_full, rows_loaded=len(out)))
        if logger:
            logger.info(f"  {f.name}: {n_full:,} rows ({split}, {label19})")
    data = pd.concat(parts, ignore_index=True)
    for c in ["label19", "label6", "label2", "split", "source_file", "dataset"]:
        data[c] = data[c].astype("category")
    return data, pd.DataFrame(manifest)



def cache_path(processed_dir, name):
    return Path(processed_dir) / f"{name}.parquet"


def save_table(df, processed_dir, name):
    p = cache_path(processed_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def load_table(processed_dir, name):
    p = cache_path(processed_dir, name)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/01_prepare.py first.")
    return pd.read_parquet(p)
