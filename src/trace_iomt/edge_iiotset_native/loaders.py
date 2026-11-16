"""Consumer-facing loader for the frozen Edge-IIoTset NATIVE artefact.

Papers 1, 2, 4 import this to read the pre-processed, pre-split, pre-scaled
NATIVE 46-col Edge-IIoTset cache. They do NOT re-split, re-dedup, or re-fit.

Typical use:
    from trace_iomt.edge_iiotset_native import load_edge_iiotset_native

    train, val, test, scaler, vocab = load_edge_iiotset_native(
        artefact_dir="data/edge_iiotset_native", seed=42,
    )
    X_tr = scaler.transform(train[NATIVE_COLUMNS].to_numpy())
    y_tr = train["label_coarse"].to_numpy()
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from trace_iomt.data.preprocess import Scaler    # imported, not modified
from .schema import NATIVE_COLUMNS


def load_edge_iiotset_native(
    artefact_dir: str | Path,
    seed: int = 42,
    verify_manifest: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Scaler, dict]:
    """Load the pre-processed Edge-IIoTset NATIVE frozen artefact.

    Returns
    -------
    (train_df, val_df, test_df, scaler, vocab)
      • scaler: already fit on train (Paper 1's Scaler class, kind='standard', clip=10)
      • vocab:  dict {categorical_col: {raw_value: int_code}} used by preprocessing;
                any downstream code that scores new raw CSV data must use the SAME vocab
    """
    root = Path(artefact_dir)
    proc = root / "processed"
    scalers = root / "scalers"

    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing MANIFEST.json at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if verify_manifest:
        pv = manifest.get("pipeline_version", "")
        if not pv.startswith("edge_iiotset_native_v"):
            raise ValueError(
                f"MANIFEST pipeline_version {pv!r} is not an Edge-IIoTset NATIVE "
                f"artefact; refusing to load.")

    scaler_path = scalers / f"standard_scaler_seed{seed}.pkl"
    if not scaler_path.exists():
        available = sorted(scalers.glob("standard_scaler_seed*.pkl"))
        raise FileNotFoundError(
            f"Scaler for seed={seed} not found at {scaler_path}. "
            f"Available: {[p.name for p in available]}")
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    train = pd.read_parquet(proc / "edge_iiotset_native_train.parquet")
    val = pd.read_parquet(proc / "edge_iiotset_native_val.parquet")
    test = pd.read_parquet(proc / "edge_iiotset_native_test.parquet")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        missing = [c for c in NATIVE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} parquet missing native columns: {missing[:5]}...")
        if "label_coarse" not in df.columns or "label_native" not in df.columns:
            raise ValueError(f"{name} parquet missing label_coarse or label_native")

    vocab = json.loads((root / "categorical_vocab.json").read_text(encoding="utf-8"))

    return train, val, test, scaler, vocab


def load_manifest_native(artefact_dir: str | Path) -> dict:
    """Return the artefact's MANIFEST.json contents."""
    return json.loads((Path(artefact_dir) / "MANIFEST.json").read_text(encoding="utf-8"))


def load_label_mapping_native(artefact_dir: str | Path) -> dict:
    """Return the artefact's label_mapping.json contents."""
    return json.loads((Path(artefact_dir) / "label_mapping.json").read_text(encoding="utf-8"))


def load_schema_native(artefact_dir: str | Path) -> dict:
    """Return the artefact's schema.json contents."""
    return json.loads((Path(artefact_dir) / "schema.json").read_text(encoding="utf-8"))
