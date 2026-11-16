"""Edge-IIoTset NATIVE-schema preprocessing and consumer loaders.

This is the AUTHORITATIVE Edge-IIoTset preprocessing path for cross-dataset
work in Papers 1, 2, and 4 (decided 2026-10-14). It uses:

  • Edge-IIoTset's own 46-feature schema (61 cols − 15 ID/payload drops per
    Ferrag 2022 Section VI). NO zero-fill fabrication.
  • Edge-IIoTset's own 6-class coarse taxonomy per paper IV-A-2:
    [Normal, DoS/DDoS, InfoGathering, MITM, Injection, Malware]
  • Paper 1's `Scaler(kind="standard", clip=10)` imported (not modified) from
    trace_iomt.data.preprocess for engineering parity (NaN safe, zero-variance
    fallback, clipping) while remaining "standard" per Edge-IIoTset paper.
  • Row-level dedup, stratified 70/15/15 split, 5 seeds {42-46}.

The parallel `trace_iomt.edge_iiotset` package (forced-mirror / CICIoMT 45-col
with 24 zero_fill) remains available as a fallback but is NOT recommended for
cross-dataset claims.

Public API:
    NATIVE_COLUMNS                 — 46-col Edge-IIoTset native schema list
    COARSE_6_NATIVE                — 6-class coarse labels per Ferrag 2022
    LABEL_MAPPING_NATIVE           — Attack_type → native coarse label
    SCHEMA_NOTES                   — per-column type (numeric/categorical) info
    load_edge_iiotset_native()     — consumer-facing frozen-artefact loader
"""
from __future__ import annotations

from .labels import COARSE_6_NATIVE, LABEL_MAPPING_NATIVE, coarse_from_attack_type_native
from .loaders import load_edge_iiotset_native
from .schema import CATEGORICAL_COLUMNS, NATIVE_COLUMNS, NUMERIC_COLUMNS, SCHEMA_NOTES

PIPELINE_VERSION = "edge_iiotset_native_v1.0_2026-10-14"

__all__ = [
    "CATEGORICAL_COLUMNS",
    "COARSE_6_NATIVE",
    "LABEL_MAPPING_NATIVE",
    "NATIVE_COLUMNS",
    "NUMERIC_COLUMNS",
    "PIPELINE_VERSION",
    "SCHEMA_NOTES",
    "coarse_from_attack_type_native",
    "load_edge_iiotset_native",
]
