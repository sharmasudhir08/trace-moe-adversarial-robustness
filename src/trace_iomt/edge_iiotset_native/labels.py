"""Edge-IIoTset NATIVE 6-class coarse taxonomy, per Ferrag 2022 Section IV-A-2.

Target coarse labels (verbatim from paper):
    Normal, DoS/DDoS, InfoGathering, MITM, Injection, Malware

This is the UNFORCED, HONEST taxonomy — no cross-dataset relabelling, no
semantic losses. Cross-dataset comparisons with CICIoMT are done at
architecture level, not label-identical.
"""
from __future__ import annotations

# Native 6-class per paper IV-A-2 (NOT the unified cross-dataset target set).
COARSE_6_NATIVE: list[str] = [
    "Normal", "DoS/DDoS", "InfoGathering", "MITM", "Injection", "Malware",
]

# Verified from DNN-EdgeIIoT-dataset.csv (2026-10-14).
EDGE_IIOTSET_ATTACK_TYPES: set[str] = {
    "Normal",
    "DDoS_UDP", "DDoS_ICMP", "DDoS_TCP", "DDoS_HTTP",
    "Port_Scanning", "Fingerprinting", "Vulnerability_scanner",
    "MITM",
    "XSS", "SQL_injection", "Uploading",
    "Backdoor", "Password", "Ransomware",
}

# Native Attack_type → native coarse 6-class.
# NO semantic losses here — this is the paper's own taxonomy.
LABEL_MAPPING_NATIVE: dict[str, str] = {
    "Normal":                 "Normal",
    "DDoS_UDP":               "DoS/DDoS",
    "DDoS_ICMP":              "DoS/DDoS",
    "DDoS_TCP":               "DoS/DDoS",
    "DDoS_HTTP":              "DoS/DDoS",
    "Port_Scanning":          "InfoGathering",
    "Fingerprinting":         "InfoGathering",
    "Vulnerability_scanner":  "InfoGathering",
    "MITM":                   "MITM",
    "XSS":                    "Injection",
    "SQL_injection":          "Injection",
    "Uploading":              "Injection",
    "Backdoor":               "Malware",     # NO loss — Malware class exists
    "Password":               "Malware",     # NO loss
    "Ransomware":             "Malware",     # NO loss
}

assert set(LABEL_MAPPING_NATIVE.keys()) == EDGE_IIOTSET_ATTACK_TYPES
assert set(LABEL_MAPPING_NATIVE.values()) == set(COARSE_6_NATIVE), \
    "Every COARSE_6_NATIVE class must be exercised by at least one Attack_type"


def coarse_from_attack_type_native(attack_type: str) -> str:
    """Map a native Attack_type to the native coarse 6-class label."""
    return LABEL_MAPPING_NATIVE[attack_type]


def validate_attack_types_native(observed: set[str]) -> dict:
    """Compare observed Attack_type values against the known set."""
    known = EDGE_IIOTSET_ATTACK_TYPES
    return {
        "known": sorted(known),
        "observed": sorted(observed),
        "unexpected": sorted(observed - known),
        "missing": sorted(known - observed),
    }
