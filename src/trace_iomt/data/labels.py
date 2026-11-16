"""Label parsing and the label hierarchy (19-class -> 6-category -> binary).

CICIoMT2024 CSVs carry no label column: the class is in the file name, e.g.
"TCP_IP-DDoS-ICMP1_train.pcap.csv" -> label19 "TCP_IP-DDoS-ICMP", category "DDoS".
"""
import re

CATEGORIES = ["Benign", "Spoofing", "Recon", "MQTT", "DoS", "DDoS"]
LEVELS = ["label2", "label6", "label19"]   # coarse -> fine


def ciciomt_label_from_filename(name):
    """Return (label19, split or None) from an official CICIoMT2024 flow-feature file name.
    Accepts .csv, .pcap.csv, .xlsx and .pcap.xlsx (the CIC release ships CSVs; the mirror in this
    repo delivered training files as .xlsx — both are read)."""
    stem = name
    for suf in (".pcap.csv", ".pcap.xlsx", ".csv", ".xlsx"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    split = None
    m = re.search(r"_(train|test)$", stem, flags=re.I)
    if m:
        split, stem = m.group(1).lower(), stem[: m.start()]
    label19 = re.sub(r"\d+$", "", stem)   # ICMP1, ICMP2 ... -> ICMP
    return label19, split


def ciciomt_category(label19):
    s = label19.lower()
    if "benign" in s:
        return "Benign"
    if "spoof" in s:
        return "Spoofing"
    if "recon" in s:
        return "Recon"
    if "mqtt" in s:          # checked before DDoS/DoS: "MQTT-DDoS-Connect_Flood" is an MQTT attack
        return "MQTT"
    if "ddos" in s:
        return "DDoS"
    if "dos" in s:
        return "DoS"
    return "Unknown"


def binary(category):
    return "Benign" if category == "Benign" else "Attack"


class Hierarchy:
    """Integer encodings for each level plus the parent maps used by the hierarchical heads."""

    def __init__(self, label19_values, label6_of):
        self.classes = {
            "label19": sorted(label19_values),
            "label6": [c for c in CATEGORIES if c in set(label6_of.values())],
            "label2": ["Benign", "Attack"],
        }
        self.index = {lvl: {c: i for i, c in enumerate(cs)} for lvl, cs in self.classes.items()}
        # parent index arrays: fine class i -> category index; category j -> binary index
        self.parent19 = [self.index["label6"][label6_of[c]] for c in self.classes["label19"]]
        self.parent6 = [self.index["label2"][binary(c)] for c in self.classes["label6"]]

    def n(self, level):
        return len(self.classes[level])

    def to_dict(self):
        return {"classes": self.classes, "parent19": self.parent19, "parent6": self.parent6}

    @classmethod
    def from_frame(cls, df):
        pairs = df[["label19", "label6"]].drop_duplicates()
        return cls(pairs["label19"].tolist(), dict(zip(pairs["label19"], pairs["label6"])))
