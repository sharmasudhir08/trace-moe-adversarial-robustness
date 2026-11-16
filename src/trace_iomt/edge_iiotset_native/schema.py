"""Edge-IIoTset NATIVE 46-column schema (post-drop of 15 ID/payload columns).

Derived from Ferrag et al. 2022, IEEE Access, Section VI:
    "we drop unnecessary flow features such as IP addresses, ports, timestamp,
     and payload information (i.e frame.time, ip.src_host, ip.dst_host,
     arp.src.proto_ipv4, arp.dst.proto_ipv4, http.file_data,
     http.request.full_uri, icmp.transmit_timestamp, http.request.uri.query,
     tcp.options, tcp.payload, tcp.srcport, tcp.dstport, udp.port, mqtt.msg)"

61 raw features in the DNN CSV − 15 drops = 46 retained features.
"""
from __future__ import annotations

# 15 columns dropped per paper Section VI (IDs, ports, timestamps, raw payloads).
DROPPED_COLUMNS: list[str] = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "http.request.uri.query",
    "icmp.transmit_timestamp",
    "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
    "udp.port", "mqtt.msg",
]
assert len(DROPPED_COLUMNS) == 15

# 46 Edge-IIoTset NATIVE columns kept for cross-dataset training.
# Order follows the DNN-EdgeIIoT-dataset.csv header order after dropping.
NATIVE_COLUMNS: list[str] = [
    # ARP (2 of 4 kept — 2 dropped: src.proto_ipv4, dst.proto_ipv4)
    "arp.opcode", "arp.hw.size",
    # ICMP (3 of 4 kept — 1 dropped: transmit_timestamp)
    "icmp.checksum", "icmp.seq_le", "icmp.unused",
    # HTTP (6 of 9 kept — 3 dropped: file_data, request.full_uri, request.uri.query)
    "http.content_length", "http.request.method", "http.referer",
    "http.request.version", "http.response", "http.tls_port",
    # TCP (11 of 15 kept — 4 dropped: options, payload, srcport, dstport)
    "tcp.ack", "tcp.ack_raw", "tcp.checksum",
    "tcp.connection.fin", "tcp.connection.rst",
    "tcp.connection.syn", "tcp.connection.synack",
    "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.seq",
    # UDP (2 of 3 kept — 1 dropped: port)
    "udp.stream", "udp.time_delta",
    # DNS (7 of 7 kept)
    "dns.qry.name", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    # MQTT (12 of 13 kept — 1 dropped: msg)
    "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags",
    "mqtt.hdrflags", "mqtt.len", "mqtt.msg_decoded_as", "mqtt.msgtype",
    "mqtt.proto_len", "mqtt.protoname", "mqtt.topic", "mqtt.topic_len",
    "mqtt.ver",
    # MBTCP (3 of 3 kept)
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
]
assert len(NATIVE_COLUMNS) == 46, f"NATIVE_COLUMNS must be 46, got {len(NATIVE_COLUMNS)}"

# Columns that contain string values in the raw CSV and need label encoding to
# integer codes before scaling. Numeric columns stay as float32.
# (Determined by inspection of the DNN CSV; a probe pass at preprocess time
# verifies this classification.)
CATEGORICAL_COLUMNS: list[str] = [
    "http.request.method",       # "GET", "POST", "0" (absent sentinel), ...
    "http.referer",              # URL strings
    "http.request.version",      # "HTTP/1.1", "0", ...
    "tcp.flags",                 # already decimal number but may contain mixed types
    "dns.qry.name",              # domain name strings
    "mqtt.msg_decoded_as",       # decode-type strings
    "mqtt.protoname",            # "MQTT", "0", ...
    "mqtt.topic",                # topic strings
]

NUMERIC_COLUMNS: list[str] = [c for c in NATIVE_COLUMNS if c not in CATEGORICAL_COLUMNS]
assert len(NUMERIC_COLUMNS) + len(CATEGORICAL_COLUMNS) == 46

# Per-column schema notes for MANIFEST reporting.
SCHEMA_NOTES: dict[str, dict] = {}
for c in NATIVE_COLUMNS:
    SCHEMA_NOTES[c] = {
        "type": "categorical_label_encoded" if c in CATEGORICAL_COLUMNS else "numeric_float32",
        "scaled": True,  # all 46 cols get scaled via Paper 1's Scaler(kind="standard")
    }


def schema_summary() -> dict:
    """Return mapping-kind counts for MANIFEST reporting."""
    return {
        "native_cols": len(NATIVE_COLUMNS),
        "numeric": len(NUMERIC_COLUMNS),
        "categorical_label_encoded": len(CATEGORICAL_COLUMNS),
        "dropped_per_paper_sec_vi": len(DROPPED_COLUMNS),
        "zero_fill": 0,  # NATIVE schema has NO zero_fill fabrication
    }
