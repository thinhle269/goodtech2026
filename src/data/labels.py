from __future__ import annotations

import re
import pandas as pd


def _norm(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


EDGE_FAMILY = {
    "normal": "Benign",
    "ddos_udp": "DoS_DDoS", "ddos_icmp": "DoS_DDoS", "ddos_tcp": "DoS_DDoS", "ddos_http": "DoS_DDoS",
    "port_scanning": "Info_Gathering", "fingerprinting": "Info_Gathering", "vulnerability_scanner": "Info_Gathering",
    "mitm": "MITM",
    "sql_injection": "Injection", "xss": "Injection", "uploading": "Injection",
    "backdoor": "Malware", "password": "Malware", "ransomware": "Malware",
}


CIC_FAMILY = {
    "benign": "Benign", "benigntraffic": "Benign",
    "ddos_icmp_flood": "DDoS", "ddos_udp_flood": "DDoS", "ddos_tcp_flood": "DDoS",
    "ddos_pshack_flood": "DDoS", "ddos_rstfinflood": "DDoS", "ddos_syn_flood": "DDoS",
    "ddos_synonymousip_flood": "DDoS", "ddos_icmp_fragmentation": "DDoS",
    "ddos_ack_fragmentation": "DDoS", "ddos_udp_fragmentation": "DDoS",
    "ddos_http_flood": "DDoS", "ddos_slowloris": "DDoS",
    "dos_udp_flood": "DoS", "dos_tcp_flood": "DoS", "dos_syn_flood": "DoS", "dos_http_flood": "DoS",
    "mirai_greeth_flood": "Mirai", "mirai_udpplain": "Mirai", "mirai_greip_flood": "Mirai",
    "vulnerabilityscan": "Recon", "recon_hostdiscovery": "Recon", "recon_osscan": "Recon",
    "recon_portscan": "Recon", "recon_pingsweep": "Recon",
    "mitm_arpspoofing": "Spoofing", "dns_spoofing": "Spoofing",
    "browserhijacking": "Web_Based", "sqlinjection": "Web_Based", "commandinjection": "Web_Based",
    "xss": "Web_Based", "backdoor_malware": "Web_Based", "uploading_attack": "Web_Based",
    "dictionarybruteforce": "Brute_Force",
}


def map_edge_family(label: object) -> str:
    s = _norm(label)
    if s in EDGE_FAMILY:
        return EDGE_FAMILY[s]
    if any(t in s for t in ["normal", "benign", "legitimate"]):
        return "Benign"
    if any(t in s for t in ["ddos", "flood"]) or re.search(r"(^|_)dos(_|$)", s):
        return "DoS_DDoS"
    if any(t in s for t in ["fingerprint", "scan", "vulnerability", "recon"]):
        return "Info_Gathering"
    if any(t in s for t in ["mitm", "man_in_the_middle", "spoof"]):
        return "MITM"
    if any(t in s for t in ["xss", "sql", "inject", "upload"]):
        return "Injection"
    if any(t in s for t in ["backdoor", "password", "ransom", "malware"]):
        return "Malware"
    return "Attack_Other"


def map_cic_family(label: object) -> str:
    s = _norm(label)
    if s in CIC_FAMILY:
        return CIC_FAMILY[s]
    if any(t in s for t in ["benign", "normal", "legitimate"]):
        return "Benign"
    if "mirai" in s:
        return "Mirai"
    if "ddos" in s:
        return "DDoS"
    if re.search(r"(^|_)dos(_|$)", s) or s.startswith("dos"):
        return "DoS"
    if any(t in s for t in ["recon", "scan", "hostdiscovery", "host_discovery", "pingsweep", "ping_sweep"]):
        return "Recon"
    if any(t in s for t in ["brute", "dictionary"]):
        return "Brute_Force"
    if "spoof" in s:
        return "Spoofing"
    if any(t in s for t in ["web", "xss", "sql", "inject", "hijack", "backdoor", "upload"]):
        return "Web_Based"
    return "Attack_Other"


def derive_labels(dataset_name: str, labels: pd.Series) -> pd.DataFrame:
    mapper = map_edge_family if dataset_name == "edge_iiotset" else map_cic_family
    family = labels.map(mapper)
    binary = (family != "Benign").astype(int)
    return pd.DataFrame({"raw_label": labels.astype(str), "binary": binary, "family": family})
