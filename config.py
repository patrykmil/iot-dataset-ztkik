"""Konfiguracja detektora IoT23. Etykiety per-flow z conn.log.labeled,
jednostka podziału: CAPTURE (scenariusz), nie flow.
"""

import os
from pathlib import Path

from threadcap import THREADS  # noqa: F401  (re-eksport, threadcap leci pierwszy)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
RAW_DIR = Path(os.environ.get("DETECTOR_RAW", str(ROOT / "raw")))
PARQUET_DIR = ART / "parquet"
MANIFEST_PATH = ART / "manifest.json"
MODEL_PATH = ART / "model.txt"
THRESHOLD_PATH = ART / "threshold.json"
FEATURES_PATH = ART / "features.json"
METRICS_PATH = ART / "metrics.json"
ROC_PATH = ART / "roc.png"

SEED = 42
LABEL = "label"
CAPTURE = "_capture"  # grupa anty-wyciekowa: jeden scenariusz = jedna grupa
FAMILY = "_family"  # rodzina malware do ewaluacji generalizacji

BASE_URL = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"

# Pełne 23 scenariusze IoT23 (20 malware + 3 benign).
# Honeypot-7-1 to split 6x Somfy-0N (rozwijane w featurize.download_capture).
CAPTURES = [
    "CTU-IoT-Malware-Capture-1-1",  # Hide and Seek, ~1M flowów
    "CTU-IoT-Malware-Capture-3-1",  # Muhstik, ~156k
    "CTU-IoT-Malware-Capture-7-1",  # Linux.Mirai, ~11M
    "CTU-IoT-Malware-Capture-8-1",  # Hakai, ~10k
    "CTU-IoT-Malware-Capture-9-1",  # Linux.Hajime, ~6M
    "CTU-IoT-Malware-Capture-17-1",  # Kenjiro, ~54M
    "CTU-IoT-Malware-Capture-20-1",  # Torii, ~3k
    "CTU-IoT-Malware-Capture-21-1",  # Torii, ~3k
    "CTU-IoT-Malware-Capture-33-1",  # Kenjiro, ~54M
    "CTU-IoT-Malware-Capture-34-1",  # Mirai, ~23k
    "CTU-IoT-Malware-Capture-35-1",  # Mirai, ~10M
    "CTU-IoT-Malware-Capture-36-1",  # Okiru, ~14M
    "CTU-IoT-Malware-Capture-39-1",  # IRCBot, ~74M
    "CTU-IoT-Malware-Capture-42-1",  # Trojan, ~4k
    "CTU-IoT-Malware-Capture-43-1",  # Mirai, ~67M
    "CTU-IoT-Malware-Capture-44-1",  # Mirai, ~200
    "CTU-IoT-Malware-Capture-48-1",  # Mirai, ~3M
    "CTU-IoT-Malware-Capture-49-1",  # Mirai, ~5M
    "CTU-IoT-Malware-Capture-52-1",  # Mirai, ~20M
    "CTU-IoT-Malware-Capture-60-1",  # Gagfyt, ~3.5M
    "CTU-Honeypot-Capture-4-1",  # benign Philips HUE
    "CTU-Honeypot-Capture-5-1",  # benign Amazon Echo
    "CTU-Honeypot-Capture-7-1",  # benign Somfy (split 6x)
]
STARTER_CAPTURES = [  # szybki start (małe, zróżnicowane) — pełny run bierze CAPTURES
    "CTU-IoT-Malware-Capture-34-1",
    "CTU-IoT-Malware-Capture-44-1",
    "CTU-IoT-Malware-Capture-20-1",
    "CTU-IoT-Malware-Capture-42-1",
    "CTU-IoT-Malware-Capture-8-1",
    "CTU-IoT-Malware-Capture-3-1",
    "CTU-Honeypot-Capture-4-1",
    "CTU-Honeypot-Capture-5-1",
    "CTU-Honeypot-Capture-7-1",
]
# Rodziny do podziału na znane (train) vs held-out (test generalizacji).
FAMILY_OF = {
    "CTU-IoT-Malware-Capture-1-1": "HideAndSeek",
    "CTU-IoT-Malware-Capture-3-1": "Muhstik",
    "CTU-IoT-Malware-Capture-7-1": "Mirai",
    "CTU-IoT-Malware-Capture-8-1": "Hakai",
    "CTU-IoT-Malware-Capture-9-1": "Hajime",
    "CTU-IoT-Malware-Capture-17-1": "Kenjiro",
    "CTU-IoT-Malware-Capture-20-1": "Torii",
    "CTU-IoT-Malware-Capture-21-1": "Torii",
    "CTU-IoT-Malware-Capture-33-1": "Kenjiro",
    "CTU-IoT-Malware-Capture-34-1": "Mirai",
    "CTU-IoT-Malware-Capture-35-1": "Mirai",
    "CTU-IoT-Malware-Capture-36-1": "Okiru",
    "CTU-IoT-Malware-Capture-39-1": "IRCBot",
    "CTU-IoT-Malware-Capture-42-1": "Trojan",
    "CTU-IoT-Malware-Capture-43-1": "Mirai",
    "CTU-IoT-Malware-Capture-44-1": "Mirai",
    "CTU-IoT-Malware-Capture-48-1": "Mirai",
    "CTU-IoT-Malware-Capture-49-1": "Mirai",
    "CTU-IoT-Malware-Capture-52-1": "Mirai",
    "CTU-IoT-Malware-Capture-60-1": "Gagfyt",
}
# Test: held-out rodziny (Torii, Gagfyt) + 1 benign. Ciągłość ze starterem (20-1).
TEST_CAPTURES = [
    "CTU-IoT-Malware-Capture-20-1",
    "CTU-IoT-Malware-Capture-60-1",
    "CTU-Honeypot-Capture-7-1__Somfy-01",
]

TARGET_FPR = 0.01

# detailed-label i IP wywalone celowo: pierwsze wynika wprost z etykiety
# (leak), drugie to adresy labu (nie transferują się na inny ruch).
DROP_COLS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "tunnel_parents",
    "detailed-label",
    "local_orig",
    "local_resp",
]

CATEGORICAL = ["proto", "service", "conn_state", "history"]
NUMERIC = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
    "orig_p",
    "resp_p",
    "bytes_per_pkt",
    "resp_orig_ratio",
    "bytes_total",
]

RF_PARAMS = dict(
    n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=5, n_jobs=THREADS, random_state=SEED
)
RF_SAMPLE_ROWS = 300_000

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    learning_rate=0.05,
    verbose=-1,
    seed=SEED,
    num_threads=THREADS,
)
LGBM_ROUNDS = 500
LGBM_EARLY_STOP = 50
