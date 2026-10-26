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
EXPERIMENT_PATH = ART / "experiment.json"

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
# Podział jawny, niezależny od wyników modeli. Wszystkie części Somfy są razem.
VALID_CAPTURES = [
    "CTU-IoT-Malware-Capture-35-1",
    "CTU-Honeypot-Capture-5-1",
]
KNOWN_TEST_CAPTURES = [
    "CTU-IoT-Malware-Capture-34-1",
    "CTU-Honeypot-Capture-4-1",
]
HELD_OUT_FAMILIES = ["Torii", "Gagfyt"]
GENERALIZATION_BENIGN = "CTU-Honeypot-Capture-7-1"

TARGET_FPR = 0.01
CAP_PER_CAPTURE = 100_000
BATCH_ROWS = 100_000

# Tylko jawnie wybrane cechy; IP, UID, czas i detailed-label nie wchodzą do modelu.
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
    use_quantized_grad=False,  # CUDA z kategoriami nie obsługuje kwantyzowanych gradientów.
)
LGBM_ROUNDS = 500
LGBM_EARLY_STOP = 50

LOGISTIC_EPOCHS = 30
LOGISTIC_ALPHA = 0.0001
MLP_HIDDEN = [64, 32]
MLP_EPOCHS = 30
MLP_BATCH_SIZE = 4096
MLP_LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 5
