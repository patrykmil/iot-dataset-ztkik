"""Wyniki per capture na oryginalnej próbce walidacji, bez zmiany progów."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import threadcap  # noqa: F401
import json
import joblib

import config as C
from data import file_signature, sample_capture, write_json
from metrics import score_metrics
from models import load_model, predict, torch_runtime

root = Path(__file__).resolve().parent
experiment = json.loads(C.EXPERIMENT_PATH.read_text())
processor = joblib.load(C.ART / "preprocessor.joblib")
torch_runtime("cpu")
metadata = {name: json.loads((C.ART / name / "metadata.json").read_text()) for name in experiment["models"]}
models = {name: load_model(name, C.ART / name, metadata[name], "cpu") for name in experiment["models"]}
report = {"experiment_id": experiment["id"], "prediction_device": "cpu", "captures": {}}
for capture in experiment["split"]["valid"]:
    assert file_signature(capture) == experiment["files"][capture]
    frame, info = sample_capture(capture, experiment["sampling"]["cap_per_capture"], seed=experiment["seed"], balance=False)
    assert info == experiment["sampling"]["valid"][capture], "inna próbka walidacyjna"
    y = frame.pop(C.LABEL).to_numpy()
    results = {}
    for name in experiment["models"]:
        assert metadata[name]["experiment_id"] == experiment["id"]
        p = predict(name, models[name], frame, metadata[name], processor, "cpu")
        results[name] = score_metrics(y, p, metadata[name]["threshold"])
        print(capture, name, {key: results[name][key] for key in ("n", "malicious", "roc_auc", "recall", "fpr")}, flush=True)
    report["captures"][capture] = results
    write_json(root / "validation_per_capture.json", report)
