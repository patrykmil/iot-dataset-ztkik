"""Powtórzenie CPU/CUDA na zachowanej próbce; uruchamiaj z katalogu głównego projektu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import threadcap  # noqa: F401
import json
import time

import joblib
import numpy as np

import config as C
from data import file_signature, write_json
from evaluate import evaluate_subset
from metrics import pick_threshold, score_metrics
from models import load_model, predict, train_lightgbm


root = Path(__file__).resolve().parent
experiment = json.loads((root / "original" / "experiment.json").read_text())
for capture, signature in experiment["files"].items():
    assert file_signature(capture) == signature, f"zmienione dane: {capture}"
Xf, yf, Xv, yv = joblib.load(root / "samples.joblib")
report = {"experiment_id": experiment["id"], "weighting": "Dataset row weights", "models": {}}
for device in ("cpu", "cuda"):
    directory = root / f"fixed_lightgbm_{device}"
    directory.mkdir(exist_ok=False)
    start = time.perf_counter()
    model, details = train_lightgbm(Xf, yf, Xv, yv, device, directory)
    seconds = time.perf_counter() - start
    p = predict("lightgbm", model, Xv, details)
    threshold, _ = pick_threshold(yv, p, C.TARGET_FPR)
    assert len(np.unique(p)) > 1, "stałe oceny po poprawce"
    reloaded = load_model("lightgbm", directory, details, "cpu")
    np.testing.assert_allclose(p, predict("lightgbm", reloaded, Xv, details))
    metadata = {
        "experiment_id": experiment["id"], "training_device": device,
        "training_seconds": seconds, "threshold": threshold,
        "validation": score_metrics(yv, p, threshold), **details,
    }
    write_json(directory / "metadata.json", metadata)
    result = {"metadata": metadata, "trees": model.num_trees(), "unique_scores": len(np.unique(p))}
    result["training"] = score_metrics(yf, predict("lightgbm", model, Xf, details), threshold)
    for subset in ("test_known", "test_unknown"):
        result[subset], _ = evaluate_subset(
            "lightgbm", model, metadata, experiment["split"][subset], experiment["families"], None, "cpu",
        )
    report["models"][device] = result
    write_json(root / "fixed_comparison.json", report)
    print(f"POPRAWKA {device}: trees={result['trees']}, unique_scores={result['unique_scores']}", flush=True)
    for subset in ("training", "test_known", "test_unknown"):
        r = result[subset]
        print(subset, {key: r[key] for key in ("roc_auc", "average_precision", "recall", "fpr")}, flush=True)

for capture, signature in experiment["files"].items():
    assert file_signature(capture) == signature, f"zmienione dane: {capture}"
print("Weryfikacja poprawki zakończona; artifacts pozostaje bez zmian.", flush=True)
