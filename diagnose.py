"""Diagnostyka zapisanego eksperymentu bez nadpisywania artifacts (CPU i CUDA kolejno)."""

import threadcap  # noqa: F401
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

import config as C
from data import FEATURES, file_signature, load_sample, sample_capture, write_json
from metrics import pick_threshold, score_metrics
from models import load_model, predict, torch_runtime, train_lightgbm


def distribution(X, y):
    result = {}
    for label in (0, 1):
        frame = X.loc[y == label]
        result[str(label)] = {
            "rows": len(frame),
            "numeric_quantiles": {
                col: frame[col].quantile([0, 0.5, 0.95, 1]).tolist() if len(frame) else []
                for col in C.NUMERIC
            },
            "top_categories": {
                col: {str(key): int(count) for key, count in frame[col].value_counts().head(5).items()}
                for col in C.CATEGORICAL
            },
        }
    return result


def scores(y, p, threshold):
    return {
        **score_metrics(y, p, threshold),
        "unique_scores": int(len(np.unique(p))),
        "score_quantiles": {
            str(label): np.quantile(p[y == label], [0, 0.5, 0.95, 1]).tolist()
            for label in (0, 1) if (y == label).any()
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path, help="nowy katalog wewnątrz projektu")
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(C.ROOT) or output == C.ROOT:
        ap.error("wyniki diagnostyki muszą być w podkatalogu projektu")
    if output.is_relative_to(C.ART.resolve()):
        ap.error("katalog diagnostyki nie może znajdować się w artifacts")
    experiment = json.loads(C.EXPERIMENT_PATH.read_text())
    if experiment.get("status") != "complete":
        raise ValueError("diagnostyka wymaga ukończonego eksperymentu")
    if experiment["features"] != FEATURES:
        raise ValueError("schemat cech różni się od zapisanego eksperymentu")
    for capture, signature in experiment["files"].items():
        if file_signature(capture) != signature:
            raise ValueError(f"zmieniony Parquet: {capture}")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "original"
    snapshot.mkdir()
    for path in C.ART.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
        elif path.name in experiment["models"]:
            shutil.copytree(path, snapshot / path.name)
    if (C.ROOT / "pipeline.log").is_file():
        shutil.copy2(C.ROOT / "pipeline.log", snapshot / "pipeline.log")

    cap = experiment["sampling"]["cap_per_capture"]
    print(f"Odtwarzam oryginalne próbki: cap={cap}", flush=True)
    balance = experiment["sampling"]["method"] == "class_behavior_priority_reservoir"
    Xf, yf, fit_info = load_sample(experiment["split"]["train"], cap, balance=balance)
    Xv, yv, valid_info = load_sample(experiment["split"]["valid"], cap, balance=False)
    # Samokontrola: oba backendy dostają dokładnie wiersze oryginalnego eksperymentu.
    assert fit_info == experiment["sampling"]["train"], "inna próbka treningowa"
    assert valid_info == experiment["sampling"]["valid"], "inna próbka walidacyjna"
    joblib.dump((Xf, yf, Xv, yv), output / "samples.joblib")
    report = {
        "experiment_id": experiment["id"], "cap_per_capture": cap,
        "train": distribution(Xf, yf), "valid": distribution(Xv, yv),
        "original_models": {}, "lightgbm_backends": {},
    }
    processor = joblib.load(snapshot / "preprocessor.joblib") if any(
        name in experiment["models"] for name in ("logistic", "mlp")
    ) else None
    if "mlp" in experiment["models"]:
        torch_runtime("cpu")
    for name in experiment["models"]:
        metadata = json.loads((snapshot / name / "metadata.json").read_text())
        if metadata["experiment_id"] != experiment["id"]:
            raise ValueError(f"model {name} pochodzi z innego eksperymentu")
        model = load_model(name, snapshot / name, metadata, "cpu")
        model_report = {}
        for subset, X, y in (("train", Xf, yf), ("valid", Xv, yv)):
            p = predict(name, model, X, metadata, processor, "cpu")
            model_report[subset] = scores(y, p, metadata["threshold"])
            print(f"Oryginalny {name}, {subset}: {model_report[subset]['roc_auc']=}", flush=True)
        report["original_models"][name] = model_report
    write_json(output / "diagnostics.json", report)

    for device in ("cpu", "cuda"):
        directory = output / f"lightgbm_{device}"
        directory.mkdir()
        print(f"LightGBM: identyczna próbka, backend={device}, ostrzeżenia włączone", flush=True)
        C.LGBM_PARAMS["verbose"] = 1
        model, details = train_lightgbm(Xf, yf, Xv, yv, device, directory)
        p = predict("lightgbm", model, Xv, details)
        threshold, _ = pick_threshold(yv, p, C.TARGET_FPR)
        result = {
            "details": details, "validation": scores(yv, p, threshold),
            "trees": model.num_trees(),
            "leaves": [tree["num_leaves"] for tree in model.dump_model()["tree_info"]],
        }
        reloaded = load_model("lightgbm", directory, details, "cpu")
        np.testing.assert_allclose(p, predict("lightgbm", reloaded, Xv, details))
        report["lightgbm_backends"][device] = result
        write_json(output / "diagnostics.json", report)
        print(f"{device}: {result['validation']}, leaves={result['leaves']}", flush=True)

    report["test_capture_samples"] = {}
    for subset in ("test_known", "test_unknown"):
        for capture in experiment["split"][subset]:
            frame, info = sample_capture(capture, cap, balance=False)
            y = frame.pop(C.LABEL).to_numpy()
            capture_report = {"subset": subset, "sampling": info, "features": distribution(frame, y), "models": {}}
            for name in experiment["models"]:
                metadata = json.loads((snapshot / name / "metadata.json").read_text())
                model = load_model(name, snapshot / name, metadata, "cpu")
                p = predict(name, model, frame, metadata, processor, "cpu")
                capture_report["models"][name] = scores(y, p, metadata["threshold"])
            report["test_capture_samples"][capture] = capture_report
            write_json(output / "diagnostics.json", report)
    for capture, signature in experiment["files"].items():
        if file_signature(capture) != signature:
            raise ValueError(f"dane zmieniły się podczas diagnostyki: {capture}")
    print(f"Zakończono diagnostykę: {output}", flush=True)


if __name__ == "__main__":
    main()
