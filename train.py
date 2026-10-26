"""Trening trzech modeli na tych samych capture'ach i wierszach. Bez cichego fallbacku GPU."""

import threadcap  # noqa: F401  MUSI być pierwsze
import argparse
import gc
import importlib.metadata
import json
import platform
import time
import uuid

import joblib
import numpy as np

import config as C
from data import FEATURES, family_of, file_signature, load_sample, make_split, write_json
from metrics import pick_threshold, score_metrics
from models import MODEL_NAMES, make_preprocessor, predict, torch_runtime, train_lightgbm, train_logistic, train_mlp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="LightGBM i MLP; logistic zawsze CPU")
    ap.add_argument("--cap-per-capture", type=int, default=C.CAP_PER_CAPTURE, help="limit wspólnych próbek train i valid")
    args = ap.parse_args()
    if args.cap_per_capture < 2 or len(set(args.models)) != len(args.models):
        ap.error("limit musi być >= 2, modele nie mogą się powtarzać")

    if "lightgbm" in args.models:
        import lightgbm  # noqa: F401  zależność sprawdzamy przed odczytem dużych danych
    torch = torch_runtime(args.device) if "mlp" in args.models else None
    manifest = json.loads(C.MANIFEST_PATH.read_text())
    limited = [capture for capture, info in manifest.items() if info.get("limited")]
    if limited:
        raise ValueError(f"Parquet z --limit nie jest reprezentatywnym datasetem: {limited}. Powtórz konwersję bez --limit.")
    missing = [capture for capture in manifest if not (C.PARQUET_DIR / f"{capture}.parquet").is_file()]
    if missing:
        raise ValueError(f"Manifest wskazuje nieobecne pliki Parquet: {missing}")
    split = make_split(manifest)
    signatures = {capture: file_signature(capture) for capture in manifest}
    print(json.dumps(split, indent=2), flush=True)
    Xf, yf, fit_sampling = load_sample(split["train"], args.cap_per_capture)
    Xv, yv, valid_sampling = load_sample(split["valid"], args.cap_per_capture)
    experiment_id = uuid.uuid4().hex
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scikit-learn")}
    for model, package in (("lightgbm", "lightgbm"), ("mlp", "torch")):
        if model in args.models:
            versions[package] = importlib.metadata.version(package)
    experiment = {
        "id": experiment_id, "status": "running", "models": args.models, "device": args.device,
        "seed": C.SEED, "threads": C.THREADS, "target_fpr": C.TARGET_FPR,
        "features": FEATURES, "split": split,
        "families": {capture: family_of(capture) for capture in manifest},
        "files": signatures, "versions": versions, "python": platform.python_version(),
        "sampling": {"method": "stratified_priority_reservoir", "cap_per_capture": args.cap_per_capture,
                     "train": fit_sampling, "valid": valid_sampling},
    }
    if torch is not None and args.device == "cuda":
        experiment["gpu"] = {"name": torch.cuda.get_device_name(0), "pytorch_cuda": torch.version.cuda}
    C.ART.mkdir(parents=True, exist_ok=True)
    # Unieważnij poprzedni eksperyment przed nadpisaniem któregokolwiek modelu/preprocessingu.
    write_json(C.EXPERIMENT_PATH, experiment)
    for filename in ("comparison.csv", "comparison.json", "test_known_curves.png", "test_unknown_curves.png"):
        (C.ART / filename).unlink(missing_ok=True)
    preprocessor, encoded_fit, encoded_valid = None, None, None
    preprocessing_seconds = 0.0
    if any(name in args.models for name in ("logistic", "mlp")):
        start = time.perf_counter()
        preprocessor = make_preprocessor()
        encoded_fit = preprocessor.fit_transform(Xf).astype(np.float32)
        encoded_valid = preprocessor.transform(Xv).astype(np.float32)
        preprocessing_seconds = time.perf_counter() - start
        joblib.dump(preprocessor, C.ART / "preprocessor.joblib")
    experiment["preprocessing_seconds"] = preprocessing_seconds

    for name in args.models:
        directory = C.ART / name
        directory.mkdir(parents=True, exist_ok=True)
        device = "cpu" if name == "logistic" else args.device
        print(f"Trening {name}, backend={device}", flush=True)
        start = time.perf_counter()
        if name == "lightgbm":
            model, details = train_lightgbm(Xf, yf, Xv, yv, device, directory)
        elif name == "logistic":
            model, details = train_logistic(encoded_fit, yf, encoded_valid, yv, directory)
        else:
            model, details = train_mlp(encoded_fit, yf, encoded_valid, yv, device, directory)
        training_seconds = time.perf_counter() - start
        proba = predict(name, model, Xv, details, preprocessor, device)
        threshold, rates = pick_threshold(yv, proba, C.TARGET_FPR)
        metadata = {
            "experiment_id": experiment_id, "model": name, "training_device": device,
            "training_seconds": training_seconds, "threshold": threshold,
            "validation": score_metrics(yv, proba, threshold), **details,
        }
        write_json(directory / "metadata.json", metadata)
        print(f"{name}: valid ROC-AUC={metadata['validation']['roc_auc']:.6f}, "
              f"próg={threshold:.8g}, FPR={rates['fpr']:.6f}, recall={rates['recall']:.6f}", flush=True)
        del model
        gc.collect()
        if name == "mlp" and device == "cuda":
            import torch

            torch.cuda.empty_cache()

    for capture, signature in signatures.items():
        if file_signature(capture) != signature:
            raise ValueError(f"Dane {capture} zmieniły się podczas treningu; powtórz eksperyment")
    experiment["status"] = "complete"
    write_json(C.EXPERIMENT_PATH, experiment)
    print(f"Zapisano eksperyment: {C.EXPERIMENT_PATH}. Następny krok: python evaluate.py", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
