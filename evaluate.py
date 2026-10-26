"""Porównanie na zapisanym teście: próg z walidacji, pełne capture'y, odczyt partiami."""

import threadcap  # noqa: F401  MUSI być pierwsze
import argparse
import gc
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config as C
from data import FEATURES, capture_batches, file_signature, validate_split, write_json
from metrics import score_metrics
from models import load_model, predict, torch_runtime


def evaluate_subset(name, model, metadata, captures, families, preprocessor, device):
    all_y, all_p, per_capture = [], [], {}
    family_y, family_p = {}, {}
    prediction_seconds = 0.0
    for capture in captures:
        labels, scores = [], []
        for df in capture_batches(capture):
            start = time.perf_counter()
            proba = predict(name, model, df[FEATURES], metadata, preprocessor, device)
            prediction_seconds += time.perf_counter() - start
            labels.append(df[C.LABEL].to_numpy())
            scores.append(np.asarray(proba, dtype=np.float64))
        if not labels:
            raise ValueError(f"{capture}: pusty test")
        y, p = np.concatenate(labels), np.concatenate(scores)
        per_capture[capture] = {"family": families[capture], **score_metrics(y, p, metadata["threshold"])}
        all_y.append(y)
        all_p.append(p)
        family_y.setdefault(families[capture], []).append(y)
        family_p.setdefault(families[capture], []).append(p)
        print(f"{name}: oceniono {capture}, n={len(y)}", flush=True)
    y, p = np.concatenate(all_y), np.concatenate(all_p)
    result = score_metrics(y, p, metadata["threshold"])
    if result["roc_auc"] is None:
        raise ValueError("Łączny zbiór testowy musi zawierać obie klasy")
    result.update({
        "prediction_seconds": prediction_seconds,
        "flows_per_second": len(y) / prediction_seconds if prediction_seconds else None,
        "prediction_device": device if name == "mlp" else "cpu",
        "per_capture": per_capture,
        "per_family": {
            family: score_metrics(np.concatenate(family_y[family]), np.concatenate(family_p[family]), metadata["threshold"])
            for family in family_y
        },
        "macro_capture_fpr": float(np.mean([r["fpr"] for r in per_capture.values() if r["fpr"] is not None])),
        "macro_capture_recall": float(np.mean([r["recall"] for r in per_capture.values() if r["recall"] is not None])),
    })
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p, drop_intermediate=True)
    return result, (fpr, tpr, recall, precision)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="backend predykcji MLP; LightGBM/sklearn przewidują na CPU")
    args = ap.parse_args()
    experiment = json.loads(C.EXPERIMENT_PATH.read_text())
    if experiment.get("status") != "complete":
        raise ValueError("Trening nie został ukończony; nie można porównywać częściowo nadpisanych modeli")
    if experiment["features"] != FEATURES:
        raise ValueError("Schemat cech różni się od treningu; powtórz trening")
    validate_split(experiment["split"])
    for capture, signature in experiment["files"].items():
        if file_signature(capture) != signature:
            raise ValueError(f"{capture}: dane zmieniły się od treningu; powtórz eksperyment")
    if "mlp" in experiment["models"]:
        torch_runtime(args.device)
    preprocessor = joblib.load(C.ART / "preprocessor.joblib") if any(
        name in experiment["models"] for name in ("logistic", "mlp")
    ) else None
    rows, results = [], {"experiment_id": experiment["id"], "models": {}}
    figures = {subset: plt.subplots(1, 2, figsize=(11, 4)) for subset in ("test_known", "test_unknown")}
    for name in experiment["models"]:
        directory = C.ART / name
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["experiment_id"] != experiment["id"]:
            raise ValueError(f"{name}: model pochodzi z innego/przerwanego eksperymentu; powtórz trening")
        model = load_model(name, directory, metadata, args.device)
        model_results = {}
        for subset in figures:
            metrics, curves = evaluate_subset(
                name, model, metadata, experiment["split"][subset], experiment["families"], preprocessor, args.device,
            )
            model_results[subset] = metrics
            rows.append({
                "model": name, "test": subset, "training_device": metadata["training_device"],
                "training_seconds": metadata["training_seconds"],
                **{key: value for key, value in metrics.items() if key not in ("per_capture", "per_family", "confusion_at_threshold")},
            })
            _, axes = figures[subset]
            fpr, tpr, recall, precision = curves
            axes[0].plot(fpr, tpr, label=f"{name} AUC={metrics['roc_auc']:.6f}")
            axes[1].plot(recall, precision, label=f"{name} AP={metrics['average_precision']:.6f}")
        results["models"][name] = {"training_seconds": metadata["training_seconds"], "training_device": metadata["training_device"], **model_results}
        write_json(directory / "metrics.json", model_results)
        del model
        gc.collect()
        if name == "mlp" and args.device == "cuda":
            import torch

            torch.cuda.empty_cache()
    for capture, signature in experiment["files"].items():
        if file_signature(capture) != signature:
            raise ValueError(f"{capture}: dane zmieniły się podczas ewaluacji; powtórz eksperyment")
    write_json(C.ART / "comparison.json", results)
    table = pd.DataFrame(rows)
    table.to_csv(C.ART / "comparison.csv", index=False)
    print(table[["model", "test", "roc_auc", "average_precision", "recall", "fpr", "training_seconds", "flows_per_second"]].to_string(index=False))
    for subset, (figure, axes) in figures.items():
        axes[0].set(xlabel="FPR", ylabel="TPR", xscale="log", xlim=(1e-5, 1), ylim=(0, 1.01))
        axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()
        figure.suptitle(subset)
        figure.tight_layout()
        figure.savefig(C.ART / f"{subset}_curves.png", dpi=120)
        plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
