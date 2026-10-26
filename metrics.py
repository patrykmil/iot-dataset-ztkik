"""Próg wybieramy na walidacji; metryki testu nie zmieniają progu."""

import threadcap  # noqa: F401
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def validate_scores(y, proba):
    y, proba = np.asarray(y), np.asarray(proba, dtype=np.float64)
    if y.ndim != 1 or proba.shape != y.shape or not len(y):
        raise ValueError("Etykiety i prawdopodobieństwa muszą być niepustymi wektorami tej samej długości")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("Wymagane etykiety 0/1 i skończone prawdopodobieństwa w [0, 1]")
    return y, proba


def threshold_rates(y, proba, threshold):
    y, proba = validate_scores(y, proba)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Próg musi być skończony i nieujemny")
    tn, fp, fn, tp = confusion_matrix(y, proba >= threshold, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else None
    return {
        "confusion_at_threshold": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": precision,
        "recall": recall,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "fpr": float(fp / (tn + fp)) if tn + fp else None,
    }


def pick_threshold(y, proba, target_fpr):
    y, proba = validate_scores(y, proba)
    if len(np.unique(y)) != 2 or not 0 <= target_fpr <= 1:
        raise ValueError("Dobór progu wymaga obu klas i FPR w [0, 1]")
    fpr, tpr, thresholds = roc_curve(y, proba, drop_intermediate=False)
    thresholds[0] = np.nextafter(1.0, np.inf)  # skończony próg „żadnych alarmów”, także gdy p=1
    eligible = np.flatnonzero(fpr <= target_fpr)
    index = eligible[np.argmax(tpr[eligible])]
    threshold = float(thresholds[index])
    rates = threshold_rates(y, proba, threshold)
    if rates["fpr"] > target_fpr:
        raise ValueError("Wybrany próg przekracza docelowy FPR")
    return threshold, rates


def score_metrics(y, proba, threshold):
    y, proba = validate_scores(y, proba)
    result = {
        "n": len(y),
        "benign": int((y == 0).sum()),
        "malicious": int((y == 1).sum()),
        "malware_rate": float(y.mean()),
        "threshold": threshold,
        **threshold_rates(y, proba, threshold),
        "roc_auc": None,
        "average_precision": None,
        "oracle_tpr_at_fpr_1pct": None,
        "oracle_tpr_at_fpr_0_1pct": None,
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, proba))
        result["average_precision"] = float(average_precision_score(y, proba))
        fpr, tpr, _ = roc_curve(y, proba)
        for target, key in ((0.01, "oracle_tpr_at_fpr_1pct"), (0.001, "oracle_tpr_at_fpr_0_1pct")):
            result[key] = float(tpr[fpr <= target].max())
    return result
