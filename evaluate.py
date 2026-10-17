"""Ewaluacja na odłożonych capture'ach: ogółem + per-capture + per-family.

Użycie:
    python evaluate.py
    python evaluate.py --captures CTU-IoT-Malware-Capture-34-1
"""

import threadcap  # noqa: F401  MUSI być pierwsze
import argparse
import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

import config as C


def tpr_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= target].max(initial=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*", default=None)
    args = ap.parse_args()

    manifest = json.loads(C.MANIFEST_PATH.read_text())
    avail = [c for c in manifest if (C.PARQUET_DIR / (c + ".parquet")).exists()]
    captures = args.captures or [c for c in C.TEST_CAPTURES if c in avail] or avail[-1:]

    feats = json.loads(C.FEATURES_PATH.read_text())
    cols = feats["categorical"] + feats["numeric"]
    thr = json.loads(C.THRESHOLD_PATH.read_text())["threshold"]
    model = lgb.Booster(model_file=str(C.MODEL_PATH))

    df = pd.concat([pd.read_parquet(C.PARQUET_DIR / (c + ".parquet")) for c in captures], ignore_index=True)
    for col in feats["categorical"]:
        if col in df.columns:
            df[col] = df[col].fillna("-").astype("category")
    X = df[[c for c in cols if c in df.columns]]
    y = df[C.LABEL].to_numpy()
    if len(set(y)) < 2:
        raise SystemExit(f"test single-class ({captures}): dołóż capture drugiej klasy.")
    p = model.predict(X)

    metrics = {
        "test_captures": captures,
        "n": int(len(y)),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "tpr_at_fpr_1%": round(tpr_at_fpr(y, p, 0.01), 4),
        "tpr_at_fpr_0.1%": round(tpr_at_fpr(y, p, 0.001), 4),
        "threshold": thr,
        "confusion_at_thr": confusion_matrix(y, (p >= thr).astype(int)).tolist(),
        "per_capture": {},
    }
    df["_p"] = p
    for cap, sub in df.groupby(C.CAPTURE):
        yy, pp = sub[C.LABEL].to_numpy(), sub["_p"].to_numpy()
        metrics["per_capture"][str(cap)] = {
            "n": int(len(sub)),
            "malware_rate": round(float(yy.mean()), 4),
            "roc_auc": round(float(roc_auc_score(yy, pp)), 4) if len(set(yy)) > 1 else None,
            "family": str(sub[C.FAMILY].iloc[0]) if C.FAMILY in sub else "?",
        }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xscale("log")
    plt.xlabel("FPR (log)")
    plt.ylabel("TPR")
    plt.title(f"ROC AUC={metrics['roc_auc']:.4f}")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(C.ROC_PATH, dpi=120)
    print("wykres ->", C.ROC_PATH)


if __name__ == "__main__":
    main()
