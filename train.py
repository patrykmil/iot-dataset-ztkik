"""Trening: RF baseline (próbka) + LightGBM CPU (docelowy).

Split PO CAPTURE'ACH (ostatni train = valid, TEST_CAPTURES odłożone).
Split po flowach = wyciek (ten sam C&C w train i teście).

Użycie:
    python train.py                    # pełny trening
    python train.py --baseline-only    # tylko RF (szybki dowód)
    python train.py --device gpu       # próba GPU (kategoryczne mogą sypnąć -> fallback CPU)
    python train.py --max-rows 50000   # smoke
"""

import threadcap  # noqa: F401  MUSI być pierwsze
import argparse
import gc
import json

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config as C

FEATURES = C.CATEGORICAL + C.NUMERIC


def load(captures, max_rows=None, cap_per_capture=None):
    frames = []
    for c in captures:
        df = pd.read_parquet(C.PARQUET_DIR / (c + ".parquet"))
        if cap_per_capture and len(df) > cap_per_capture:
            # stratified cap per capture: zachowuje malware_rate
            mal = df[df[C.LABEL] == 1].sample(
                n=min((df[C.LABEL] == 1).sum(), int(cap_per_capture * df[C.LABEL].mean())), random_state=C.SEED
            )
            ben = df[df[C.LABEL] == 0].sample(
                n=min((df[C.LABEL] == 0).sum(), cap_per_capture - len(mal)), random_state=C.SEED
            )
            df = pd.concat([mal, ben]).sample(frac=1, random_state=C.SEED).reset_index(drop=True)
        frames.append(df)
        gc.collect()
    df = pd.concat(frames, ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=C.SEED).reset_index(drop=True)
    for col in C.CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("-").astype("category")
    X = df[[c for c in FEATURES if c in df.columns]]
    return X, df[C.LABEL].to_numpy()


def pick_threshold(y_true, proba):
    order = proba.argsort()[::-1]
    y = y_true[order]
    P, N = y.sum(), (1 - y).sum()
    tpr = y.cumsum() / max(P, 1)
    fpr = (1 - y).cumsum() / max(N, 1)
    ok = fpr <= C.TARGET_FPR
    if not ok.any():
        return 0.5, float(tpr[-1]), float(fpr[-1])
    i = int(tpr[ok].argmax())
    thr = float(sorted(proba, reverse=True)[int(ok.sum()) - 1])
    return thr, float(tpr[ok][i]), float(fpr[ok][i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument(
        "--cap-per-capture",
        type=int,
        default=None,
        help="stratified cap flowów na capture (pełna skala: 1000000)",
    )
    ap.add_argument("--test-captures", nargs="*", default=None)
    args = ap.parse_args()

    manifest = json.loads(C.MANIFEST_PATH.read_text())
    avail = [c for c in manifest if (C.PARQUET_DIR / (c + ".parquet")).exists()]
    if not avail:
        raise SystemExit(
            "brak parquetów: manifest wskazuje pliki, których nie ma "
            "(featurize przerwany?). Uruchom featurize.py do końca."
        )
    test = args.test_captures or [c for c in C.TEST_CAPTURES if c in avail] or avail[-1:]
    train = [c for c in avail if c not in test]
    if not train:
        raise SystemExit(
            f"pusty train: wszystko w teście ({test}). "
            f"Manifest ma tylko: {avail}. Dokończ featurize.py --download."
        )
    # Valid = ostatni MIeszany capture, PO którym fit wciąż ma obie klasy.
    # (Czysty valid sypie AUC; valid zjadający jedyny miks sypie fit.)
    # Brak takiego -> stratified 15% fita.
    rates = {c: manifest[c].get("malware_rate", 0.5) for c in train}
    valid = fit = None
    for cand in [c for c in train if 0.0 < rates[c] < 1.0][::-1]:
        f = [c for c in train if c != cand]
        rf = [rates[x] for x in f]
        if any(r < 1.0 for r in rf) and any(r > 0.0 for r in rf):
            valid, fit = [cand], f
            break
    if valid is not None:
        Xf, yf = load(fit, args.max_rows, args.cap_per_capture)
        Xv, yv = load(valid, args.max_rows, args.cap_per_capture)
    else:
        fit, valid = train, ["stratified-15%-of-fit"]
        Xa, ya = load(train, args.max_rows, args.cap_per_capture)
        Xf, Xv, yf, yv = train_test_split(Xa, ya, test_size=0.15, random_state=C.SEED, stratify=ya)
        print("brak mieszanego capture'a -> valid = stratified 15% fita")
    if len(set(yf)) < 2:
        raise SystemExit(f"fit single-class ({fit}): dołóż capture'y drugiej klasy.")
    if len(set(yv)) < 2:
        raise SystemExit(f"valid single-class ({valid}): dołóż mieszany capture.")
    print("fit:", fit, "\nvalid:", valid, "\ntest(odlozone):", test)

    idx = (
        pd.Series(range(len(Xf)))
        .sample(min(C.RF_SAMPLE_ROWS, len(Xf)), random_state=C.SEED)
        .sort_index()
        .to_numpy()
    )
    Xs, ys = Xf.iloc[idx].reset_index(drop=True), yf[idx]
    Xr = pd.get_dummies(Xs[C.CATEGORICAL].astype(str)).join(
        Xs[[c for c in C.NUMERIC if c in Xs]].reset_index(drop=True)
    )
    Xvr = pd.get_dummies(Xv[C.CATEGORICAL].astype(str)).join(
        Xv[[c for c in C.NUMERIC if c in Xv]].reset_index(drop=True)
    )
    Xvr = Xvr.reindex(columns=Xr.columns, fill_value=0)
    rf = RandomForestClassifier(**C.RF_PARAMS).fit(Xr, ys)
    print(f"RF baseline ROC-AUC (valid): {roc_auc_score(yv, rf.predict_proba(Xvr)[:, 1]):.4f}")
    if args.baseline_only:
        return

    device = args.device
    if device == "auto":
        device = "cpu"  # GPU-learner nie wspiera kategorycznych (lekcja z Sysmona)
    params = {
        **C.LGBM_PARAMS,
        "device": device,
        "scale_pos_weight": float((yf == 0).sum() / max((yf == 1).sum(), 1)),
    }
    print("lgbm device:", device)
    dfit = lgb.Dataset(Xf, label=yf)
    dval = lgb.Dataset(Xv, label=yv, reference=dfit)
    try:
        model = lgb.train(
            params,
            dfit,
            C.LGBM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(C.LGBM_EARLY_STOP, verbose=False)],
        )
    except lgb.basic.LightGBMError as e:
        print(f"trening {device} nieudany ({e}), fallback do CPU...")
        params["device"] = "cpu"
        model = lgb.train(
            params,
            dfit,
            C.LGBM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(C.LGBM_EARLY_STOP, verbose=False)],
        )
    proba = model.predict(Xv)
    auc = roc_auc_score(yv, proba)
    thr, tpr, fpr = pick_threshold(yv, proba)
    print(
        f"LightGBM ROC-AUC (valid): {auc:.4f} thr@{C.TARGET_FPR:.0%}FPR={thr:.4f} (TPR={tpr:.4f} FPR={fpr:.4f})"
    )
    model.save_model(C.MODEL_PATH)
    C.THRESHOLD_PATH.write_text(
        json.dumps(
            {"threshold": thr, "valid_auc": float(auc), "valid_tpr": float(tpr), "valid_fpr": float(fpr)}
        )
    )
    C.FEATURES_PATH.write_text(json.dumps({"categorical": C.CATEGORICAL, "numeric": C.NUMERIC}))
    print("zapisano:", C.MODEL_PATH, C.THRESHOLD_PATH, C.FEATURES_PATH)


if __name__ == "__main__":
    main()
