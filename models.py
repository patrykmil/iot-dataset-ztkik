"""Trzy modele, jeden preprocessing dla regresji logistycznej i MLP."""

import threadcap  # noqa: F401
import copy

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config as C
from data import FEATURES

MODEL_NAMES = ("lightgbm", "logistic", "mlp")


def make_preprocessor():
    counts = [col for col in C.NUMERIC if col not in ("orig_p", "resp_p")]
    return ColumnTransformer(
        [
            ("counts", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), counts),
            ("ports", StandardScaler(), ["orig_p", "resp_p"]),
            # shortcut: najwyżej 256 kategorii na cechę; zwiększ po analizie utraty informacji w history.
            ("categories", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=256, dtype=np.float32), C.CATEGORICAL),
        ],
        sparse_threshold=1.0,
    )


def positive_weight(y):
    counts = np.bincount(y, minlength=2)
    if not counts.all():
        raise ValueError("Trening wymaga obu klas")
    return float(counts[0] / counts[1])


def categorical_frame(X, categories=None):
    X = X[FEATURES].copy()
    for col in C.CATEGORICAL:
        X[col] = X[col].astype("category") if categories is None else X[col].astype(
            pd.CategoricalDtype(categories=categories[col])
        )
    return X


def train_lightgbm(Xf, yf, Xv, yv, device, directory):
    import lightgbm as lgb

    Xf = categorical_frame(Xf)
    categories = {col: Xf[col].cat.categories.tolist() for col in C.CATEGORICAL}
    Xv = categorical_frame(Xv, categories)
    params = {**C.LGBM_PARAMS, "device_type": device, "scale_pos_weight": positive_weight(yf)}
    if device == "cuda":
        params["gpu_device_id"] = 0
    fit = lgb.Dataset(Xf, label=yf, categorical_feature=C.CATEGORICAL)
    valid = lgb.Dataset(Xv, label=yv, reference=fit)
    try:
        model = lgb.train(
            params, fit, C.LGBM_ROUNDS, valid_sets=[valid],
            callbacks=[lgb.early_stopping(C.LGBM_EARLY_STOP), lgb.log_evaluation(25)],
        )
    except lgb.basic.LightGBMError as exc:
        raise RuntimeError(
            f"LightGBM ({device}) nie zakończył treningu. Dla CUDA wymagany jest działający sterownik "
            "i build USE_CUDA=ON. Nie przechodzę automatycznie na CPU."
        ) from exc
    model.save_model(str(directory / "model.txt"))
    return model, {"parameters": params, "best_iteration": model.best_iteration, "categories": categories}


def train_logistic(Xf, yf, Xv, yv, directory):
    params = {
        "loss": "log_loss", "alpha": C.LOGISTIC_ALPHA, "average": True,
        "random_state": C.SEED, "n_jobs": 1,
    }
    model = SGDClassifier(**params)
    rng = np.random.default_rng(C.SEED)
    weight = positive_weight(yf)
    best_auc, best_model, stale, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, C.LOGISTIC_EPOCHS + 1):
        order = rng.permutation(len(yf))
        for start in range(0, len(order), C.MLP_BATCH_SIZE):
            idx = order[start:start + C.MLP_BATCH_SIZE]
            model.partial_fit(Xf[idx], yf[idx], classes=np.array([0, 1]), sample_weight=np.where(yf[idx] == 1, weight, 1.0).astype(np.float32))
        auc = float(roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))
        print(f"logistic epoch={epoch} valid_auc={auc:.6f}", flush=True)
        if auc > best_auc:
            best_auc, best_model, stale, best_epoch = auc, copy.deepcopy(model), 0, epoch
        else:
            stale += 1
        if stale >= C.EARLY_STOP_PATIENCE:
            break
    joblib.dump(best_model, directory / "model.joblib")
    return best_model, {"parameters": params, "positive_weight": weight, "best_epoch": best_epoch}


def torch_runtime(device):
    import torch

    torch.set_num_threads(C.THREADS)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch nie widzi CUDA. Sprawdź sterownik NVIDIA i pakiet PyTorch; brak fallbacku na CPU.")
    return torch


def make_mlp(input_features):
    import torch

    layers = []
    sizes = [input_features, *C.MLP_HIDDEN]
    for left, right in zip(sizes, sizes[1:]):
        layers.extend([torch.nn.Linear(left, right), torch.nn.ReLU()])
    layers.append(torch.nn.Linear(sizes[-1], 1))
    return torch.nn.Sequential(*layers)


def dense_batch(X):
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


def predict_mlp(model, X, device):
    import torch

    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, X.shape[0], C.MLP_BATCH_SIZE):
            tensor = torch.from_numpy(dense_batch(X[start:start + C.MLP_BATCH_SIZE])).to(device)
            result.append(torch.sigmoid(model(tensor)).flatten().cpu().numpy())
    return np.concatenate(result)


def train_mlp(Xf, yf, Xv, yv, device, directory):
    import torch

    torch.manual_seed(C.SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(C.SEED)
    model = make_mlp(Xf.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.MLP_LEARNING_RATE)
    weight = positive_weight(yf)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    rng = np.random.default_rng(C.SEED)
    best_auc, best_state, stale, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, C.MLP_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(yf))
        for start in range(0, len(order), C.MLP_BATCH_SIZE):
            idx = order[start:start + C.MLP_BATCH_SIZE]
            batch = torch.from_numpy(dense_batch(Xf[idx])).to(device)
            target = torch.from_numpy(yf[idx].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch).flatten(), target)
            if not torch.isfinite(loss):
                raise RuntimeError("MLP: nieskończona strata, sprawdź dane i learning rate")
            loss.backward()
            optimizer.step()
        auc = float(roc_auc_score(yv, predict_mlp(model, Xv, device)))
        print(f"mlp epoch={epoch} valid_auc={auc:.6f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale, best_epoch = 0, epoch
        else:
            stale += 1
        if stale >= C.EARLY_STOP_PATIENCE:
            break
    model.load_state_dict(best_state)
    torch.save({"input_features": Xf.shape[1], "hidden": C.MLP_HIDDEN, "state_dict": best_state}, directory / "model.pt")
    return model, {
        "hidden": C.MLP_HIDDEN, "learning_rate": C.MLP_LEARNING_RATE,
        "batch_size": C.MLP_BATCH_SIZE, "positive_weight": weight, "best_epoch": best_epoch,
    }


def load_model(name, directory, metadata, device):
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(directory / "model.txt"))
    if name == "logistic":
        return joblib.load(directory / "model.joblib")
    import torch

    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["hidden"] != C.MLP_HIDDEN:
        raise ValueError("Architektura MLP różni się od zapisanego modelu; użyj konfiguracji treningowej")
    model = make_mlp(checkpoint["input_features"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def predict(name, model, X, metadata, preprocessor=None, device="cpu"):
    if name == "lightgbm":
        return model.predict(categorical_frame(X, metadata["categories"]), num_threads=C.THREADS)
    X = preprocessor.transform(X).astype(np.float32)
    if name == "logistic":
        return model.predict_proba(X)[:, 1]
    return predict_mlp(model, X, device)
