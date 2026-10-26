"""Wspólne dane eksperymentu: rozłączne scenariusze i próbka bez pełnego odczytu do RAM."""

import threadcap  # noqa: F401
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C

FEATURES = C.CATEGORICAL + C.NUMERIC


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def capture_group(capture):
    return capture.split("__", 1)[0]


def family_of(capture):
    group = capture_group(capture)
    if group in C.FAMILY_OF:
        return C.FAMILY_OF[group]
    if group in C.CAPTURES and "Honeypot" in group:
        return "benign"
    raise ValueError(f"Nieznany scenariusz {capture}: uzupełnij config.py")


def make_split(captures):
    captures = sorted(captures)
    groups = {capture_group(c) for c in captures}
    required = set(C.VALID_CAPTURES + C.KNOWN_TEST_CAPTURES + [C.GENERALIZATION_BENIGN])
    required.update(c for c, family in C.FAMILY_OF.items() if family in C.HELD_OUT_FAMILIES)
    missing = sorted(required - groups)
    if missing:
        raise ValueError(f"Brak scenariuszy wymaganych do podziału: {missing}")
    split = {name: [] for name in ("train", "valid", "test_known", "test_unknown")}
    for capture in captures:
        group, family = capture_group(capture), family_of(capture)
        if family in C.HELD_OUT_FAMILIES or group == C.GENERALIZATION_BENIGN:
            subset = "test_unknown"
        elif group in C.VALID_CAPTURES:
            subset = "valid"
        elif group in C.KNOWN_TEST_CAPTURES:
            subset = "test_known"
        else:
            subset = "train"
        split[subset].append(capture)
    validate_split(split)
    return split


def validate_split(split):
    seen = set()
    for name in ("train", "valid", "test_known", "test_unknown"):
        captures = split[name]
        if not captures or len(captures) != len(set(captures)):
            raise ValueError(f"Pusty lub powielony podział: {name}")
        groups = {capture_group(c) for c in captures}
        if seen & groups:
            raise ValueError(f"Wyciek scenariuszy między podziałami: {seen & groups}")
        seen.update(groups)
    train_families = {family_of(c) for c in split["train"]} - {"benign"}
    valid_families = {family_of(c) for c in split["valid"]} - {"benign"}
    known_families = {family_of(c) for c in split["test_known"]} - {"benign"}
    unknown_families = {family_of(c) for c in split["test_unknown"]} - {"benign"}
    if unknown_families & (train_families | valid_families | known_families):
        raise ValueError("Wyciek odłożonej rodziny malware")
    if not known_families or not known_families <= train_families:
        raise ValueError("Test znanych rodzin zawiera rodzinę nieobecną w treningu")


def file_signature(capture):
    path = C.PARQUET_DIR / f"{capture}.parquet"
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "rows": pq.ParquetFile(path).metadata.num_rows}


def capture_batches(capture, batch_rows=C.BATCH_ROWS):
    path = C.PARQUET_DIR / f"{capture}.parquet"
    parquet = pq.ParquetFile(path)
    missing = set(FEATURES + [C.LABEL]) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{capture}: brak kolumn {sorted(missing)}; powtórz featurize.py")
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=FEATURES + [C.LABEL], use_threads=False):
        df = batch.to_pandas(use_threads=False)
        if df[C.LABEL].isna().any() or not df[C.LABEL].isin([0, 1]).all():
            raise ValueError(f"{capture}: etykiety muszą mieć wartości 0/1")
        df[C.LABEL] = df[C.LABEL].astype("int8")
        for col in C.CATEGORICAL:
            df[col] = df[col].astype("string").fillna("-").astype(object)
        for col in C.NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("float32")
        if not np.isfinite(df[C.NUMERIC].to_numpy()).all() or (df[C.NUMERIC] < 0).any().any():
            raise ValueError(f"{capture}: niepoprawne cechy liczbowe; powtórz featurize.py")
        for col in ("orig_p", "resp_p"):
            if (df[col] > 65535).any() or (df[col] % 1 != 0).any():
                raise ValueError(f"{capture}: niepoprawny port {col}")
        yield df


def sample_capture(capture, cap, seed=C.SEED):
    if cap < 2:
        raise ValueError("Limit na capture musi być >= 2")
    digest = hashlib.sha256(f"{seed}:{capture}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    counts = [0, 0]
    kept = [pd.DataFrame(), pd.DataFrame()]
    offset = 0
    for df in capture_batches(capture):
        df["_row"] = np.arange(offset, offset + len(df), dtype=np.int64)
        df["_key"] = rng.random(len(df))
        offset += len(df)
        for label in (0, 1):
            part = df[df[C.LABEL] == label]
            counts[label] += len(part)
            if len(part):
                candidates = pd.concat([kept[label], part], ignore_index=True) if len(kept[label]) else part
                kept[label] = candidates.nsmallest(cap, "_key")
    total = sum(counts)
    if not total:
        raise ValueError(f"{capture}: pusty Parquet")
    size = min(cap, total)
    positives = round(size * counts[1] / total)
    if all(counts):
        positives = max(1, min(size - 1, positives))
    quotas = [size - positives, positives]
    sample = pd.concat([kept[label].head(quotas[label]) for label in (0, 1)], ignore_index=True)
    sample = sample.sort_values("_row").reset_index(drop=True)
    fingerprint = hashlib.sha256(sample["_row"].to_numpy(dtype="<i8").tobytes()).hexdigest()
    info = {
        "source_counts": {"benign": counts[0], "malicious": counts[1]},
        "sample_counts": {"benign": quotas[0], "malicious": quotas[1]},
        "row_indices_sha256": fingerprint,
    }
    return sample.drop(columns=["_row", "_key"]), info


def load_sample(captures, cap):
    frames, info = [], {}
    for capture in captures:
        frame, info[capture] = sample_capture(capture, cap)
        frames.append(frame)
        print(f"{capture}: próbka={len(frame)}, klasy={info[capture]['sample_counts']}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    y = df.pop(C.LABEL).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError(f"Podział {captures} musi zawierać obie klasy")
    return df[FEATURES], y, info
