"""Zeek conn.log.labeled -> Parquet partiami + aktualizowany manifest.

Bez --download czyta tylko istniejące pliki. --limit jest prefiksem do smoke testów,
nie reprezentatywną próbką do raportowania wyników.
"""

import threadcap  # noqa: F401  MUSI być pierwsze
import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as C
from data import FEATURES, family_of, write_json

HREF_DIR = re.compile(r'href="([A-Za-z0-9._-]+/)"')
DERIVED = ["bytes_per_pkt", "resp_orig_ratio", "bytes_total"]
SOURCE_NUMERIC = [col for col in C.NUMERIC if col not in DERIVED]
SCHEMA = pa.schema(
    [(col, pa.string()) for col in C.CATEGORICAL]
    + [(col, pa.float32()) for col in C.NUMERIC]
    + [(C.LABEL, pa.int8()), (C.CAPTURE, pa.string()), (C.FAMILY, pa.string())]
)


def list_dir(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def exists(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return False
            if exc.code < 500 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(5 * (attempt + 1))


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        return dest
    temporary = dest.with_name(dest.name + ".part")
    print(f"GET {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
            expected = response.headers.get("Content-Length")
        if temporary.stat().st_size == 0 or (expected is not None and temporary.stat().st_size != int(expected)):
            raise ValueError(f"Niekompletny plik pobrany z {url}")
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest


def download_capture(name, dest):
    """Obsługuje bro/, labeled/ oraz podkatalogi Somfy."""
    base = f"{C.BASE_URL}/{name}"
    for sub in ("bro", "labeled"):
        url = f"{base}/{sub}/conn.log.labeled"
        if exists(url):
            return [(fetch(url, dest), name)]
    out = []
    for subdir in sorted(set(HREF_DIR.findall(list_dir(base + "/")))):
        for sub in ("bro", "labeled"):
            url = f"{base}/{subdir}{sub}/conn.log.labeled"
            if exists(url):
                part = f"{name}__{subdir.rstrip('/')}"
                out.append((fetch(url, dest.parent / f"{part}.conn.log.labeled"), part))
                break
    if not out:
        raise ValueError(f"{name}: nie znaleziono conn.log.labeled")
    return out


def validate_fields(fields):
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Brak lub powielone pola w nagłówku #fields")
    renamed = ["orig_p" if col == "id.orig_p" else "resp_p" if col == "id.resp_p" else col for col in fields]
    if len(renamed) != len(set(renamed)):
        raise ValueError("Powielone pola po mapowaniu portów Zeek")
    missing = set(C.CATEGORICAL + SOURCE_NUMERIC + [C.LABEL]) - set(renamed)
    if missing:
        raise ValueError(f"Brak wymaganych pól Zeek: {sorted(missing)}")


def engineer(df):
    df = df.rename(columns={"id.orig_p": "orig_p", "id.resp_p": "resp_p"}).copy()
    missing = set(C.CATEGORICAL + SOURCE_NUMERIC) - set(df.columns)
    if missing:
        raise ValueError(f"Brak cech: {sorted(missing)}")
    for col in SOURCE_NUMERIC:
        values = pd.to_numeric(df[col], errors="raise").fillna(0)
        if not np.isfinite(values.to_numpy()).all() or (values < 0).any():
            raise ValueError(f"Niepoprawne wartości {col}")
        if col in ("orig_p", "resp_p") and ((values > 65535).any() or (values % 1 != 0).any()):
            raise ValueError(f"Niepoprawny port {col}")
        df[col] = values.astype("float32")
    op, ob, rb = df["orig_pkts"], df["orig_bytes"], df["resp_bytes"]
    df["bytes_per_pkt"] = ob / op.replace(0, 1)
    df["resp_orig_ratio"] = rb / ob.replace(0, 1)
    df["bytes_total"] = ob + rb
    if not np.isfinite(df[C.NUMERIC].to_numpy()).all():
        raise ValueError("Cechy liczbowe przekraczają zakres float32")
    for col in C.CATEGORICAL:
        df[col] = df[col].astype("string").fillna("-")
    return df


def parse_block(lines, fields, sep, capture):
    validate_fields(fields)
    rows = []
    for line in lines:
        values = line.rstrip("\r\n").split(sep)
        if len(values) != len(fields):
            # IoT23 bywa oznaczony etykietami dołączonymi przez kilka spacji do końca TSV.
            values = values[:-1] + re.split(r" {2,}", values[-1])
        if len(values) != len(fields):
            raise ValueError(f"{capture}: wiersz ma {len(values)} pól, oczekiwano {len(fields)}")
        rows.append(values)
    df = pd.DataFrame(rows, columns=fields).replace({"-": pd.NA, "(empty)": pd.NA})
    labels = df[C.LABEL].astype("string").str.strip().str.lower()
    y = pd.Series(-1, index=df.index, dtype="int8")
    y[labels.str.startswith("malicious", na=False)] = 1
    y[labels.eq("benign").fillna(False)] = 0
    df = engineer(df.loc[y != -1].copy())
    df[C.LABEL] = y[y != -1]
    df[C.CAPTURE] = capture
    df[C.FAMILY] = family_of(capture)
    return df[FEATURES + [C.LABEL, C.CAPTURE, C.FAMILY]]


def parse_zeek(path, capture, limit=None, block=C.BATCH_ROWS):
    """Generator (ramka, liczba wierszy źródłowych), bez gromadzenia capture'a w RAM."""
    if block < 1 or (limit is not None and limit < 1):
        raise ValueError("block i limit muszą być dodatnie")
    fields, sep, buf, read = None, "\t", [], 0
    with Path(path).open(encoding="utf-8", errors="strict") as source:
        for line in source:
            if line.startswith("#separator"):
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"{path}: brak wartości #separator")
                value = parts[1].strip().encode().decode("unicode_escape")
                if len(value) != 1 or (fields is not None and value != sep):
                    raise ValueError(f"{path}: niepoprawny lub zmieniony separator")
                sep = value
            elif line.startswith("#fields"):
                new_fields = line.strip().split()[1:]
                validate_fields(new_fields)
                if fields is not None and new_fields != fields:
                    raise ValueError(f"{path}: zmiana schematu w środku pliku")
                fields = new_fields
            elif line.startswith("#") or not line.strip():
                continue
            else:
                if fields is None:
                    raise ValueError(f"{path}: dane przed #fields")
                buf.append(line)
                read += 1
                if len(buf) >= block or (limit is not None and read >= limit):
                    yield parse_block(buf, fields, sep, capture), len(buf)
                    buf = []
                if limit is not None and read >= limit:
                    break
    if fields is None:
        raise ValueError(f"{path}: brak nagłówka #fields")
    if buf:
        yield parse_block(buf, fields, sep, capture), len(buf)


def convert(path, capture, out_path, limit=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".tmp")
    counts, source_rows = [0, 0], 0
    try:
        with pq.ParquetWriter(temporary, SCHEMA, compression="snappy") as writer:
            for df, read in parse_zeek(path, capture, limit):
                source_rows += read
                for label in (0, 1):
                    counts[label] += int((df[C.LABEL] == label).sum())
                if len(df):
                    writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
        if not sum(counts):
            raise ValueError(f"{capture}: brak rozpoznanych etykiet; poprzedni Parquet pozostaje bez zmian")
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "flows": sum(counts), "benign": counts[0], "malicious": counts[1],
        "malware_rate": counts[1] / sum(counts), "source_rows": source_rows,
        "dropped_labels": source_rows - sum(counts), "limited": limit is not None,
    }


def capture_name(path):
    if path.name.endswith(".conn.log.labeled") and path.name != "conn.log.labeled":
        name = path.name.removesuffix(".conn.log.labeled")
        family_of(name)
        return name
    for parent in path.parents:
        if parent.name in C.CAPTURES:
            if parent.name == C.GENERALIZATION_BENIGN:
                parts = [part for part in path.relative_to(parent).parts[:-1] if part.startswith("Somfy-")]
                if parts:
                    return f"{parent.name}__{parts[0]}"
            return parent.name
    raise ValueError(f"Nie można ustalić scenariusza dla {path}; umieść plik w katalogu z nazwą capture'a")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help="istniejące conn.log.labeled lub katalogi")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--captures", nargs="+")
    ap.add_argument("--limit", type=int, help="maksymalna liczba wierszy źródłowych; tylko smoke test")
    args = ap.parse_args()
    if (not args.inputs and not args.download) or (args.limit is not None and args.limit < 1):
        ap.error("podaj pliki/katalogi albo --download; limit musi być dodatni")
    if args.captures and not args.download:
        ap.error("--captures jest dostępne tylko z --download")
    if args.captures and set(args.captures) - set(C.CAPTURES):
        ap.error("nieznane capture'y; uzupełnij config.py")
    C.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(C.MANIFEST_PATH.read_text()) if C.MANIFEST_PATH.exists() else {}
    jobs = []
    if args.download:
        for name in args.captures or C.CAPTURES:
            jobs.extend(download_capture(name, C.RAW_DIR / name / "conn.log.labeled"))
    for inp in args.inputs:
        path = Path(inp).resolve()
        files = sorted(path.rglob("*conn.log.labeled")) if path.is_dir() else [path]
        for file in files:
            jobs.append((file, capture_name(file)))
    if not jobs:
        raise ValueError("Nie znaleziono plików conn.log.labeled")
    if len({capture for _, capture in jobs}) != len(jobs):
        raise ValueError("Kilka plików przypisanych do tego samego capture'a; usuń duplikaty lub nazwij części Somfy")
    for path, capture in jobs:
        manifest[capture] = convert(path, capture, C.PARQUET_DIR / f"{capture}.parquet", args.limit)
        write_json(C.MANIFEST_PATH, manifest)
        print(f"{capture}: {manifest[capture]}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
