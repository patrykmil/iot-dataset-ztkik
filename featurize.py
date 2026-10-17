"""conn.log.labeled (Zeek TSV) -> parquet per capture + manifest.

Użycie:
    python featurize.py --download            # starter wg config.STARTER_CAPTURES
    python featurize.py --download --captures CTU-IoT-Malware-Capture-34-1
    python featurize.py raw/                  # lokalne pliki/katalogi z conn.log*
    python featurize.py --limit 20000         # + limit flowów na capture (smoke)

Format Zeek: linie '#' to nagłówki (#fields z nazwami kolumn),
dane rozdzielone tabulatorem, '-' = brak wartości.
label: 'Benign' -> 0, 'Malicious*' -> 1, reszta (w tym '-') DROP.
"""

import threadcap  # noqa: F401  MUSI być pierwsze
import argparse
import gc
import io
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

import config as C

CHUNK_LINES = 200_000
HREF_DIR = re.compile(r'href="([A-Za-z0-9._-]+/)"')
HREF_FILE = re.compile(r'href="(conn\.log\.labeled)"')


def list_dir(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def exists(url, retries=3):
    import time

    for i in range(retries):
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
            return True
        except urllib.error.HTTPError:
            return False  # 404 = na pewno nie ma, nie retryujemy
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5 * (i + 1))  # chwilowy problem sieciowy -> retry
    return False


def download_capture(name, dest: Path):
    """Zwraca listę (plik, nazwa_capture). Layouty CTU: bro/conn.log.labeled,
    labeled/conn.log.labeled (np. 43-1) albo split na podkatalogi (Somfy-0N)."""
    base = f"{C.BASE_URL}/{name}"
    for sub in ("bro", "labeled"):
        url = f"{base}/{sub}/conn.log.labeled"
        if exists(url):
            return [(fetch(url, dest), name)]
    print(f"{name}: szukam w podkatalogach...")
    out = []
    for subdir in sorted(set(HREF_DIR.findall(list_dir(base + "/")))):
        for sub in ("bro", "labeled"):
            furl = f"{base}/{subdir}{sub}/conn.log.labeled"
            if exists(furl):
                part = f"{name}__{subdir.rstrip('/')}"
                out.append((fetch(furl, dest.parent / (part + ".conn.log.labeled")), part))
                break
    if not out:
        raise SystemExit(f"{name}: nie znaleziono conn.log.labeled (bro/, labeled/ ani podkatalogi).")
    return out


def fetch(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"GET {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def parse_block(lines, fields, sep, capture):
    """Jeden blok linii TSV -> ramka (kolumny keep + label/capture/family)."""
    text = re.sub(r"[ ]{2,}", "\t", "".join(lines))
    df = pd.read_csv(io.StringIO(text), sep=sep, names=fields, na_values=["-", "(empty)"], low_memory=False)
    del text
    # Etykiety case-insensitive: malware 'Malicious*', benign 'benign'/'Benign'.
    lab = df["label"].astype(str).str.lower() if "label" in df else "?"
    y = pd.Series(-1, index=df.index, dtype="int8")
    y[lab.str.startswith("malicious")] = 1
    y[lab == "benign"] = 0
    df = df[y != -1].copy()
    df[C.LABEL] = y[y != -1].astype("int8")
    df[C.CAPTURE] = capture
    df[C.FAMILY] = C.FAMILY_OF.get(capture, "benign" if "Honeypot" in capture else capture)
    df = engineer(df)
    return df[[c for c in C.CATEGORICAL + C.NUMERIC if c in df] + [C.LABEL, C.CAPTURE, C.FAMILY]]


def parse_zeek(path, capture, limit=None, block=500_000):
    """Stream parser TSV: czyta blokami (giganty: 67M flowów nie wejdą na raz)."""
    fields, sep, buf, frames, n = None, "\t", [], [], 0
    stop = limit * 5 if limit else None  # zapas: część wierszy odpadnie
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#separator"):
                sep = line.strip().split(" ", 1)[1].encode().decode("unicode_escape")
            elif line.startswith("#fields"):
                fields = line.strip().split()[1:]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                buf.append(line)
                n += 1
                if len(buf) >= block or (stop and n >= stop):
                    frames.append(parse_block(buf, fields, sep, capture))
                    buf = []
                    gc.collect()
                if stop and n >= stop:
                    break
    assert fields, f"brak nagłówka #fields w {path}"
    if buf:
        frames.append(parse_block(buf, fields, sep, capture))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if limit is not None and len(df) > limit:  # stride: bloki klas
        df = df.iloc[:: len(df) // limit].head(limit).reset_index(drop=True)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    for c in C.NUMERIC:
        if c in ("bytes_per_pkt", "resp_orig_ratio", "bytes_total"):
            continue
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    op = pd.to_numeric(df.get("orig_pkts", 0), errors="coerce").fillna(0)
    ob = pd.to_numeric(df.get("orig_bytes", 0), errors="coerce").fillna(0)
    rb = pd.to_numeric(df.get("resp_bytes", 0), errors="coerce").fillna(0)
    df["bytes_per_pkt"] = (ob / op.replace(0, 1)).astype("float32")
    df["resp_orig_ratio"] = (rb / ob.replace(0, 1)).astype("float32")
    df["bytes_total"] = (ob + rb).astype("float32")
    for c in ("orig_p", "resp_p"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype("int32")
    return df


def convert(path: Path, capture, out_path, limit=None):
    df = parse_zeek(path, capture, limit)
    df.to_parquet(out_path, index=False)
    return len(df), float(df[C.LABEL].mean()) if len(df) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=None, help="pliki conn.log* lub katalogi (rekurencyjnie)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--captures", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    C.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    jobs = []
    if args.download:
        for name in args.captures or C.CAPTURES:  # default: pełne 23 scenariusze
            dest = C.RAW_DIR / name / "conn.log.labeled"
            for path, capture in download_capture(name, dest):
                jobs.append((path, capture))
    for inp in args.inputs or []:
        p = Path(inp)
        files = sorted(p.rglob("conn.log*")) if p.is_dir() else [p]
        for f in files:
            jobs.append((f, f.parent.name if f.parent != p else f.stem))
    for path, capture in jobs:
        out = C.PARQUET_DIR / (capture + ".parquet")
        n, rate = convert(path, capture, out, args.limit)
        manifest[capture] = {"flows": n, "malware_rate": round(rate, 4)}
        print(f"{capture}: flows={n} malware_rate={manifest[capture]['malware_rate']}")
    C.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print("manifest ->", C.MANIFEST_PATH)


if __name__ == "__main__":
    main()
