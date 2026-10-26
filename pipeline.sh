#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export DETECTOR_THREADS="${DETECTOR_THREADS:-4}"
export DETECTOR_RAW="${DETECTOR_RAW:-raw}"

raw_file=''
if [[ -d "$DETECTOR_RAW" ]]; then
    raw_file="$(find -H "$DETECTOR_RAW" -type f -name '*conn.log.labeled' -size +0c -print -quit)"
elif [[ -e "$DETECTOR_RAW" ]]; then
    echo "Błąd: DETECTOR_RAW nie jest katalogiem: $DETECTOR_RAW" >&2
    exit 1
fi

if [[ -n "$raw_file" ]]; then
    # shortcut: częściowy raw nie jest uzupełniany; użyj featurize.py --download przy brakujących scenariuszach.
    echo '[1/3] Konwersja istniejących danych raw do Parquet'
    .venv/bin/python -u featurize.py "$DETECTOR_RAW"
else
    echo '[1/3] Brak danych raw: pobieranie 23 scenariuszy IoT23 i konwersja do Parquet'
    .venv/bin/python -u featurize.py --download
fi

echo '[2/3] Domyślny trening LightGBM, logistic i MLP'
.venv/bin/python -u train.py

echo '[3/3] Ewaluacja i raport porównawczy'
.venv/bin/python -u evaluate.py

echo 'Pipeline zakończony. Wyniki: artifacts/comparison.csv i artifacts/comparison.json'
