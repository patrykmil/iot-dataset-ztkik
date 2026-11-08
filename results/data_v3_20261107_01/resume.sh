#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export DETECTOR_THREADS=4
export DETECTOR_ARTIFACTS="$PWD/results/data_v3_20261107_01/artifacts"
export TMPDIR="$PWD/results/data_v3_20261107_01/tmp"
test ! -e "$DETECTOR_ARTIFACTS/experiment.json"
.venv/bin/python -u - <<'PY'
import threadcap
import json
import sys
import config as C
from featurize import capture_name, main

manifest = json.loads(C.MANIFEST_PATH.read_text())
inputs = [str(path) for path in sorted(C.RAW_DIR.rglob("*conn.log.labeled")) if capture_name(path) not in manifest]
assert inputs, "Brak pozostałych plików do konwersji"
sys.argv = ["featurize.py", *inputs]
main()
PY
.venv/bin/python -u train.py --device cuda --cap-per-capture 100000
.venv/bin/python -u evaluate.py --device cuda
