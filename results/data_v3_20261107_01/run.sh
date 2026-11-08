#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export DETECTOR_THREADS=4
export DETECTOR_ARTIFACTS="$PWD/results/data_v3_20261107_01/artifacts"
export TMPDIR="$PWD/results/data_v3_20261107_01/tmp"
test ! -e "$DETECTOR_ARTIFACTS/experiment.json"
.venv/bin/python -u featurize.py raw/
.venv/bin/python -u train.py --device cuda --cap-per-capture 100000
.venv/bin/python -u evaluate.py --device cuda
