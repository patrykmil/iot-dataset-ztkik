# iot23-detector

Detektor malware w ruchu sieciowym IoT na bazie **IoT23** (Stratosphere Lab, 20 capture'ów malware + 3 benign, etykiety per-flow w `conn.log.labeled`).
Model: **LightGBM** (baseline: Random Forest). Cel: ROC-AUC > 0.94 przy progu strojonym pod niski FP.

Cap wątków (`threadcap.py`, torus ma `ulimit -u=100`).

## Start (torus)

```bash
cd iot23-detector
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python featurize.py --download                                    # pełne 23 scenariusze (~9 GB)
.venv/bin/python train.py --baseline-only                                   # szybki dowód RF
.venv/bin/python -u train.py > train.log 2>&1 &                             # pełny LightGBM (nohup)
.venv/bin/python -u train.py --cap-per-capture 1000000 > train.log 2>&1 &   # pełna skala: stratified cap 1M flowów/capture (~20M wierszy)
.venv/bin/python evaluate.py                                                # metryki + rozbicie per-capture
```

lub

pełny pipeline (nohup przy wyrzuceniu z torusa)

```bash
nohup bash -c '.venv/bin/python -u featurize.py --download && .venv/bin/python -u train.py --cap-per-capture 1000000 && .venv/bin/python -u evaluate.py' > pipeline.log 2>&1 &
```

Wyniki: `artifacts/model.txt`, `threshold.json`, `features.json`, `metrics.json`, `roc.png`.

## Dwa reżimy wyniku (oba w `metrics.json` → `per_capture`)

- **(a) held-out capture'y tych samych rodzin** — tu celujemy w >0.94.
- **(b) held-out rodzina** (`CTU-IoT-Malware-Capture-20-1` = Torii, spoza train) — uczciwa generalizacja; spodziewanie niższa, do raportu w pracy.

`TEST_CAPTURES` w `config.py` definiuje reżim (b). Dołożenie capture'ów:
`featurize.py --download --captures <nazwy>` (linki per scenariusz na stratosphereips.org/datasets-iot23).

## Założenia

- `label`: `Benign`→0, `Malicious*`→1, brak/`-`→DROP (raportuje `manifest.json`).
- `detailed-label` i IP odrzucane celowo (leak adresów labu, leak etykiety).
- Cytowanie datasetu: Garcia et al. 2020, Zenodo doi:10.5281/zenodo.4743746.
