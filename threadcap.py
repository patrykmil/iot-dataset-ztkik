"""Cap wątków. MUSI być importowany jako PIERWSZA linia skryptu, przed
pandas/sklearn/lightgbm — te inicjalizują pule wątków już przy imporcie
(tyle co rdzeni), a późniejsze ustawianie env niczego nie cofa.
Torus: ulimit -u=100 przy 32 rdzeniach -> bez tego libgomp EAGAIN.

Nadpisanie bez zmian kodu: DETECTOR_THREADS=4 python train.py
"""

import os

THREADS = int(os.environ.get("DETECTOR_THREADS", 4))
if THREADS < 1:
    raise ValueError("DETECTOR_THREADS musi być >= 1")
for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[name] = str(THREADS)
os.environ["OMP_THREAD_LIMIT"] = str(THREADS)
try:
    import pyarrow as pa

    pa.set_cpu_count(THREADS)
    pa.set_io_thread_count(THREADS)
except ImportError:
    pass  # ścieżki bez pyarrow
