"""Cap wątków. MUSI być importowany jako PIERWSZA linia skryptu, przed
pandas/sklearn/lightgbm — te inicjalizują pule wątków już przy imporcie
(tyle co rdzeni), a późniejsze ustawianie env niczego nie cofa.
Torus: ulimit -u=100 przy 32 rdzeniach -> bez tego libgomp EAGAIN.

Nadpisanie bez zmian kodu: DETECTOR_THREADS=4 python train.py
"""

import os

THREADS = max(1, int(os.environ.get("DETECTOR_THREADS", 8)))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(THREADS))
try:
    import pyarrow as pa

    try:  # nie każdy build ma setter CPU; krytyczny jest io (odczyt parquet)
        pa.set_cpu_thread_count(THREADS)
    except AttributeError:
        pass
    pa.set_io_thread_count(THREADS)
except ImportError:
    pass  # ścieżki bez pyarrow
