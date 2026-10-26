# iot23-detector

Porównanie trzech klasyfikatorów ruchu IoT23: **LightGBM**, **regresji logistycznej
trenowanej SGD** i **sieci MLP (64 → 32 → 1)**. Klasy: Benign = 0, Malicious* = 1.

| Model | Metoda | Trening | Predykcja |
|---|---|---|---|
| LightGBM | boosting drzew, natywne kategorie | CUDA na NVIDIA L4 lub jawnie CPU | CPU |
| logistic | liniowy model, `SGDClassifier(loss="log_loss")` | CPU | CPU |
| mlp | nieliniowa sieć PyTorch, minibatch 4096 | CUDA na NVIDIA L4 lub jawnie CPU | CUDA/CPU |

Modele trenują **kolejno**, na dokładnie tych samych wierszach. CUDA jest domyślna;
błąd GPU zatrzymuje trening, bez cichego fallbacku na CPU. Nie ma automatycznego
strojenia hiperparametrów. Early stopping każdego modelu korzysta z ROC-AUC walidacji.

## Środowisko torusa

Kod przygotowany lokalnie można przesłać tą komendą (bez danych, venv i wyników,
bez usuwania czegokolwiek z serwera). Uruchom ją **lokalnie z katalogu repozytorium**:

```bash
rsync -av --exclude='.git/' --exclude='.venv/' --exclude='raw/' \
  --exclude='artifacts/' --exclude='__pycache__/' ./ \
  miler.patryk@torus.uck.pk.edu.pl:/home/inf1s-22z/miler.patryk/ztk/projekt/iot23-detector/
```

Pozostałe komendy wykonuje się na serwerze:

```bash
ssh miler.patryk@torus.uck.pk.edu.pl
cd /home/inf1s-22z/miler.patryk/ztk/projekt/iot23-detector
```

Znane wersje: Python **3.13.5**, CUDA Toolkit **13.4**, GPU **NVIDIA L4**.
`Failed to initialize NVML: Driver/library version mismatch` wymaga rozwiązania przez
administrację. **Nie instaluj sterowników ani nie zmieniaj konfiguracji serwera w ramach
tego projektu. Poniższe komendy są instrukcją do późniejszego wykonania, nie zostały uruchomione.**

### Instalacja zależności

Użyj istniejącego `.venv`; jeśli go nie ma:

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

PyTorch jest zależnością opcjonalną, potrzebną tylko do MLP. Wybierz aktualny pakiet
Linux/Pip/Python/CUDA obsługujący Python 3.13 z [oficjalnego selektora](https://pytorch.org/get-started/locally/).
Użyj `.venv/bin/python -m pip` zamiast systemowego `pip`. Sam `torch` wystarczy;
`torchvision` i `torchaudio` nie są potrzebne.

Przykładowa kompletna komenda używająca gotowego runtime CUDA 12.8, obsługującego L4:

```bash
.venv/bin/python -m pip install 'torch>=2.7' --index-url https://download.pytorch.org/whl/cu128
```

Wersja CUDA w gotowym pakiecie PyTorch **nie musi wynosić 13.4**. Pakiet zawiera własny
runtime, ale wymaga zgodnego sterownika. Możesz wybrać inny aktualny wariant z selektora;
nie kompilujemy PyTorch z lokalnym Toolkit. Ta komenda także pozostaje nieprzetestowana
na torusie do czasu naprawy sterownika.

### LightGBM CUDA

`device_type="gpu"` oznacza OpenCL, a `device_type="cuda"` osobny backend CUDA.
Używamy **CUDA**, z natywnymi cechami kategorycznymi i `use_quantized_grad=False`.
Kwantyzowane gradienty wraz z kategoriami nie są obsługiwane przez ten backend.

Zwykły pakiet LightGBM nie gwarantuje obsługi CUDA. Jeśli trzeba go zbudować, poniższa
komenda instaluje go **wyłącznie w venv**, używając dostępnego CUDA Toolkit i kompilatora C++:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 .venv/bin/python -m pip install \
  --force-reinstall --no-deps --no-binary=lightgbm 'lightgbm>=4.6' \
  --config-settings=cmake.define.USE_CUDA=ON \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

Wymaga zgodnych CMake, kompilatora C++ i CUDA; build isolation może pobrać narzędzia
budowania. L4 ma compute capability 8.9. **Zgodność konkretnego wydania LightGBM
z CUDA 13.4 pozostaje do sprawdzenia**. Jeśli kompilacja się nie powiedzie, zachowaj log
i skonsultuj zgodny zestaw z administracją, zamiast zmieniać systemowy Toolkit.

## Przygotowanie danych bez pobierania

**Dotychczasowy parser pomijał porty** (`id.orig_p`/`id.resp_p`). Przed nowym treningiem
ponownie wygeneruj Parquet z istniejących plików raw na serwerze:

```bash
DETECTOR_THREADS=4 .venv/bin/python -u featurize.py raw/
```

Jeśli dane są gdzie indziej, zamiast `raw/` podaj ich katalog. Obsługiwane układy:

```text
raw/CTU-IoT-Malware-Capture-34-1/conn.log.labeled
raw/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled
raw/CTU-Honeypot-Capture-7-1/Somfy-01/bro/conn.log.labeled
raw/CTU-Honeypot-Capture-7-1/CTU-Honeypot-Capture-7-1__Somfy-01.conn.log.labeled
```

Konwersja i zapis odbywają się partiami. Błąd parsera pozostawia poprzedni Parquet.
Manifest jest aktualizowany po każdym capture, zawiera liczebności obu klas i liczbę
wierszy odrzuconych z powodu etykiety. Nieznane etykiety są pomijane, brak wymaganych
kolumn, niepoprawne liczby i uszkodzone wiersze zatrzymują konwersję.
Braki wartości liczbowych są zastępowane zerem, kategorii — `-`, jak w poprzednim pipeline.
Liczby są zapisywane jako float32, aby ograniczyć pamięć treningu.

`--limit` ogranicza **prefiks wierszy źródłowych**, tylko do sprawdzenia parsera.
Trening odrzuca takie pliki, aby nie raportować wyników na niereprezentatywnym prefiksie.
Po tej próbie powtórz konwersję bez limitu.

Tylko jeśli faktycznie brakuje plików, opcjonalne pobieranie wybranych scenariuszy:

```bash
DETECTOR_THREADS=4 .venv/bin/python -u featurize.py --download \
  --captures CTU-IoT-Malware-Capture-20-1 CTU-IoT-Malware-Capture-21-1
```

Bez `--captures` pobierane są wszystkie 23 scenariusze. Istniejące niepuste pliki raw
nie są pobierane ponownie. Pliki pozostawione przez starą, przerwaną operację pobierania
mogą być niekompletne — sprawdź je przed konwersją.

## Podział i uczciwe porównanie

Jawny podział w `config.py`, niezależny od wyników i kolejności manifestu:

- **Walidacja:** Mirai `35-1` + benign Amazon Echo `5-1`.
- **Test znanych rodzin:** Mirai `34-1` + benign Philips HUE `4-1`.
- **Test nieznanych rodzin:** oba Torii (`20-1`, `21-1`), Gagfyt `60-1` i wszystkie
  dostępne części benign Somfy `7-1`.
- **Trening:** pozostałe dostępne scenariusze. Musi zawierać Mirai i obie klasy.

Brak wymaganych scenariuszy zatrzymuje eksperyment; nie ma losowego podziału po flowach.
Części Somfy należą do jednej grupy. Rodziny testu generalizacji nie mogą trafić do
treningu ani walidacji. Przed raportowaniem wyników zadbaj o kompletność całego datasetu;
program dopuszcza brak innych, niewymaganych capture'ów treningowych i zapisuje ich listę.

Train i valid są próbkowane stratyfikacyjnie do domyślnie **100 tys. flowów na capture**,
z zachowaniem co najmniej jednego wiersza każdej obecnej klasy. Próbkowanie odczytuje
cały plik partiami, utrzymując w pamięci ograniczony rezerwuar. Nadal oznacza to pełny
skan dużego datasetu i może długo trwać. Test obejmuje **wszystkie flowy** odłożonych capture'ów.

Preprocessing jest dopasowany tylko na treningu:
- LightGBM: natywne kategorie, nowe wartości traktowane jako brak.
- Logistic i MLP: ten sam one-hot (do 256 kategorii/cechę; rzadkie łączone),
  `log1p` i skalowanie liczników, osobne skalowanie portów. Macierz pozostaje rzadka;
  dla MLP zagęszczany jest tylko aktualny minibatch.
- Wszystkie modele używają tej samej wagi klasy pozytywnej `n_benign / n_malicious`.

IP, czas, UID i szczegółowe etykiety nie są cechami. Jeden stały podział nie jest
cross-validation; wynik generalizacji dotyczy tych konkretnych rodzin/scenariuszy,
nie gwarantuje jakości dla wszystkich urządzeń i nowych rodzin malware.

## Trening i ewaluacja — po naprawie sterownika

Cały domyślny pipeline (pobranie danych w razie pustego raw → konwersja → trening trzech modeli → ewaluacja):

```bash
nohup bash pipeline.sh > pipeline.log 2>&1 &
```

Podgląd postępu:

```bash
tail -f pipeline.log
```

`pipeline.sh` używa istniejącego `.venv`, domyślnego limitu 100 tys. flowów na capture
i CUDA dla LightGBM/MLP. Nie instaluje zależności. Jeśli `raw/` nie istnieje albo nie
zawiera żadnego niepustego pliku `*conn.log.labeled` (także w podkatalogach), uruchamia
`featurize.py --download`, pobierając wszystkie 23 scenariusze i konwertując je do
Parquet. Wymaga to dostępu do internetu oraz miejsca na raw i Parquet. Jeśli dane
już istnieją, używa ich bez pobierania; częściowego datasetu nie uzupełnia automatycznie.
Do jego uzupełnienia użyj opisanego wyżej `featurize.py --download`. Zatrzymuje
się przy pierwszym błędzie. Działa również uruchomiony przez pełną ścieżkę z innego
katalogu. Domyślnie używa `raw/` i 4 wątków; istniejące `DETECTOR_RAW` oraz
`DETECTOR_THREADS` są respektowane. Nowy trening nadpisuje poprzedni eksperyment.

Mniejszy eksperyment na początek:

```bash
DETECTOR_THREADS=4 .venv/bin/python -u train.py \
  --models lightgbm logistic mlp --device cuda --cap-per-capture 100000

DETECTOR_THREADS=4 .venv/bin/python -u evaluate.py --device cuda
```

Po ocenie pamięci i czasu można zwiększyć limit treningu/walidacji do miliona:

```bash
DETECTOR_THREADS=4 .venv/bin/python -u train.py --device cuda --cap-per-capture 1000000
```

Przy rozłączeniu SSH:

```bash
nohup env DETECTOR_THREADS=4 .venv/bin/python -u train.py \
  --device cuda --cap-per-capture 100000 > train.log 2>&1 &
```

Ewaluację uruchom dopiero po zakończeniu treningu. Jawny CPU jest dostępny przez
`--device cpu` w obu skryptach. Do treningu tylko regresji nie potrzeba PyTorch:

```bash
DETECTOR_THREADS=4 .venv/bin/python train.py --models logistic --device cpu
DETECTOR_THREADS=4 .venv/bin/python evaluate.py --device cpu
```

`threadcap.py` ustawia limity BLAS/OpenMP/NumExpr/Arrow przed importami bibliotek.
PyTorch ma dodatkowo jeden wątek inter-op, brak procesów DataLoader. Treningi nie są
równoległe. Limit 4 to zachowawczy punkt startowy, nie gwarancja zmieszczenia się w
`ulimit -u=100`, który dotyczy też innych procesów/wątków tego użytkownika.
Nie uruchamiaj kilku eksperymentów jednocześnie w tym samym katalogu artifacts.

## Wyniki

```text
artifacts/experiment.json          # podział, seed, liczebności, wersje, fingerprint próbki
artifacts/preprocessor.joblib      # współdzielony preprocessing logistic/MLP
artifacts/lightgbm/model.txt
artifacts/logistic/model.joblib
artifacts/mlp/model.pt
artifacts/<model>/metadata.json    # backend, czas treningu, parametry, próg i walidacja
artifacts/<model>/metrics.json     # test znanych i nieznanych rodzin
artifacts/comparison.csv
artifacts/comparison.json
artifacts/test_known_curves.png
artifacts/test_unknown_curves.png
```

Nowy trening **nadpisuje** modele i unieważnia stare porównanie. Przerwany trening
nie pozwala uruchomić ewaluacji częściowo nadpisanych modeli. Stare pliki pojedynczego
detektora (`artifacts/model.txt`, `threshold.json`, `features.json`, `metrics.json`, `roc.png`)
nie są już używane. Pliki `joblib` ładuj wyłącznie z zaufanego źródła.

Próg każdego modelu jest wybierany **wyłącznie na walidacji**, maksymalizując recall przy
FPR ≤ 1%, z uwzględnieniem remisów. Gdy żaden alarm nie spełnia limitu, wybierany jest
próg ponad 1. Limit FPR nie jest gwarantowany na teście.

Raport zawiera ROC-AUC, **Average Precision** (nie trapezową PR-AUC), precision, recall,
F1, FPR i macierz `[[TN, FP], [FN, TP]]` przy zapisanym progu, a także wyniki per capture,
per rodzina oraz średnie FPR/recall po capture'ach. AUC/AP dla grup jednoklasowych są
`null`, ale ich FPR lub recall nadal są mierzone. `oracle_tpr_at_fpr_*` to diagnostyka
krzywej testowej, **nie** wynik operacyjny progu ustalonego na walidacji.

Czas treningu obejmuje przygotowanie danych specyficzne dla modelu i zapis modelu,
bez wspólnego próbkowania. Czas wspólnego preprocessingu jest osobno w experiment.json.
Przepustowość predykcji obejmuje transformację cech i transfery MLP CPU↔GPU, bez
odczytu Parquet i obliczania metryk. Wyniki nie są zaokrąglane przed zapisaniem.
Dokładne ROC/AP wymagają przechowania etykiet i wyników testowych w RAM; same cechy
są odczytywane partiami. Kontrola danych używa rozmiaru, mtime i liczby wierszy,
nie pełnego kryptograficznego hasha wielogigabajtowych Parquetów.

## Testy — komendy do późniejszego wykonania

```bash
DETECTOR_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

Testy obejmują próg z remisami, wyciek rodzin/scenariuszy, rzadkie klasy w próbkowaniu,
porty i parser, bezpieczny zapis, aktualizację manifestu, nieznane kategorie i odczyt
modeli. Mały test całego pipeline trenuje trzy modele **na syntetycznych danych CPU**,
jeśli LightGBM i PyTorch są zainstalowane; w przeciwnym razie jest pomijany.
Nie sprawdza działania CUDA ani kompatybilności Toolkit 13.4.

Dataset: Garcia et al. 2020, Zenodo DOI: [10.5281/zenodo.4743746](https://doi.org/10.5281/zenodo.4743746).
