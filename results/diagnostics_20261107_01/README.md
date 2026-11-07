# Diagnostyka eksperymentu IoT-23

Eksperyment źródłowy: `15eb5ea381d74d8cb0a5a73e82ea170e`.
Diagnostyka wykonana na torusie 2026-11-07, tylko w katalogu projektu.
Bez instalowania zależności, zmian ustawień systemu i nadpisywania `artifacts/`.
Modele, raporty i kod źródłowy sprzed zmian zachowano na serwerze w `original/`.

## Oryginalny eksperyment

| Zbiór | Benign | Malicious | Malicious % |
|---|---:|---:|---:|
| Train (próbka) | 90 629 | 1 124 437 | 92,54 |
| Valid (próbka) | 80 457 | 20 917 | 20,63 |
| Test known (pełny) | 2 375 | 21 222 | 89,94 |
| Test unknown (pełny) | 9 071 | 3 578 582 | 99,75 |

Te proporcje są zgodne z laboratoryjnymi scenariuszami infekcji IoT-23,
ale nie reprezentują codziennego ruchu całej sieci.
Źródło i oryginalne liczebności: https://www.stratosphereips.org/datasets-iot23.

### Dwa niezależne problemy

1. **CUDA i wagi klas LightGBM.** Odtworzono dokładnie oryginalne próbki
   train/valid, sprawdzając fingerprinty i liczebności obu klas. Oryginalna
   konfiguracja CPU tworzy 3 drzewa (115 różnych ocen na walidacji); CUDA
   tworzy tylko jedno drzewo z jednym liściem, ocena zawsze 0,9254122821.
   AUC walidacyjne: CPU 0,122707, CUDA 0,5. CPU również nie generalizuje dobrze;
   sam brak podziałów jest jednak specyficzny dla starej ścieżki CUDA.

   W źródle LightGBM 4.7.0 kernel `GetGradientsKernel_BinaryLogloss` używa
   `cuda_label_weights[label]`, gdzie `label` dla benign wynosi -1.
   Bufor ma tylko dwa elementy (indeksy 0 i 1), więc jest to odczyt poza buforem.
   Ścieżkę uruchamia m.in. użyty `scale_pos_weight=0,0805994467`.
   Źródła:
   - https://github.com/microsoft/LightGBM/blob/v4.7.0/src/objective/cuda/cuda_binary_objective.cu
   - https://github.com/microsoft/LightGBM/blob/v4.7.0/src/objective/cuda/cuda_binary_objective.cpp

   Obejście w projekcie: te same wagi malicious/benign przekazywane jako
   wagi wierszy `Dataset`, bez `scale_pos_weight`, dla obu backendów.
   Wagi wierszy zmieniają też inicjalizację modelu na średnią ważoną;
   wyniki sprzed i po zmianie nie muszą być identyczne nawet na CPU.
   Włączono ostrzeżenia i odrzucanie modeli bez żadnego podziału przed zapisem.

2. **Różne zachowania w treningu i walidacji.** W treningu malicious to
   głównie `history=S` (1 118 587 / 1 124 437). W walidacji aż 20 909 / 20 917
   malicious (99,96%) ma `history=I` lub `DTT`, nieobecne w treningu.
   Medianą `duration` malicious jest 0 w train i 3,4655 s w valid.
   Logistic: AUC train 0,9372, valid 0,2381. MLP: train 0,9555, valid 0,1044.
   To dowód silnej różnicy rozkładów, a nie potwierdzenie odwróconych etykiet.

### Interpretacja metryk

- AP nierozróżniającego modelu na test unknown = 0,997472, dokładnie udział
  malicious. AP blisko 1 nie oznacza tu dobrego detektora.
- Przy progach z walidacji logistic wykrywa 35, MLP 33 z 3 578 582 malicious.
- Logistic daje 95 fałszywych alarmów na 452 czystych flowach Philips HUE (21%).
- Torii ma tylko 30 malicious. Wynik nie dowodzi generalizacji na całą rodzinę.
- Gagfyt dominuje liczebnie test unknown; należy pokazywać wyniki per capture
  i rodzinę, a nie wyłącznie jeden wynik zbiorczy.

## Pliki i odtwarzanie

- `diagnostics.json`: rozkłady i porównanie oryginalnej ścieżki CPU/CUDA.
- `logs/diagnostics_20261107_01.log` (od katalogu głównego repo): log diagnostyki.
- Na serwerze `samples.joblib`: oryginalne próbki, bez ponownego skanowania Parquet.
- `verify_fix.py`: weryfikacja poprawki na tych próbkach i pełnych testach.
  Uruchamiana z katalogu głównego projektu; tworzy nowe `fixed_lightgbm_*`.
  Ponowne uruchomienie wymaga innego katalogu wynikowego, aby nie nadpisać modeli.

## Wynik weryfikacji poprawki

| LightGBM po poprawce | CPU | CUDA |
|---|---:|---:|
| Liczba drzew | 49 | 70 |
| Różne oceny na walidacji | 542 | 946 |
| AUC train | 0,99937 | 0,99946 |
| AUC valid | 0,01473 | 0,01402 |
| AUC test known | 0,94206 | 0,94298 |
| AUC test unknown | 0,91131 | 0,91391 |
| Recall test known (próg valid) | 0,4995% | 0% |
| Recall test unknown (próg valid) | 0% | 0% |

Obejście usuwa degenerację CUDA: model tworzy podziały i jego wyniki są zbliżone
do CPU. Nie rozwiązuje różnicy zachowań między treningiem i walidacją.
Próg CUDA wybrany na walidacji to ponownie wariant „bez alarmów” (ponad 1),
tym razem mimo niestałego modelu. Nie oznacza to błędu doboru progu:
żaden próg wykrywający malicious nie spełnił ograniczenia FPR na tej walidacji.
Nie odwracano ocen, nie dobierano progu na teście i nie zmieniano podziału.

Dokładne wyniki: `fixed_comparison.json` oraz
`logs/verify_fix_20261107_01.log` (od katalogu głównego repo).
Na serwerze modele diagnostyczne są w `fixed_lightgbm_cpu/` i `fixed_lightgbm_cuda/`.
Sprawdzono zgodność predykcji po zapisaniu i ponownym wczytaniu każdego modelu.

Końcowe sprawdzenie: **33 testy przeszły**, w tym test pełnego pipeline'u CPU,
wag wierszy, odrzucania stałego modelu i bezpiecznej lokalizacji diagnostyki.
Log: `logs/tests_20261107_02.log`. Dodatkowo porównano SHA-256 wszystkich
16 oryginalnych plików modeli/raportów z kopią: `artifacts/` pozostało bez zmian.
Parquet sprawdzano przez zapisane sygnatury (rozmiar, mtime, liczba wierszy),
nie pełny hash. Nie uruchamiano od nowa treningu logistic/MLP na rzeczywistych danych.

## Następny eksperyment — propozycja, nie wykonano

Zachować bieżący podział jako benchmark i dodać osobno test z przewagą benign,
np. 99% benign / 1% malicious. Proporcję ustalić przed nowym eksperymentem,
nie dobierać jej pod oczekiwany wynik. Ograniczać malicious bez powielania benign,
z zachowaniem rozłączności capture'ów oraz raportowaniem liczebności per rodzina.
Przy 9 071 benign w obecnym teście unknown wariant 99:1 pozwala na około
91 malicious — będzie mały i nie zastąpi pełnego testu ataków.

Trening nie musi mieć proporcji rzeczywistej sieci. Ważniejsze jest pokrycie
normalnych zachowań i rodzajów ataków; dopasowanie proporcji nie tworzy nowych
zachowań benign. Rozszerzenie źródeł benign wymaga zgodnego schematu Zeek oraz
kontroli różnic źródeł danych. Nie przenosić obecnych capture'ów testowych do
treningu po obejrzeniu ich wyników bez wydzielenia nowego, nietkniętego testu.
