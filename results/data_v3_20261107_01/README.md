# Zmiany danych — eksperyment rozwojowy v3

Protokół ustalony przed treningiem, 2026-11-07. Nie dobieramy podziału ani progu
po otrzymaniu wyników tego eksperymentu. Stare `artifacts/` pozostaje bez zmian.

## Co zmieniono

- `detailed-label` zachowane w Parquet i liczebności rodzajów ataku w manifeście.
  Szczegółowa etykieta nigdy nie trafia do cech modelu.
- Trening: osobne limity do 50 tys. benign i 50 tys. malicious na capture.
  Dostępne rzadkie benign nie są usuwane po to, by zachować proporcje 99% malware.
  Limit malicious dzielony między szczegółowe etykiety, z oddawaniem
  niewykorzystanego limitu pozostałym rodzajom. Nie powielamy wierszy.
- Walidacja: `35-1` (DDoS), `48-1` (skanowanie), `44-1` (C&C/pobieranie plików)
  i czysty Amazon Echo `5-1`. Próbkowanie walidacji zachowuje proporcje klas
  wewnątrz każdego capture. Mirai pozostaje w treningu m.in. w `43-1`, `49-1`, `52-1`.
- Pełne testy pozostały bez zmian: znane rodziny `34-1` i Philips HUE;
  nieznane Torii `20-1`/`21-1`, Gagfyt `60-1` i Somfy.
- Dodatkowy wariant co najmniej 99% benign: wszystkie benign i jednorodna losowa
  próbka malicious, do `n_benign // 99`, bez zwracania. Te same wiersze dla modeli,
  bez zmiany progów. Pełne testy i wyniki per rodzina są nadal raportowane.
- AP raportowana z punktem odniesienia `average_precision_baseline`.

## Co sprawdzono przed uruchomieniem

41 testów przeszło na serwerze. Obejmują zachowanie rzadkich klas/zachowań,
reprodukowalność niezależnie od granic batchy, brak szczegółowej etykiety w cechach,
zapis i odczyt modeli oraz wspólny wariant 99:1 w pełnym syntetycznym pipeline CPU.
Log: `logs/tests_data_v3_20261107_02.log` (od katalogu głównego projektu).

## Uruchomienie

`run.sh` konwertuje istniejące raw do osobnego Parquet, trenuje kolejno trzy modele
i wykonuje ewaluację. Wyniki w `artifacts/` wewnątrz tego katalogu eksperymentu.
Kod sprzed zmian zachowano na serwerze w `original_code/`.
Nie zmieniano zależności ani żadnych ustawień systemowych.

Pierwsza konwersja zatrzymała się na `34-1`, który w oryginalnym nagłówku
używa `det_label`, a nie `detailed-label`. Dodano mapowanie tej nazwy oraz test
regresyjny. Sprawdzono nagłówki wszystkich 23 lokalnych plików raw: pozostałe
używają `detailed-label`. `resume.sh` konwertuje tylko pliki, których nie ma
w nowym manifeście, zachowując wcześniej ukończone Parquety. Nie zmieniono
wartości etykiet ani protokołu eksperymentu.

## Wynik ukończonego eksperymentu

Id: `a6eefe68d97e45dfbac9855f450e0c2f`, status `complete`.
Wyniki i modele: `artifacts/` wewnątrz tego katalogu (również pobrane lokalnie,
bez kopiowania Parquet). Log zakończonego pipeline'u:
`logs/data_v3_20261107_01_resume.log` od katalogu głównego projektu.

Próbka treningowa ma **280 582 benign i 558 228 malicious** (66,55% malicious).
W poprzednim eksperymencie było 90 629 benign i 1 124 437 malicious (92,54%).
Zachowano wszystkie obecne w źródłach szczegółowe etykiety każdego capture'u
treningowego, w tym 2 C&C-Mirai, 3 Okiru-Attack i 5 PartOfAHorizontalPortScan-Attack.
Ogółem w treningu znalazło się m.in. 51 991 DDoS i 33 578 C&C-HeartBeat.
Liczebności pełnych 23 capture'ów i etykiety 0/1 są zgodne z poprzednią konwersją.

| Model | AUC known | Recall known | AUC unknown | Recall unknown |
|---|---:|---:|---:|---:|
| LightGBM | 0,96075 | 0,5560% | 0,92774 | 0% |
| Logistic | 0,31120 | 0,1131% | 0,21886 | 0,000782% |
| MLP | 0,32579 | 0,3016% | 0,29242 | 0,000782% |

Przy zapisanych progach LightGBM wykrywa 118 z 21 222 malicious na known
i 0 z 3 578 582 na unknown. Logistic wykrywa odpowiednio 24 i 28, MLP 64 i 28.
**Zmiany nie dały skutecznego detektora operacyjnego.** Dobre rankingowe AUC
LightGBM nie oznacza użytecznych alarmów przy progu walidacyjnym.

Wariant known benign-majority: 2 375 benign i 23 malicious (0,9591% malicious).
Wariant unknown: 9 071 benign i 91 malicious (0,9932%). Wszystkie modele mają
recall 0 na tych małych próbkach. AP LightGBM wynosi 0,15873 i 0,07589, przy
baseline 0,00959 i 0,00993. AP logistic/MLP na unknown (0,00644 / 0,00702)
jest poniżej baseline. Próbki są identyczne dla modeli, co potwierdzają fingerprinty.

### Dlaczego lepsza zbiorcza walidacja nie wystarczyła

Zbiorcze AUC walidacyjne wynosi około 0,83, ale po rozbiciu na capture:

- `48-1` (skanowanie): wszystkie modele wykrywają około 99,88% malicious;
- `35-1` (DDoS): LightGBM recall 0%, logistic/MLP około 2,95%;
- `44-1` (C&C): LightGBM recall 0%, logistic około 46%, MLP 50%, tylko 26 malicious;
- czysty Amazon Echo: FPR LightGBM 0%, logistic około 14,99%, MLP 3,93%.

Skanowanie stanowi około 82,67% malicious w nowej próbce walidacji i dominuje
wynik zbiorczy. Benign z `35-1` dominuje globalną kontrolę FPR, maskując duże
FPR na Amazon Echo. Nie należy interpretować wzrostu zbiorczej walidacji jako
poprawy wykrywania każdego zachowania. Dodatkowy raport:
`validation_per_capture.json`, odtworzono dokładnie zapisane próbki; macierze
pomyłek sumują się do metryk oryginalnej walidacji. Predykcja diagnostyczna na CPU.

Na czystym Philips HUE w teście logistic daje 63 fałszywe alarmy z 452 flowów
(13,94%), MLP 62 (13,72%), LightGBM 0. Na Somfy logistic daje 66 z 130 (50,77%).
Sama przewaga benign w próbie testowej nie usuwa tej różnicy rozkładów.

Stare 16 plików modeli/raportów `artifacts/` sprawdzono SHA-256: bez zmian.
Nie zmieniano już splitu ani progów po obejrzeniu tych wyników. Dalsza praca
wymaga lepszego pokrycia niezależnego benign i kontroli FPR per urządzenie/capture,
a ocena końcowa nowego nietkniętego testu.

## Ograniczenia

Nie pozyskano nowych sesji czystych urządzeń: nie wskazano źródła takich danych.
Trening nadal korzysta z benign ze scenariuszy infekcji; zachowanie większej liczby
wierszy nie zastępuje nowych normalnych zachowań. Próbkowanie nie gwarantuje
przewagi benign w całym treningu.

Obecne testy były wcześniej oglądane i użyte do diagnostyki. Wyniki mają rolę
**rozwojową**, nie są niezależną oceną końcową. Wariant 99:1 to kontrola proporcji,
nie realistyczna symulacja sieci. Przy 9 071 benign w teście nieznanych rodzin
otrzymamy tylko 91 malicious; złośliwe flowy Torii mogą nie trafić do tej próbki. Jego pełny wynik
pozostaje w raporcie. Próg dobierany wyłącznie na walidacji, FPR na teście nie jest
gwarantowany. Nie odwracamy ocen ani nie zmieniamy splitu w poszukiwaniu lepszego wyniku.
