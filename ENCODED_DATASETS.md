# Encoded Dataset Pipeline

## Übersicht

Die finale Datentabelle mit One-Hot-Encoding der Station-IDs wird nun in `data/processed/` gespeichert, statt nur im RAM während des Trainings erstellt zu werden.

## Dateipfade

Die folgenden Dateien werden von `05_create_encoded_dataset.py` in `data/processed/` erstellt:

- `encoded_train.csv` - Trainingsdaten mit One-Hot-Encoding
- `encoded_validation.csv` - Validierungsdaten mit One-Hot-Encoding
- `encoded_test.csv` - Testdaten mit One-Hot-Encoding
- `encoded_feature_names.csv` - Liste der finalen Feature-Namen nach Encoding

## Pipeline

### Schritt 1: Reduzierter Datensatz erstellen
```bash
python src/scripts/04_build_reduced_feature_dataset.py
```
Erstellt `data/processed/daily_rentals_top20_reduced.csv` mit numerischen Features.

### Schritt 2: Encoded Datensätze erstellen
```bash
python src/scripts/05_create_encoded_dataset.py
```
Erstellt die finalen Datentabellen mit:
- Chronologischem Split (70% Train, 15% Val, 15% Test)
- One-Hot-Encoding der `start_station_id`
- Speichern in `data/processed/`

### Schritt 3: Modelle trainieren
Die Trainingsskripte in `modelling/` laden automatisch die vorkodierten Daten, wenn sie verfügbar sind:

```bash
python modelling/00_dummy_regressor/train_dummy.py
python modelling/01_linear_regression/train_linear_regression.py
# ... etc
```

## Technische Details

### load_encoded_datasets()
Lädt die vorkodierten Trainings-, Validierungs- und Testdaten:

```python
from modelling.common.preprocessing import load_encoded_datasets
from modelling.common.config import (
    ENCODED_TRAIN_PATH,
    ENCODED_VAL_PATH,
    ENCODED_TEST_PATH,
    TARGET_COL,
)

train_df, val_df, test_df = load_encoded_datasets(
    train_path=ENCODED_TRAIN_PATH,
    val_path=ENCODED_VAL_PATH,
    test_path=ENCODED_TEST_PATH,
    target_col=TARGET_COL,
)
```

### prepare_encoded_feature_matrices_for_model()
Bereitet die Daten für das Modelltraining vor (z.B. mit optionalem Scaling):

```python
from modelling.common.preprocessing import prepare_encoded_feature_matrices_for_model

scaler, feature_names, X_train, X_val, X_test = (
    prepare_encoded_feature_matrices_for_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=TARGET_COL,
        feature_names=None,  # Automatisch ermitteln
        scale_numeric=False,  # True für lineare Modelle
    )
)
```

## Rückwärtskompatibilität

Die Trainingsskripte unterstützen weiterhin beide Methoden:
1. **Mit vorkodiertem Datensatz** (schneller, empfohlen)
2. **Mit On-The-Fly-Encoding** (falls vorkodierte Daten nicht vorhanden)

Das System erkennt automatisch, welche Methode verwendet werden soll.

## Wichtige Notizen

- Die Station-IDs werden als **Strings** behandelt, bevor sie One-Hot-encodiert werden
- Das `drop="first"` Parameter verhindert Perfect Multicollinearity
- Der Scaler wird nur auf Trainingsdaten trainiert (methodologisch sauber)
- Zeitstempel und Station-IDs werden für die Analyse gespeichert, sind aber keine Model Features

## Aktualisierte Config

Die `modelling/common/config.py` enthält jetzt:

```python
# Original reduced dataset (before encoding)
DATA_PATH = DATA_DIR / "daily_rentals_top20_reduced.csv"

# Encoded datasets (after one-hot encoding of station IDs)
ENCODED_TRAIN_PATH = DATA_DIR / "encoded_train.csv"
ENCODED_VAL_PATH = DATA_DIR / "encoded_validation.csv"
ENCODED_TEST_PATH = DATA_DIR / "encoded_test.csv"
ENCODED_FEATURES_PATH = DATA_DIR / "encoded_feature_names.csv"
```
