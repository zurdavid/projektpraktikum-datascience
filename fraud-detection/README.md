# Fraud Detection

Das Projekt enthält den Code für Phase 3 des Projekts (Modellentwicklung).


## Installation

Idealerweise wird der Package-Manager [uv](https://docs.astral.sh/uv) verwendet, um die Abhängigkeiten zu installieren. Dann reicht es aus, den folgenden Befehl auszuführen:

```bash
uv sync
```

Es ist auch möglich, die Abhängigkeiten manuell zu installieren. Dazu muss die Datei `requirements.txt` im root-Ordner verwendet werden:

```bash
pip install -r requirements.txt
```

Um die Notebooks auszuführen, muss das Projekt als package installiert sein.

```bash
uv pip install -e .
```

## Aufbau

`models.md` enthält eine Auflistung der getestenen Modelle mit Hyperparametern.

### src

Im Ordner `src` befindet sich der Quellcode des Projekts. Das Projekt enthält hilfreiche Klassen und Funktionen, für das Training, die Evaluation und den Vergleich von Modellen. Außerdem sind dort die Datenvorverarbeitung und die Encoder-Klassen zu finden.

### Notebooks

Im Ordner `notebooks` befinden sich die Jupyter-Notebooks, die für die Entwicklung der Modelle verwendet wurden. Unter anderem gibt es folgende Notebooks:

- `compare_models.ipynb`: Vergleich der getesteten Modelle
- `combined_model`: Testläufe mit dem kombinierten Modell (Klassifikation und Regression)
- `summary.ipynb`: Zusammenfassung der Ergebnisse
- `final_model_rabatte.ipynb`: Analyse, wieviel Rabattbetrugsfälle das Modell erkennt
- `train_final_models.ipynb`: Training und Speichern der finalen Modelle und Encoder
- `vorhersage_modelergebnis.ipynb`: Abschätzung der Bewertungsfunktion

- `shap_xgboost.ipynb`: SHAP-Werte für XGBoost-Klassifikationsmodell
- `shap_xgboost_reg.ipynb`: SHAP-Werte für XGBoost-Regressionsmodell

- `kalibrierung.ipynb`: Untersuchung ob das Modell kalibriert werden sollte
- `classifier_optimize_threshold.ipynb`: Untersuchung ob eine threshold-Optimierung zu besseren Ergebnissen führt
- `clf_only_vorhersage_modelergebnis.ipynb`: Abschätzung der Bewertungsfunktion (nur Klassifikationsmodell)
- `vorhersage_modelergebnis_calibrated.ipynb`: Abschätzung der Bewertungsfunktion (kalibriertes Modell)

- `combined_model_eval_damage_predictions.ipynb`: wie gut schneiden die Regressions-Varianten bei der Vorhersage von Schaden ab, beschränkt auf tatsächliche Fraud-Fälle, die vom Klassifkationsmodell erkennt werden)

- Notebooks für die Hyperparametersuche
- `select_clf_features.ipynb`: Feature selection
