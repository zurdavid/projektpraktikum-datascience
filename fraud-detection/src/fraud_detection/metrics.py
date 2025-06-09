from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def bewertung(yhat, y, damage) -> dict[str, float]:
    assert yhat.shape == y.shape, f"shapes yhat: {y.shape} == y: {yhat.shape}"
    assert y.shape == damage.shape, f"shapes yhat: {y.shape} == y: {damage.shape}"

    metrics = {}

    metrics["cm"] = confusion_matrix(y, yhat)

    metrics["precision"] = precision_score(y, yhat, zero_division=0.0)
    metrics["recall"] = recall_score(y, yhat)
    metrics["f1"] = f1_score(y, yhat)

    metrics["damage_total"] = damage.sum()

    metrics["damage_prevented"] = (((y == 1) & (yhat == 1)) * damage).sum()
    metrics["damage_missed"] = (((y == 1) & (yhat == 0)) * damage).sum()

    metrics["detected bonus"] = (((y == 1) & (yhat == 1)) * 5).sum()

    res = np.zeros(damage.shape)
    # Case 1: FRAUD caught
    res += ((y == 1) & (yhat == 1)) * 5
    # res += ((y == 1) & (yhat == 1)) * damage
    # Case 2: False positive
    fp_penalty = ((y == 0) & (yhat == 1)) * 10
    metrics["fp penalty"] = fp_penalty.sum()
    res -= fp_penalty
    # Case 3: FRAUD missed
    res -= ((y == 1) & (yhat == 0)) * damage
    metrics["Bewertung"] = res.sum()

    return metrics


def print_metrics(metrics_dict: dict[str, Any]):
    for text, value in metrics_dict.items():
        try:
            print(f"{text: <20} {value:>8.2f}")
        except TypeError:
            # Handle cases where value is not a number (e.g., confusion matrix)
            print(f"{text:}\n{value}")


def print_metrics_comp(train_metrics: dict[str, Any], test_metrics: dict[str, Any]):
    print(f"{'Metric': <18}| {'train': <9} | {'test': <8}|")
    for text, value in train_metrics.items():
        value_test = test_metrics[text]
        try:
            print(f"{text: <18}| {value:>9.2f} | {value_test:>8.2f}|")
        except TypeError:
            pass
    print("Confusion marices:")
    for row_a, row_b in zip(train_metrics["cm"], test_metrics["cm"], strict=True):

        print(" ".join(f"{x:6}" for x in row_a) + "   |   " + " ".join(f"{x:6}" for x in row_b))


def regression(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
