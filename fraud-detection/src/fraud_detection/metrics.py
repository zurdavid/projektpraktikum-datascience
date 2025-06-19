from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def bewertung(y_probs, yhat, y, damage) -> dict[str, float]:
    assert yhat.shape == y.shape, f"shapes yhat: {y.shape} == y: {yhat.shape}"
    assert y.shape == damage.shape, f"shapes yhat: {y.shape} == y: {damage.shape}"

    metrics = {}

    metrics["cm"] = confusion_matrix(y, yhat)

    metrics["precision"] = precision_score(y, yhat, zero_division=0.0)
    metrics["recall"] = recall_score(y, yhat)
    metrics["f1"] = f1_score(y, yhat)
    metrics["mcc"] = matthews_corrcoef(y, yhat)
    metrics["auc-pr:"] = average_precision_score(y, y_probs)


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
        print(
            " ".join(f"{x:6}" for x in row_a)
            + "   |   "
            + " ".join(f"{x:6}" for x in row_b)
        )


def regression(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def propability_histogram(
    predictions,
    targets,
    name: str,
    epoch: int = -1,
    bins=50,
):
    # replace plot with seaborn
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.histplot(
        predictions[targets == 0],
        bins=bins,
        alpha=0.8,
        label="Normal",
        )
    sns.histplot(
        predictions[targets == 1],
        bins=bins,
        alpha=0.4,
        label="Fraud",
        color="red",
        )
    plt.yscale("log")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Predicted Fraud Probabilities by label: " + name)
    plt.legend()
    epochstr = f"epoch_{epoch}" if epoch != -1 else "final"
    plt.savefig(f"plots/clf_probability_histogram_{name}_{epochstr}.png")
    plt.close()


def plot_roc_curve(
    preds,
    targets,
    name: str,
    epoch: int = -1,
):
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(targets, preds)

    # Compute the AUC score (optional, but useful)
    roc_auc = roc_auc_score(targets, preds)

    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    epochstr = f"epoch_{epoch}" if epoch != -1 else "final"
    plt.savefig(f"plots/clf_roc_{name}_{epochstr}.png")
    plt.close()
