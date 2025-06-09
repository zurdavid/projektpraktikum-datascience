from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import metrics
from .data_loader import load_data_np
from .models.types import FraudDetectionModel


def train_model(
    clf: FraudDetectionModel,
    X: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
):
    X_train, X_test, y_train, y_test = map(
        np.asarray, train_test_split(X, targets, test_size=test_size, random_state=seed)
    )

    clf.fit(X_train, y_train, X_test, y_test)
    preds = clf.predict(X_test)

    bew = metrics.bewertung(preds, y_test[:, 0], y_test[:, 1])
    metrics.print_metrics(bew)


def train_classifier_with_cross_validation(
    name: str,
    clf: FraudDetectionModel,
    X: np.ndarray,
    targets: np.ndarray,
    skf: StratifiedKFold,
):
    print(f"\n\nStart trainings for {name}")

    model_metrics = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, targets[:, 0])):
        print(f"Round {i}")
        clf.fit(X[train_idx], targets[train_idx], X[test_idx], targets[test_idx])
        preds = clf.predict(X[test_idx])

        labels = targets[:, 0]
        damage = targets[:, 1]
        bew = metrics.bewertung(preds, labels[test_idx], damage[test_idx])
        model_metrics.append(bew)

    avg_metrics = dict()
    for metric_name in [k for k in model_metrics[0].keys() if k != "cm"]:
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in model_metrics])
    avg_metrics["cm"] = np.stack([m["cm"] for m in model_metrics], axis=2).mean(axis=2)
    return avg_metrics


def compare_models(
    models: Sequence[tuple[str, FraudDetectionModel]],
    datapath: Path,
    n_splits: int = 5,
    random_state: int = 42,
):
    X, targets = load_data_np(datapath)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    model_metrics = dict()
    for name, model in models:
        model_metrics[name] = train_classifier_with_cross_validation(
            name, model, X, targets, skf
        )
    return model_metrics
