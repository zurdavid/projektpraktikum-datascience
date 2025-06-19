from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)

from . import metrics
from .data_loader import load_data_np
from .models.types import DamagePredictionModel, FraudDetectionModel


def train_model(
    clf: FraudDetectionModel,
    X: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
):
    X_train, X_test, y_train, y_test = map(
        np.asarray,
        train_test_split(
            X, targets, test_size=test_size, random_state=seed, stratify=targets[:, 0]
        ),
    )

    clf.fit(X_train, y_train, X_test, y_test)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    metrics.propability_histogram(probs, y_test[:, 0], clf.name(), bins=20)
    metrics.plot_roc_curve(probs, y_test[:, 0], clf.name())

    bew = metrics.bewertung(probs, preds, y_test[:, 0], y_test[:, 1])
    metrics.print_metrics(bew)


def train_classifier_with_cross_validation(
    name: str,
    clf_creator,
    X: np.ndarray,
    targets: np.ndarray,
    skf: RepeatedStratifiedKFold,
):
    print(f"\n\nStart trainings for {name}")

    model_metrics = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, targets[:, 0])):
        # Create a new instance of the classifier for each fold
        clf = clf_creator()
        print(f"Round {i}")
        clf.fit(X[train_idx], targets[train_idx], X[test_idx], targets[test_idx])

        probs = clf.predict_proba(X[test_idx])
        preds = clf.predict(X[test_idx])

        labels = targets[:, 0]
        damage = targets[:, 1]
        bew = metrics.bewertung(probs, preds, labels[test_idx], damage[test_idx])
        model_metrics.append(bew)

    metrics_dict = dict()
    for metric_name in [k for k in model_metrics[0].keys() if k != "cm"]:
        metrics_dict[metric_name + "_mean"] = np.mean(
            [m[metric_name] for m in model_metrics]
        )
        metrics_dict[metric_name + "_max"] = np.max(
            [m[metric_name] for m in model_metrics]
        )
        metrics_dict[metric_name + "_min"] = np.min(
            [m[metric_name] for m in model_metrics]
        )
        metrics_dict[metric_name + "_var"] = np.var(
            [m[metric_name] for m in model_metrics]
        )
    metrics_dict["cm"] = np.stack([m["cm"] for m in model_metrics], axis=2).mean(axis=2)
    return metrics_dict


def compare_models(
    models,
    datapath: Path,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    drop_features=None,
    select_features=None,
):
    X, targets = load_data_np(
        datapath, features=select_features, drop_features=drop_features
    )
    skf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    model_metrics = dict()
    for name, model in models:
        model_metrics[name] = train_classifier_with_cross_validation(
            name, model, X, targets, skf
        )
    return model_metrics


def select_features_to_drop(
    model: FraudDetectionModel,
    datapath: Path,
    features_to_drop: Sequence[str] = [],
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
):
    skf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    model_metrics = dict()

    for f in features_to_drop:
        X, targets = load_data_np(datapath, drop_features=[f])
        name = model.name()
        model_metrics[f] = train_classifier_with_cross_validation(
            name, model, X, targets, skf
        )
    return model_metrics


def train_regression_model(
    model: DamagePredictionModel,
    X: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
):
    X_train, X_test, y_train, y_test = map(
        np.asarray, train_test_split(X, targets, test_size=test_size, random_state=seed)
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    bew = metrics.regression(preds, y_test[:, 1])
    metrics.print_metrics(bew)

    return model, bew
