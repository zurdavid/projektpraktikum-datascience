import logging
import sys

import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .. import data_loader, metrics


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 30),
        "eval_metric": "logloss",
    }

    clf = XGBClassifier(**params)

    clf.fit(X_train, y_train[:, 0])
    preds = clf.predict(X_test)
    bew = metrics.bewertung(preds, y_test[:, 0], y_test[:, 1])
    return -bew["Bewertung"]


def objective_reg(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 30),
        "eval_metric": "logloss",
    }

    clf = XGBRegressor(**params)

    clf.fit(X_train, y_train[:, 1])
    preds = clf.predict(X_test)
    bew = metrics.regression(preds, y_test[:, 1])
    return -bew["R2"]


def optimize_persistent(path, seed=42):
    X, targets = data_loader.load_data_np(path)

    X_train, X_test, y_train, y_test = map(
        np.asarray, train_test_split(X, targets, test_size=0.2, random_state=seed)
    )

    def wrapped_objective(trial):
        return objective_reg(trial, X_train, X_test, y_train, y_test)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(wrapped_objective, show_progress_bar=True, n_trials=10)

    # Best result
    print("Best params:", study.best_params)
    print("Beste Bewertung:", study.best_value)


def load_study():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    print("Best params:", study.best_params)
    print("Beste Bewertung:", study.best_value)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
