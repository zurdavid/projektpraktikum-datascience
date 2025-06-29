from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fraud_detection as fd
from fraud_detection.model_comparison import train_regression_model

path = Path("./data/transformed_label_and_damage.parquet")

seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

features_linear_modell = [
    "payment_medium",
    "has_positive_price_difference",
    "calculated_price_difference",
]

useless_features = fd.data_loader.useless_features


def compare():
    X, _ = fd.data_loader.load_data_np(path, drop_features=useless_features)
    n_input = X.shape[1]

    models = [
        ("Decision Tree", lambda: fd.models.classifiers.get_decsion_tree()),
        ("Random Forest", lambda: fd.models.classifiers.get_random_forest()),
        ("LGBMClassifier", lambda: fd.models.classifiers.get_lgmb()),
        ("XGBoost simple", lambda: fd.models.classifiers.get_xgb_simple()),
        ("XGBoost simple", lambda: fd.models.classifiers.get_xgb()),
        ("CatBoost", lambda: fd.models.classifiers.get_catboost()),
        # ("NeuralNet", lambda: fd.neuralnets.train_nn.getNN(n_input)),
    ]

    model_metrics = fd.model_comparison.compare_models(
        models,
        path,
        n_splits=5,
        n_repeats=5,
        random_state=seed,
        drop_features=useless_features,
    )
    df = pd.DataFrame(model_metrics)
    df.to_csv("model_comparison.csv", index=True)
    print(df)


def train_regression():
    X, y = fd.data_loader.load_data_np(path, drop_features=useless_features)
    model = fd.neuralnets.train_nn.getNN_regressor(X.shape[1])

    fd.model_comparison.train_regression_model(model, X, y, test_size=0.2, seed=seed)


def train_single():
    X, y = fd.data_loader.load_data_np(path, drop_features=useless_features)
    clf = fd.models.classifiers.get_xgb_simple()
    # clf = fd.neuralnets.train_nn.getNN(X.shape[1])
    clf = fd.neuralnets.train_nn.getvanillaNN(X.shape[1])
    # clf = fd.models.classifiers.get_catboost()
    # clf = fd.models.classifiers.get_decsion_tree()

    # clf = fd.models.classifiers.get_random_forest()

    # clf = fd.models.classifiers.get_lgmb()

    # clf = fd.neuralnets.train_nn.getNN2(X.shape[1])

    fd.model_comparison.train_model(clf, X, y, seed=seed)


def train_linear():
    X, y = fd.data_loader.load_data_np(path, features=features_linear_modell)
    # take equal parts of data with target[:, 1] == 0 and target[:, 1] == 1
    clf = fd.models.classifiers.get_logistic_regression()
    fd.model_comparison.train_model(clf, X, y, seed=seed)


if __name__ == "__main__":
    # fd.models.hyperparam.optimize(path)
    # fd.models.hyper2.optimize_persistent(path)
    # fd.models.hyper2.load_study()
    # fd.neuralnets.hyperparam_search.optimize(path, 10)
    # fd.single_cat_boost.miau(path, useless_features=useless_features)
    train_single()
    # train_linear()

    # train_regression()

    # compare()
