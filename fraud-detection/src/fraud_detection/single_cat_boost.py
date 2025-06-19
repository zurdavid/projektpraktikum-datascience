from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

import fraud_detection as fd
from fraud_detection import metrics

seed = 4


def miau(path: Path, useless_features=None):
    X, y = fd.data_loader.load_pandas_data(path, drop_features=useless_features)

    # Convert label from string to numerical 0/1
    y["label"] = y["label"].map({"FRAUD": 1, "NORMAL": 0})

    clf = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        verbose=0,
        random_seed=42,
    )

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y["label"]
    )

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    clf.fit(
        X_train,
        y_train["label"],
        cat_features=cat_features,
        # eval_set=(X_test, y_test["label"]),
        verbose=10,
    )

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    bew = metrics.bewertung(probs, preds, y_test["label"], y_test["damage"])
    metrics.print_metrics(bew)


def miau2(path: Path, useless_features=None):
    X, y = fd.data_loader.load_pandas_data(path, drop_features=useless_features)

    # dummpy encode categorical features
    X = X.select_dtypes(exclude=["object", "category"]).join(
        pd.get_dummies(X.select_dtypes(include=["object", "category"]), drop_first=True)
    )
    # drop categorical features that were converted to dummies
    X = X.drop(columns=X.select_dtypes(include=["object", "category"]).columns)

    # Convert label from string to numerical 0/1
    y["label"] = y["label"].map({"FRAUD": 1, "NORMAL": 0})

    clf = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        verbose=0,
        random_seed=42,
    )

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y["label"]
    )

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    clf.fit(
        X_train,
        y_train["label"],
        cat_features=cat_features,
        # eval_set=(X_test, y_test["label"]),
        verbose=10,
    )

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    bew = metrics.bewertung(probs, preds, y_test["label"], y_test["damage"])
    metrics.print_metrics(bew)

