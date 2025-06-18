from pathlib import Path

import polars as pl
import polars.selectors as cs

classification_features = [
    "payment_medium",
    "hour",
    "has_positive_price_difference",
    "has_camera_detected_wrong_product_high_certainty",
    "has_snacks",
]

regression_features = [
    "payment_medium",
    "hour",
    "has_voided",
    "n_voided",
    "has_camera_detected_wrong_product_high_certainty",
    "calculated_price_difference",
    "has_positive_price_difference",
    "has_snacks",
]


def load_data_df(path: Path, filter_has_unscanned: bool = True, drop_features=None):
    drop_features = drop_features or []
    df = pl.read_parquet(path).drop("transaction_id").drop(drop_features)

    if filter_has_unscanned:
        df = df.filter(pl.col("has_unscanned").not_()).drop("has_unscanned")

    return df


def load_data(
    path: Path, features=None, filter_has_unscanned: bool = True, drop_features=None
):
    df = load_data_df(path, filter_has_unscanned, drop_features=drop_features)
    # targets
    y = (
        df.select(["label", "damage"])
        .to_dummies(cs.categorical(), drop_first=True)
        .select(pl.col("label_FRAUD").alias("label"), "damage")
    )

    # features
    X = (
        df.select(features or df.columns)
        .drop("label", "damage", strict=False)
        .with_columns(
            pl.col(pl.Boolean).cast(pl.Int8),
        )
        .to_dummies(cs.categorical(), drop_first=True)
    )
    return X, y


def load_data_np(
    path: Path,
    features=None,
    filter_has_unscanned: bool = True,
    drop_features=None,
):
    X, y = load_data(
        path,
        features=features,
        filter_has_unscanned=filter_has_unscanned,
        drop_features=drop_features,
    )
    X, y = X.to_numpy(), y.to_numpy()
    return X, y


def load_data_for_regression(path: Path, filter_has_unscanned: bool = True):
    # lade nur FRAUD
    df = load_data_df(path, filter_has_unscanned).filter(pl.col("label") == "FRAUD")

    # targets
    y = df.select(["damage"]).to_dummies(cs.categorical(), drop_first=True).to_numpy()

    # features
    X = (
        df.drop("label", "damage")
        #  .select(regression_features)
        .with_columns(
            pl.col(pl.Boolean).cast(pl.Int8),
        )
        .to_dummies(cs.categorical(), drop_first=True)
        .to_numpy()
    )
    return X, y
