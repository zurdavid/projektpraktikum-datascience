from pathlib import Path

import polars as pl
import polars.selectors as cs

# Features that are not useful for the model and should be dropped
useless_features = [
    "max_product_price",
    "has_positive_price_difference",
    "has_bakery",
    "time_to_first_scan",
    "popularity_max",
    "has_age_restricted",
    "cash_desk",
    "transaction_duration_seconds",
    "feedback_low",
    "feedback_middle",
    "feedback_high",
    "feedback_top",
    "store_id",
    "location",
    "urbanization",
    "has_voided",
    "has_sold_by_weight",
    "has_limited_time_offers",
    "has_fruits_vegetables",
    "has_missing",
    "has_camera_detected_wrong_product",
    "day_of_week",
    "hour_categorical",
]


def load_data_df(path: Path, filter_has_unscanned: bool = True, drop_features=None):
    """
    Load the data from a parquet file into a Polars DataFrame.

    Args:
        path (Path): Path to the parquet file.
        filter_has_unscanned (bool): If True, filter out rows where 'has_unscanned'
            is True and drop the 'has_unscanned' column.
        drop_features (list, optional): List of features to drop from the DataFrame.

    Returns:
        pl.DataFrame: Polars DataFrame containing the loaded data.
    """
    drop_features = drop_features or []
    df = pl.read_parquet(path).drop("transaction_id").drop(drop_features)

    if filter_has_unscanned:
        df = df.filter(pl.col("has_unscanned").not_()).drop("has_unscanned")

    return df


def load_pandas_data(path: Path, filter_has_unscanned: bool = True, drop_features=None):
    """
    Load the data from a parquet file into Pandas DataFrames and two dataframes
    for features and targets.

    Args:
        path (Path): Path to the parquet file.
        filter_has_unscanned (bool): If True, filter out rows where 'has_unscanned'
            is True and drop the 'has_unscanned' column.
        drop_features (list, optional): List of features to drop from the DataFrame.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): DataFrame with features.
            - y (pd.DataFrame): DataFrame with targets ('label' and 'damage').
    """
    df = load_data_df(path, filter_has_unscanned, drop_features=drop_features)
    df = df.to_pandas()

    X = df.drop(columns=["label", "damage"])
    y = df[["label", "damage"]]
    return X, y


def load_data_pl(
    path: Path, features=None, filter_has_unscanned: bool = True, drop_features=None
):
    """
    Load the data from a parquet file into Polars DataFrames and two dataframes
    for features and targets. The categorical features are converted to dummy variables.

    Args:
        path (Path): Path to the parquet file.
        features (list, optional): List of features to select from the DataFrame.
        filter_has_unscanned (bool): If True, filter out rows where 'has_unscanned'
            is True and drop the 'has_unscanned' column.
        drop_features (list, optional): List of features to drop from the DataFrame.

    Returns:
        tuple: A tuple containing:
            - X (pl.DataFrame): DataFrame with features.
            - y (pl.DataFrame): DataFrame with targets ('label' and 'damage').
    """
    df = load_data_df(path, filter_has_unscanned, drop_features=drop_features)

    # targets
    y = (
        df.select(["label", "damage"])
        .to_dummies(cs.categorical())
        .select(pl.col("label_FRAUD").alias("label"), "damage")
    )

    X = df.select(features or df.columns).drop("label", "damage", strict=False)

    return X, y


def load_data(
    path: Path, features=None, filter_has_unscanned: bool = True, drop_features=None
):
    """
    Load the data from a parquet file into Polars DataFrames and two dataframes
    for features and targets. The categorical features are converted to dummy variables.

    Args:
        path (Path): Path to the parquet file.
        features (list, optional): List of features to select from the DataFrame.
        filter_has_unscanned (bool): If True, filter out rows where 'has_unscanned'
            is True and drop the 'has_unscanned' column.
        drop_features (list, optional): List of features to drop from the DataFrame.

    Returns:
        tuple: A tuple containing:
            - X (pl.DataFrame): DataFrame with features.
            - y (pl.DataFrame): DataFrame with targets ('label' and 'damage').
    """
    df = load_data_df(path, filter_has_unscanned, drop_features=drop_features)
    # targets
    y = (
        df.select(["label", "damage"])
        .to_dummies(cs.categorical())
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
    """
    Load the data from a parquet file into NumPy arrays for features and targets.
    Categorical features are converted to dummy variables.

    Args:
        path (Path): Path to the parquet file.
        features (list, optional): List of features to select from the DataFrame.
        filter_has_unscanned (bool): If True, filter out rows where 'has_unscanned'
            is True and drop the 'has_unscanned' column.
        drop_features (list, optional): List of features to drop from the DataFrame.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): NumPy array with features.
            - y (np.ndarray): NumPy array with targets ('label' and 'damage').
    """
    X, y = load_data(
        path,
        features=features,
        filter_has_unscanned=filter_has_unscanned,
        drop_features=drop_features,
    )
    X, y = X.to_numpy(), y.to_numpy()
    return X, y


def load_data_for_regression(path: Path, filter_has_unscanned: bool = True):
    """
    Load the data for regression tasks, specifically for predicting 'damage'.

    Args:
        path (Path): Path to the parquet file.
        filter_has_unscanned (bool): If True, filter out rows where 'has_unscanned'
            is True and drop the 'has_unscanned' column.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): NumPy array with features.
            - y (np.ndarray): NumPy array with targets ('damage').
    """
    # lade nur FRAUD
    df = load_data_df(path, filter_has_unscanned).filter(pl.col("label") == "FRAUD")

    # targets
    y = df.select(["damage"]).to_dummies(cs.categorical(), drop_first=True).to_numpy()

    # features
    X = (
        df.drop("label", "damage")
        .with_columns(
            pl.col(pl.Boolean).cast(pl.Int8),
        )
        .to_dummies(cs.categorical(), drop_first=True)
        .to_numpy()
    )
    return X, y
