import os
from pathlib import Path
import polars as pl

from app.model.transform import transform
from app.model.ml_model import load_models, FraudDetectionModel

data_dir = Path("../data")


def load_transform(datapath, testdata=False, only_labeled=False):
    transactions_file = "transactions_train_3.parquet"
    lines_file = "transaction_lines_train_3.parquet"
    stores_file = "stores.csv"
    products_file = "products.csv"

    if testdata:
        transactions_file = "transactions_test_3.parquet"
        lines_file = "transaction_lines_test_3.parquet"

    transactions = pl.scan_parquet(datapath / transactions_file)
    lines = pl.scan_parquet(datapath / lines_file)
    stores = pl.scan_csv(datapath / stores_file)
    products = pl.scan_csv(datapath / products_file)

    if testdata:
        transactions = transactions.with_columns(
            pl.lit("UNKNOWN").alias("label"),
            pl.lit(0.0).alias("damage"),
        )
    elif not testdata and only_labeled:
        transactions = transactions.filter(pl.col("label") != "UNKNOWN")

    transformed_df = transform(transactions, lines, stores, products)
    return transformed_df.collect()


def classify_static(df):
    df_unscanned = (
        df.filter(pl.col("has_missing") | pl.col("has_unscanned"))
        .with_columns(
            pl.lit(True).alias("is_fraud"),
            pl.lit(1.0).cast(pl.Float32).alias("fraud_proba"),
            pl.col("price_unscanned_articles")
            .cast(pl.Float32)
            .alias("estimated_damage"),
        )
        .select("transaction_id", "is_fraud", "fraud_proba", "estimated_damage")
    )
    return df_unscanned


def classify_model(df):
    classifier_path = "models/xgb_fraud_classifier.joblib"
    regressor_path = "models/xgb_damage_regressor.joblib"
    encoder_path = "models/encoder.json"

    model = FraudDetectionModel(
        *load_models(
            classifier_path=classifier_path,
            regressor_path=regressor_path,
            encoder_path=encoder_path,
        )
    )
    return model.predict_df(df)


def main():
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            "Data directory not found. Script should be run from the root directory of the api project."
        )

    df = load_transform(data_dir, testdata=True)
    certain_frauds = classify_static(df)
    df_remain = df.join(certain_frauds, on="transaction_id", how="anti")
    predictions = classify_model(df_remain)
    predictions = pl.concat([certain_frauds, predictions], how="vertical")
    predictions.write_parquet("predictions_test.parquet")


if __name__ == "__main__":
    main()
