import polars as pl
from pathlib import Path

data_dir = Path("../data")

transactions = pl.scan_parquet(data_dir / "transactions_train_3.parquet")
lines = pl.scan_parquet(data_dir / "transaction_lines_train_3.parquet")
stores = pl.scan_csv(data_dir / "stores.csv")
products = pl.scan_csv(data_dir / "products.csv")

# lazily join the four dataframes and save it to a new parquet file
(
    transactions.with_columns(
        # binäre Spalte "has_feedback" erstellen
        pl.col("customer_feedback").is_not_null().cast(pl.Int8).alias("has_feedback")
    )
    .join(
        lines,
        left_on="id",
        right_on="transaction_id",
        suffix="__lines",
        how="left",
    )
    # entferne Transaktionen ohne gültige lines
    .filter(pl.col("id__lines").is_not_null())
    .join(
        products,
        left_on="product_id",
        right_on="id",
        suffix="__products",
        how="left",
    )
    .join(
        stores,
        left_on="store_id",
        right_on="id",
        suffix="__stores",
        how="left",
    )
).sink_parquet(data_dir / "transactions_train_3_joined.parquet")
