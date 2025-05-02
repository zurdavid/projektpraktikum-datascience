import polars as pl
from pathlib import Path

data_dir = Path("../data")

full = pl.scan_parquet(data_dir / "joined_data_train_3.parquet")
df = full.with_columns(pl.col("category").fill_null("MISSING"))

columns_to_encode = ["category", "urbanization"]

# Get unique values for each column (this is done eagerly)
unique_values = {
    col: df.select(col).unique().collect().to_series(0).to_numpy()
    for col in columns_to_encode
}

# Convert df to LazyFrame
lf = df.lazy()

# Build one-hot encoded expressions
one_hot_exprs = []
for col, values in unique_values.items():
    one_hot_exprs.extend([
        pl.when(pl.col(col) == val).then(1).otherwise(0).alias(f"{col}_{val}")
        for val in values
    ])

# Apply one-hot encoding lazily
lf_one_hot = lf.with_columns(one_hot_exprs)

# Collect column names to aggregate
one_hot_col_names = [f"{col}_{val}" for col, vals in unique_values.items() for val in vals]

# Group by transaction_id and aggregate with max to detect presence
result = (
    lf_one_hot
    .group_by("transaction_id")
    .agg([
        pl.col(col).max().alias(col) for col in one_hot_col_names
    ])
)

# Execute the lazy pipeline
result.sink_parquet(data_dir / "one_hot_encoded.parquet")
