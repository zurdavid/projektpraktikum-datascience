import polars as pl

datapath = "../data"

transactions = pl.scan_parquet(f"{datapath}/transactions_train_3.parquet")
lines = pl.scan_parquet(f"{datapath}/transaction_lines_train_3.parquet")

transactions = (
    transactions.filter(pl.col("label") != "UNKNOWN")
    # .filter(pl.col("label") == "FRAUD")
    # .filter(pl.col("customer_feedback").is_not_null())
    .collect()
)


def generate_test_data():
    # select a random transaction
    transaction = transactions.sample(n=1)

    transaction_id = transaction["id"][0]
    print(f"Selected transaction ID: {transaction_id}")
    transaction_lines = (
        lines.filter(pl.col("transaction_id") == transaction_id)
        .with_columns(
            pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        .collect()
    )

    transaction = transaction.with_columns(
        pl.col("transaction_start").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        pl.col("transaction_end").dt.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    payload = dict()
    transaction = transaction.drop(["label", "damage", "n_lines"])
    payload["transaction_header"] = transaction.to_dicts()[0]
    payload["transaction_lines"] = transaction_lines.to_dicts()
    return payload


if __name__ == "__main__":
    generate_test_data()
