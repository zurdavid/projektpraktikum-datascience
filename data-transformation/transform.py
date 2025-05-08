import polars as pl
import numpy as np
from pathlib import Path

# Pfad zum Ordner mit den Input Daten
data_dir = Path("../data")
# Pfad der Ausgabedatei
output_path = "transformed_transactions.parquet"

transactions = pl.scan_parquet(data_dir / "transactions_train_3.parquet")
lines = pl.scan_parquet(data_dir / "transaction_lines_train_3.parquet")
stores = pl.scan_csv(data_dir / "stores.csv")
products = pl.scan_csv(data_dir / "products.csv")

CAMERA_CERTAINTY_THRESHOLD = 0.8


def transform_duration_to_seconds(duration: pl.Duration):
    """Transform a duration column into seconds and round to the nearest second."""
    return duration.dt.total_microseconds() / 1_000_000


# Mapping from number to weekday name
weekday_map = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}

# Mapping from hour to daytime name
morning = "morning"
noon = "noon"
afternoon = "afternoon"
evening = "evening"
daytime_map = {
    8: morning,
    9: morning,
    10: morning,
    11: morning,
    12: noon,
    13: noon,
    14: noon,
    15: noon,
    16: afternoon,
    17: afternoon,
    18: afternoon,
    19: afternoon,
    20: evening,
    21: evening,
    22: evening,
}


####################################################################################################
# Transformiere und wähle die Spalten aus der Tabelle mit den Transaktionen
####################################################################################################

transactions_transformed = (
    transactions.with_columns(
        [
            # Features aus den Spalten transaction_start und transaction_end
            # Wochentag
            pl.col("transaction_start")
            .dt.weekday()
            .alias("day_of_week"),  # monday = 1 and sunday = 7
            # Monat
            pl.col("transaction_start")
            .dt.month()
            .cast(pl.Utf8)
            .cast(pl.Categorical)
            .alias("month"),  # monday = 1 and sunday = 7
            # Tageszeit
            pl.col("transaction_start")
            .dt.hour()
            .replace_strict(daytime_map, default=None)
            .cast(pl.Categorical)
            .alias("daytime"),  # morning, noon, afternoon, evening
            # Stunde
            pl.col("transaction_start").dt.hour().alias("hour"),
            pl.col("transaction_start")
            .dt.hour()
            .cast(pl.Utf8)
            .cast(pl.Categorical)
            .alias("hour_categorical"),
            # Dauer in Sekunden
            (pl.col("transaction_end") - pl.col("transaction_start"))
            .dt.total_seconds()
            .alias("transaction_duration_seconds"),
            # Variationen der Spalte customer_feedback
            # es gibt eine Spalte has_feedback und 4 Spalten für die verschiedenen Feedback-Kategorien
            # (One-Hot-Encoding), da die viele null-Werte enthalten
            pl.col("customer_feedback")
            .is_not_null()
            .cast(pl.Boolean)
            .alias("has_feedback"),
            # low feedback = 1, 2, 3
            (pl.col("customer_feedback").is_in([1, 2, 3]))
            .fill_null(False)
            .alias("feedback_low")
            .cast(pl.Boolean),
            # middle feedback = 4, 5, 6
            (pl.col("customer_feedback").is_in([4, 5, 6]))
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("feedback_middle"),
            # high feedback = 7, 8, 9
            (pl.col("customer_feedback").is_in([7, 8, 9]))
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("feedback_high"),
            # Top feedback - als FRAUD gelabelte Daten enthalten vor allem 10
            (pl.col("customer_feedback") == 10)
            .cast(pl.Boolean)
            .fill_null(False)
            .alias("feedback_top"),
            pl.col("cash_desk").cast(pl.Utf8).cast(pl.Categorical),
        ]
    )
    .with_columns(
        pl.col("day_of_week")
        .replace_strict(weekday_map, default=None)
        .cast(pl.Categorical)
    )
    .select(
        [
            "id",
            "cash_desk",
            "total_amount",
            "n_lines",
            "payment_medium",
            "has_feedback",
            "feedback_low",
            "feedback_middle",
            "feedback_high",
            "feedback_top",
            "daytime",
            "hour",
            "hour_categorical",
            "day_of_week",
            "month",
            "transaction_duration_seconds",
            "damage",
            "label",
            "store_id",
            "transaction_start",
            "transaction_end",
        ]
    )
)

####################################################################################################
# Stores
####################################################################################################

stores_transformed = stores.with_columns(
    pl.col("sco_introduction").str.strptime(pl.Datetime, "%Y-%m-%d")
).select(
    [
        pl.col("id").alias("store_id"),
        "location",
        "urbanization",
        "sco_introduction",
    ]
)

####################################################################################################
# Lines
####################################################################################################

lines_with_products = lines.join(
    products,
    left_on="product_id",
    right_on="id",
    suffix="__products",
    how="left",
).select(
    [
        pl.col("transaction_id"),
        pl.col("product_id"),
        pl.col("was_voided"),
        pl.col("camera_product_similar"),
        pl.col("camera_certainty"),
        pl.col("price"),
        pl.col("category"),
        pl.col("popularity"),
        pl.col("sold_by_weight"),
        pl.col("age_restricted"),
    ]
)

# Kategorien für One-Hot-Encoding
categories = products.select("category").unique().collect().to_series(0).to_numpy()
categories = np.append(categories, "MISSING")
category_columns = [f"has_{category.lower()}" for category in categories]
category_one_hot_exprs = [
    pl.when(pl.col("category") == category)
    .then(1)
    .otherwise(0)
    .alias(f"has_{category.lower()}")
    for category in categories
]


lines_with_products_grouped = (
    lines_with_products
    # Replace missing category with "MISSING" (siginificant feature)
    .with_columns(pl.col("category").fill_null("MISSING"))
    .with_columns(category_one_hot_exprs)
    .group_by("transaction_id")
    .agg(
        [
            pl.col("was_voided").max().alias("has_voided").cast(pl.Boolean),
            pl.col("was_voided").sum().alias("n_voided"),
            pl.col("age_restricted").sum().alias("n_age_restricted"),
            pl.col("age_restricted").max().alias("has_age_restricted").cast(pl.Boolean),
            pl.col("popularity").max().alias("popularity_max"),
            pl.col("popularity").min().alias("popularity_min"),
            pl.col("price").max().alias("max_product_price"),
            pl.col("sold_by_weight").sum().alias("n_sold_by_weight"),
            pl.col("sold_by_weight").max().alias("has_sold_by_weight").cast(pl.Boolean),
            pl.col("camera_product_similar")
            .min()
            .alias("has_camera_detected_wrong_product")
            .cast(pl.Boolean),
            (
                pl.col("camera_product_similar")
                & (pl.col("camera_certainty") >= CAMERA_CERTAINTY_THRESHOLD)
            )
            .min()
            .alias("has_camera_detected_wrong_product_high_certainty")
            .cast(pl.Boolean),
        ]
        # Produkt-Kategorien als One-Hot-Encoding
        + [pl.col(col).max().alias(col).cast(pl.Boolean) for col in category_columns]
    )
)

####################################################################################################
# Berechne Werte aus den Line Timestamps
####################################################################################################

lines_with_timestamps_grouped = (
    lines.sort(["transaction_id", "timestamp"])
    .with_columns(
        (pl.col("timestamp") - pl.col("timestamp").shift(1))
        .over("transaction_id")
        .alias("time_betweeen_scans")
    )
    .group_by("transaction_id")
    .agg(
        [
            transform_duration_to_seconds(pl.col("time_betweeen_scans").mean()).alias(
                "mean_time_between_scans"
            ),
            transform_duration_to_seconds(pl.col("time_betweeen_scans").max()).alias(
                "max_time_between_scans"
            ),
            # für spätere Berechnung
            pl.col("timestamp").min().alias("first_timestamp"),
            pl.col("timestamp").max().alias("last_timestamp"),
        ]
    )
)

####################################################################################################
# Joine alle Dataframes
####################################################################################################

joined_table = (
    transactions_transformed.join(
        stores_transformed,
        left_on="store_id",
        right_on="store_id",
        suffix="__stores",
        how="left",
    )
    .join(
        lines_with_products_grouped,
        left_on="id",
        right_on="transaction_id",
        suffix="__lines",
        how="left",
    )
    .join(
        lines_with_timestamps_grouped,
        left_on="id",
        right_on="transaction_id",
        suffix="__lines_timestamps",
        how="left",
    )
)

# Operationen die auf den gejointen Daten durchgeführt werden müssen
joined_table = (
    joined_table.with_columns(
        [
            # Berechne die Zeiten zwischen dem Beginn der Transaktion und dem ersten Scan,
            # sowie dem letzten Scan und dem Ende der Transaktion
            transform_duration_to_seconds(
                pl.col("first_timestamp") - pl.col("transaction_start")
            ).alias("time_to_first_scan"),
            transform_duration_to_seconds(
                pl.col("transaction_end") - pl.col("last_timestamp")
            ).alias("time_from_last_scan_to_end"),
            # Tage seit der Einführung des Self-Checkout
            (pl.col("transaction_start") - pl.col("sco_introduction"))
            .dt.total_days()
            .alias("days_since_sco_introduction"),
        ]
    )
    # caste kategorische Spalten
    .with_columns(
        [
            pl.col(col).cast(pl.Categorical)
            for col in [
                "cash_desk",
                "payment_medium",
                "daytime",
                "day_of_week",
                "month",
                "label",
                "store_id",
                "location",
                "urbanization",
            ]
        ]
    )
    # Drop Spalten die nicht mehr benötigt werden
    .drop(
        [
            "transaction_start",
            "transaction_end",
            "first_timestamp",
            "last_timestamp",
            "sco_introduction",
        ]
    )
)

joined_table.sink_parquet(output_path)
