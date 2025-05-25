import polars as pl

df_damage = pl.scan_parquet("transformed_damage_first.parquet")
df_label = pl.scan_parquet("transformed_label_first.parquet")

cols_damage_list = df_damage.collect_schema().names()
cols_label_list = df_label.collect_schema().names()

cols_damage = set(cols_damage_list)
cols_label = set(cols_label_list)

expected_cols = {
  "transaction_id",
  "calculated_price_difference",
  "cash_desk",
  "day_of_week",
  "days_since_sco_introduction",
  "daytime",
  "feedback_categorical",
  "feedback_high",
  "feedback_low",
  "feedback_middle",
  "feedback_top",
  "has_age_restricted",
  "has_alcohol",
  "has_bakery",
  "has_beverages",
  "has_camera_detected_wrong_product",
  "has_camera_detected_wrong_product_high_certainty",
  "has_convenience",
  "has_dairy",
  "has_feedback",
  "has_frozen_goods",
  "has_fruits_vegetables",
  "has_fruits_vegetables_pieces",
  "has_household",
  "has_limited_time_offers",
  "has_long_shelf_life",
  "has_missing",
  "has_personal_care",
  "has_positive_price_difference",
  "has_snacks",
  "has_sold_by_weight",
  "has_tobacco",
  "has_voided",
  "has_unscanned",
  "hour",
  "hour_categorical",
  "location",
  "max_product_price",
  "max_time_between_scans",
  "mean_time_between_scans",
  "month",
  "n_age_restricted",
  "n_lines",
  "n_sold_by_weight",
  "n_voided",
  "payment_medium",
  "popularity_max",
  "popularity_min",
  "store_id",
  "time_from_last_scan_to_end",
  "time_to_first_scan",
  "total_amount",
  "transaction_duration_seconds",
  "urbanization"
}


def test_first_column():
    assert cols_damage_list[0] == "damage"
    assert cols_label_list[0] == "label"


def test_contains_correct_targets():
    assert "label" in cols_label
    assert "label" not in cols_damage

    assert "damage" in cols_damage
    assert "damage" not in cols_label


def test_same_length():
    assert len(cols_damage) == len(cols_label)


def test_contains_expected_columns():
    diff_label_first = expected_cols.symmetric_difference(cols_label)
    assert  diff_label_first == {"label"}, diff_label_first
    diff_damage_first = expected_cols.symmetric_difference(cols_damage)
    assert diff_damage_first == {"damage"}


def test_n_unscanned():
    n_unscanned = df_damage.filter(pl.col("has_unscanned")).select(pl.len()).collect()[0, 0]
    assert n_unscanned == 377
