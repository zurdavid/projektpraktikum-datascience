"""
encoder.py

Module for encoding categorical and boolean features in a Polars DataFrame.

Classes:
    PolarsEncoder: Encoder for transforming categorical and boolean features into a format suitable for machine learning models.
"""

import json
from typing import Self

import polars as pl


class PolarsEncoder:
    """
    Encoder for transforming categorical and boolean features in a Polars DataFrame.
    """

    def __init__(self, drop_first=False):
        """
        Initialize the PolarsEncoder.

        Args:
            drop_first (bool): If True, the first category of each categorical column will be dropped to avoid multicollinearity.
                               Defaults to False.

        Attributes:
            drop_first (bool): Whether to drop the first category of categorical columns.
            categories_ (dict): Dictionary mapping column names to their unique categories.
            columns_ (list): List of all expected output columns after transformation.
            numeric_columns_ (list): List of numeric columns in the DataFrame.
            bool_columns_ (list): List of boolean columns in the DataFrame.
        """
        self.drop_first = drop_first
        self.categories_ = {}
        self.columns_ = []
        self.numeric_columns_ = []
        self.bool_columns_ = []

    def fit(self, df: pl.DataFrame):
        """
        Fit the encoder to the DataFrame by identifying categorical and boolean columns.

        Args:
            df (pl.DataFrame): The DataFrame to fit the encoder on.
        Returns:
            self: Returns the fitted encoder instance.
        """
        # Identify columns by dtype
        for col, dtype in zip(df.columns, df.dtypes, strict=True):
            if dtype in [pl.Utf8, pl.Categorical]:
                # Categorical column
                cats = df[col].unique().to_list()
                if self.drop_first and cats:
                    cats = cats[1:]
                self.categories_[col] = cats
            elif dtype == pl.Boolean:
                self.bool_columns_.append(col)
            elif dtype.is_numeric():
                self.numeric_columns_.append(col)
            else:
                pass

        # Build expected final columns list
        self.columns_ = []
        for col in self.numeric_columns_:
            self.columns_.append(col)
        for col in self.bool_columns_:
            self.columns_.append(col)
        for col in self.categories_:
            for cat in self.categories_[col]:
                self.columns_.append(f"{col}_{cat}")

        return self

    def transform(self, df: pl.DataFrame):
        """
        Transform the DataFrame by encoding categorical and boolean features.

        Args:
            df (pl.DataFrame): The DataFrame to transform.
        Returns:
            pl.DataFrame: Transformed DataFrame with encoded features.
        """
        result = df.clone()

        # Cast booleans to Int8
        for col in self.bool_columns_:
            result = result.with_columns(pl.col(col).cast(pl.Int8))

        # Add dummy columns for categoricals
        for col in self.categories_:
            for cat in self.categories_[col]:
                dummy_name = f"{col}_{cat}"
                result = result.with_columns(
                    (df[col] == cat).cast(pl.Int8).alias(dummy_name)
                )

        # Ensure all expected columns are present, add missing as 0
        final_cols = []
        for col in self.columns_:
            if col in result.columns:
                final_cols.append(pl.col(col))
            else:
                final_cols.append(pl.lit(0).alias(col))

        return result.select(final_cols)

    def fit_transform(self, df: pl.DataFrame):
        """
        Fit the encoder to the DataFrame and then transform it.

        Args:
            df (pl.DataFrame): The DataFrame to fit and transform.
        Returns:
            pl.DataFrame: Transformed DataFrame with encoded features.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path):
        """
        Save the encoder state to a JSON file.

        Args:
            path (str): Path to the file where the encoder state will be saved.
        """
        state = {
            "drop_first": self.drop_first,
            "categories_": self.categories_,
            "columns_": self.columns_,
            "numeric_columns_": self.numeric_columns_,
            "bool_columns_": self.bool_columns_,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path) -> Self:
        """
        Load the encoder state from a JSON file.

        Args:
            path (str): Path to the file from which the encoder state will be loaded.
        Returns:
            self: Returns the encoder instance with loaded state.
        """
        with open(path, "r") as f:
            state = json.load(f)
        self.drop_first = state["drop_first"]
        self.categories_ = state["categories_"]
        self.columns_ = state["columns_"]
        self.numeric_columns_ = state["numeric_columns_"]
        self.bool_columns_ = state["bool_columns_"]
        return self
