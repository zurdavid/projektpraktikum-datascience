"""
datastore.py

Module for managing the data layer of the application, including loading and accessing product and store data.

classes:
    DataStore: Manages the loading and access of product and store data using Polars DataFrames.

fucntions:
    load_datastore: Loads the data store and returns an instance of DataStore.
"""

import polars as pl


class DataStore:
    def __init__(self):
        self._products: pl.DataFrame
        self._stores: pl.DataFrame

    def load(self):
        self._products = pl.read_csv("data/products.csv")
        self._stores = pl.read_csv("data/stores.csv")

    def stores_df(self) -> pl.DataFrame:
        if self._stores is None:
            raise ValueError("Stores data not loaded")
        return self._stores

    def products_df(self) -> pl.DataFrame:
        if self._products is None:
            raise ValueError("Products data not loaded")
        return self._products


def load_datastore():
    data_store = DataStore()
    data_store.load()
    return data_store
