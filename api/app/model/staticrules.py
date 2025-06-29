"""
staticrules.py

Module for handling static rules in fraud detection.

Classes:
    StaticRulesHandler: Handles static rules for fraud detection.

functions:
    handle_no_lines: Handles cases where no transaction lines are provided.
"""

import logging

import polars as pl

from app.schemas import (
    Explanation,
    FraudPrediction,
    FraudPredictionRequest,
    TransactionLine,
)
from app.version import __version__ as VERSION

log = logging.getLogger("uvicorn.app")


class StaticRulesHandler:
    """
    Class handles static rules for fraud detection.
    These rules are applied before the machine learning model.
    They are used to detect fraud based on static rules that do not require a model.
    """

    def __init__(self, config):
        """
        Initialize the StaticRulesHandler with configuration.

        Args:
            config: Configuration dictionary containing store information and no discount categories.
        """
        self.name = "StaticRulesHandler"
        if not config["discounts"]["enable_excluded_categories"]:
            self.no_discount = {}
            log.info("StaticRulesHandler initialized without excluded categories")
            return

        self.no_discount = {
            s["id"]: s["categories_excluded_from_discount"] for s in config["stores"]
        }
        log.info(
            "StaticRulesHandler initialized with no discount categories: %s",
            config["stores"],
        )

    def check_static_rules(
        self,
        request_df: pl.DataFrame,
        request: FraudPredictionRequest,
        products_df: pl.DataFrame,
    ) -> FraudPrediction | None:
        """
        Check static rules for fraud detection.

        Args:
            request_df: DataFrame containing the transaction.
            request: FraudPredictionRequest object containing the transaction header and lines.

        Returns:
            FraudPrediction object if a static rule is triggered, None otherwise.
        """
        if request_df["has_missing"].any():
            return _handle_missing()
        if request_df["has_unscanned"].any():
            return _handle_unscanned(request, products_df)
        if request_df["has_positive_price_difference"].any():
            return self._handle_positive_price_difference(
                request_df, request, products_df
            )
        return None

    def _handle_positive_price_difference(
        self,
        request_df: pl.DataFrame,
        request: FraudPredictionRequest,
        products: pl.DataFrame,
    ) -> FraudPrediction | None:
        """
        Handle lines with a positive price difference. Inserted by camera when it detects a product that was scanned but the price is different from the expected price.
        """
        log.info("Discount used")
        store_id = str(request.transaction_header.store_id)
        no_discount_cats = self.no_discount.get(store_id, [])
        if len(no_discount_cats) == 0:
            log.info("No discount categories configured for store %s", store_id)
            return None

        lines = (
            pl.DataFrame(
                [
                    {**line.model_dump(), "product_id": str(line.product_id)}
                    for line in request.transaction_lines
                ]
            )
            .join(
                products,
                left_on="product_id",
                right_on="id",
                how="left",
            )
            .filter(pl.col("category").is_in(no_discount_cats))
            .with_columns(
                (
                    pl.col("price") * pl.col("pieces_or_weight") - pl.col("sales_price")
                ).alias("price_difference")
            )
            # threshold to filter out minor differences
            .filter(pl.col("price_difference") > 0.05)
        )
        print(lines.select("product_id", "category", "price_difference"))
        if len(lines) > 0:
            log.info("Dicsount fraud detected")
            damage = lines["price_difference"].sum()
            offending_products = lines["product_id"].to_list()
            offending_categories = lines["category"].to_list()
            offending_products = [
                f"{p} ({c})" for p, c in zip(offending_products, offending_categories)
            ]

            explanation = Explanation(
                human_readable_reason="Rabatt auf von Rabatt ausgenommene Produkte angewendet",
                offending_products=offending_products,
            )
            fp = FraudPrediction(
                version=VERSION,
                is_fraud=True,
                fraud_proba=1.0,
                estimated_damage=damage,
                explanation=explanation,
            )
            return fp

        return None


def handle_no_lines(_: FraudPredictionRequest) -> FraudPrediction:
    """
    Handle cases where no transaction lines are provided.

    Args:
        _: FraudPredictionRequest object (currently not used in this function).

    Returns:
        FraudPrediction object indicating the absence of transaction lines.
    """
    log.info("No transaction lines provided")
    explanation = Explanation(
        human_readable_reason="Keine Transaktionszeilen vorhanden",
    )
    fp = FraudPrediction(
        version=VERSION,
        is_fraud=True,
        fraud_proba=None,
        explanation=explanation,
    )
    return fp


def _handle_unscanned(
    request: FraudPredictionRequest, products_df: pl.DataFrame
) -> FraudPrediction:
    """
    Handle lines with unscanned products. Inserted by camera when it detects a recognizable product that was not scanned. Damage is calculated based on the price of the unscanned products.

    """
    log.info("Unscanned products detected")
    unscanned = _unscanned_products(request.transaction_lines)
    damage = _calculate_damage_from_unscanned_products(unscanned, products_df)
    explanation = Explanation(
        human_readable_reason="Kamera hat nicht gescannte Produkte erkannt",
        offending_products=[str(p) for p in unscanned],
    )
    fp = FraudPrediction(
        version=VERSION,
        is_fraud=True,
        fraud_proba=1.0,
        estimated_damage=damage,
        explanation=explanation,
    )
    return fp


def _handle_missing() -> FraudPrediction:
    """
    Handle lines with missing product details. Inserted by camera when it detects a product that was not scanned but can't determine a product-id. Damage is unknown.
    """
    log.info("Missing products detected")
    explanation = Explanation(
        human_readable_reason="Kamera hat nicht gescannte Produkte erkannt",
        offending_products=["unbekannt"],
    )
    fp = FraudPrediction(
        version=VERSION,
        is_fraud=True,
        fraud_proba=1.0,
        explanation=explanation,
    )
    return fp


def _unscanned_products(lines: list[TransactionLine]) -> list[str]:
    """
    Check if there are any unscanned products in the transaction lines.

    Args:
        lines: List of TransactionLine objects representing the transaction lines.

    Returns:
        A list of product IDs that were unscanned, or None if there are no unscanned products.
    """
    unscanned_products = list(
        map(
            lambda line: str(line.product_id),
            filter(
                lambda line: line.was_voided
                and not line.camera_product_similar
                and line.sales_price == 0,
                lines,
            ),
        )
    )
    return unscanned_products


def _calculate_damage_from_unscanned_products(
    unscanned_products: list[str], products: pl.DataFrame
) -> float:
    """
    Calculate the estimated damage from unscanned products.
    Args:
        unscanned_products: List of product IDs that were unscanned.
    Returns:
        The total estimated damage from unscanned products.
    """
    damage = (
        products.filter(pl.col("id").is_in(unscanned_products))
        .select(pl.col("price"))
        .to_series()
        .sum()
    )
    return damage
