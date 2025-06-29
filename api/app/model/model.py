"""
model.py

This module contains the main logic for fraud detection using a machine learning model and static rules.

classes:
    FraudDetector: Main class for fraud detection, which uses a machine learning model and static rules to detect fraud.

functions:
    load_model: Load the fraud detection model and return an instance of FraudDetector.
"""

import logging

import polars as pl

from app.schemas import Explanation, FraudPrediction, FraudPredictionRequest
from app.version import __version__ as VERSION

from . import staticrules
from .ml_model import FraudDetectionModel, PredictionResult, load_models
from .staticrules import StaticRulesHandler
from .transform import transform

# transformation code needs a transaction_id to join transaction and lines
TRANSACTION_ID = 42

log = logging.getLogger("uvicorn.app")


def load_model(config):
    """
    Load the fraud detection model and return an instance of FraudDetectionModel.
    """
    classifier_model_path = config["model"]["classifier_path"]
    regression_model_path = config["model"]["regressor_path"]
    encoder_path = config["model"]["encoder_path"]
    cost_fp = config["costfunction"]["cost_false_positive"]
    gain_tp = config["costfunction"]["gain_true_positive"]
    clf, reg, encoder = load_models(
        classifier_model_path, regression_model_path, encoder_path
    )
    model = FraudDetectionModel(clf, reg, encoder, cost_fp=cost_fp, gain_tp=gain_tp)
    static_rules = StaticRulesHandler(config)
    return FraudDetector(model, static_rules)


class FraudDetector:
    """
    Class for fraud detection, that uses a machine learning model and static rules to detect fraud.
    It is the main entry point for fraud detection requests.
    It handles the request, checks static rules, and predicts possible fraud using the model.
    """

    def __init__(self, model: FraudDetectionModel, static_rules: StaticRulesHandler):
        self.model = model
        self.static_rules = static_rules

    def detect_fraud(
        self,
        request: FraudPredictionRequest,
        stores_df: pl.DataFrame,
        products_df: pl.DataFrame,
    ) -> FraudPrediction:
        # Transacation without lines -> indicator for fraud
        # check necessary before transforming the request
        if len(request.transaction_lines) == 0:
            return staticrules.handle_no_lines(request)
        request_df = _to_dataframe(request, stores_df, products_df)
        log.info("Check static rules")
        check_result = self.static_rules.check_static_rules(
            request_df, request, products_df
        )
        if check_result:
            return check_result
        log.info("Predicting fraud with model")
        prediction_result = self.model.predict(request_df)
        return _map_prediction_result(prediction_result)


def _map_prediction_result(result: PredictionResult) -> FraudPrediction:
    """
    Map the prediction result to FraudPrediction return value.
    """
    explanation = None
    if result.fraud and result.feature_importances:
        feature_importance = result.feature_importances
        reasons = "\n - ".join(
            [
                f"Merkmal {feature} hat den Wert {value:.2f} ({shap:.2f} Einfluss)"
                for feature, value, shap in zip(
                    feature_importance["feature"],
                    feature_importance["value"],
                    feature_importance["shap_value"],
                )
            ]
        )
        explanation = Explanation(
            human_readable_reason=f"Das Modell hat einen möglichen Betrugsfall erkannt: {reasons}",
        )
    elif result.fraud:
        explanation = Explanation(
            human_readable_reason="Das Modell hat einen möglichen Betrugsfall erkannt.",
        )

    return FraudPrediction(
        version=VERSION,
        is_fraud=result.fraud,
        fraud_proba=result.probability,
        estimated_damage=result.damage,
        explanation=explanation,
    )


def _to_dataframe(
    request: FraudPredictionRequest, stores_df: pl.DataFrame, products_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Convert the FraudPredictionRequest to a Polars DataFrame for processing.
    """
    n_lines = len(request.transaction_lines)
    lines = pl.DataFrame(
        [
            {**line.model_dump(), "product_id": str(line.product_id)}
            for line in request.transaction_lines
        ]
    ).with_columns(
        pl.lit(TRANSACTION_ID).alias("transaction_id"),
    )

    transaction = pl.DataFrame(
        {
            **request.transaction_header.model_dump(),
            "store_id": str(request.transaction_header.store_id),
        }
    ).with_columns(
        pl.lit(TRANSACTION_ID).alias("id"),
        pl.lit(n_lines).alias("n_lines"),
        # need to be set for transform
        pl.lit("UNKNOWN").alias("label"),
        pl.lit(0.0).alias("damage"),
        # make sure customer_feedback has not type Null
        pl.col("customer_feedback").cast(pl.Int64).alias("customer_feedback"),
    )

    transaction_df = transform(transaction, lines, stores_df, products_df)
    return transaction_df
