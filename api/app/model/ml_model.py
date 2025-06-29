"""
ml_model.py

Module for loading and using machine learning models for fraud detection.

classes:
    FraudDetectionModel: Class for fraud detection using machine learning models.
    PredictionResult: Data class to hold the prediction results.

functions:
    load_models: Load the XGBoost models and encoder from specified file paths.
    cost_fn: Calculate the cost function for fraud detection.
"""

from dataclasses import dataclass

import joblib
import polars as pl
import shap

from .encoder import PolarsEncoder


def load_models(classifier_path: str, regressor_path: str, encoder_path: str) -> tuple:
    """
    Load the XGBoost models from the specified file paths.

    Args:
        classifier_path (str): Path to the fraud classifier model.
        regressor_path (str): Path to the damage regressor model.

    Returns:
        tuple: A tuple containing the loaded models (model1, model2).
    """
    clf = joblib.load(classifier_path)
    reg = joblib.load(regressor_path)

    encoder = PolarsEncoder(drop_first=True)
    encoder.load(encoder_path)
    return clf, reg, encoder


def cost_fn(
    fraud_probability, predicted_damage, cost_fp: float = 10.0, tain_tp: float = 5.0
):
    """
    Calculate the cost function for fraud detection, to determine if the cost of false positives is acceptable.

    Args:
        fraud_probability (array-like): Predicted probabilities of fraud.
        damage (array-like): Predicted damage values.
        cost_fp: Cost of false positive.
        tain_tp: Gain from true positive.

    Returns:
        bool: True if the cost of false positive is less than the expected gain from true positive, otherwise False.
    """
    return fraud_probability > cost_fp / (tain_tp + cost_fp + predicted_damage)


@dataclass
class PredictionResult:
    """
    Data class to hold the prediction results.

    Attributes:
        fraud (bool): Indicates if fraud is detected.
        damage (float): Predicted damage amount.
    """

    fraud: bool
    probability: float
    damage: float
    feature_importances: dict | None = None


class FraudDetectionModel:
    def __init__(
        self,
        clf_model,
        reg_model,
        encoder: PolarsEncoder,
        cost_fp: float = 10.0,
        gain_tp: float = 5.0,
    ):
        self.clf = clf_model
        self.reg = reg_model
        self.encoder = encoder
        self.cost_fp = cost_fp
        self.gain_tp = gain_tp
        self.explainer = shap.TreeExplainer(self.clf)

    def predict(self, X) -> PredictionResult:
        """
        Predict fraud and damage using the loaded models.

        Args:
            X (array-like): Input features for prediction.

        Returns:
            tuple: A tuple containing the fraud prediction and damage prediction.
        """
        Xt = self.encoder.transform(X)
        fraud_probability = self.clf.predict_proba(Xt)[:, 1]
        damage_prediction = self.reg.predict(Xt)
        prediction = cost_fn(
            fraud_probability, damage_prediction, self.cost_fp, self.gain_tp
        )

        # Build feature importances
        shap_values = self.explainer.shap_values(Xt).flatten()
        feature_names = Xt.columns
        feature_importances = (
            (
                pl.DataFrame(
                    {
                        "feature": feature_names,
                        "value": Xt.to_numpy().flatten(),
                        "shap_value": shap_values,
                    }
                )
                .with_columns(abs_shap=pl.col("shap_value").abs())
                .sort("abs_shap", descending=True)
            )
            .head(5)
            .select("feature", "value", "shap_value")
            .to_dict(as_series=False)
        )

        return PredictionResult(
            prediction.item(),
            fraud_probability.item(),
            damage_prediction.item(),
            feature_importances,
        )
