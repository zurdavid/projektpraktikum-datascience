from typing import Protocol

import numpy as np


class FraudDetectionModel(Protocol):
    """Protocol for classification models used in fraud detection."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Train the model using the provided training data and evaluate on test data.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test feature matrix for validation.
            y_test (np.ndarray): Test labels for validation.
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels for the given samples.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities for the given samples.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        ...


class DamagePredictionModel(Protocol):
    """Protocol for regression models used in damage prediction."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the provided training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training target values.
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict damage values for the given samples.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        ...
