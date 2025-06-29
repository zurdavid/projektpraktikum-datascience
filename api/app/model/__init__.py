"""
model Module for fraud detection.

classes:
    FraudDetector: Main class for fraud detection, which uses a machine learning model and static rules to detect fraud.

fucntions:
    load_model: Load the fraud detection model and return an instance of FraudDetector.
"""

from .model import FraudDetector, load_model

___all__ = [
    "load_model",
    "FraudDetector",
]
