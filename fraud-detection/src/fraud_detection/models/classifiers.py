import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.models.costoptim import find_optimal_threshhold

from .types import FraudDetectionModel


class FraudDetector(FraudDetectionModel):
    def __init__(
        self,
        name: str,
        clf,
        threshold: float = 0.5,
    ):
        self._name = name
        self.clf = clf
        self.threshold = threshold

    def name(self) -> str:
        return self._name

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        # fit only on label
        self.clf.fit(X_train, y_train[:, 0])

        probs = self.clf.predict_proba(X_test)[:, 1]
        threshold = find_optimal_threshhold(probs, y_test[:, 0], y_test[:, 1])
        print(f"Optimal threshold for {self.name()}: {threshold}")
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.clf.predict_proba(X)[:, 1]
        preds = (y_probs > self.threshold).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


class LinearFraudDetector(FraudDetector):
    def __init__(
        self,
        name: str,
        clf,
        threshold: float = 0.5,
    ):
        self._name = name
        self.clf = clf
        self.threshold = threshold

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        # fit only on label
        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train)

        self.clf.fit(X_train, y_train[:, 0])

        probs = self.clf.predict_proba(X_test)[:, 1]
        threshold = find_optimal_threshhold(probs, y_test[:, 0], y_test[:, 1])
        print(f"Optimal threshold for {self.name()}: {threshold}")
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.predict_proba(X)
        preds = (y_probs > self.threshold).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self.scaler.transform(X)
        return self.clf.predict_proba(Xt)[:, 1]


def get_lin_reg() -> FraudDetectionModel:
    clf = LogisticRegression(solver="saga", max_iter=5000)
    return FraudDetector("Logistic Regression", clf)


def get_xgb() -> FraudDetector:
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=1,
        reg_alpha=0.0,
        reg_lambda=0.04,
        min_child_weight=4,
        objective="binary:logistic",
    )
    return FraudDetector("XGBClassifier", clf)


def get_lgmb() -> FraudDetectionModel:
    clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=45,
        verbosity=-1,
    )
    return FraudDetector("LGMB Classifier", clf)


def get_xgb_simple() -> FraudDetectionModel:
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
    )
    return FraudDetector("XGB Simple Classifier", clf, threshold=0.50)


def get_gradient_boosting():
    clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    return FraudDetector("GradientBoosting Classifier", clf, threshold=0.5)


def get_catboost() -> FraudDetectionModel:
    clf = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        verbose=0,
        random_seed=42,
    )
    return FraudDetector("CatBoost Classifier", clf, threshold=0.5)


def get_logistic_regression() -> FraudDetectionModel:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    # clf = LogisticRegression(
    # penalty="l2",
    # C=1.0,
    # solver="lbfgs",
    # max_iter=1000,
    # class_weight="balanced",
    # )
    return LinearFraudDetector("Logistic Regression", clf, threshold=0.95)


def get_random_forest() -> FraudDetectionModel:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
    )
    return FraudDetector("Random Forest Classifier", clf, threshold=0.75)
