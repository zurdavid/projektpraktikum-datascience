import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy.sql.operators import op
from xgboost import XGBClassifier

from fraud_detection.models.costoptim import find_optimal_threshhold

from .types import FraudDetectionModel


class FraudDetector(FraudDetectionModel):
    def __init__(
        self,
        name: str,
        clf,
        threshold: float = 0.5,
        optimize_threshold: bool = False,
    ):
        self._name = name
        self.clf = clf
        self.threshold = threshold
        self.optimize_threshold = optimize_threshold

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

        if self.optimize_threshold:
            probs = self.clf.predict_proba(X_train)[:, 1]
            threshold = find_optimal_threshhold(probs, y_train[:, 0], y_train[:, 1])
            self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.clf.predict_proba(X)[:, 1]
        preds = (y_probs > self.threshold).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


class NoScaler:
    def fit() -> None:
        pass

    def fit_transform(self, X: np.ndarray, _) -> np.ndarray:
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


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
    name = "Logistic Regression"
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga")
    clf = Pipeline(
        [
            ("scaler", RobustScaler()),
            (name, clf),
        ]
    )
    return FraudDetector("Logistic Regression", clf, optimize_threshold=True)


def get_decsion_tree() -> FraudDetectionModel:
    clf = DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
    )
    return FraudDetector("Decision Tree Classifier", clf, optimize_threshold=True)


def get_random_forest() -> FraudDetectionModel:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
    )
    return FraudDetector("Random Forest Classifier", clf, optimize_threshold=True)
