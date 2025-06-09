import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from .costoptim import find_params
from .types import DamagePredictionModel, FraudDetectionModel


class FraudDetector(FraudDetectionModel):
    def __init__(
        self,
        clf,
        threshold: float = 0.5,
    ):
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
        self.clf.fit(X_train, y_train[:, 0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.clf.predict_proba(X)[:, 1]
        preds = (y_probs > self.threshold).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


class DamageRegressor(DamagePredictionModel):
    def __init__(
        self,
        clf,
        threshold: float = 0.5,
    ):
        self.clf = clf
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        damage = y[:, 1]
        idx = np.where(damage > 0.0)
        self.clf.fit(X[idx], damage[idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)


class FraudDetectorWithDamagePrediction(FraudDetectionModel):
    def __init__(self, clf, regressor, cost_function):
        self.clf = clf
        self.regressor = regressor
        self.cost_function = cost_function

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        # fit only on label
        labels = y_train[:, 0]
        self.clf.fit(X_train, labels)

        damage = y_train[:, 1]
        self.regressor.fit(X_train, damage)

        ps = self.predict_proba(X_train)[:, 1]
        ds = self.regressor.predict(X_train)
        malus = find_params(ps, ds, labels, damage) * 1.3
        self.cost_function = lambda p, d: p > malus / (5 + malus + d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.clf.predict_proba(X)[:, 1]
        damage = self.regressor.predict(X)
        return self.cost_function(y_probs, damage)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


def get_xgb(
    scale_pos_weight: int,
) -> FraudDetector:
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        reg_alpha=1.0,
        reg_lambda=2.0,
    )
    return FraudDetector(clf, threshold=0.90)


def cost_fn(probs, damage, malus):
    bonus = 5
    bewertung = probs > malus / (bonus + malus + damage)
    return bewertung


def get_xgb_clf_with_reg(
    scale_pos_weight: int,
) -> FraudDetectionModel:
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        reg_alpha=1.0,
        reg_lambda=2.0,
    )

    regressor = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
        reg_alpha=1.0,
        reg_lambda=10.0,
    )

    return FraudDetectorWithDamagePrediction(clf, regressor, cost_fn)


def get_lgmb() -> FraudDetectionModel:
    clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        is_unbalance=True,
        verbosity=-1,
    )
    return FraudDetector(clf, threshold=0.95)


def get_lgmb_clf_with_reg() -> FraudDetectionModel:
    clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        is_unbalance=True,
        verbosity=-1,
    )

    regressor = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)

    return FraudDetectorWithDamagePrediction(clf, regressor, cost_fn)
