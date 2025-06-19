import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from .costoptim import find_params
from .types import DamagePredictionModel, FraudDetectionModel


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

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.clf.predict_proba(X)[:, 1]
        preds = (y_probs > self.threshold).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


class DamageRegressor(DamagePredictionModel):
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        damage = y[:, 1]
        idx = np.where(damage > 0.0)
        self.clf.fit(X[idx], damage[idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)


class FraudDetectorWithDamagePrediction(FraudDetectionModel):
    def __init__(
        self,
        name: str,
        clf,
        regressor,
        cost_function,
    ):
        self._name = name
        self.clf = clf
        self.regressor = regressor
        self.cost_function = cost_function

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
        labels = y_train[:, 0]
        self.clf.fit(X_train, labels)

        damage = y_train[:, 1]
        self.regressor.fit(X_train, damage)

        ps = self.predict_proba(X_train)
        ds = self.regressor.predict(X_train)
        malus = find_params(ps, ds, labels, damage) * 1.3
        print(f"malus {malus}")
        self.cost_function = lambda p, d: p > malus / (5 + malus + d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_probs = self.clf.predict_proba(X)[:, 1]
        damage = self.regressor.predict(X)
        return self.cost_function(y_probs, damage)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


def cost_fn(probs, damage, malus):
    bonus = 5
    bewertung = probs > malus / (bonus + malus + damage)
    return bewertung


def get_xgb_clf_with_reg() -> FraudDetectionModel:
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=0.04,
        min_child_weight=4,
        objective="binary:logistic",
    )

    best_params_reg = {
        "n_estimators": 291,
        "max_depth": 7,
        "learning_rate": 0.015007267730048334,
        "subsample": 0.9064202506084049,
        "colsample_bytree": 0.7480099831239908,
        "gamma": 4.22979445500876,
        "reg_alpha": 0.6348383668703481,
        "reg_lambda": 4.326244708355912,
        "min_child_weight": 6,
        "scale_pos_weight": 28,
        "objective": "reg:squarederror",
    }

    regressor = XGBRegressor(**best_params_reg)

    regressor = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=0.05,
        objective="reg:squarederror",
    )

    return FraudDetectorWithDamagePrediction(
        "XGB Combinded Classifier", clf, regressor, cost_fn
    )


def get_lgmb_clf_with_reg() -> FraudDetectionModel:
    clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        is_unbalance=True,
        verbosity=-1,
    )

    regressor = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)

    return FraudDetectorWithDamagePrediction("LGMB Combined", clf, regressor, cost_fn)


def get_xgb_clf_with_reg_from_params(
    clf_params: dict,
) -> FraudDetectionModel:
    clf_params["eval_metric"] = "logloss"
    clf = XGBClassifier(**clf_params)

    regressor = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
        reg_alpha=1.0,
        reg_lambda=10.0,
    )

    return FraudDetectorWithDamagePrediction(
        "XGB Combined with Reg from params", clf, regressor, cost_fn
    )
