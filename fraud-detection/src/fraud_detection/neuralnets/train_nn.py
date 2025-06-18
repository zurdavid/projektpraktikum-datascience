import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import batch_norm, nn, optim
from torch.utils.data import DataLoader, Sampler, TensorDataset
from xgboost import XGBRegressor

from fraud_detection.models.types import DamagePredictionModel, FraudDetectionModel

from ..models.costoptim import find_optimal_threshhold, find_params
from .loss import FocalLoss, PenalizedLoss, WertkaufLoss
from .model import FFNN, load_model, train_classifier, train_regression

device = "cpu"


class Scheduler:
    def __init__(self, scheduler_type: str, scheduler):
        self.scheduler_type = scheduler_type
        self.scheduler = scheduler

    def type(self) -> str:
        return self.scheduler_type

    def step(self):
        if self.scheduler_type == "OneCycleLR":
            self.scheduler.step()
        elif self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler.step(metrics=None)


class NNFraudDetector(FraudDetectionModel):
    def __init__(
        self,
        model: FFNN,
        name: str,
        loss_fn,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
        scheduler=None,
        threshold=0.5,
    ):
        if scheduler is None:
            scheduler = Scheduler("None", None)
        self.model = model
        self._name = name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
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
        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).squeeze(1)

        X_test = self.scaler.transform(X_test)  # pyright: ignore
        Xt_test = torch.tensor(X_test, dtype=torch.float32)
        yt_test = torch.tensor(y_test, dtype=torch.float32).squeeze(1)

        dataset = TensorDataset(Xt, yt)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(Xt_test, yt_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        if self.scheduler.type() == "OneCycleLR":
            self.scheduler.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=1e-2,
                steps_per_epoch=len(train_loader),
                epochs=self.epochs,
            )

        self.model, _ = train_classifier(
            self.model,
            train_loader,
            test_loader,
            self.epochs,
            self.optimizer,
            self.loss_fn,
            self.scheduler,
        )
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self._predict_proba(X)
        preds = (probs > self.threshold).long().view(-1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._predict_proba(X)
        return probs.cpu().numpy()

    def _predict_proba(self, X: np.ndarray) -> torch.Tensor:
        X_transformed = self.scaler.transform(X)
        Xt = torch.tensor(X_transformed, dtype=torch.float32)
        outputs = self.model.predict(Xt)
        probs = torch.sigmoid(outputs)
        return probs


class DamageRegressor(DamagePredictionModel):
    def __init__(
        self,
        model: FFNN,
        name: str,
        loss_fn,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
    ):
        self.model = model
        self._name = name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    def name(self) -> str:
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        print(y.shape)

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y[:, 1], dtype=torch.float32)

        dataset = TensorDataset(Xt, yt)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = train_regression(
            self.model,
            train_loader,
            self.epochs,
            self.optimizer,
            self.loss_fn,
        )
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_transformed = self.scaler.transform(X)
        Xt = torch.tensor(X_transformed, dtype=torch.float32)
        return self.model(Xt)


def getNN_regressor(input_size: int):
    inner_layers = [64, 32, 16]

    model = FFNN(input_size, 1, inner_layers, dropout=0.4)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-1, weight_decay=1e-5)

    batch_size = 256
    epochs = 100

    return DamageRegressor(
        model, "NN Regressor", loss_fn, optimizer, batch_size, epochs
    )


class NNFraudDetectorWithDamagePrediction(FraudDetectionModel):
    def __init__(
        self,
        model: FFNN,
        name: str,
        loss_fn,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
        regressor,
    ):
        self.model = model
        self._name = name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.regressor = regressor

    def name(self) -> str:
        return self._name

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train)

        # Xt = torch.tensor(X_train, dtype=torch.float32)
        # yt = torch.tensor(y_train, dtype=torch.float32).squeeze(1)

        # X_test = self.scaler.transform(X_test)  # pyright: ignore
        # Xt_test = torch.tensor(X_test, dtype=torch.float32)
        # yt_test = torch.tensor(y_test, dtype=torch.float32).squeeze(1)

        # dataset = TensorDataset(Xt, yt)
        # train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # test_dataset = TensorDataset(Xt_test, yt_test)
        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # self.model = train_classifier(
        # self.model,
        # train_loader,
        # test_loader,
        # self.epochs,
        # self.optimizer,
        # self.loss_fn,
        # )

        # self.model, _ = load_model("models/model_final.pth", optim.AdamW)
        self.model, _ = load_model("models/model_epoch_10.pth", optim.AdamW)
        self.model.eval()

        probs = self.predict_proba(X_test).squeeze()
        threshold = find_optimal_threshhold(probs, y_test[:, 0], y_test[:, 1])
        print(f"Optimal threshold for {self.name()}: {threshold}")
        # threshold = 0.5
        self.cost_function = lambda p, d: p > threshold

        labels = y_train[:, 0]
        damage = y_train[:, 1]
        self.regressor.fit(X_train, damage)

        # ps = self.predict_proba(X_train).squeeze()
        # ds = self.regressor.predict(X_train)
        # malus = find_params(ps, ds, labels, damage)
        # print(f"malus {malus}")
        # # malus = 11
        # #self.cost_function = lambda p, d: p > malus / (5 + malus + d)
        # self.cost_function = lambda p, d: p > 0.75

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X).squeeze()
        damage = self.regressor.predict(X)
        return self.cost_function(probs, damage)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._predict_proba(X)
        return probs.cpu().numpy()

    def _predict_proba(self, X: np.ndarray) -> torch.Tensor:
        X_transformed = self.scaler.transform(X)
        Xt = torch.tensor(X_transformed, dtype=torch.float32)
        outputs = self.model.predict(Xt)
        probs = torch.sigmoid(outputs)
        return probs


def getvanillaNN(input_size: int):
    inner_layers = [64]

    model = FFNN(input_size, 1, inner_layers, dropout=0.2)

    pos_weight = torch.tensor([2.0], dtype=torch.float).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # loss_fn = PenalizedLoss(pos_weight=pos_weight, false_positive_penalty=4.0)
    # loss_fn = WertkaufLoss(base_loss_weight=0.01, fp_penalty_weight=30.0, malus=100)
    # loss_fn = FocalLoss(alpha=0.25, gamma=1.5, pos_weight=pos_weight)

    # optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    batch_size = 256
    epochs = 15
    return NNFraudDetector(model, "NN Vanilla", loss_fn, optimizer, batch_size, epochs)


def getNN(input_size: int):
    # inner_layers = [64, 16, 8]
    # inner_layers = [64, 16]
    inner_layers = [64]
    # inner_layers = [64, 16]
    # inner_layers = [32, 8, 4]
    # inner_layers = [128, 64, 16, 16]
    # inner_layers = [256, 128, 65]
    # inner_layers = [32]

    model = FFNN(input_size, 1, inner_layers, dropout=0.4)

    pos_weight = torch.tensor([1.0], dtype=torch.float).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # loss_fn = PenalizedLoss(pos_weight=pos_weight, false_positive_penalty=4.0)
    # loss_fn = WertkaufLoss(base_loss_weight=0.01, fp_penalty_weight=30.0, malus=100)
    loss_fn = FocalLoss(alpha=0.1, gamma=2.0, pos_weight=pos_weight)

    # optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = Scheduler("OneCycleLR", None)
    batch_size = 256
    epochs = 12 

    return NNFraudDetector(
        model,
        "NN Classifier",
        loss_fn,
        optimizer,
        batch_size,
        epochs,
        scheduler=scheduler,
    )


def getNN2(input_size: int):
    # inner_layers = [64, 32, 16, 8]
    inner_layers = [128, 32, 16, 16]

    model = FFNN(input_size, 1, inner_layers, dropout=0.4)

    pos_weight = torch.tensor([10.0], dtype=torch.float).to(device)
    loss_fn = PenalizedLoss(pos_weight=pos_weight, false_positive_penalty=3.0)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    batch_size = 256
    epochs = 16

    regressor = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
    )

    return NNFraudDetectorWithDamagePrediction(
        model,
        "NN Combined Classifier",
        loss_fn,
        optimizer,
        batch_size,
        epochs,
        regressor,
    )
