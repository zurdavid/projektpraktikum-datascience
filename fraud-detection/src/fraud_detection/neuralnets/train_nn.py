import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from fraud_detection.models.types import DamagePredictionModel, FraudDetectionModel

from ..models.costoptim import find_params
from .loss import FocalLoss, PenalizedLoss, WertkaufLoss
from .model import FFNN, load_model, train_classifier, train_regression

device = "cpu"


class NNFraudDetector(FraudDetectionModel):
    def __init__(
        self,
        model: FFNN,
        loss_fn,
        optimizer: optim.Optimizer,
        scheduler,
        batch_size: int,
        epochs: int,
        threshold=0.8,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).squeeze(1)

        X_test = self.scaler.transform(X_test)  # pyright: ignore
        Xt_test = torch.tensor(X_test, dtype=torch.float32)
        yt_test = torch.tensor(y_test, dtype=torch.float32).squeeze(1)

        dataset = TensorDataset(Xt, yt)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(Xt_test, yt_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=self.epochs
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
        loss_fn,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y[:, 1], dtype=torch.float32).squeeze(1)

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
    inner_layers = [128, 32, 16, 16]

    model = FFNN(input_size, 1, inner_layers, dropout=0.3)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    batch_size = 256
    epochs = 1000

    return DamageRegressor(model, loss_fn, optimizer, batch_size, epochs)


class NNFraudDetectorWithDamagePrediction(FraudDetectionModel):
    def __init__(
        self,
        model: FFNN,
        loss_fn,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
        regressor,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.regressor = regressor

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.scaler = StandardScaler()
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

        self.model, opt = load_model("models/model_final.pth", optim.AdamW)
        self.model.eval()

        labels = y_train[:, 0]
        damage = y_train[:, 1]
        self.regressor.fit(X_train, damage)

        ps = self.predict_proba(X_train).squeeze()
        ds = self.regressor.predict(X_train)
        malus = find_params(ps, ds, labels, damage)
        print(f"malus {malus}")
        malus = 90
        self.cost_function = lambda p, d: p > malus / (5 + malus + d)

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


def getNN(input_size: int):
    inner_layers = [64, 32, 16, 8]
    # inner_layers = [128, 32, 16, 16]

    model = FFNN(input_size, 1, inner_layers, dropout=0.5)

    pos_weight = torch.tensor([3.0], dtype=torch.float).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # loss_fn = PenalizedLoss(pos_weight=pos_weight, false_positive_penalty=3.0)
    # loss_fn = WertkaufLoss(base_loss_weight=0.01, fp_penalty_weight=30.0, malus=100)
    #  loss_fn = FocalLoss(alpha=.25, gamma=2, pos_weight=pos_weight.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    batch_size = 128 
    epochs = 20 

    return NNFraudDetector(model, loss_fn, optimizer, scheduler, batch_size, epochs)


def getNN2(input_size: int):
    # inner_layers = [64, 32, 16, 8]
    inner_layers = [128, 32, 16, 16]

    model = FFNN(input_size, 1, inner_layers, dropout=0.4)

    pos_weight = torch.tensor([10.0], dtype=torch.float).to(device)
    loss_fn = PenalizedLoss(pos_weight=pos_weight, false_positive_penalty=3.0)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    batch_size = 256
    epochs = 10

    regressor = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
        reg_alpha=1.0,
        reg_lambda=10.0,
    )

    return NNFraudDetectorWithDamagePrediction(
        model, loss_fn, optimizer, batch_size, epochs, regressor
    )
