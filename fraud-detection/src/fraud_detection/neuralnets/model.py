import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .. import metrics

device = "cpu"


class FFNN(nn.Module):
    """Feed-Forward Neural Network (FFNN) for classification or regression tasks."""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers: list[int],
        dropout=0.4,
        activation=nn.LeakyReLU,
    ):
        super(FFNN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(activation())

        # Hidden layers
        for in_size, out_size in zip(hidden_layers, hidden_layers[1:], strict=False):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            return self.network(x)


def train_classifier(
    model: FFNN,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
) -> FFNN:
    train_losses = dict()
    model.to(device)

    print("=> Starting training")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_losses = []

        for X, y in train_data:
            X, y = X.to(device), y.to(device)
            y, damage = y[:, 0], y[:, 1]

            optimizer.zero_grad()
            yhat = model(X)

            if hasattr(loss_fn, "loss_fn_takes_damage"):
                loss = loss_fn(yhat.view(-1), y, damage)
            else:
                loss = loss_fn(yhat.view(-1), y)

            loss.backward()

            optimizer.step()

            epoch_losses.append(loss.detach().item())

        mean_train_loss = torch.tensor(epoch_losses).mean().item()
        train_losses[epoch] = mean_train_loss
        elapsed = time.time() - start_time
        print(
            f"=> epoch: {epoch + 1}, loss: {mean_train_loss:.4f}, duration: {elapsed}"
        )
        metrics_train = evaluate_classifier(model, train_data, loss_fn)
        metrics_test = evaluate_classifier(model, test_data, loss_fn)
        metrics.print_metrics_comp(metrics_train, metrics_test)

        elapsed = time.time() - start_time
        print(f"-- train and test duration: {elapsed}")
        print("\n" + "-" * 50 + "\n")
    return model


def evaluate_classifier(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    all_damages = []
    all_losses = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y, damage = y[:, 0], y[:, 1]

            outputs = model(X)
            if hasattr(loss_fn, "loss_fn_takes_damage"):
                loss = loss_fn(outputs.view(-1), y, damage)
            else:
                loss = loss_fn(outputs.view(-1), y)

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long().view(-1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y.long().cpu().numpy())
            all_damages.append(damage)
            all_losses.append(loss.cpu().item())

    loss = np.array(all_losses).mean()
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    damages = np.concatenate(all_damages, axis=0)

    bew = metrics.bewertung(preds, labels, damages)
    bew["loss"] = loss
    return bew


def l1_regularization(model, lambda_l1=1e-5):
    l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    return lambda_l1 * l1_norm


def train_regression(
    model: FFNN,
    train_data: DataLoader,
    epochs: int,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
):
    train_losses = dict()
    model.to(device)

    print("=> Starting training")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_losses = []

        for X, y in train_data:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            yhat = model(X)
            loss = loss_fn(yhat.view(-1), y) + l1_regularization(model, lambda_l1=1e-2)

            loss.backward()

            optimizer.step()

            epoch_losses.append(loss.detach().item())

        mean_train_loss = torch.tensor(epoch_losses).mean().item()
        train_losses[epoch] = mean_train_loss
        elapsed = time.time() - start_time

        if epoch % 10 == 0:
            print(
                f"=> epoch: {epoch + 1}, loss: {mean_train_loss:.4f}, duration: {elapsed}"
            )
            evaluate_regression_model(model, train_data)

            elapsed = time.time() - start_time
            print(f"-- train and test duration: {elapsed}")
            print("\n" + "-" * 50 + "\n")

    return model


def evaluate_regression_model(model, dataloader, device="cpu", final=False):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            predictions.append(outputs.view(-1).cpu())
            targets.append(labels.view(-1).cpu())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    m = metrics.regression(predictions, targets)
    for text, value in m.items():
        print(f"{text:<10} {value:>6.2f}")
