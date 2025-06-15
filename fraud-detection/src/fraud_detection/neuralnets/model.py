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
            layers.append(nn.LayerNorm(out_size))
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


def save_model(model: FFNN, optimizer: torch.optim.Optimizer, path: str):
    """Save model and optimizer state to a file."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                "input_size": model.network[0].in_features,
                "output_size": model.network[-1].out_features,
                "hidden_layers": [
                    layer.out_features
                    for layer in model.network
                    if isinstance(layer, nn.Linear)
                ][:-1],
                "dropout": next(
                    (m.p for m in model.network if isinstance(m, nn.Dropout)), 0.4
                ),
                "activation": type(model.network[1]),
            },
        },
        path,
    )
    print(f"=> Model saved to {path}")


def load_model(
    path: str, optimizer_func, device="cpu"
) -> tuple[FFNN, torch.optim.Optimizer]:
    """Load model and optimizer state from a file."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]

    model = FFNN(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        activation=config["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    optimizer = optimizer_func(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"=> Model loaded from {path}")
    return model, optimizer


def train_classifier(
    model: FFNN,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scheduler,
) -> tuple[FFNN, float]:
    train_losses = dict()

    model.to(device)

    print("=> Starting training")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_losses = []

        for X_, y_ in train_data:
            X, y = X_.to(device), y_.to(device)
            y, damage = y[:, 0], y[:, 1]

            optimizer.zero_grad()
            yhat = model(X)

            if hasattr(loss_fn, "loss_fn_takes_damage"):
                loss = loss_fn(yhat.view(-1), y, damage)
            else:
                loss = loss_fn(yhat.view(-1), y)

            loss.backward()

            optimizer.step()
            scheduler.step()

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

        # scheduler.step(metrics_test["loss"])

        elapsed = time.time() - start_time
        print(f"-- train and test duration: {elapsed}")
        print("\n" + "-" * 50 + "\n")

        save_model(model, optimizer, f"models/model_epoch_{epoch + 1}.pth")

    metrics_test = evaluate_classifier(model, test_data, loss_fn)

    print("save model")
    save_model(model, optimizer, "models/model_final.pth")
    return model, metrics_test["Bewertung"]


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
