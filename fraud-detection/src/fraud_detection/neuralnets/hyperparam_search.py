import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import Trial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .. import data_loader, metrics

DEVICE = "cpu"
THRESHOLD = 0.5
SAMPLER_PATH = "sampler.pkl"


def define_model(trial: Trial, input_size: int):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []

    norm_dict = {
        "none": nn.Identity,
        "batchnorm": nn.BatchNorm1d,
        "layernorm": nn.LayerNorm,
    }

    activation_dict = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "SiLU": nn.SiLU}

    norm = trial.suggest_categorical("norm", list(norm_dict.keys()))
    activation = trial.suggest_categorical("activataion", list(activation_dict.keys()))

    in_features = input_size
    for i in range(n_layers):
        out_features = trial.suggest_categorical(
            "n_units_l{}".format(i), [4, 16, 32, 64, 128, 256]
        )
        layers.append(nn.Linear(in_features, out_features))
        layers.append(norm_dict[norm](out_features))
        layers.append(activation_dict[activation]())
        p = trial.suggest_float("dropout_{}".format(i), 0.1, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))

    return nn.Sequential(*layers)


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_data: DataLoader,
):
    model.train()

    epoch_losses = []
    for X_, y_ in train_data:
        X, y = X_.to(DEVICE), y_.to(DEVICE)
        y, damage = y[:, 0], y[:, 1]

        optimizer.zero_grad()
        yhat = model(X)

        if hasattr(loss_fn, "loss_fn_takes_damage"):
            loss = loss_fn(yhat.view(-1), y, damage)
        else:
            loss = loss_fn(yhat.view(-1), y)

        loss.backward()

        optimizer.step()
        # scheduler.step()
        epoch_losses.append(loss.detach().item())

    return np.sum(epoch_losses)


def evaluate_classifier(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    all_preds = []
    all_labels = []
    all_damages = []
    all_losses = []

    with torch.no_grad():
        for X_, y_ in data_loader:
            X, y = X_.to(DEVICE), y_.to(DEVICE)

            y, damage = y[:, 0], y[:, 1]

            outputs = model(X)
            if hasattr(loss_fn, "loss_fn_takes_damage"):
                loss = loss_fn(outputs.view(-1), y, damage)
            else:
                loss = loss_fn(outputs.view(-1), y)

            probs = torch.sigmoid(outputs)
            predicted = (probs > THRESHOLD).long().view(-1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y.long().cpu().numpy())
            all_damages.append(damage)
            all_losses.append(loss.cpu().item())

    loss = np.array(all_losses).mean()
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    damages = np.concatenate(all_damages, axis=0)

    return metrics.bewertung(preds, labels, damages)


def train_and_eval_single_run(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
) -> float:
    score = []

    for i in range(n_epochs):
        train_model(model, optimizer, loss_fn, train_loader)
        if i >= n_epochs - 3:
            bew = evaluate_classifier(model, test_loader, loss_fn)
            score.append(bew["Bewertung"])

    return np.mean(score).astype("float")


def train_and_eval(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
) -> float:
    scores = []
    for _ in range(3):
        score = train_and_eval_single_run(
            model, optimizer, loss_fn, train_loader, test_loader, n_epochs
        )
        scores.append(score)
    return np.mean(scores).astype("float")


def objective(
    trial: Trial, dataset: Dataset, test_dataset: Dataset, input_size: int
) -> float:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = define_model(trial, input_size=input_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
    )

    pos_weight = trial.suggest_float("pos_weight", 1.0, 20.0)
    pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20, 30])

    return train_and_eval(
        model, optimizer, loss_fn, train_loader, test_loader, n_epochs
    )


def get_sampler(sampler_path: str, seed: int) -> BaseSampler:
    """
    Load an Optuna sampler from a file if it exists.
    Otherwise, create a new TPESampler with the given seed and save it.

    Args:
        sampler_path (str): Path to the sampler pickle file.
        seed (int): Random seed for the TPESampler.

    Returns:
        BaseSampler: The loaded or newly created sampler.
    """
    if os.path.exists(sampler_path):
        print(f"Load sampler from {sampler_path}")
        return pickle.load(open("sampler.pkl", "rb"))
    else:
        print(f"Created new TPESampler and saved to {sampler_path}")
        return TPESampler(seed=seed)


def save_sampler(study, path) -> None:
    with open(path, "wb") as fout:
        pickle.dump(study.sampler, fout)


def save_sampler_callback(sampler, path: str):
    def callback(study, trial):
        with open(path, "wb") as f:
            pickle.dump(sampler, f)
        print(f"Sampler saved after trial {trial.number}")

    return callback


def optimize(path: Path, n_trials: int, seed: int = 42) -> None:
    X, targets = data_loader.load_data_np(path)
    input_size = X.shape[1]

    X_train, X_test, y_train, y_test = map(
        np.asarray, train_test_split(X, targets, test_size=0.2, random_state=seed)
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).squeeze(1)

    X_test = scaler.transform(X_test)  # pyright: ignore
    Xt_test = torch.tensor(X_test, dtype=torch.float32)
    yt_test = torch.tensor(y_test, dtype=torch.float32).squeeze(1)

    dataset = TensorDataset(Xt, yt)
    test_dataset = TensorDataset(Xt_test, yt_test)

    def wrapped_objective(trial):
        return objective(trial, dataset, test_dataset, input_size)

    sampler = get_sampler(SAMPLER_PATH, seed)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "neuralnet_clf"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(
        wrapped_objective,
        show_progress_bar=True,
        n_trials=n_trials,
        callbacks=[save_sampler_callback(sampler, SAMPLER_PATH)],
    )

    save_sampler(study, path)
