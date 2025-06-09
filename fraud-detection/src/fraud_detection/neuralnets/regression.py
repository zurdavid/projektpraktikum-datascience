from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..data_loader import load_data_for_regression
from .model import FFNN, train_regression

device = "cpu"


def start(datapath: Path):
    torch.manual_seed(43)

    X_unscaled, y = load_data_for_regression(datapath)

    print(X_unscaled.shape)
    print(X_unscaled)

    print("Fit scaler")
    scaler = StandardScaler()
    print("Scale data")
    X = scaler.fit_transform(X_unscaled)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).squeeze(1)

    dataset = TensorDataset(X, y)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    input_size = X.shape[1]
    output_size = 1
    hidden_layers = [64, 16, 8]
    m = FFNN(input_size, output_size, hidden_layers)

    loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=5e-2)

    train_regression(m, train_loader, test_loader, 1000, optimizer, loss_fn)
