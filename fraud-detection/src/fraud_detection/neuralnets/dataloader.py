from pathlib import Path

import torch.nn as nn

from .. import data_loader


def load_data_with_embeddings(path: Path):
    df = data_loader.load_data_df(path)


    embedding = nn.Embedding(
        num_embeddings=num_categories, embedding_dim=embedding_size
    )
    embedded = embedding(x)  # shape: (3, embedding_dim)
