"""
utils/data_loader.py
Descarca si preproceseaza Car Evaluation Dataset de la UCI.

Dataset: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
- 1728 inregistrari, 6 atribute categorice, 4 clase
- Nu are valori lipsa
"""

import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


URL       = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
SAVE_PATH = "./data/car.data"

ATTRIBUTE_VALUES = {
    "buying"  : ["vhigh", "high", "med", "low"],
    "maint"   : ["vhigh", "high", "med", "low"],
    "doors"   : ["2", "3", "4", "5more"],
    "persons" : ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety"  : ["low", "med", "high"],
}

CLASS_LABELS = ["unacc", "acc", "good", "vgood"]
CLASS_NAMES  = ["Unacceptable", "Acceptable", "Good", "Very Good"]


def download_data():
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(SAVE_PATH):
        print("  Descarcare Car Evaluation Dataset de la UCI...")
        urllib.request.urlretrieve(URL, SAVE_PATH)
        print(f"  Salvat la: {SAVE_PATH}")
    else:
        print(f"  Dataset deja existent: {SAVE_PATH}")


def encode_row(row: list) -> np.ndarray:
    encoded = []
    for col, val in zip(list(ATTRIBUTE_VALUES.keys()), row[:-1]):
        options = ATTRIBUTE_VALUES[col]
        one_hot = [1.0 if val == opt else 0.0 for opt in options]
        encoded.extend(one_hot)
    return np.array(encoded, dtype=np.float32)


def load_car_data(batch_size: int = 64, val_split: float = 0.15,
                  test_split: float = 0.15, seed: int = 42):
    download_data()

    X_list, y_list = [], []
    with open(SAVE_PATH, "r") as f:
        for line in f:
            row = line.strip().split(",")
            if len(row) != 7:
                continue
            X_list.append(encode_row(row))
            y_list.append(CLASS_LABELS.index(row[-1]))

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)

    n       = len(X)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    dataset = TensorDataset(X, y)
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=gen
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    counts  = torch.bincount(y)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()

    print(f"  Total: {n} | Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"  Distributie clase:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {counts[i].item()} exemple")

    return train_loader, val_loader, test_loader, weights
