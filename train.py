"""
train.py — Antrenare MLP pe Car Evaluation Dataset

Dataset  : UCI Car Evaluation (1728 inregistrari, 6 atribute, 4 clase)
           Descarcat automat la prima rulare din internet
Framework: PyTorch
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.mlp import CarMLP
from utils.data_loader import load_car_data, CLASS_NAMES
from utils.metrics import (compute_metrics, plot_training_curves,
                            plot_confusion_matrix, plot_class_distribution,
                            plot_prediction_summary)
from utils.logger import get_logger

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = "./results"
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(SEED)
logger = get_logger("train", os.path.join(RESULTS_DIR, "train.log"))


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        preds = out.argmax(1)
        correct    += preds.eq(y).sum().item()
        total      += X.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    logger.info(f"Device: {DEVICE}")
    logger.info("Incarcare date...")

    train_loader, val_loader, test_loader, class_weights = load_car_data(
        batch_size=BATCH_SIZE, seed=SEED
    )

    model     = CarMLP(input_size=21, num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    logger.info(f"Parametri model: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Epoci: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    logger.info("-" * 65)

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:5.2f}% | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:5.2f}% | "
            f"{time.time()-t0:.2f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, "best_model.pth"))
            logger.info(f"  >> Model salvat (val_acc={vl_acc*100:.2f}%)")

    logger.info("")
    logger.info("=" * 65)
    logger.info("EVALUARE FINALA PE TEST SET")
    logger.info("=" * 65)

    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth"),
                                     map_location=DEVICE))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    metrics = compute_metrics(labels, preds)

    logger.info(f"Cel mai bun epoch : {best_epoch}")
    logger.info(f"Test Accuracy     : {metrics['accuracy']*100:.2f}%")
    logger.info(f"Macro F1          : {metrics['macro_f1']:.4f}")
    logger.info(f"Macro Precision   : {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall      : {metrics['macro_recall']:.4f}")
    logger.info("")
    logger.info(metrics["report"])

    logger.info("Generare grafice...")
    plot_training_curves(history, RESULTS_DIR)
    plot_confusion_matrix(labels, preds, RESULTS_DIR)
    plot_class_distribution(RESULTS_DIR)
    plot_prediction_summary(labels, preds, RESULTS_DIR)
    logger.info("Gata! Graficele sunt in results/")


if __name__ == "__main__":
    main()
