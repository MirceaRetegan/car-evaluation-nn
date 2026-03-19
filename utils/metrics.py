"""utils/metrics.py — Calcul metrici + grafice"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

CLASS_NAMES = ["Unacceptable", "Acceptable", "Good", "Very Good"]
COLORS      = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy"        : sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true),
        "macro_f1"        : f1_score(y_true, y_pred, average="macro"),
        "macro_precision" : precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall"    : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "report"          : classification_report(
                                y_true, y_pred,
                                target_names=CLASS_NAMES, digits=4
                            ),
    }


def plot_training_curves(history: dict, save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curbe de antrenare — Car Evaluation MLP",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            color="#2563eb", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",
            color="#dc2626", linewidth=2, linestyle="--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]],
            label="Train Acc", color="#2563eb", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],
            label="Val Acc",   color="#dc2626", linewidth=2, linestyle="--")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {path}")


def plot_confusion_matrix(y_true, y_pred, save_dir: str):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Matrice de Confuzie (normalizata)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predictie", fontsize=11)
    ax.set_ylabel("Eticheta reala", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {path}")


def plot_class_distribution(save_dir: str):
    counts = [1210, 384, 69, 65]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_NAMES, counts, color=COLORS,
                  edgecolor="white", linewidth=1.5, width=0.6)
    ax.set_title("Distributia claselor in Car Evaluation Dataset",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Clasa"); ax.set_ylabel("Numar exemple")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(cnt), ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {path}")


def plot_prediction_summary(y_true, y_pred, save_dir: str):
    true_counts = [y_true.count(i) for i in range(4)]
    pred_counts = [y_pred.count(i) for i in range(4)]
    x     = np.arange(4)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, true_counts, width, label="Real",
           color=COLORS, alpha=0.9, edgecolor="white")
    ax.bar(x + width/2, pred_counts, width, label="Predictie",
           color=COLORS, alpha=0.5, edgecolor="white", hatch="//")
    ax.set_title("Real vs Predictie per clasa", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Numar exemple"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "prediction_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {path}")
