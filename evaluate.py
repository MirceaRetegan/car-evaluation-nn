"""
evaluate.py — Evaluare model antrenat

Utilizare:
    python evaluate.py
    python evaluate.py --model results/best_model.pth
"""

import argparse, os, torch
from utils.data_loader import load_car_data
from utils.metrics import compute_metrics, plot_confusion_matrix, plot_prediction_summary
from utils.logger import get_logger
from models.mlp import CarMLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/best_model.pth")
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        preds = model(X.to(device)).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    return all_labels, all_preds


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("evaluate")

    if not os.path.exists(args.model):
        logger.info(f"Modelul nu a fost gasit la: {args.model}")
        logger.info("Ruleaza mai intai: python train.py")
        return

    _, _, test_loader, _ = load_car_data()
    model = CarMLP(input_size=21, num_classes=4).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    logger.info(f"Model incarcat: {args.model}")

    y_true, y_pred = run_eval(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(f"Test Accuracy   : {metrics['accuracy']*100:.2f}%")
    logger.info(f"Macro F1        : {metrics['macro_f1']:.4f}")
    logger.info(f"Macro Precision : {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall    : {metrics['macro_recall']:.4f}")
    logger.info("")
    logger.info(metrics["report"])

    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, "results")
    plot_prediction_summary(y_true, y_pred, "results")


if __name__ == "__main__":
    main()
