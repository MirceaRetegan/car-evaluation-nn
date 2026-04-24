"""
baseline.py — Comparatie modele clasice vs MLP

Ruleaza 3 modele simple (scikit-learn) pe acelasi split train/test
ca si train.py, apoi afiseaza un tabel comparativ cu MLP-ul antrenat.

Utilizare:
    python baseline.py
    python baseline.py --mlp_acc 96.15 --mlp_f1 0.9412

Nota: Nu modifica nimic din proiectul existent.
      Foloseste exact acelasi seed=42 si acelasi split ca train.py.
"""

import argparse
import time
import numpy as np

from sklearn.dummy        import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.metrics      import (accuracy_score, f1_score,
                                  precision_score, recall_score,
                                  classification_report)

from utils.data_loader import load_car_data, CLASS_NAMES


# ─── Argumente optionale (pentru a afisa MLP-ul in tabel) ─────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp_acc", type=float, default=None,
                   help="Test Accuracy a MLP-ului (ex: 96.15)")
    p.add_argument("--mlp_f1",  type=float, default=None,
                   help="Macro F1 a MLP-ului (ex: 0.9412)")
    return p.parse_args()


# ─── Extrage X, y din DataLoader (tensor → numpy) ─────────────────────────────
def loader_to_numpy(loader):
    X_all, y_all = [], []
    for X_batch, y_batch in loader:
        X_all.append(X_batch.numpy())
        y_all.append(y_batch.numpy())
    return np.vstack(X_all), np.concatenate(y_all)


# ─── Calculeaza si returneaza metrici ─────────────────────────────────────────
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred) * 100
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES, zero_division=0)

    return {
        "name":    name,
        "acc":     acc,
        "f1":      f1,
        "prec":    prec,
        "rec":     rec,
        "report":  report,
        "elapsed": elapsed,
    }


# ─── Afiseaza tabelul comparativ ──────────────────────────────────────────────
def print_table(results, mlp_acc=None, mlp_f1=None):
    sep = "-" * 72
    print()
    print("=" * 72)
    print("   COMPARATIE MODELE — BASELINE vs MLP")
    print("=" * 72)
    print(f"  {'Model':<28} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
    print(sep)

    for r in results:
        print(f"  {r['name']:<28} {r['acc']:>9.2f}% {r['f1']:>10.4f} {r['prec']:>10.4f} {r['rec']:>10.4f}")

    # Linie separatoare inainte de MLP
    if mlp_acc is not None or mlp_f1 is not None:
        print(sep)
        acc_str = f"{mlp_acc:>9.2f}%" if mlp_acc else f"{'N/A':>10}"
        f1_str  = f"{mlp_f1:>10.4f}"  if mlp_f1  else f"{'N/A':>10}"
        print(f"  {'★  MLP (PyTorch)':<28} {acc_str} {f1_str}   (vezi results/)")

    print("=" * 72)
    print()


# ─── Afiseaza raport detaliat per clasa ───────────────────────────────────────
def print_details(results):
    for r in results:
        print(f"\n{'─' * 55}")
        print(f"  {r['name']}  |  Timp antrenare: {r['elapsed']:.3f}s")
        print(f"{'─' * 55}")
        print(r["report"])


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\nIncarcare date (seed=42, split 70/15/15 — identic cu train.py)...")
    train_loader, _, test_loader, _ = load_car_data(seed=42)

    X_train, y_train = loader_to_numpy(train_loader)
    X_test,  y_test  = loader_to_numpy(test_loader)

    print(f"  Train: {len(y_train)} exemple  |  Test: {len(y_test)} exemple\n")

    # ── Cele 3 modele baseline ────────────────────────────────────────────────
    models = [
        (
            "DummyClassifier (majority)",
            DummyClassifier(strategy="most_frequent", random_state=42)
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42,
                               class_weight="balanced")
        ),
        (
            "Random Forest (100 arbori)",
            RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight="balanced")
        ),
    ]

    results = []
    for name, model in models:
        print(f"  Antrenare: {name}...")
        r = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(r)
        print(f"    Acc: {r['acc']:.2f}%  |  Macro F1: {r['f1']:.4f}  "
              f"(timp: {r['elapsed']:.2f}s)")

    # ── Tabel comparativ ─────────────────────────────────────────────────────
    print_table(results, mlp_acc=args.mlp_acc, mlp_f1=args.mlp_f1)

    # ── Detalii per clasa ────────────────────────────────────────────────────
    print_details(results)

    # ── Concluzie automata ───────────────────────────────────────────────────
    best_baseline = max(results, key=lambda r: r["f1"])
    print(f"\n  Cel mai bun baseline: {best_baseline['name']}")
    print(f"    Acc: {best_baseline['acc']:.2f}%  |  Macro F1: {best_baseline['f1']:.4f}")

    if args.mlp_acc and args.mlp_f1:
        gain_acc = args.mlp_acc - best_baseline["acc"]
        gain_f1  = args.mlp_f1  - best_baseline["f1"]
        print(f"\n  MLP vs cel mai bun baseline:")
        print(f"    +{gain_acc:.2f}% Accuracy  |  +{gain_f1:.4f} Macro F1")
    print()


if __name__ == "__main__":
    main()
