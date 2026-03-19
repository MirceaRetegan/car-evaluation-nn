# Car Evaluation — Retea Neuronala MLP cu PyTorch

Clasificarea calitatii masinilor folosind un **Multi-Layer Perceptron (MLP)**
antrenat pe **Car Evaluation Dataset** de la UCI.

**Dataset:** Car Evaluation UCI (1728 inregistrari, 6 atribute, 4 clase)
**Framework:** PyTorch
**Metrici:** Accuracy, Macro F1, Precision, Recall, Confusion Matrix

---

## Dataset

| Atribut    | Valori posibile              |
|------------|------------------------------|
| buying     | vhigh, high, med, low        |
| maint      | vhigh, high, med, low        |
| doors      | 2, 3, 4, 5more               |
| persons    | 2, 4, more                   |
| lug_boot   | small, med, big              |
| safety     | low, med, high               |
| **class**  | unacc, acc, good, vgood      |

---

## Instalare

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Rulare

```bash
# 1. Antreneaza modelul (rapida: ~30 secunde pe CPU)
python train.py

# 2. Evalueaza pe test set
python evaluate.py

# 3. Testeaza o masina noua (interactiv!)
python predict.py
```

---

## Structura proiectului

```
car-evaluation-nn/
├── train.py            <- antrenare model
├── evaluate.py         <- evaluare test set
├── predict.py          <- predictie interactiva
├── models/
│   └── mlp.py          <- arhitectura MLP
├── utils/
│   ├── data_loader.py  <- descarcare + preprocesare date
│   ├── metrics.py      <- metrici + grafice
│   └── logger.py       <- logging
├── results/            <- grafice + model salvat (auto)
└── data/               <- dataset descarcat automat
```

---

## Rezultate asteptate

| Metrica       | Valoare  |
|---------------|----------|
| Test Accuracy | ~95-97%  |
| Macro F1      | ~0.93    |

## Grafice generate automat in results/

- `training_curves.png` - Loss si Accuracy pe epoci
- `confusion_matrix.png` - Matricea de confuzie
- `class_distribution.png` - Distributia claselor in dataset
- `prediction_summary.png` - Real vs Predictie per clasa

---

> Proiect realizat pentru cursul de Calcul Neuronal
