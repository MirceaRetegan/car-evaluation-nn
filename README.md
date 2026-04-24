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

Distributia claselor:
- **Unacceptable**: 1210 exemple (70%) — clasa majoritara
- **Acceptable**: 384 exemple (22%)
- **Good**: 69 exemple (4%)
- **Very Good**: 65 exemple (4%)

> Dezechilibrul claselor este gestionat prin `CrossEntropyLoss` cu ponderi
> inverse frecventei (`class_weight = 1 / frecventa`).

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
# 1. Antreneaza modelul (~30 secunde pe CPU)
python train.py

# 2. Evalueaza pe test set
python evaluate.py

# 3. Compara MLP cu modele clasice baseline
python baseline.py --mlp_acc 96.15 --mlp_f1 0.9412

# 4. Testeaza o masina noua (interactiv!)
python predict.py
```

---

## Structura proiectului

```
car-evaluation-nn/
├── train.py            <- antrenare model MLP
├── evaluate.py         <- evaluare test set (metrici + grafice)
├── baseline.py         <- comparatie cu modele clasice sklearn
├── predict.py          <- predictie interactiva
├── models/
│   └── mlp.py          <- arhitectura MLP (CarMLP)
├── utils/
│   ├── data_loader.py  <- descarcare + preprocesare + split date
│   ├── metrics.py      <- metrici + grafice
│   └── logger.py       <- logging
├── results/            <- grafice + model salvat (generat automat)
└── data/               <- dataset descarcat automat la prima rulare
```

---

## Decizii tehnice

### Arhitectura MLP: 128 → 64 → 32 → 4
Dataset mic (1728 exemple) → model compact pentru a preveni overfitting.
Reducerea progresiva a dimensiunii captureaza ierarhia features.
Un MLP mai adanc (256→128→64→32) testat a dat acuratete similara cu
de 4x mai multi parametri — nu justifica complexitatea adaugata.

### Batch Normalization dupa fiecare strat
Stabilizeaza gradientii, permite learning rate mai mare si reduce
sensibilitatea la initializare. A adus o imbunatatire de ~2% fata de
reteaua fara BatchNorm.

### One-Hot Encoding (nu Ordinal Encoding)
Atributele nu au ordine naturala clara (ex: `doors`: 2, 3, 4, 5more
nu are o relatie numerica). One-hot nu induce relatii false intre valori.
Ordinal encoding testat: -4% accuracy pe clasele rare.

### CrossEntropyLoss cu class weights
70% din date sunt clasa Unacceptable. Fara ponderi, reteaua ar ignora
clasele rare (Good, Very Good). Cu ponderi inverse frecventei,
F1-score pe clasele rare a crescut cu ~15%.

### Split reproductibil: 70% train / 15% val / 15% test
`torch.manual_seed(42)` + generator seed in `random_split` garanteaza
acelasi split la fiecare rulare — rezultatele sunt reproductibile.

---

## Rezultate finale

### MLP (PyTorch)

| Metrica        | Valoare  |
|----------------|----------|
| Test Accuracy  | 98.46%   |
| Macro F1       | 0.9740   |
| Macro Precision| 0.9586   |
| Macro Recall   | 0.9913   |

### Comparatie cu modele baseline (sklearn)

| Model                      | Accuracy | Macro F1 |
|----------------------------|----------|----------|
| DummyClassifier (majority) | 71.81%   | 0.2090   |
| Logistic Regression        | 90.73%   | 0.8560   |
| Random Forest (100 arbori) | 97.30%   | 0.9449   |
| **MLP PyTorch**            | **98.46%**| **0.9740**|

> **Observatie:** Random Forest obtine rezultate similare MLP-ului pe acest
> dataset deoarece datele sunt tabelare, putine (1728 exemple) si toate
> atributele sunt categorice — conditii in care modelele bazate pe arbori
> de decizie exceleaza natural. MLP-ul ramane relevant prin flexibilitate
> si scalabilitate pe seturi de date mai mari si mai complexe.

### Grafice generate automat in `results/`

- `training_curves.png` — Loss si Accuracy pe epoci
- `confusion_matrix.png` — Matricea de confuzie
- `class_distribution.png` — Distributia claselor in dataset
- `prediction_summary.png` — Real vs Predictie per clasa

---

## Concluzii

- MLP cu BatchNorm + Dropout atinge **98.46% Accuracy** si **Macro F1 = 0.974**
- Obiectivele proiectului (Accuracy ≥ 95%, F1 ≥ 0.90) au fost atinse
- Dezechilibrul claselor a fost rezolvat prin `class_weight` in loss function
- Modelul compact (13,540 parametri) este mai eficient decat variante mai adanci
- Pe date tabelare mici, Random Forest ramane un competitor puternic

## Limitari si directii viitoare

- Dataset mic: clasele Good si Very Good au doar 65-69 exemple de antrenare
- Directii: comparatie cu SVM, augmentare date (SMOTE), interfata web (Streamlit),
  explicabilitate prin SHAP values

---

> Proiect realizat pentru cursul de Calcul Neuronal
