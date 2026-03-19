"""
predict.py — Testeaza modelul cu o masina introdusa manual

Utilizare:
    python predict.py

Vei fi intrebat interactiv despre caracteristicile masinii,
apoi modelul iti spune daca masina este recomandata sau nu.
"""

import torch
from models.mlp import CarMLP
from utils.data_loader import ATTRIBUTE_VALUES, CLASS_NAMES

QUESTIONS = {
    "buying"  : ("Pretul de cumparare",   ["vhigh", "high", "med", "low"],
                 ["Foarte mare", "Mare", "Mediu", "Mic"]),
    "maint"   : ("Costul de intretinere", ["vhigh", "high", "med", "low"],
                 ["Foarte mare", "Mare", "Mediu", "Mic"]),
    "doors"   : ("Numarul de usi",        ["2", "3", "4", "5more"],
                 ["2 usi", "3 usi", "4 usi", "5 sau mai multe"]),
    "persons" : ("Capacitate persoane",   ["2", "4", "more"],
                 ["2 persoane", "4 persoane", "Mai mult de 4"]),
    "lug_boot": ("Spatiu portbagaj",      ["small", "med", "big"],
                 ["Mic", "Mediu", "Mare"]),
    "safety"  : ("Siguranta",             ["low", "med", "high"],
                 ["Scazuta", "Medie", "Ridicata"]),
}

RESULT_EMOJI = {
    "Unacceptable": "❌", "Acceptable": "✅", "Good": "⭐", "Very Good": "🌟"
}


def get_input() -> list:
    print()
    print("=" * 52)
    print("   EVALUARE MASINA NOUA — Car Evaluation NN")
    print("=" * 52)
    print("  Raspunde la intrebarile de mai jos:")
    row = []
    for attr, (question, options, labels_ro) in QUESTIONS.items():
        print(f"
  {question}:")
        for i, (opt, label) in enumerate(zip(options, labels_ro), 1):
            print(f"    {i}. {label}")
        while True:
            try:
                choice = int(input(f"  Alegere (1-{len(options)}): "))
                if 1 <= choice <= len(options):
                    row.append(options[choice - 1])
                    break
                else:
                    print(f"  Te rog alege intre 1 si {len(options)}")
            except ValueError:
                print("  Introdu un numar valid!")
    return row


def encode(row: list) -> torch.Tensor:
    encoded = []
    for col, val in zip(list(ATTRIBUTE_VALUES.keys()), row):
        options = ATTRIBUTE_VALUES[col]
        encoded.extend([1.0 if val == opt else 0.0 for opt in options])
    return torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)


def main():
    model = CarMLP(input_size=21, num_classes=4)
    try:
        model.load_state_dict(torch.load("results/best_model.pth",
                                         map_location="cpu"))
        model.eval()
    except FileNotFoundError:
        print("Modelul nu a fost gasit! Ruleaza mai intai: python train.py")
        return

    row = get_input()
    X   = encode(row)

    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1).squeeze()
        pred  = probs.argmax().item()

    print()
    print("=" * 52)
    print("   REZULTAT")
    print("=" * 52)
    emoji = RESULT_EMOJI[CLASS_NAMES[pred]]
    print(f"
  {emoji}  Clasa prezisa: {CLASS_NAMES[pred].upper()}")
    print(f"
  Probabilitati pentru fiecare clasa:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar    = chr(9608) * int(prob.item() * 25)
        marker = "  <--" if i == pred else ""
        print(f"  {name:<15} {prob*100:5.1f}%  {bar}{marker}")
    print()


if __name__ == "__main__":
    main()
