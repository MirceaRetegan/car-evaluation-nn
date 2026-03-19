"""
models/mlp.py
Retea neuronala MLP (Multi-Layer Perceptron) pentru clasificarea masinilor.

Arhitectura:
    Input (21) -> FC(128) -> BN -> ReLU -> Dropout
               -> FC(64)  -> BN -> ReLU -> Dropout
               -> FC(32)  -> ReLU
               -> Output (4 clase)
"""

import torch
import torch.nn as nn


class CarMLP(nn.Module):
    """
    MLP simplu cu Batch Normalization si Dropout pentru regularizare.
    Input: 21 neuroni (6 caracteristici one-hot encoded)
    Output: 4 clase (unacceptable, acceptable, good, vgood)
    """

    def __init__(self, input_size: int = 21, num_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Strat 1
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Strat 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Strat 3
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            # Output
            nn.Linear(32, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    model = CarMLP()
    dummy = torch.randn(8, 21)
    out   = model(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Parametri    : {sum(p.numel() for p in model.parameters()):,}")
