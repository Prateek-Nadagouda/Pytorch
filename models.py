# Model definitions: lightweight MLPs for regression, binary classification, and multiclass classification.

import torch
from torch import nn

# Small MLP builder reused across models to keep consistent architecture.
class MLPBase(nn.Module):
    # in_features: input dimensionality. hidden_sizes: tuple of hidden layer sizes.
    def __init__(self, in_features: int, hidden_sizes=(128, 64), activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_features
        # Build Linear -> Activation blocks
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        # Pack into a sequential module
        self.body = nn.Sequential(*layers)

    # Forward pass through the MLP body.
    def forward(self, x):
        return self.body(x)

# Regression model: MLP body + scalar head.
class RegressionModel(nn.Module):
    def __init__(self, in_features: int, hidden_sizes=(128, 64)):
        super().__init__()
        # Base MLP to learn features
        self.base = MLPBase(in_features, hidden_sizes)
        # Final linear head producing one output per sample
        last = hidden_sizes[-1] if len(hidden_sizes) else in_features
        self.head = nn.Linear(last, 1)

    # Squeeze to return shape (N,) rather than (N,1)
    def forward(self, x):
        return self.head(self.base(x)).squeeze(-1)

# Binary classification model: single-logit output per sample.
class BinaryClassificationModel(nn.Module):
    def __init__(self, in_features: int, hidden_sizes=(128, 64)):
        super().__init__()
        self.base = MLPBase(in_features, hidden_sizes)
        last = hidden_sizes[-1] if len(hidden_sizes) else in_features
        # Head returns a single value (logit) per sample
        self.head = nn.Linear(last, 1)

    def forward(self, x):
        return self.head(self.base(x)).squeeze(-1)

# Multiclass classification model: returns raw logits for each class.
class MultiClassModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_sizes=(256, 128)):
        super().__init__()
        self.base = MLPBase(in_features, hidden_sizes)
        last = hidden_sizes[-1] if len(hidden_sizes) else in_features
        # Head returns logits for each class (N, num_classes)
        self.head = nn.Linear(last, num_classes)

    def forward(self, x):
        return self.head(self.base(x))
