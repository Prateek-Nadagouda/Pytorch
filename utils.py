# Utilities for reproducibility, device selection, checkpointing, and factories for optimizers/losses.
import os
import time
import random
from typing import Optional, Dict

import numpy as np
import torch
from torch.optim import SGD, Adam, RMSprop, Adagrad, AdamW
from torch import nn

# Set deterministic random seeds across python, numpy, and torch for reproducibility.
def set_seed(seed: int = 42):
    # Seed Python random module
    random.seed(seed)
    # Seed NumPy RNG
    np.random.seed(seed)
    # Seed PyTorch CPU RNG
    torch.manual_seed(seed)
    # Seed all available CUDA devices (if any)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic where possible (may slow down)
    torch.backends.cudnn.deterministic = True
    # Disable CuDNN auto-tuner which can introduce non-determinism
    torch.backends.cudnn.benchmark = False


# Return a torch.device object (cuda if available else cpu).
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ensure checkpoint directory exists and return its path.
def make_checkpoint_dir(base: str = "checkpoints"):
    os.makedirs(base, exist_ok=True)
    return base


# Save a model + optimizer checkpoint at the given path. Useful to resume training.
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict] = None,
):
    state = {"model_state": model.state_dict()}
    # Optionally include optimizer state to resume training exactly.
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    # Optionally record the epoch number and other metadata.
    if epoch is not None:
        state["epoch"] = epoch
    if extra:
        state.update(extra)
    # Write the state dict to disk.
    torch.save(state, path)


# Load checkpoint into model (and optimizer if provided). Returns the loaded checkpoint dict.
def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
):
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# Factory function to construct optimizers by name. Keeps CLI code tidy.
def get_optimizer(
    name: str,
    parameters,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
):
    name_l = name.lower()
    if name_l == "sgd":
        # SGD with momentum is a strong baseline for many problems.
        return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name_l == "adam":
        # Adam adapts learning rates per-parameter; good default for many tasks.
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name_l == "adamw":
        # AdamW decouples weight decay from gradient update rule.
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name_l == "rmsprop":
        # RMSprop sometimes works well for recurrent nets or particular losses.
        return RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name_l == "adagrad":
        return Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


# Factory to pick appropriate loss functions by name.
def get_loss(name: str):
    name_l = name.lower()
    if name_l in ("mse", "mse_loss", "mean_squared_error"):
        # Mean squared error for regression.
        return nn.MSELoss()
    if name_l in ("mae", "l1", "l1_loss"):
        # Mean absolute error is robust to outliers.
        return nn.L1Loss()
    if name_l in ("huber", "smooth_l1", "smooth_l1_loss"):
        # Huber (smooth L1) balances MAE and MSE behavior.
        return nn.SmoothL1Loss()
    if name_l in ("bce", "bce_with_logits", "binary_cross_entropy"):
        # BCEWithLogitsLoss expects raw logits — numerically stable.
        return nn.BCEWithLogitsLoss()
    if name_l in ("cross_entropy", "ce", "nll"):
        # CrossEntropyLoss expects raw logits and integer class labels.
        return nn.CrossEntropyLoss()
    if name_l in ("kl", "kldiv"):
        return nn.KLDivLoss()
    raise ValueError(f"Unsupported loss: {name}")


# Return a compact timestamp string for naming checkpoints.
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


# Ensure input value is a torch tensor.
def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)
