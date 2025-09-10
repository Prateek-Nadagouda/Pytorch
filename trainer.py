# trainer.py
# Train / validate / predict orchestration with AMP and scheduler integration.

import os
from typing import Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split

from utils import get_device, save_checkpoint, timestamp
from evaluate import regression_metrics, binary_classification_metrics, multiclass_classification_metrics


# Trainer is a thin wrapper to keep training code organized and reusable.
class Trainer:
    # Initialize with model components and runtime configuration.
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer,
        device: Optional[torch.device] = None,
        scheduler=None,
        amp: bool = False,
        grad_clip: Optional[float] = None,
    ):
        # Save references to training components.
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # Choose computing device (GPU if available).
        self.device = device or get_device()
        # Scheduler may be None or an LR scheduler instance.
        self.scheduler = scheduler
        # Mixed precision flag (only effective if CUDA available).
        self.amp = amp and torch.cuda.is_available()
        self.grad_clip = grad_clip

        # Move model weights to the chosen device immediately.
        self.model.to(self.device)
        # Prepare GradScaler for AMP if used.
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    # Single epoch training loop. Uses AMP when requested.
    def train_one_epoch(self, dataloader, epoch: int = 0, task: str = 'regression'):
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # Reset gradients before backpropagation.
            self.optimizer.zero_grad()

            if self.amp:
                # Use autocast for mixed-precision forward pass.
                with torch.cuda.amp.autocast():
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)
                # Scale, backward, unscale and step optimizer via scaler.
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(xb)
                loss = self.loss_fn(out, yb)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            pbar.set_postfix({"loss": total_loss / n_samples})

        avg_loss = total_loss / n_samples
        return avg_loss

    # Validation loop to compute average loss and metrics on provided dataset.
    @torch.no_grad()
    def validate(self, dataloader, task='regression'):
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        all_preds = []
        all_probs = []
        all_targets = []

        for xb, yb in tqdm(dataloader, desc="Validate"):
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            if self.amp:
                with torch.cuda.amp.autocast():
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)
            else:
                out = self.model(xb)
                loss = self.loss_fn(out, yb)

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

            # Convert outputs to preds and probs depending on the task.
            if task == 'regression':
                preds = out.detach().cpu().numpy()
                all_preds.append(preds)
            elif task == 'binary':
                probs = torch.sigmoid(out).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_preds.append(preds)
                all_probs.append(probs)
            elif task == 'multiclass':
                probs = torch.softmax(out, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
                all_preds.append(preds)
                all_probs.append(probs)
            else:
                raise ValueError("Unknown task for validation")

            all_targets.append(yb.detach().cpu().numpy())

        avg_loss = total_loss / n_samples
        if all_preds:
            preds = np.concatenate(all_preds, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            probs = np.concatenate(all_probs, axis=0) if all_probs else None
        else:
            preds = targets = probs = None

        # Compute metrics using evaluate.py helpers.
        if task == 'regression':
            metrics = regression_metrics(targets, preds)
        elif task == 'binary':
            metrics = binary_classification_metrics(targets, preds, y_prob=probs)
        elif task == 'multiclass':
            metrics = multiclass_classification_metrics(targets, preds)
        else:
            metrics = {}

        return avg_loss, metrics

    # Fit full training process for given number of epochs, with optional validation and checkpointing.
    def fit(
        self,
        train_dl,
        val_dl=None,
        epochs: int = 10,
        task: str = 'regression',
        ckpt_dir: str = 'checkpoints',
        ckpt_name: Optional[str] = None,
        save_best: bool = True,
    ):
        best_val = float('inf')
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = ckpt_name or f"{task}_{timestamp()}.pt"

        for ep in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_dl, epoch=ep, task=task)
            val_loss, val_metrics = (None, None)
            if val_dl is not None:
                val_loss, val_metrics = self.validate(val_dl, task=task)

            # Scheduler step: handle ReduceLROnPlateau specially which requires a metric.
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_for_scheduler = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric_for_scheduler)
                else:
                    # Step epoch-based schedulers (StepLR, CosineAnnealingLR, etc.) once per epoch.
                    self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            print(f"Epoch {ep}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss if val_loss is None else f'{val_loss:.6f}'}")
            if val_metrics is not None:
                print("Val metrics:", val_metrics)

            # Save best model by validation loss if requested.
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if save_best and val_loss is not None and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_path, self.model, self.optimizer, epoch=ep, extra={"val_loss": val_loss})
                print("Saved best checkpoint to", ckpt_path)

        return history

   # Predict on dataloader and return predictions and optionally probabilities.
    @torch.no_grad()
    def predict(self, dataloader, task='regression'):
        self.model.eval()
        preds = []
        probs = []

        for xb, _ in tqdm(dataloader, desc="Predict"):
            xb = xb.to(self.device)
            if self.amp:
                with torch.cuda.amp.autocast():
                    out = self.model(xb)
            else:
                out = self.model(xb)

            if task == 'regression':
                preds.append(out.detach().cpu().numpy())
            elif task == 'binary':
                p = torch.sigmoid(out).detach().cpu().numpy()
                preds.append((p >= 0.5).astype(int))
                probs.append(p)
            elif task == 'multiclass':
                p = torch.softmax(out, dim=-1).detach().cpu().numpy()
                preds.append(p.argmax(axis=1))
                probs.append(p)
            else:
                raise ValueError("Unknown task")

        preds = np.concatenate(preds, axis=0)
        probs = np.concatenate(probs, axis=0) if probs else None
        return preds, probs
