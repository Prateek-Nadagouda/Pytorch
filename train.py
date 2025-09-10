# train.py
# CLI entrypoint: parse args, build model/dataloaders, run training, evaluate on test set.

import argparse
import os
import numpy as np
import torch
from utils import set_seed, get_optimizer, get_loss, make_checkpoint_dir, timestamp
from dataset import get_dataloaders
from models import RegressionModel, BinaryClassificationModel, MultiClassModel
from trainer import Trainer


# Build a model appropriate for the requested task.
def build_model_for_task(task: str, in_features: int, num_classes: int = None):
    if task == 'regression':
        return RegressionModel(in_features=in_features)
    if task == 'binary':
        return BinaryClassificationModel(in_features=in_features)
    if task == 'multiclass':
        assert num_classes is not None, "num_classes required for multiclass"
        return MultiClassModel(in_features=in_features, num_classes=num_classes)
    raise ValueError("Unknown task")


# Main function configures training via CLI arguments.
def main():
    parser = argparse.ArgumentParser(description="Train template for regression/binary/multiclass")
    parser.add_argument("--task", type=str, choices=["regression", "binary", "multiclass"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision if available")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "step", "plateau", "cosine"])
    parser.add_argument("--num_classes", type=int, default=4, help="Only for multiclass")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Set global seeds for reproducibility.
    set_seed(args.seed)

    # Get dataloaders (synthetic by default). Replace dataset.py for real data.
    train_dl, val_dl, test_dl = get_dataloaders(task=args.task, batch_size=args.bs)

    # Infer input feature size from a batch sample of the train dataloader.
    sample_batch = next(iter(train_dl))[0]
    in_features = sample_batch.shape[1]

    # Instantiate the appropriate model.
    model = build_model_for_task(args.task, in_features=in_features, num_classes=args.num_classes)

    # Choose default loss if user did not specify one.
    if args.loss is None:
        if args.task == 'regression':
            loss_name = 'mse'
        elif args.task == 'binary':
            loss_name = 'bce_with_logits'
        else:
            loss_name = 'cross_entropy'
    else:
        loss_name = args.loss

    # Create loss and optimizer objects using the factories in utils.py
    loss_fn = get_loss(loss_name)
    optimizer = get_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )

    # Configure LR scheduler if requested from CLI
    scheduler = None
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    elif args.scheduler == 'cosine':
        # CosineAnnealingLR will anneal learning rate smoothly across T_max epochs.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Prepare trainer and run training loop.
    ckpt_dir = make_checkpoint_dir(args.save_dir)
    trainer = Trainer(model, loss_fn, optimizer, scheduler=scheduler, amp=args.amp)
    ckpt_name = f"{args.task}_{timestamp()}.pt"
    trainer.fit(train_dl, val_dl=val_dl, epochs=args.epochs, task=args.task, ckpt_dir=ckpt_dir, ckpt_name=ckpt_name)

    # After training, evaluate on the test set and print metrics.
    preds, probs = trainer.predict(test_dl, task=args.task)

    # Collect true labels from the test dataloader.
    y_list = []
    for xb, yb in test_dl:
        y_list.append(yb.numpy())
    y_true = np.concatenate(y_list, axis=0)

    # Compute metrics using evaluate.py helpers and print them.
    from evaluate import regression_metrics, binary_classification_metrics, multiclass_classification_metrics
    if args.task == 'regression':
        metrics = regression_metrics(y_true, preds)
    elif args.task == 'binary':
        metrics = binary_classification_metrics(y_true, preds, y_prob=probs)
    else:
        metrics = multiclass_classification_metrics(y_true, preds)

    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
