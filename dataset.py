# dataset.py
# Synthetic data generators and dataloader factory for quick testing.


# Create a synthetic linear regression dataset with gaussian noise.
def make_synthetic_regression(n_samples=2000, in_features=10, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, in_features).astype(np.float32)
    # Random linear weights
    w = rng.randn(in_features).astype(np.float32)
    # Linear target + gaussian noise
    y = X.dot(w) + noise * rng.randn(n_samples).astype(np.float32)
    return X, y

# Create a synthetic binary classification dataset via a linear decision boundary.
def make_synthetic_binary(n_samples=2000, in_features=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, in_features).astype(np.float32)
    logits = X[:, 0] * 2.0 - 0.3 * X[:, 1] + 0.1 * rng.randn(n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(np.float32)
    return X, y

# Create a multiclass dataset via random linear projection and argmax.
def make_synthetic_multiclass(n_samples=3000, in_features=12, n_classes=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, in_features).astype(np.float32)
    W = rng.randn(in_features, n_classes).astype(np.float32)
    logits = X.dot(W)
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y

# Return train/val/test dataloaders using random splits.
def get_dataloaders(task: str = 'regression', batch_size: int = 128, val_frac: float = 0.2, test_frac: float = 0.1, **kwargs):
    # Select dataset generator based on task.
    if task == 'regression':
        X, y = make_synthetic_regression(**kwargs)
    elif task == 'binary':
        X, y = make_synthetic_binary(**kwargs)
    elif task == 'multiclass':
        X, y = make_synthetic_multiclass(**kwargs)
    else:
        raise ValueError("Unsupported task")

    # Convert to tensors and wrap in a TensorDataset
    X = torch.tensor(X)
    y = torch.tensor(y)
    ds = TensorDataset(X, y)

    # Compute sizes for train/val/test split
    n = len(ds)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    # random_split returns Subset objects that preserve order randomness
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    # Create DataLoader objects. Set num_workers > 0 for better performance in real runs.
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dl, val_dl, test_dl
