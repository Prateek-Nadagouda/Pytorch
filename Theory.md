# PyTorch Neural Network Approaches & Data Handling

I'll beautify the content and organize it into a more readable format with properly styled code blocks and clear section headers.

## Neural Network Architecture Approaches in PyTorch

PyTorch offers flexibility in how you define and connect neural network layers. Here are the main approaches:

### 1. Explicit Layer Definition

Define each layer explicitly in `__init__` and manually connect them in the `forward` method:

```python
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        # Define layers explicitly
        self.fc1 = nn.Linear(in_features, hidden_size)  # input → hidden
        self.relu = nn.ReLU()  # activation
        self.fc2 = nn.Linear(hidden_size, out_features)  # hidden → output

    def forward(self, x):
        # Manually pass data through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

**When to use:**
- When you need maximum control (custom architectures, skip connections, conditional logic)
- Research prototyping or complex networks like ResNet, transformers

**Why:**
- You can insert arbitrary operations, combine outputs, apply dropout at specific places—with complete flexibility

### 2. Sequential Container Approach

Pack layers in order using `nn.Sequential`:

```python
class Net(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        # Pack everything in order in one container
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x):
        return self.layers(x)  # One-liner
```

**When to use:**
- Standard feedforward networks where data flows strictly one way
- Quick prototypes, MLPs, CNN stacks (Conv → ReLU → Pool → Flatten → Linear)

**Why:**
- Cleaner code, less boilerplate
- Limited flexibility—can't easily add skip connections or concatenate inside the sequence

### 3. Dynamic Layer Creation

Use `nn.ModuleList` or `nn.ModuleDict` for dynamic architectures:

```python
class Net(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super().__init__()
        # Build layers dynamically based on config
        self.layers = nn.ModuleList()
        prev_size = in_features
        for h in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, h))
            prev_size = h
        self.out = nn.Linear(prev_size, out_features)

    def forward(self, x):
        # Manually loop through layers
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.out(x)
        return x
```

**When to use:**
- Variable-length architectures (e.g., configurable number of hidden layers)
- Hyperparameter sweeps, AutoML pipelines, configurable models

**Why:**
- Flexibility: you can loop over them, store them, or build them programmatically

### 4. Functional API Approach

Use `torch.nn.functional` instead of module objects for activations:

```python
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        # Use functional API for activation instead of module objects
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**When to use:**
- Lightweight models, where you don't need activation layers as objects
- Simple feedforward networks

**Why:**
- More concise code, avoids storing unnecessary activation objects
- Note: Use module versions (nn.Module) when you need to track state (e.g., dropout probability, batch norm statistics)

### 5. Composite Building Blocks

Create reusable blocks of layers:

```python
class Block(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.block1 = Block(in_features, hidden_size)
        self.block2 = Block(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x)
```

**When to use:**
- Deep networks with repeating structures (ResNet, DenseNet, Transformers)
- When you want to organize code into readable "building blocks"

## Understanding Layer Types

- **Input layer**: The first `nn.Linear` layer that takes input features. Its input size must match your dataset feature dimension.
- **Hidden layers**: The core of your network. Use `nn.Linear` for dense connections, `nn.Conv2d` for images, `nn.LSTM` for sequences.
- **Output layer**: Matches the shape of your task:
  - Regression: `nn.Linear(hidden, 1)`
  - Binary classification: `nn.Linear(hidden, 1)` + `BCEWithLogitsLoss`
  - Multiclass: `nn.Linear(hidden, num_classes)` + `CrossEntropyLoss`
  - Multilabel: `nn.Linear(hidden, num_classes)` + `BCEWithLogitsLoss`

## TL;DR — When/what/how

- Use explicit layers when you need custom control
- Use `nn.Sequential` for quick and simple one-way networks
- Use `ModuleList` for dynamic architecture definitions
- Use functional API (`F.relu`) when you don't need stateful modules
- Use blocks of modules when building complex architectures for better reuse and clarity

## Data Handling in PyTorch

PyTorch's data pipeline consists of two main components:
- **Dataset**: Defines what the data is and how to access individual samples
- **DataLoader**: Handles batching, shuffling, and iterating through the Dataset

### 1. Creating Custom Datasets

Subclass `torch.utils.data.Dataset` and implement required methods:

```python
from torch.utils.data import Dataset
import pandas as pd
import torch


class MyCSVData(Dataset):
    def __init__(self, csv_path):
        # Load entire CSV into memory
        self.data = pd.read_csv(csv_path)

        # Separate features (all columns except 'label') and target
        self.X = self.data.drop(columns=['label']).values.astype('float32')
        self.y = self.data['label'].values.astype('int64')

    def __len__(self):
        # Number of rows in CSV = number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return sample at given index as torch tensors
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y


# Usage
dataset = MyCSVData("mydata.csv")
print(len(dataset))  # total number of samples
print(dataset[0])    # first sample (x, y)
```

### 2. Using DataLoader

Wrap a Dataset and handle batching, shuffling, and parallel loading:

```python
from torch.utils.data import DataLoader

# Wrap dataset into a dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over it
for batch in dataloader:
    x_batch, y_batch = batch
    print(x_batch.shape, y_batch.shape)
    break
```

### 3. Using Built-in Datasets

PyTorch provides ready-to-use datasets through libraries like torchvision:

```python
from torchvision import datasets, transforms

# Define transformation (convert to tensor, normalize)
transform = transforms.Compose([transforms.ToTensor()])

# Download and load dataset
train_dataset = datasets.MNIST(
    root="./data", 
    train=True, 
    transform=transform, 
    download=True
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Iterate through batches
for images, labels in train_loader:
    print(images.shape)  # (64, 1, 28, 28)
    print(labels.shape)  # (64,)
    break
```

### 4. How It Works Under the Hood

1. You define a `Dataset` that specifies how to access individual samples
2. You wrap it in a `DataLoader` that fetches samples, batches them, shuffles them, and returns one batch at a time
3. Your training loop consumes the `DataLoader` like a generator:
   ```python
   for xb, yb in train_loader:
       preds = model(xb)  # forward pass
       loss = loss_fn(preds, yb)
       # ...optimizer steps, etc.
   ```

## Summary

- **Dataset**: "What's my data and how do I get one sample?"
- **DataLoader**: "Give me batches of samples in the order I want (shuffled or not), potentially using parallel workers."

The decoupling of storage (Dataset) from iteration/batching (DataLoader) makes your code modular and reusable.
