# Alright, let’s cut to the bone. You’re asking: “How do I define input, hidden, and output layers in PyTorch with nn? And when/what/how to use each approach?”
# The short answer is: PyTorch gives you flexibility — you can wire layers together in three main ways. Each comes with trade-offs in readability, flexibility, and complexity. Let’s go through them with precision.

# ---
# 1. Explicit nn.Linear layers in __init__ and write your forward manually
import torch
from torch import nn

class Net(nn.Module):    
    def __init__(self, in_features, hidden_size, out_features):        
        super().__init__()        
        # Define layers explicitly        
        self.fc1 = nn.Linear(in_features, hidden_size) # input → hidden        
        self.relu = nn.ReLU() # activation        
        self.fc2 = nn.Linear(hidden_size, out_features) # hidden → output
    
    def forward(self, x):        
        # Manually pass data through layers        
        x = self.fc1(x)        
        x = self.relu(x)        
        x = self.fc2(x)        
        return x

# When to use:
# You need maximum control (custom architectures, skip connections, conditional logic).
# Research prototyping or complex nets like ResNet, transformers.

# Why:
# You can insert arbitrary operations, combine outputs, apply dropout at weird places — you’re not constrained.

# ---
# 2. Using nn.Sequential (stacking layers)
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
        return self.layers(x) # One-liner

# When to use:
# Standard feedforward nets where data flows strictly one way, layer after layer.
# Quick prototypes, MLPs, CNN stacks (Conv → ReLU → Pool → Flatten → Linear).

# Why:
# Cleaner code, less boilerplate. But you lose flexibility — can’t easily do “add this skip connection” or “concatenate here” inside the sequence.

# ---
# 3. Dynamic layer creation with nn.ModuleList or nn.ModuleDict
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

# When to use:
# You want variable-length architectures (e.g., “n hidden layers defined by user”).
# Useful for hyperparameter sweeps, AutoML pipelines, configurable models.

# Why:
# Flexibility: you can loop over them, store them, or build them on the fly.

# ---
# 4. Functional API (torch.nn.functional) instead of defining activations
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

# When to use:
# Lightweight models, where you don’t need activation layers as objects (e.g., for sequential).
# Often preferred for simple feedforward.

# Why:
# Saves lines, avoids storing useless nn.ReLU() objects. But note: if you need to track module state (like dropout probability, batch norm statistics), use the nn.Module version, not F.*.

# ---
# 5. Container blocks inside container blocks
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

# When to use:
# Deep networks with repeating structures (ResNet, DenseNet, Transformers).
# Helps you organize code into readable “building blocks.”

# ---
# Input layer vs hidden layer vs output layer
# Input layer: Just the first nn.Linear that takes input features. Its input size must match your dataset feature dimension. You rarely think of it separately — it’s just the first linear projection.
# Hidden layers: The “meat” of the net. Use nn.Linear for dense, nn.Conv2d for images, nn.LSTM for sequences. Choose size and number based on task complexity + regularization.
# Output layer: Matches the shape of your task:
# Regression: nn.Linear(hidden, 1)
# Binary classification: nn.Linear(hidden, 1) + BCEWithLogitsLoss
# Multiclass: nn.Linear(hidden, num_classes) + CrossEntropyLoss
# Multilabel: nn.Linear(hidden, num_classes) + BCEWithLogitsLoss

# ---
# TL;DR — When/what/how
# Use explicit layers if you need custom control.
# Use nn.Sequential for quick and simple one-way nets.
# Use ModuleList for dynamic architecture definitions.
# Use functional API (F.relu) when you don’t need stateful modules (dropout, batchnorm).
# Use blocks of modules when building complex architectures — makes reuse and clarity better.

# ---
# Want me to make a side-by-side code comparison of all 5 approaches using the same simple MLP architecture (say, 2 hidden layers, 10 → 64 → 32 → 3 classes), so you can see how they differ in structure?

# Let’s crack this open systematically.
# In PyTorch, Dataset and DataLoader are two halves of the input pipeline:
# Dataset = defines what the data is and how to get one sample.
# DataLoader = defines how to batch, shuffle, and iterate through the Dataset.

# Think of Dataset as a library of books, and DataLoader as the librarian who fetches a stack of books (batch) for you, in order or shuffled, and maybe in parallel.

# ---
# 1. torch.utils.data.Dataset
# This is an abstract class you subclass. You need to implement two methods:
# __len__ → how many samples in total.
# __getitem__(idx) → how to get one sample by index.

# PyTorch already provides datasets like MNIST, CIFAR, ImageNet. But for your custom data (CSV, images, text), you write your own.
# Example: Custom dataset for a CSV with features + labels
from torch.utils.data import Dataset
import pandas as pd

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
print(len(dataset)) # total number of samples
print(dataset[0]) # first sample (x, y)
# Here, dataset[0] returns one row. No batching yet — that’s DataLoader’s job.

# ---
# 2. torch.utils.data.DataLoader
# This wraps a Dataset and handles:
# Batching: groups samples into tensors of shape (batch_size, …).
# Shuffling: randomizes sample order.
# Parallel loading: can load multiple samples in parallel with workers.
# Iteration: gives you a Python iterable in your training loop.

# Example continuing above:
from torch.utils.data import DataLoader

# Wrap dataset into a dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over it
for batch in dataloader:    
    x_batch, y_batch = batch    
    print(x_batch.shape, y_batch.shape)    
    break

# Output (say dataset has 10 features):
# torch.Size([4, 10]) torch.Size([4])
# That’s 4 samples grouped into one mini-batch.

# ---
# 3. Built-in datasets (don’t reinvent the wheel)
# PyTorch has torchvision.datasets (images), torchtext.datasets (text), etc. They already subclass Dataset.
# Example with MNIST:
from torchvision import datasets, transforms

# Define transformation (convert to tensor, normalize)
transform = transforms.Compose([transforms.ToTensor()])

# Download and load dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Iterate through batches
for images, labels in train_loader:    
    print(images.shape) # (64, 1, 28, 28)    
    print(labels.shape) # (64,)    
    break

# ---
# 4. How it all works under the hood
# 1. You define a Dataset → tells PyTorch how to get a single sample.

# 2. You wrap it in a DataLoader → PyTorch uses __getitem__ repeatedly to fetch samples, batches them up, shuffles them, and returns them one batch at a time.

# 3. Training loop consumes DataLoader like a generator:
# for xb, yb in train_loader:    
#     preds = model(xb) # forward pass    
#     loss = loss_fn(preds, yb)    
#     ...

# ---
# TL;DR
# Dataset = “What’s my data and how do I get one sample?”
# DataLoader = “Give me batches of samples in the order I want (shuffled or not), maybe using parallel workers.”
# Why separate? Because decoupling storage (Dataset) from iteration/batching (DataLoader) makes code modular and reusable.

# ---
# Would you like me to show you a minimal full training loop (model + optimizer + dataset + dataloader) so you see how Dataset and DataLoader plug into the training process end-to-end?
