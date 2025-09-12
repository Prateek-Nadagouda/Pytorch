# Project: Text Field Classifier (5 Classes)

This file bundles multiple module files into one document for convenience. Each code line is preceded by a comment explaining what it does and why.

---

## File: requirements.txt

The following packages are required to run this project:

- `torch`: Core deep learning framework (PyTorch) used for model definition and training.

---

## File: README.md

A concise README that explains usage, pipeline steps, and practical notes about labeling and model choices. Keep the README with the repo for easy reference.

---

## File: data_prep.py

```python
# Import regular expressions module for string pattern matching and substitution.
import re

# Import argparse to parse command-line arguments.
import argparse

# Define the set of labels we expect. 'unknown' is a catch-all for things heuristics can't decide.
LABELS = [
    'serial_number',
    'make',
    'customer_id',
    'dealer_code',
    'dealer_customer_number',
    'unknown'
]

# Heuristic function: determine if a token looks like a serial number.
# Serial numbers are typically alphanumeric, reasonably long, and have both letters and digits.
def is_serial(s):
    # Remove non-alphanumeric characters.
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    # Check if the cleaned string is alphanumeric and sufficiently long.
    return len(s_clean) > 5 and any(c.isdigit() for c in s_clean) and any(c.isalpha() for c in s_clean)

# Heuristic function: determine if token looks like a customer ID.
# Customer IDs are often numeric or prefix+digits (e.g., CUST12345) and moderate length.
def is_customer_id(s):
    # Strip non-alphanumeric characters.
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    # Check if it matches customer ID patterns.
    return len(s_clean) > 4 and s_clean.isalnum()

# Heuristic function: detect dealer codes which tend to be short uppercase alpha/numeric codes.
def is_dealer_code(s):
    # Remove non-alphanumeric characters.
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    # Dealer codes are short and uppercase.
    return s_clean.isupper() and 2 <= len(s_clean) <= 6

# Heuristic function: detect dealer customer numbers which often are numeric possibly with separators.
def is_dealer_customer_number(s):
    # Remove separators like '-' and '/'.
    s_clean = s.replace("-", "").replace("/", "")
    # Check if the cleaned string is numeric.
    return s_clean.isdigit()

# Set of known makes to get high precision on make detection; extend with your domain's makes.
KNOWN_MAKES = set([
    "sony", "samsung", "lg", "dell", "hp", "lenovo", "bosch", "bajaj", "maruti", "tata"
])

# Heuristic function: determine if token is a 'make' (brand).
def is_make(s):
    # Keep only alphabetic characters and convert to lowercase for normalization.
    s_clean = re.sub(r"[^A-Za-z]", "", s).lower()
    # Check against the known makes.
    return s_clean in KNOWN_MAKES

# Main labeling function applying heuristics in a priority order.
# Priority matters: more specific patterns first (serials) then looser (make).
def label_text(s):
    # Ensure we work with a trimmed string.
    s = s.strip()
    if is_serial(s):
        return 'serial_number'
    elif is_make(s):
        return 'make'
    elif is_customer_id(s):
        return 'customer_id'
    elif is_dealer_code(s):
        return 'dealer_code'
    elif is_dealer_customer_number(s):
        return 'dealer_customer_number'
    else:
        return 'unknown'

# Utility to split compound input strings into candidate tokens and label each.
# Many raw rows contain multiple fields separated by commas/pipes; this attempts to split them.
def split_and_label(text):
    tokens = re.split(r"[,\|]", text)
    return [(token, label_text(token)) for token in tokens]

# The CLI entry point for data_prep.py. Reads raw CSV and outputs token-level labeled CSV.
def main(input_csv, out_csv, text_col='text', max_rows=None):
    # Load the CSV; nrows optional for debugging.
    import pandas as pd
    df = pd.read_csv(input_csv, nrows=max_rows)
    # Generate labeled data.
    df['labeled'] = df[text_col].apply(split_and_label)
    # Save the output CSV.
    df.to_csv(out_csv, index=False)

# Only execute main when running as a script, not when imported as a module.
if __name__ == '__main__':
    # Set up command-line arguments to let user specify input and output paths.
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', required=True, help="Path to input CSV file.")
    p.add_argument('--out_csv', required=True, help="Path to output CSV file.")
    p.add_argument('--text_col', default='text', help="Column name containing text.")
    p.add_argument('--max_rows', type=int, help="Maximum number of rows to process (for debugging).")
    args = p.parse_args()
    main(args.input_csv, args.out_csv, args.text_col, args.max_rows)
```

---

## File: dataset.py

```python
# Import PyTorch core to handle tensors and dataset constructs.
import torch

# Import Dataset base class to create custom dataset object.
from torch.utils.data import Dataset

# Define a dataset object that yields tokenized inputs and labels for training.
class FieldDataset(Dataset):
    # Initialize with CSV path, optional label mapping, and tokenizer/model parameters.
    def __init__(self, csv_path, tokenizer, label2id):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label2id = label2id

    # Return dataset size to allow DataLoader to know how many samples exist.
    def __len__(self):
        return len(self.df)

    # Implement item access to return a dictionary of tensors required by the model.
    def __getitem__(self, idx):
        # Fetch the row by integer position.
        row = self.df.iloc[idx]
        # Tokenize the input text and convert labels to IDs.
        tokens = self.tokenizer(row['text'], padding="max_length", truncation=True, return_tensors="pt")
        label_id = self.label2id[row['label']]
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label_id)
        }
```

---

## File: model.py

```python
# Import PyTorch modules for defining neural network layers.
import torch
import torch.nn as nn

# Import AutoModel to load pre-trained Transformer models.
from transformers import AutoModel

# Define the classification model wrapping a Transformer encoder and a linear head.
class FieldClassifier(nn.Module):
    # Constructor: load backbone, create dropout and classification head sized to number of labels.
    def __init__(self, model_name, num_labels):
        super(FieldClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    # Forward pass that optionally returns loss when labels provided.
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass inputs through transformer to obtain last hidden states.
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return logits
```

---

## File: train.py

```python
# Import standard libraries and training utilities.
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW

# Train function encapsulates dataset creation, model init, training loop, validation, and checkpointing.
def train(train_csv, model_out_dir, model_name='distilbert-base-uncased', epochs=3, batch_size=32, lr=5e-5):
    # Create data loaders for batched iteration.
    from dataset import FieldDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset and split into training and validation sets.
    dataset = FieldDataset(train_csv, tokenizer, label2id={'label': 0})  # Example label2id
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Choose device: GPU if available else CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model.
    model = FieldClassifier(model_name, num_labels=5)  # Example num_labels
    model.to(device)

    # Use AdamW optimizer as recommended for transformer fine-tuning.
    optimizer = AdamW(model.parameters(), lr=lr)

    # Compute total training steps for learning rate scheduler.
    total_steps = len(train_loader) * epochs

    # Track best validation accuracy to save best checkpoint.
    best_val_acc = 0.0

    # Ensure output directory exists for saving models.
    import os
    os.makedirs(model_out_dir, exist_ok=True)

    # Training loop.
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            # Move input tensors to the chosen device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward pass.
            loss, _ = model(input_ids, attention_mask, labels)

            # Backward pass and optimization.
            loss.backward()
            optimizer.step()

        # After each epoch evaluate on validation data.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc}")

        # Save the best model.
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), os.path.join(model_out_dir, "best_model.pth"))
            best_val_acc = val_acc

    # Save final model regardless, useful for diagnostics or further fine-tuning.
    torch.save(model.state_dict(), os.path.join(model_out_dir, "final_model.pth"))

# CLI for train.py when executed as script.
if __name__ == '__main__':
    # Argument parsing for training parameters and paths.
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True, help="Path to training CSV file.")
    p.add_argument('--model_out_dir', required=True, help="Directory to save trained model.")
    p.add_argument('--model_name', default='distilbert-base-uncased', help="Pre-trained model name.")
    p.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    p.add_argument('--batch_size', type=int, default=32, help="Training batch size.")
    p.add_argument('--lr', type=float, default=5e-5, help="Learning rate.")
    args = p.parse_args()
    train(args.train_csv, args.model_out_dir, args.model_name, args.epochs, args.batch_size, args.lr)
```

---

# Text Field Classifier Project

This document provides a comprehensive overview of the text classification system, with well-formatted code and a detailed execution flow.

## Requirements and Setup

### requirements.txt
```
torch==1.9.0
transformers==4.8.2
pandas==1.3.0
scikit-learn==0.24.2
tqdm==4.61.2
```

## Code Components

### data_prep.py
```python name=data_prep.py
# Import regular expressions module for string pattern matching and substitution
import re
import argparse

# Define the set of labels we expect
LABELS = [
    'serial_number',
    'make',
    'customer_id',
    'dealer_code',
    'dealer_customer_number',
    'unknown'
]

# Heuristic function: determine if a token looks like a serial number
def is_serial(s):
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    return len(s_clean) > 5 and any(c.isdigit() for c in s_clean) and any(c.isalpha() for c in s_clean)

# Heuristic function: determine if token looks like a customer ID
def is_customer_id(s):
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    return len(s_clean) > 4 and s_clean.isalnum()

# Heuristic function: detect dealer codes
def is_dealer_code(s):
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    return s_clean.isupper() and 2 <= len(s_clean) <= 6

# Heuristic function: detect dealer customer numbers
def is_dealer_customer_number(s):
    s_clean = s.replace("-", "").replace("/", "")
    return s_clean.isdigit()

# Set of known makes to get high precision on make detection
KNOWN_MAKES = set([
    "sony", "samsung", "lg", "dell", "hp", "lenovo", 
    "bosch", "bajaj", "maruti", "tata"
])

# Heuristic function: determine if token is a 'make' (brand)
def is_make(s):
    s_clean = re.sub(r"[^A-Za-z]", "", s).lower()
    return s_clean in KNOWN_MAKES

# Main labeling function applying heuristics in a priority order
def label_text(s):
    s = s.strip()
    if is_serial(s):
        return 'serial_number'
    elif is_make(s):
        return 'make'
    elif is_customer_id(s):
        return 'customer_id'
    elif is_dealer_code(s):
        return 'dealer_code'
    elif is_dealer_customer_number(s):
        return 'dealer_customer_number'
    else:
        return 'unknown'

# Utility to split compound input strings into candidate tokens and label each
def split_and_label(text):
    tokens = re.split(r"[,\|]", text)
    return [(token, label_text(token)) for token in tokens]

# The CLI entry point for data_prep.py
def main(input_csv, out_csv, text_col='text', max_rows=None):
    import pandas as pd
    df = pd.read_csv(input_csv, nrows=max_rows)
    df['labeled'] = df[text_col].apply(split_and_label)
    df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', required=True, help="Path to input CSV file.")
    p.add_argument('--out_csv', required=True, help="Path to output CSV file.")
    p.add_argument('--text_col', default='text', help="Column name containing text.")
    p.add_argument('--max_rows', type=int, help="Maximum number of rows to process (for debugging).")
    args = p.parse_args()
    main(args.input_csv, args.out_csv, args.text_col, args.max_rows)
```

### dataset.py
```python name=dataset.py
import torch
from torch.utils.data import Dataset

class FieldDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label2id):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = self.tokenizer(
            row['text'], 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        label_id = self.label2id[row['label']]
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label_id)
        }
```

### model.py
```python name=model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class FieldClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(FieldClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return logits
```

### train.py
```python name=train.py
import argparse
import torch
import os
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, AutoTokenizer

# Import local modules
from dataset import FieldDataset
from model import FieldClassifier

def train(train_csv, model_out_dir, model_name='distilbert-base-uncased', 
          epochs=3, batch_size=32, lr=5e-5):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset and split into training and validation sets
    dataset = FieldDataset(train_csv, tokenizer, label2id={'label': 0})  # Example label2id
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = FieldClassifier(model_name, num_labels=5)  # Example num_labels
    model.to(device)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Compute total training steps
    total_steps = len(train_loader) * epochs

    # Track best validation accuracy
    best_val_acc = 0.0

    # Create output directory
    os.makedirs(model_out_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 
                      os.path.join(model_out_dir, "best_model.pth"))
            best_val_acc = val_acc

    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(model_out_dir, "final_model.pth"))
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True, help="Path to training CSV file.")
    p.add_argument('--model_out_dir', required=True, help="Directory to save trained model.")
    p.add_argument('--model_name', default='distilbert-base-uncased', help="Pre-trained model name.")
    p.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    p.add_argument('--batch_size', type=int, default=32, help="Training batch size.")
    p.add_argument('--lr', type=float, default=5e-5, help="Learning rate.")
    args = p.parse_args()
    train(args.train_csv, args.model_out_dir, args.model_name, 
          args.epochs, args.batch_size, args.lr)
```

## System Execution Flow

### High-level Pipeline

1. **data_prep.py**: Produces token-level labeled CSV using heuristics
2. **dataset.py**: Loads CSV and tokenizes text into model-ready tensors
3. **train.py**: Builds DataLoaders, instantiates model, runs training loop
4. **evaluate.py**: Loads checkpoint, runs model on test data, prints metrics
5. **predict.py**: Loads checkpoint and tokenizer, returns label for single text

### Detailed Execution Flow

#### A. Data Preparation (data_prep.py)

1. **Script start**:
   - Read raw CSV file into dataframe
   - Apply field detection and labeling functions

2. **Processing each row**:
   - Split text into tokens using delimiters
   - Apply heuristic functions to label each token
   - Each token-label pair becomes a row in output CSV

3. **Output**:
   - Produces a CSV with tokenized text and corresponding labels

#### B. Dataset Object (FieldDataset)

1. **Initialization**:
   - Loads CSV data into dataframe
   - Creates tokenizer from model name
   - Maps labels to numeric IDs

2. **Item retrieval**:
   - Tokenizes text with padding and truncation
   - Converts labels to tensor IDs
   - Returns dictionary with input_ids, attention_mask, and label

#### C. DataLoader Behavior

- Creates batches from individual samples
- Stacks tensors into batch tensors with shapes:
  - input_ids: (batch_size, max_len)
  - attention_mask: (batch_size, max_len)
  - labels: (batch_size,)

#### D. Model Architecture (FieldClassifier)

1. **Initialization**:
   - Loads pretrained transformer model
   - Creates dropout layer and classification head

2. **Forward Pass**:
   - Process inputs through transformer
   - Take [CLS] token representation
   - Apply dropout and linear classification
   - Calculate loss if labels provided

#### E. Training Loop (train.py)

1. **Setup**:
   - Load dataset and create train/val split
   - Initialize model, optimizer, and device

2. **Epoch Loop**:
   - Training phase: process batches, compute loss, update weights
   - Validation phase: evaluate accuracy
   - Save best checkpoint based on validation accuracy

#### F. Evaluation and Prediction

- **evaluate.py**: Loads model checkpoint, runs inference on test data, computes metrics
- **predict.py**: Loads model for single-text prediction

### Important Considerations

- **Label mapping consistency**: Must use same label2id mapping across training and inference
- **Tokenization**: All samples padded/truncated to max_length
- **Model pooling**: Uses [CLS] token for sentence representation
- **Heuristic quality**: Data preparation heuristics can affect model quality
