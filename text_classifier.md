Here’s the formatted and beautified version of your markdown document with Python code segments made more readable:

```markdown name=text_classifier.md
# Project: Text Field Classifier (5 Classes)

This file bundles multiple module files into one document for convenience. Each code line is preceded by a comment explaining what it does and why.

---

## file: requirements.txt

The following packages are required to run this project:
- `torch`: Core deep learning framework (PyTorch) used for model definition.

---

## file: README.md

A concise README that explains usage, pipeline steps, and practical notes about labeling and model choices. Keep the README with the repository.

---

## file: data_prep.py

### Import Libraries
```python
# Import regular expressions module for string pattern matching and substitution.
import re

# Import argparse to parse command-line arguments.
import argparse
```

### Define the Set of Labels
```python
# Define the set of labels we expect. 'unknown' is a catch-all for things heuristics can't decide.
LABELS = [
    'serial_number',
    'make',
    'customer_id',
    'dealer_code',
    'dealer_customer_number',
    'unknown'
]
```

### Heuristic Functions
```python
# Determine if a token looks like a serial number.
def is_serial(s):
    # Remove non-alphanumeric characters.
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    # Check if it contains both letters and digits and has a reasonable length.
    return len(s_clean) > 5 and any(c.isdigit() for c in s_clean) and any(c.isalpha() for c in s_clean)

# Determine if a token looks like a customer ID.
def is_customer_id(s):
    # Strip non-alphanumeric characters.
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    # Check for common patterns like numeric or prefix+digits (e.g., CUST12345).
    return s_clean.isdigit() or (len(s_clean) > 4 and s_clean[:4].isalpha() and s_clean[4:].isdigit())

# Detect dealer codes which tend to be short uppercase alpha/numeric codes.
def is_dealer_code(s):
    s_clean = re.sub(r"[^A-Za-z0-9]", "", s)
    return len(s_clean) <= 5 and s_clean.isalnum() and s_clean.isupper()

# Detect dealer customer numbers which are often numeric with possible separators.
def is_dealer_customer_number(s):
    s_clean = s.replace("-", "").replace("/", "")
    return s_clean.isdigit()

# Determine if a token is a 'make' (brand).
KNOWN_MAKES = set(["sony", "samsung", "lg", "dell", "hp", "lenovo", "bosch", "bajaj", "maruti", "tata"])

def is_make(s):
    # Keep only alphabetic characters and normalize to lowercase.
    s_clean = re.sub(r"[^A-Za-z]", "", s).lower()
    return s_clean in KNOWN_MAKES
```

### Main Labeling Function
```python
# Apply heuristics in a priority order.
def label_text(s):
    s = s.strip()
    if is_serial(s):
        return 'serial_number'
    elif is_customer_id(s):
        return 'customer_id'
    elif is_dealer_code(s):
        return 'dealer_code'
    elif is_dealer_customer_number(s):
        return 'dealer_customer_number'
    elif is_make(s):
        return 'make'
    else:
        return 'unknown'
```

### Utility for Splitting and Labeling
```python
# Split compound input strings into candidate tokens and label each.
def split_and_label(text):
    tokens = re.split(r"[,\|]", text)
    return [(token, label_text(token)) for token in tokens]
```

### CLI Entry Point
```python
# Read raw CSV and output token-level labeled CSV.
def main(input_csv, out_csv, text_col='text', max_rows=None):
    import pandas as pd
    # Load the CSV; nrows optional for debugging.
    df = pd.read_csv(input_csv, nrows=max_rows)
    labeled_rows = []

    for _, row in df.iterrows():
        text = row[text_col]
        labeled_rows.extend(split_and_label(text))

    # Save labeled data to CSV.
    labeled_df = pd.DataFrame(labeled_rows, columns=['token', 'label'])
    labeled_df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    # Set up command-line arguments.
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', required=True, help="Path to raw input CSV.")
    p.add_argument('--out_csv', required=True, help="Path to output labeled CSV.")
    p.add_argument('--text_col', default='text', help="Column name containing text.")
    p.add_argument('--max_rows', type=int, default=None, help="Optional limit on rows to process.")
    args = p.parse_args()

    # Execute main function with parsed arguments.
    main(args.input_csv, args.out_csv, args.text_col, args.max_rows)
```

---

## file: dataset.py

### Import Libraries
```python
import torch
from torch.utils.data import Dataset
```

### Define Custom Dataset
```python
class FieldDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label_to_id, max_len=128):
        import pandas as pd
        # Load CSV and initialize tokenizer and label mapping.
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = row['label']

        # Tokenize the text.
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.label_to_id[label], dtype=torch.long)
        }
```

---

## file: model.py

### Import Libraries
```python
import torch
import torch.nn as nn
from transformers import AutoModel
```

### Define Classification Model
```python
class FieldClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super(FieldClassifier, self).__init__()
        # Load pre-trained Transformer backbone.
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass inputs through transformer to obtain last hidden states.
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits
```

---

The code is now formatted for readability with well-structured sections and formatted Python code blocks. Let me know if you'd like additional edits or clarifications!
