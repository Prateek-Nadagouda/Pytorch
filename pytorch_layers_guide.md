# Different Ways to Create Layers in PyTorch & Their Differences

## **1. Using nn.Module Classes vs nn.functional**

### **nn.Module Approach (Stateful)**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Has learnable parameters
        self.relu = nn.ReLU()           # Stateless but consistent interface
        
    def forward(self, x):
        return self.relu(self.linear(x))
```

### **nn.functional Approach (Stateless)**
```python
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 10))
        self.bias = nn.Parameter(torch.randn(5))
        
    def forward(self, x):
        return F.relu(F.linear(x, self.weight, self.bias))
```

**Key Differences:**
- **nn.Module**: Automatic parameter management, easier to use, consistent interface
- **nn.functional**: More control, explicit parameters, functional programming style
- **Memory**: nn.Module stores parameters, nn.functional requires explicit parameter passing
- **Use Case**: Use nn.Module for standard layers, nn.functional for custom operations

---

## **2. Sequential vs ModuleList vs ModuleDict**

### **nn.Sequential (Linear Flow)**
```python
# Method 1: Constructor arguments
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Method 2: OrderedDict
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(10, 5)),
    ('relu', nn.ReLU()),
    ('linear2', nn.Linear(5, 1))
]))

# Method 3: Add modules dynamically
model = nn.Sequential()
model.add_module('linear1', nn.Linear(10, 5))
model.add_module('relu', nn.ReLU())
```

### **nn.ModuleList (List Container)**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### **nn.ModuleDict (Dictionary Container)**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            'linear1': nn.Linear(10, 5),
            'activation': nn.ReLU(),
            'linear2': nn.Linear(5, 1)
        })
        
    def forward(self, x):
        x = self.layers['linear1'](x)
        x = self.layers['activation'](x)
        x = self.layers['linear2'](x)
        return x
```

**Key Differences:**
- **Sequential**: Fixed order, automatic forward pass, simplest for linear architectures
- **ModuleList**: Flexible control flow, manual forward pass, good for repeated blocks
- **ModuleDict**: Named access, flexible architecture, good for conditional paths

---

## **3. Linear/Dense Layers**

### **Standard Linear Layer**
```python
# Basic linear transformation: y = xW^T + b
linear = nn.Linear(in_features=10, out_features=5, bias=True)

# Without bias
linear_no_bias = nn.Linear(10, 5, bias=False)

# Custom initialization
linear = nn.Linear(10, 5)
nn.init.xavier_uniform_(linear.weight)
nn.init.zeros_(linear.bias)
```

### **Bilinear Layer**
```python
# Bilinear transformation: y = x1^T A x2 + b
bilinear = nn.Bilinear(in1_features=10, in2_features=15, out_features=5)

# Usage
output = bilinear(input1, input2)
```

### **Identity Layer**
```python
# Placeholder layer that returns input unchanged
identity = nn.Identity()

# Useful in conditional architectures
def create_layer(use_layer=True):
    return nn.Linear(10, 10) if use_layer else nn.Identity()
```

---

## **4. Activation Layers**

### **Standard Activations**
```python
# ReLU variants
relu = nn.ReLU(inplace=False)           # Standard ReLU
leaky_relu = nn.LeakyReLU(0.01)         # Leaky ReLU
elu = nn.ELU(alpha=1.0)                 # Exponential Linear Unit
gelu = nn.GELU()                        # Gaussian Error Linear Unit

# Sigmoid-based
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# Softmax
softmax = nn.Softmax(dim=1)             # Specify dimension
log_softmax = nn.LogSoftmax(dim=1)      # Numerically stable
```

### **Advanced Activations**
```python
# Swish/SiLU
swish = nn.SiLU()                       # x * sigmoid(x)

# Mish
mish = nn.Mish()                        # x * tanh(softplus(x))

# Parametric ReLU
prelu = nn.PReLU(num_parameters=1)      # Learnable negative slope
```

**Differences:**
- **ReLU**: Fast, simple, but can cause dead neurons
- **LeakyReLU**: Prevents dead neurons, small negative slope
- **ELU**: Smooth, mean activation closer to zero
- **GELU**: Probabilistic activation, good for transformers
- **Swish**: Smooth, self-gated, performs well in deep networks

---

## **5. Convolutional Layers**

### **Standard Convolution**
```python
# 2D Convolution
conv2d = nn.Conv2d(
    in_channels=3,          # Input channels (RGB = 3)
    out_channels=32,        # Number of filters
    kernel_size=3,          # Filter size (3x3)
    stride=1,               # Step size
    padding=1,              # Padding amount
    dilation=1,             # Dilation rate
    groups=1,               # Group convolution
    bias=True              # Include bias
)

# Different kernel sizes
conv_3x3 = nn.Conv2d(3, 32, kernel_size=3)
conv_5x5 = nn.Conv2d(3, 32, kernel_size=5)
conv_1x1 = nn.Conv2d(32, 64, kernel_size=1)  # Pointwise conv
```

### **Specialized Convolutions**
```python
# Depthwise Separable Convolution
depthwise = nn.Conv2d(32, 32, 3, groups=32)    # groups = in_channels
pointwise = nn.Conv2d(32, 64, 1)               # 1x1 conv

# Transposed Convolution (Deconvolution)
deconv = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)

# 1D and 3D versions
conv1d = nn.Conv1d(10, 20, kernel_size=5)      # For sequences
conv3d = nn.Conv3d(1, 8, kernel_size=3)        # For volumes
```

### **Grouped and Dilated Convolutions**
```python
# Grouped Convolution (reduces parameters)
grouped_conv = nn.Conv2d(32, 64, 3, groups=8)

# Dilated/Atrous Convolution (larger receptive field)
dilated_conv = nn.Conv2d(32, 64, 3, dilation=2)

# Separable Convolution
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

---

## **6. Pooling Layers**

### **Standard Pooling**
```python
# Max Pooling
maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
maxpool_with_indices = nn.MaxPool2d(2, return_indices=True)  # For unpooling

# Average Pooling
avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)

# Global Average Pooling (any input size -> fixed output)
global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
```

### **Advanced Pooling**
```python
# Fractional Max Pooling (random pooling)
frac_maxpool = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=(0.5, 0.5))

# LP Pooling (generalized pooling)
lp_pool = nn.LPPool2d(norm_type=2, kernel_size=2)  # L2 pooling

# Adaptive Max Pooling
adaptive_maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
```

---

## **7. Normalization Layers**

### **Batch Normalization**
```python
# Standard Batch Norm
bn1d = nn.BatchNorm1d(num_features=100)         # For FC layers
bn2d = nn.BatchNorm2d(num_features=32)          # For Conv layers
bn3d = nn.BatchNorm3d(num_features=16)          # For 3D Conv

# Custom parameters
bn_custom = nn.BatchNorm2d(
    num_features=32,
    eps=1e-5,              # Small constant for numerical stability
    momentum=0.1,          # Momentum for running statistics
    affine=True,           # Learnable affine parameters
    track_running_stats=True  # Track running mean/var
)
```

### **Other Normalization Methods**
```python
# Layer Normalization (good for RNNs/Transformers)
layer_norm = nn.LayerNorm(normalized_shape=[100])

# Group Normalization (alternative to batch norm)
group_norm = nn.GroupNorm(num_groups=8, num_channels=32)

# Instance Normalization (for style transfer)
instance_norm = nn.InstanceNorm2d(num_features=32)

# Local Response Normalization
local_response_norm = nn.LocalResponseNorm(size=5)
```

**Key Differences:**
- **BatchNorm**: Normalizes across batch dimension, good for CNNs
- **LayerNorm**: Normalizes across feature dimension, good for RNNs/Transformers
- **GroupNorm**: Independent of batch size, good for small batches
- **InstanceNorm**: Normalizes each instance separately, good for style transfer

---

## **8. Recurrent Layers**

### **Standard RNN Layers**
```python
# Basic RNN
rnn = nn.RNN(
    input_size=10,          # Input feature size
    hidden_size=20,         # Hidden state size
    num_layers=2,           # Number of stacked layers
    bias=True,              # Include bias
    batch_first=False,      # Input format (seq, batch, features) vs (batch, seq, features)
    dropout=0.0,            # Dropout between layers
    bidirectional=False     # Bidirectional processing
)

# LSTM (Long Short-Term Memory)
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,       # Common to use batch_first=True
    bidirectional=True      # Double the output size
)

# GRU (Gated Recurrent Unit)
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1)
```

### **RNN Cells (Single Step)**
```python
# For custom loops and more control
rnn_cell = nn.RNNCell(input_size=10, hidden_size=20)
lstm_cell = nn.LSTMCell(input_size=10, hidden_size=20)
gru_cell = nn.GRUCell(input_size=10, hidden_size=20)

# Usage in custom forward pass
def forward(self, input_seq):
    hidden = torch.zeros(batch_size, hidden_size)
    outputs = []
    for input_t in input_seq:
        hidden = self.rnn_cell(input_t, hidden)
        outputs.append(hidden)
    return torch.stack(outputs)
```

---

## **9. Attention Layers**

### **Multi-Head Attention**
```python
# Transformer-style attention
multihead_attn = nn.MultiheadAttention(
    embed_dim=512,          # Embedding dimension
    num_heads=8,            # Number of attention heads
    dropout=0.1,            # Attention dropout
    bias=True,              # Include bias
    batch_first=True        # Input format
)

# Usage
output, attn_weights = multihead_attn(query, key, value, key_padding_mask=mask)
```

### **Custom Attention**
```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
        context = torch.sum(attention_weights * hidden_states, dim=1)
        return context, attention_weights
```

---

## **10. Regularization Layers**

### **Dropout Variants**
```python
# Standard Dropout
dropout = nn.Dropout(p=0.5)                    # General dropout

# Spatial Dropout (for CNNs)
dropout2d = nn.Dropout2d(p=0.5)                # Drops entire feature maps
dropout3d = nn.Dropout3d(p=0.5)                # For 3D data

# Alpha Dropout (for SELU networks)
alpha_dropout = nn.AlphaDropout(p=0.5)         # Maintains mean/variance
```

---

## **11. Custom Layer Creation**

### **Method 1: Simple Function Wrapper**
```python
class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sigmoid(x) * torch.tanh(x)  # Custom activation
```

### **Method 2: Parametric Layer**
```python
class WeightedLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size))
        self.bias = nn.Parameter(torch.zeros(input_size))
    
    def forward(self, x):
        return x * self.weight + self.bias
```

### **Method 3: Complex Custom Layer**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
```

---

## **12. Layer Initialization Methods**

### **Built-in Initialization**
```python
# Weight initialization
nn.init.xavier_uniform_(layer.weight)           # Xavier/Glorot uniform
nn.init.xavier_normal_(layer.weight)            # Xavier/Glorot normal
nn.init.kaiming_uniform_(layer.weight)          # He uniform (good for ReLU)
nn.init.kaiming_normal_(layer.weight)           # He normal

# Bias initialization
nn.init.zeros_(layer.bias)                      # Zero initialization
nn.init.constant_(layer.bias, 0.1)              # Constant value

# Custom initialization
nn.init.uniform_(layer.weight, -0.1, 0.1)      # Uniform distribution
nn.init.normal_(layer.weight, mean=0, std=0.01) # Normal distribution
```

### **Custom Initialization Function**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Apply to entire model
model.apply(init_weights)
```

---

## **Summary of Key Differences**

| Aspect | nn.Module | nn.functional | Sequential | ModuleList/Dict |
|--------|-----------|---------------|------------|-----------------|
| **Parameter Management** | Automatic | Manual | Automatic | Automatic |
| **Flexibility** | High | Highest | Low | High |
| **Code Complexity** | Medium | High | Low | Medium |
| **Memory Usage** | Efficient | Explicit | Efficient | Efficient |
| **Best For** | Standard layers | Custom ops | Simple models | Complex architectures |

**Choosing the Right Approach:**
- **Simple models**: Use `nn.Sequential`
- **Standard architectures**: Use `nn.Module` classes
- **Custom operations**: Use `nn.functional` or custom modules
- **Dynamic architectures**: Use `nn.ModuleList/Dict`
- **Research/experimentation**: Custom modules with full control