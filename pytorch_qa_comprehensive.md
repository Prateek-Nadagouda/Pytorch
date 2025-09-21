# Top 100 PyTorch Interview Questions & Answers

## **1. Basics of PyTorch (Questions 1-15)**

**Q1: What is PyTorch? How is it different from TensorFlow?**
**A:** PyTorch is an open-source machine learning library based on Torch. It provides dynamic computation graphs, making it more flexible than TensorFlow's static graphs (though TensorFlow now supports eager execution).

**Q2: What are Tensors in PyTorch?**
**A:** Tensors are multi-dimensional arrays similar to NumPy arrays but with GPU support and automatic differentiation capabilities. They are the fundamental data structure in PyTorch.

**Q3: How do you create tensors in PyTorch?**
**A:** `torch.tensor([1,2,3])`, `torch.zeros(2,3)`, `torch.ones(2,3)`, `torch.randn(2,3)`, `torch.arange(10)`, `torch.linspace(0,1,10)`, `torch.from_numpy(np_array)`.

**Q4: What's the difference between torch.tensor() and torch.Tensor()?**
**A:** `torch.tensor()` is a function that infers dtype and creates a tensor. `torch.Tensor()` is a class constructor that creates FloatTensor by default.

**Q5: How do you check tensor properties?**
**A:** Use `.shape` or `.size()` for dimensions, `.dtype` for data type, `.device` for location, `.requires_grad` for gradient tracking.

**Q6: What is the difference between .size() and .shape?**
**A:** Both return tensor dimensions. `.size()` is a method, `.shape` is a property. They're functionally equivalent.

**Q7: How do you convert between NumPy and PyTorch?**
**A:** NumPy to PyTorch: `torch.from_numpy(np_array)`. PyTorch to NumPy: `tensor.numpy()` (CPU tensors only).

**Q8: What are the different ways to create tensors with specific values?**
**A:** `torch.zeros()`, `torch.ones()`, `torch.full()`, `torch.eye()` (identity), `torch.empty()` (uninitialized), `torch.zeros_like()`, `torch.ones_like()`.

**Q9: How do you change tensor data type?**
**A:** Use `.to(dtype)`, `.float()`, `.int()`, `.long()`, `.double()`, or during creation: `torch.tensor([1,2], dtype=torch.float32)`.

**Q10: What is tensor indexing and slicing?**
**A:** Similar to NumPy: `tensor[0]`, `tensor[0:2]`, `tensor[:, 1]`, `tensor[tensor > 0]` (boolean indexing), `tensor[[0,2]]` (fancy indexing).

**Q11: How do you concatenate tensors?**
**A:** `torch.cat([tensor1, tensor2], dim=0)` along existing dimension, `torch.stack([tensor1, tensor2], dim=0)` creates new dimension.

**Q12: What is broadcasting in PyTorch?**
**A:** Broadcasting allows operations between tensors of different shapes by automatically expanding smaller tensors following specific rules.

**Q13: How do you perform element-wise operations?**
**A:** Use `+`, `-`, `*`, `/` operators or functions like `torch.add()`, `torch.mul()`, `torch.div()`. Also `torch.sin()`, `torch.exp()`, etc.

**Q14: What is the difference between * and @ operators?**
**A:** `*` performs element-wise multiplication, `@` performs matrix multiplication (same as `torch.matmul()`).

**Q15: How do you handle different tensor devices?**
**A:** Use `.to(device)`, `.cuda()`, `.cpu()`. Check with `torch.cuda.is_available()`, get device with `tensor.device`.

## **2. Tensor Operations & Manipulation (Questions 16-25)**

**Q16: How do you reshape tensors?**
**A:** `tensor.view(new_shape)`, `tensor.reshape(new_shape)`, `tensor.squeeze()` (remove size-1 dims), `tensor.unsqueeze(dim)` (add size-1 dim).

**Q17: What's the difference between view() and reshape()?**
**A:** `view()` requires contiguous memory and returns a view. `reshape()` may return a copy if memory isn't contiguous.

**Q18: How do you transpose tensors?**
**A:** `tensor.t()` (2D only), `tensor.transpose(dim1, dim2)`, `tensor.permute(*dims)` for reordering all dimensions.

**Q19: What are reduction operations?**
**A:** Operations that reduce tensor dimensions: `torch.sum()`, `torch.mean()`, `torch.max()`, `torch.min()`, `torch.std()`, `torch.var()`.

**Q20: How do you perform matrix multiplication?**
**A:** `torch.mm()` (2D), `torch.matmul()` or `@` (general), `torch.bmm()` (batch), `torch.einsum()` (Einstein summation).

**Q21: What is torch.einsum() and when to use it?**
**A:** Einstein summation for complex tensor operations. Useful for custom operations: `torch.einsum('ij,jk->ik', A, B)` for matrix multiplication.

**Q22: How do you handle missing or NaN values?**
**A:** Check with `torch.isnan()`, `torch.isinf()`. Handle with `torch.nan_to_num()`, masking, or custom logic.

**Q23: What are tensor memory layouts?**
**A:** Tensors can be contiguous or non-contiguous in memory. Check with `.is_contiguous()`, make contiguous with `.contiguous()`.

**Q24: How do you clone and detach tensors?**
**A:** `tensor.clone()` creates a copy, `tensor.detach()` removes from computation graph, `tensor.clone().detach()` for both.

**Q25: What is advanced indexing in PyTorch?**
**A:** Boolean indexing: `tensor[tensor > 0]`, fancy indexing: `tensor[[0,2,1]]`, multiple indices: `tensor[indices1, indices2]`.

## **3. Autograd & Gradients (Questions 26-35)**

**Q26: What is autograd in PyTorch?**
**A:** Automatic differentiation engine that builds dynamic computational graphs and computes gradients using backpropagation.

**Q27: How do you enable gradient computation?**
**A:** Set `requires_grad=True`: `x = torch.tensor([1.0], requires_grad=True)` or use `.requires_grad_()` method.

**Q28: What is a computation graph?**
**A:** A directed graph representing mathematical operations. Nodes are operations, edges are tensors. Built dynamically during forward pass.

**Q29: How do you compute gradients?**
**A:** Call `.backward()` on a scalar tensor. Gradients are stored in `.grad` attribute of leaf tensors.

**Q30: What happens when you call backward() multiple times?**
**A:** Gradients accumulate. Use `.zero_grad()` to clear gradients before each backward pass.

**Q31: How do you prevent gradient computation?**
**A:** Use `torch.no_grad()` context manager or `.detach()` method: `with torch.no_grad(): output = model(input)`.

**Q32: What is the difference between .detach() and .clone()?**
**A:** `.detach()` removes tensor from computation graph, `.clone()` creates a copy that still tracks gradients.

**Q33: What are leaf tensors?**
**A:** Tensors created by user (not by operations). They store gradients in `.grad` attribute. Check with `.is_leaf`.

**Q34: How do you implement custom gradients?**
**A:** Create custom `torch.autograd.Function` with static `forward()` and `backward()` methods.

**Q35: What is gradient checkpointing?**
**A:** Trade computation for memory by recomputing activations during backward pass. Use `torch.utils.checkpoint.checkpoint()`.

## **4. Neural Networks & Layers (Questions 36-50)**

**Q36: What is nn.Module?**
**A:** Base class for all neural networks. Provides parameter management, device handling, and training/evaluation modes.

**Q37: How do you create a custom neural network?**
**A:** Inherit from `nn.Module`, implement `__init__()` (define layers) and `forward()` (define forward pass).

**Q38: What's the difference between nn.functional and nn modules?**
**A:** `nn` modules have learnable parameters and state, `nn.functional` are stateless functions that require explicit parameters.

**Q39: What are the common linear layers?**
**A:** `nn.Linear` (fully connected), `nn.Bilinear` (bilinear transformation), `nn.Identity` (identity mapping).

**Q40: What are activation functions available?**
**A:** `nn.ReLU`, `nn.LeakyReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax`, `nn.LogSoftmax`, `nn.ELU`, `nn.GELU`, `nn.Swish`.

**Q41: How do you implement dropout?**
**A:** Use `nn.Dropout(p=0.5)` for regular dropout, `nn.Dropout2d` for spatial dropout. Only active during training.

**Q42: What is batch normalization?**
**A:** `nn.BatchNorm1d/2d/3d` normalizes inputs to have zero mean and unit variance. Improves training stability.

**Q43: How do you access model parameters?**
**A:** `.parameters()` (iterator), `.named_parameters()` (name-parameter pairs), `.state_dict()` (ordered dict), `.modules()` (all modules).

**Q44: What is the purpose of model.train() and model.eval()?**
**A:** `model.train()` enables training mode (dropout, batch norm behave normally). `model.eval()` enables evaluation mode.

**Q45: How do you freeze model parameters?**
**A:** Set `param.requires_grad = False` or use context manager `torch.no_grad()` during forward pass.

**Q46: What are parameter groups in optimizers?**
**A:** Different parameter sets with different optimization settings: `optimizer = SGD([{'params': model.features.parameters(), 'lr': 1e-3}])`.

**Q47: How do you implement weight initialization?**
**A:** Use `torch.nn.init` functions: `nn.init.xavier_uniform_()`, `nn.init.kaiming_normal_()`, or custom initialization in `__init__()`.

**Q48: What is nn.Sequential?**
**A:** Container for stacking layers sequentially: `nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))`.

**Q49: How do you create modular networks?**
**A:** Use `nn.ModuleList` for lists of modules, `nn.ModuleDict` for dictionary of modules, or create custom module classes.

**Q50: What are hooks in PyTorch?**
**A:** Functions that run during forward/backward pass. Register with `.register_forward_hook()`, `.register_backward_hook()` for debugging/visualization.

## **5. Convolutional Neural Networks (Questions 51-60)**

**Q51: What are the main CNN layers in PyTorch?**
**A:** `nn.Conv1d/2d/3d`, `nn.ConvTranspose2d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`, `nn.BatchNorm2d`.

**Q52: How do you calculate conv output size?**
**A:** Formula: `(input_size + 2*padding - kernel_size) / stride + 1`. For 'same' padding: `padding = (kernel_size - 1) / 2`.

**Q53: What is the difference between Conv2d and ConvTranspose2d?**
**A:** `Conv2d` reduces spatial dimensions, `ConvTranspose2d` (deconv) increases spatial dimensions. Used in upsampling/generation.

**Q54: How do you implement residual connections?**
**A:** Store input, apply transformations, add back: `out = F.relu(self.conv1(x)); out = self.conv2(out); return out + x`.

**Q55: What is adaptive pooling?**
**A:** `nn.AdaptiveAvgPool2d(output_size)` pools to fixed output size regardless of input size. Useful for variable input sizes.

**Q56: How do you handle different input sizes in CNNs?**
**A:** Use adaptive pooling, global average pooling, or fully convolutional networks (replace FC layers with conv layers).

**Q57: What is dilated convolution?**
**A:** Convolution with gaps between kernel elements. Use `dilation` parameter in `nn.Conv2d` to increase receptive field.

**Q58: How do you implement attention mechanisms?**
**A:** Self-attention: `attention = F.softmax(Q @ K.T / sqrt(d_k), dim=-1) @ V`. Use `nn.MultiheadAttention` for built-in implementation.

**Q59: What is group convolution?**
**A:** `groups` parameter in `nn.Conv2d` splits input channels into groups, reducing parameters. Depthwise conv: `groups=in_channels`.

**Q60: How do you implement custom CNN architectures?**
**A:** Combine basic layers in custom `nn.Module`, use skip connections, attention, different conv types based on problem requirements.

## **6. Recurrent Neural Networks (Questions 61-70)**

**Q61: What RNN layers are available in PyTorch?**
**A:** `nn.RNN`, `nn.LSTM`, `nn.GRU`. All support bidirectional processing (`bidirectional=True`) and multiple layers (`num_layers`).

**Q62: What is the difference between RNN, LSTM, and GRU?**
**A:** RNN: simple recurrence, vanishing gradient problem. LSTM: gates control information flow. GRU: simplified LSTM with fewer parameters.

**Q63: How do you handle variable-length sequences?**
**A:** Use `nn.utils.rnn.pack_padded_sequence()` before RNN and `nn.utils.rnn.pad_packed_sequence()` after to handle padding efficiently.

**Q64: What does LSTM return?**
**A:** `(output, (h_n, c_n))` where output contains all timesteps, h_n is final hidden state, c_n is final cell state.

**Q65: How do you implement bidirectional RNNs?**
**A:** Set `bidirectional=True` in RNN constructor. Output size becomes `2 * hidden_size`. Handle concatenated forward/backward states.

**Q66: What is teacher forcing?**
**A:** Training technique where ground truth is used as input at each timestep instead of model predictions. Improves training stability.

**Q67: How do you handle batch processing in RNNs?**
**A:** Use `batch_first=True` for (batch, seq, features) format, or default (seq, batch, features). Pad sequences to same length.

**Q68: What is attention in sequence models?**
**A:** Mechanism allowing model to focus on different parts of input sequence. Computed as weighted combination based on query-key similarity.

**Q69: How do you implement sequence-to-sequence models?**
**A:** Use encoder-decoder architecture. Encoder processes input sequence, decoder generates output sequence, often with attention mechanism.

**Q70: What are common RNN applications?**
**A:** Language modeling, machine translation, sentiment analysis, time series forecasting, speech recognition, text generation.

## **7. Loss Functions & Optimization (Questions 71-80)**

**Q71: What are common loss functions for classification?**
**A:** `nn.CrossEntropyLoss` (multi-class), `nn.BCELoss` (binary), `nn.BCEWithLogitsLoss` (binary with logits), `nn.NLLLoss` (negative log-likelihood).

**Q72: What loss functions are used for regression?**
**A:** `nn.MSELoss` (L2), `nn.L1Loss` (MAE), `nn.SmoothL1Loss` (Huber), `nn.PoissonNLLLoss` (Poisson regression).

**Q73: What's the difference between CrossEntropyLoss and NLLLoss?**
**A:** `CrossEntropyLoss` combines `LogSoftmax` and `NLLLoss`. Use with raw logits. `NLLLoss` expects log-probabilities as input.

**Q74: How do you handle class imbalance in loss functions?**
**A:** Use `weight` parameter in loss functions: `nn.CrossEntropyLoss(weight=class_weights)` or implement focal loss for severe imbalance.

**Q75: What optimizers are available in PyTorch?**
**A:** `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, `LBFGS`. Each has different convergence properties and hyperparameters.

**Q76: What is the difference between Adam and AdamW?**
**A:** AdamW decouples weight decay from gradient updates, providing better regularization than Adam's L2 penalty approach.

**Q77: How do you implement learning rate scheduling?**
**A:** Use `torch.optim.lr_scheduler`: `StepLR`, `ExponentialLR`, `ReduceLROnPlateau`, `CosineAnnealingLR`, `OneCycleLR`.

**Q78: What is gradient clipping and how to implement it?**
**A:** Prevents exploding gradients by limiting gradient magnitude: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`.

**Q79: How do you implement custom loss functions?**
**A:** Create function or inherit from `nn.Module`: `def custom_loss(pred, target): return torch.mean((pred - target) ** 2)`.

**Q80: What is the typical training loop structure?**
**A:** 1) `optimizer.zero_grad()` 2) Forward pass 3) Compute loss 4) `loss.backward()` 5) `optimizer.step()` 6) Optional: scheduler step.

## **8. Data Loading & Processing (Questions 81-90)**

**Q81: What is torch.utils.data.Dataset?**
**A:** Abstract class for datasets. Must implement `__len__()` and `__getitem__()`. Base class for custom datasets.

**Q82: How do you create a custom Dataset?**
**A:** Inherit from `Dataset`, implement `__init__()`, `__len__()`, `__getitem__()`. Handle data loading and preprocessing in `__getitem__()`.

**Q83: What is DataLoader and its key parameters?**
**A:** Provides batching, shuffling, parallel loading. Key params: `batch_size`, `shuffle`, `num_workers`, `pin_memory`, `drop_last`.

**Q84: How do you handle different data types in Dataset?**
**A:** Convert to appropriate tensor types in `__getitem__()`: images to float32, labels to long, apply transforms as needed.

**Q85: What are transforms in torchvision?**
**A:** Data preprocessing: `Resize`, `ToTensor`, `Normalize`, `RandomCrop`, `RandomHorizontalFlip`. Use `Compose` to chain transforms.

**Q86: How do you implement data augmentation?**
**A:** Use torchvision transforms: `RandomRotation`, `ColorJitter`, `RandomAffine`, or create custom transforms by inheriting from `Transform`.

**Q87: What is the difference between map-style and iterable-style datasets?**
**A:** Map-style implements `__getitem__()` and `__len__()`. Iterable-style implements `__iter__()` for streaming data.

**Q88: How do you handle memory-efficient data loading?**
**A:** Use `num_workers > 0` for parallel loading, `pin_memory=True` for GPU transfer, lazy loading in `__getitem__()`.

**Q89: What are samplers in DataLoader?**
**A:** Control data ordering: `RandomSampler`, `SequentialSampler`, `WeightedRandomSampler`, `SubsetRandomSampler`. Mutually exclusive with shuffle.

**Q90: How do you handle distributed data loading?**
**A:** Use `DistributedSampler` to ensure each process gets different data subset in multi-GPU training.

## **9. Model Saving, Loading & Deployment (Questions 91-100)**

**Q91: How do you save and load models in PyTorch?**
**A:** Save state_dict: `torch.save(model.state_dict(), 'model.pth')`. Load: `model.load_state_dict(torch.load('model.pth'))`.

**Q92: What's the difference between saving state_dict vs entire model?**
**A:** state_dict saves only parameters (recommended). Saving entire model saves architecture too but is less portable and flexible.

**Q93: How do you save training checkpoints?**
**A:** Save dict with model, optimizer, scheduler, epoch: `torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint.pth')`.

**Q94: What is TorchScript and when to use it?**
**A:** Production deployment format. Use `torch.jit.script()` for Python control flow or `torch.jit.trace()` for simple models.

**Q95: How do you optimize models for inference?**
**A:** Use `model.eval()`, `torch.no_grad()`, TorchScript, quantization, pruning, ONNX export, or specialized inference engines.

**Q96: What is ONNX and how to export to it?**
**A:** Open Neural Network Exchange format for interoperability. Export: `torch.onnx.export(model, dummy_input, 'model.onnx')`.

**Q97: How do you implement model quantization?**
**A:** Post-training quantization: `torch.quantization.quantize_dynamic()`. Quantization-aware training: use `torch.quantization.fakeQuantize`.

**Q98: What is mixed precision training?**
**A:** Uses float16 for forward pass, float32 for gradients. Use `torch.cuda.amp.GradScaler` and `autocast()` for automatic mixed precision.

**Q99: How do you handle model versioning?**
**A:** Save metadata with models, use semantic versioning, maintain backward compatibility, document model architecture and requirements.

**Q100: What are best practices for model deployment?**
**A:** Use TorchScript/ONNX, implement proper error handling, monitor performance, use appropriate hardware, implement model serving APIs, version control.