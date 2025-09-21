# Complete guide — Hyperparameter tuning with PyTorch (practical, no fluff)

Deliberately direct: below I’ll give you principled explanations, pragmatic tips, and copy-pasteable code for each technique you asked for. Where the technique is a big system (DARTS, PBT, NAS), I’ll give a compact, runnable *starter* example + notes on scaling it to production. Use this as a blueprint — adapt search spaces, budgets and infra to your problem.

---

# 1. Quick common setup (model / dataloaders / training loop)

A tiny, reproducible PyTorch training loop we’ll reuse across examples. Keep this file as `train_utils.py`.

```python
# train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any

def get_toy_data(n=1000, d=20, n_classes=2, batch_size=64):
    X = torch.randn(n, d)
    y = (X[:, 0].abs() + 0.1*torch.randn(n) > 0.5).long()
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(ds, batch_size=batch_size)

class SimpleNet(nn.Module):
    def __init__(self, d=20, hidden=64, n_layers=2, n_classes=2, act=nn.ReLU):
        super().__init__()
        layers = []
        in_ch = d
        for _ in range(n_layers):
            layers.append(nn.Linear(in_ch, hidden))
            layers.append(act())
            in_ch = hidden
        layers.append(nn.Linear(in_ch, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    correct = total = 0
    loss_f = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_f(out, yb)
            total_loss += loss.item()*xb.size(0)
            pred = out.argmax(dim=1)
            correct += (pred==yb).sum().item()
            total += xb.size(0)
    return total_loss/total, correct/total
```

---

# 2. Manual / Grid Search — Systematic exploration

When: baseline, small discrete search spaces.

Example: brute-force grid search wrapper.

```python
# grid_search.py
from itertools import product
import torch, random
from train_utils import SimpleNet, get_toy_data, evaluate
import torch.nn as nn, torch.optim as optim

def train_one(cfg):
    train_dl, val_dl = get_toy_data(batch_size=cfg['batch_size'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet(d=20, hidden=cfg['hidden'], n_layers=cfg['n_layers']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_f = nn.CrossEntropyLoss()
    for epoch in range(cfg['epochs']):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_f(model(xb), yb)
            loss.backward()
            opt.step()
    val_loss, val_acc = evaluate(model, val_dl, device)
    return {'val_loss': val_loss, 'val_acc': val_acc}

def grid_search(space: dict):
    keys = list(space.keys())
    best = None
    for vals in product(*(space[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        res = train_one(cfg)
        print(cfg, res)
        if best is None or res['val_loss'] < best[0]['val_loss']:
            best = (res, cfg)
    print("Best:", best)

if __name__ == "__main__":
    space = {
      'hidden': [32, 64],
      'n_layers': [1,2],
      'lr': [1e-3, 1e-4],
      'batch_size': [64],
      'epochs': [5]
    }
    grid_search(space)
```

**Pros:** Simple, exhaustive.
**Cons:** Exponential cost; wasteful when some dims matter more than others.

---

# 3. Random Search — efficient baseline

When: high-D spaces; often beats grid for same compute budget.

Example: random sampling wrapper:

```python
# random_search.py
import random, math
from train_utils import SimpleNet, get_toy_data, evaluate
import torch.nn as nn, torch.optim as optim, torch
from functools import partial

def sample(space):
    return {k: (random.choice(v) if isinstance(v, list) else v()) for k,v in space.items()}

def train_one(cfg):
    train_dl, val_dl = get_toy_data(batch_size=cfg['batch_size'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet(d=20, hidden=cfg['hidden'], n_layers=cfg['n_layers']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_f = nn.CrossEntropyLoss()
    for epoch in range(cfg['epochs']):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_f(model(xb), yb)
            loss.backward()
            opt.step()
    val_loss, val_acc = evaluate(model, val_dl, device)
    return {'val_loss': val_loss, 'val_acc': val_acc}

if __name__ == "__main__":
    space = {
      'hidden': [16,32,64,128],
      'n_layers': [1,2,3],
      'lr': lambda: 10**random.uniform(-5,-2),
      'batch_size': [32,64,128],
      'epochs': [5]
    }
    trials = 20
    best = None
    for _ in range(trials):
        cfg = sample(space)
        res = train_one(cfg)
        print(cfg, res)
        if best is None or res['val_loss'] < best[0]['val_loss']:
            best = (res, cfg)
    print("Best:", best)
```

**Why random > grid**: if only a few hyperparameters matter, random allocates budget better across dimensions.

---

# 4. Bayesian Optimization — Optuna (smart search)

When: expensive runs, need to find good hyperparams with fewer trials.

Install: `pip install optuna`

Minimal Optuna example (optimize lr, hidden, layers):

```python
# optuna_example.py
import optuna
import torch, torch.nn as nn, torch.optim as optim
from train_utils import SimpleNet, get_toy_data, evaluate

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden = trial.suggest_categorical('hidden', [32,64,128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    batch_size = trial.suggest_categorical('batch_size', [32,64])
    epochs = 8

    train_dl, val_dl = get_toy_data(batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet(d=20, hidden=hidden, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_f(model(xb), yb)
            loss.backward()
            opt.step()

    val_loss, val_acc = evaluate(model, val_dl, device)
    trial.report(val_loss, epoch)

    # Prune unpromising trials
    if trial.should_prune():
        raise optuna.TrialPruned()
    return val_loss

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=50, timeout=600)
    print("Best:", study.best_trial.params, study.best_value)
```

**Notes:**

* Use pruning (Optuna integration with intermediate metrics) to stop poor trials early.
* Optuna supports multi-objective optimization (Pareto) — see section on multi-objective.

---

# 5. Population-Based Training (PBT) — Ray Tune

When: need dynamic hyperparameters (e.g., learning-rate schedules, mutation), large parallel resources.

Install: `pip install ray[tune] ray[default]` (versions matter).

Small Ray Tune PBT example (conceptual):

```python
# pbt_tune.py
# pip install ray[tune]
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from train_utils import SimpleNet, get_toy_data, evaluate
import torch, torch.nn as nn, torch.optim as optim

def train_tune(config):
    train_dl, val_dl = get_toy_data(batch_size=config["batch_size"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet(d=20, hidden=config["hidden"], n_layers=config["n_layers"]).to(device)
    opt = optim.Adam(model.parameters(), lr=config["lr"])
    loss_f = nn.CrossEntropyLoss()
    for epoch in range(1, 11):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_f(model(xb), yb)
            loss.backward()
            opt.step()
        val_loss, val_acc = evaluate(model, val_dl, device)
        tune.report(val_loss=val_loss, val_acc=val_acc)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=2,
        metric="val_loss",
        mode="min",
        hyperparam_mutations={
            "lr": [1e-4, 5e-4, 1e-3, 5e-3],
            "hidden": [32, 64, 128],
        })
    analysis = tune.run(
        train_tune,
        name="pbt_test",
        scheduler=pbt,
        num_samples=4,
        resources_per_trial={"cpu":1, "gpu": 0.25}
    )
    print("Best config:", analysis.get_best_config(metric="val_loss", mode="min"))
```

**PBT idea:** training runs evolve: underperformers copy weights+hyperparams from better runs then mutate hyperparams. Good for non-stationary schedules.

---

# 6. Neural Architecture Search — Random NAS & DARTS (sketch + starter)

NAS is huge. Two pragmatic options:

* **Random NAS**: sample architectures randomly and evaluate (cheap baseline).
* **DARTS** (Differentiable ARchitecture Search): a gradient-based continuous relaxation of the architecture search space.

## Random NAS example (simple)

We sample hyperparams + small architectures (layer sizes, skip connections):

```python
# random_nas.py
import random
from train_utils import get_toy_data, evaluate
import torch, torch.nn as nn, torch.optim as optim

def build_net(d=20, config=None):
    layers = []
    in_ch = d
    for s in config['sizes']:
        layers.append(nn.Linear(in_ch, s))
        layers.append(nn.ReLU())
        in_ch = s
    layers.append(nn.Linear(in_ch, 2))
    return nn.Sequential(*layers)

def random_search_arch(trials=20):
    best=None
    for _ in range(trials):
        cfg = {'sizes': [random.choice([16,32,64,128]) for _ in range(random.choice([1,2,3]))],
               'lr': 10**random.uniform(-5,-2),
               'batch': random.choice([32,64])}
        train_dl, val_dl = get_toy_data(batch_size=cfg['batch'])
        model = build_net(config=cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
        opt = optim.Adam(model.parameters(), lr=cfg['lr'])
        loss_f = nn.CrossEntropyLoss()
        for epoch in range(5):
            model.train()
            for xb,yb in train_dl:
                xb,yb = xb.to(next(model.parameters()).device), yb.to(next(model.parameters()).device)
                opt.zero_grad()
                loss_f(model(xb), yb).backward()
                opt.step()
        val_loss, val_acc = evaluate(model, val_dl)
        if best is None or val_loss < best[0]:
            best = (val_loss, cfg)
    print("Best arch:", best)
```

## DARTS (notes & pointers)

* DARTS requires a separate architecture-parameter optimization loop (bi-level optimization). Implementing DARTS from scratch is non-trivial; use research implementations (e.g., `nni`, `darts` repo).
* Practical: try **ProxylessNAS**, **FBNet**, or **NAS-Bench** libraries for reproducible search.

If you want a DARTS runnable start, I can provide a compact DARTS skeleton (longer) — but use a tested repo for production.

---

# 7. Multi-objective optimization (Pareto fronts, NSGA-II)

When: optimize tradeoffs (accuracy vs. latency, accuracy vs. model size).

## Optuna multi-objective (simple)

```python
# optuna_multi.py
import optuna
from train_utils import SimpleNet, get_toy_data, evaluate
import torch, torch.nn as nn, torch.optim as optim

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden = trial.suggest_categorical('hidden', [32,64,128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    # objective1: val_loss, objective2: model size (params)
    model = SimpleNet(d=20, hidden=hidden, n_layers=n_layers)
    size = sum(p.numel() for p in model.parameters())
    # train quickly (toy)
    train_dl, val_dl = get_toy_data(batch_size=64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.CrossEntropyLoss()
    for _ in range(3):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_f(model(xb), yb).backward(); opt.step()
    val_loss, val_acc = evaluate(model, val_dl, device)
    return val_loss, float(size)

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=30)
pareto = study.best_trials
print("Pareto set:", pareto)
```

## NSGA-II (pymoo)

For heavy multi-objective evolutionary search, use `pymoo` or `DEAP` NSGA-II. Sketch:

```python
# pseudo: use pymoo to define objectives function that trains model and returns [1-acc, size]
# from pymoo.optimize import minimize
# from pymoo.algorithms.moo.nsga2 import NSGA2
# etc.
```

**Practical advice:** start with Optuna multi-objective for simplicity; move to NSGA-II when you need evolutionary operators.

---

# 8. Learning rate finding, cyclical schedules, and one-cycle

* **LR finder**: sweep lr exponentially and pick lr where loss decreases fastest. fastai has `lr_find`. Here’s a simple LR finder implementation outline:

```python
# lr_finder.py (sketch)
# approach: increase lr every batch from 1e-7 -> 1, record loss, pick lr a bit before loss explodes.
```

* **Cyclical LR**: `torch.optim.lr_scheduler.CyclicLR` (triangular, triangular2).
* **One-cycle**: `torch.optim.lr_scheduler.OneCycleLR` — often gives strong speedups.

Example (OneCycle + AMP):

```python
# onecycle_example.py
import torch, torch.nn as nn, torch.optim as optim
from train_utils import SimpleNet, get_toy_data, evaluate
from torch.cuda.amp import GradScaler, autocast

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleNet().to(device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
train_dl, val_dl = get_toy_data()
total_steps = len(train_dl) * 10
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=total_steps)
scaler = GradScaler()

for epoch in range(10):
    model.train()
    for xb,yb in train_dl:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        with autocast():
            loss = nn.CrossEntropyLoss()(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()
    print("epoch done")
```

**Rule of thumb:** combine OneCycle with SGD for best results on many vision/NLP tasks.

---

# 9. Mixed precision, AMP and gradient scaling (practical)

* Use `torch.cuda.amp.autocast()` and `GradScaler`.
* Benefits: memory savings, faster throughput on modern GPUs.

(Seen above in OneCycle example.)

**Pitfalls:** reduced numerical stability for some ops (e.g., LSTM layernorm) — test.

---

# 10. Automated pipelines with progressive search

Progressive search: start with cheap proxies (smaller dataset / fewer epochs / smaller model) and progressively increase fidelity for promising candidates.

Example pattern:

1. Phase A: 1 epoch on 10% of data; cheap architectures.
2. Phase B: 3 epochs on 50% of data for top-k candidates.
3. Phase C: full training on final candidates.

Implementable with Optuna pruners, Ray Tune’s `ConcurrencyLimiter` + schedulers, or custom orchestration.

**Code design:** create evaluation function that accepts `budgets` and use conditional re-eval for progressed candidates.

---

# 11. W\&B Sweeps integration (experiment tracking)

Install: `pip install wandb`

YAML sweep example:

```yaml
# sweep.yaml
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  hidden:
    values: [32,64,128]
  n_layers:
    values: [1,2,3]
```

Code:

```python
import wandb
from train_utils import SimpleNet, get_toy_data, evaluate
import torch, torch.nn as nn, torch.optim as optim

def main():
    wandb.init()
    config = wandb.config
    train_dl, val_dl = get_toy_data(batch_size=config.batch_size or 64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet(hidden=config.hidden, n_layers=config.n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(5):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            opt.step()
        val_loss, val_acc = evaluate(model, val_dl, device)
        wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch})

if __name__ == "__main__":
    sweep_id = wandb.sweep("sweep.yaml", project="pytorch-tune")
    wandb.agent(sweep_id, function=main)
```

**Why use W\&B:** centralized experiment logs, compare runs, visualize hyperparam importance, easily integrate with Optuna/Ray.

---

# 12. AutoML pipelines & hands-off optimization

Options:

* Compose Optuna + Ray Tune + W\&B for automated pipeline with progressive fidelity, parallelism, and experiment tracking.
* Use AutoML libraries (e.g., `AutoGluon`, `Auto-Keras`) if you want high-level automation; they often wrap models and search internally. For PyTorch-first teams, build Optuna+Ray pipeline.

Pattern:

* Use Ray Tune for distributed schedule + PBT.
* Use Optuna for sophisticated samplers and pruning.
* Log everything to W\&B + store artifacts (models, checkpoints).
* Use progressive search budgets (smaller proxy, enlarge for top trials).

---

# 13. Advanced training loops with AMP & gradient scaling

We touched on this. Extra tips:

* Use `torch.backends.cudnn.benchmark = True` for variable input sizes improvement.
* Use `scaler.unscale_(optimizer)` + `torch.nn.utils.clip_grad_norm_` when clipping gradients with scaler.

Snippet for clipping:

```python
scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(opt)
scaler.update()
```

---

# 14. Comprehensive reporting & analysis tools

* **Tools:** W\&B, TensorBoard, MLflow.
* **Metrics to log:** val loss, val acc, train loss, lr, GPU mem, throughput, epoch time, model size, FLOPs (profile).
* **Post-hoc analysis:** hyperparameter importance (Optuna), parallel coordinates plots (W\&B), Pareto front plotting for multi-objective.

---

# 15. Multi-objective example: plotting Pareto (concept)

After multi-objective Optuna, you get trials forming a Pareto front — visualize via W\&B or Matplotlib.

---

# 16. Best Practices (decision trees, search space design, validation)

Be direct — this is the decision tree boiled to essentials:

1. **Budget tiny (few GPUs, quick):** Random search + LR finder + OneCycle → baseline.
2. **Medium budget:** Optuna (TPE) with pruning + progressive fidelity + W\&B tracking.
3. **Large infra (many GPUs, need adaptive policies):** Ray Tune with PBT or Evolutionary + NAS + multi-objective NSGA-II.
4. **Need automated production pipelines:** Combine Optuna/Tune + W\&B + checkpointing + CI jobs.

Search space design principles:

* **Parameterize sensibly:** ranges in log-space for lr / weight decay; ints for layers/units; categorical for activations/optimizers.
* **Constrain coupling:** avoid letting meaningless combos exist (e.g., `batch_size` too big with tiny GPU).
* **Use conditional search spaces:** only sample `dropout` if model supports it.
* **Start coarse → refine**: broad ranges first, then narrow after few studies.

Validation & pitfalls:

* **Holdout vs CV:** use k-fold for small datasets; holdout stratified split for large datasets.
* **Leakage:** ensure no label leakage — e.g., time series needs time-based split.
* **Overfitting hyperparams**: use nested CV if you tune heavily.
* **Test set**: never touch test set until final evaluation.
* **Reproducibility:** log random seeds, env, CUDA versions, package versions.

Resource management & efficiency:

* **Use shorter budgets early.**
* **Pruning**: early stop unpromising trials.
* **Parallelism**: run multiple cheap trials in parallel rather than one expensive.
* **Mixed precision**: improve throughput and memory to allow larger batch sizes.
* **Checkpointing**: resume failed runs, save top-K models.
* **Monitor GPU utilization**: avoid CPU/GPU bottlenecks (data pipeline often the limiter).

---

# 17. Example: Combined pipeline (Optuna + W\&B + OneCycle + AMP)

A compact integrated example:

```python
# combined_pipeline.py
import optuna, wandb, torch
from train_utils import SimpleNet, get_toy_data, evaluate
import torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

def objective(trial):
    wandb.init(reinit=True)
    config = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'hidden': trial.suggest_categorical('hidden', [32,64,128]),
        'n_layers': trial.suggest_int('n_layers',1,3),
        'batch': trial.suggest_categorical('batch', [32,64]),
    }
    wandb.config.update(config)
    train_dl, val_dl = get_toy_data(batch_size=config['batch'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet(hidden=config['hidden'], n_layers=config['n_layers']).to(device)
    opt = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    total_steps = len(train_dl)*5
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config['lr'], total_steps=total_steps)
    scaler = GradScaler()

    for epoch in range(5):
        model.train()
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                loss = nn.CrossEntropyLoss()(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sch.step()
        val_loss, val_acc = evaluate(model, val_dl, device)
        wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch})
        trial.report(val_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.TrialPruned()
    wandb.finish()
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=20)
    print("Best:", study.best_trial.params)
```

---

# 18. Reporting / Analysis checklist (practical)

* Save top-K checkpoints + corresponding hyperparams (W\&B artifacts or S3).
* Log learning curves, lr schedules, gradient norms.
* Produce hyperparam importance (Optuna: `study.trials_dataframe()`).
* For multi-objective, plot Pareto front (accuracy vs size vs latency).
* Generate final reproducibility script: env + seed + best config.

---

# 19. Final checklist & concise cheat-sheet

* Start: run **LR finder** → set max\_lr for OneCycle.
* Baseline: **Random search** (cheap).
* Optimize: **Optuna** (TPE + pruning).
* Scale: **Ray Tune** (PBT, distributed).
* NAS: Start with **Random NAS**, then try DARTS or existing NAS libraries for production.
* Multi-objective: **Optuna multi-objective** → NSGA-II (pymoo) for evolutionary search.
* Training: **OneCycle + AMP + GradScaler**.
* Track: **W\&B** for experiments; log everything.
* Efficiency: progressive budgets, pruning, mixed precision, data pipeline tuning.
* Validation: nested CV for heavy tuning; strict test set for final evaluation.

---

# 20. Wantable extras (I can produce on request)

* Full DARTS runnable skeleton (long).
* NSGA-II example using `pymoo` with full code.
* A Dockerized reproducible pipeline that runs Optuna + W\&B + resume/restore.
* A Ray Tune + W\&B production example on Azure / AWS with spot instances and autoscaling.

Pick one and I’ll give a full end-to-end script you can run immediately. If you want immediate priorities: tell me your **compute budget**, **dataset size**, and **goal(s)** (accuracy, latency, size) and I’ll convert this into a concrete pipeline with exact configs and expected run-time tradeoffs.
