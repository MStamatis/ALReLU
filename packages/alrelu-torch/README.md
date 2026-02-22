# alrelu-torch

PyPI-installable PyTorch package for the ALReLU activation with 2 variants:

1. `ALReLU` (default): fixed `alpha=0.01`
2. `TrainableALReLU`: trainable `alpha` parameter

Formula:

`ALReLU(x, alpha) = max(abs(alpha * x), x)`

## Reference

ALReLU paper:

`ALReLU: A different approach on Leaky ReLU activation function to improve Neural Networks Performance`

https://arxiv.org/abs/2012.07564

## Installation

After publishing:

```bash
pip install alrelu-torch
```

From source:

```bash
pip install .
```

## Usage

### 1) Functional API

```python
import torch
from alrelu_torch import alrelu

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = alrelu(x)  # alpha=0.01 by default
```

### 2) Fixed module

```python
import torch.nn as nn
from alrelu_torch import ALReLU

model = nn.Sequential(
    nn.Linear(32, 64),
    ALReLU(alpha=0.01),
    nn.Linear(64, 10),
)
```

### 3) Trainable alpha module

```python
import torch.nn as nn
from alrelu_torch import TrainableALReLU

model = nn.Sequential(
    nn.Linear(32, 64),
    TrainableALReLU(alpha_init=0.01, non_negative=True),
    nn.Linear(64, 10),
)
```

## Development

Install dev tools:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Build package:

```bash
python -m build
```
